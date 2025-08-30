import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dynsys.inverted_pendulum.ip_uncertain import IP_UNCERTAIN
from dynsys.inverted_pendulum.ip_uncertain_sindy import IP_UNCERTAIN_SINDY
from dynsys.utils import wrapToPi
import pickle

USE_CP = 1 # 1 or 0: whether to use conformal prediction
USE_ADAPTIVE = 1 # 1 or 0: whether to use adaptive control

# Load the SINDy model ##########################################################################
with open('sindy_models/model_inverted_pendulum_traj_sindy', 'rb') as file:
    model = pickle.load(file)

feature_names = model["feature_names"]
n_features = len(feature_names)
for i in range(n_features):
    feature_names[i] = feature_names[i].replace(" ", "*")
    feature_names[i] = feature_names[i].replace("^", "**")
    feature_names[i] = feature_names[i].replace("sin", "np.sin")
    feature_names[i] = feature_names[i].replace("cos", "np.cos")

coefficients = model["coefficients"]

idx_x = [] # Indices for f(x)
idx_u = [] # Indices for g(x)*u
for i in range(len(feature_names)):
    if 'u0' in feature_names[i]:
        idx_u.append(i)
    else:
        idx_x.append(i)
        
cp_quantile = model["model_error"]['quantile']
print("cp_quantile = ", cp_quantile)

cp_quantile = cp_quantile * USE_CP # setting cp_quantile = 0 is equivalent to using the regular cbf

# Time setup
dt = 0.01
sim_T = 10
tt = np.arange(0, sim_T, dt)

# ========= Set up the learned and true models =========
params = {
    "l": 1.0,    # length of pendulum [m]
    "m": 1.0,    # mass [kg]
    "g": 9.81,   # gravity [m/s^2]
    "b": 0.01,   # friction [s*Nm/rad]
    #"u_max": 7.0,
    #"u_min": -7.0,
    "Kp": 8,
    "Kd": 5,
    "clf": {"rate": 0.5},
    "weight": {"slack": 500},
    "feature_names": feature_names,
    "coefficients": coefficients,
    "idx_x": idx_x,
    "idx_u": idx_u,
}
params["I"] = params["m"] * params["l"]**2 / 3

params["a_true"] = np.array([
    [(params["m"]*params['g']*params["l"]/2/params["I"])*0.5],
    [-params['b']/params["I"]*0.5]
]) # true a(Theta)

ip_true = IP_UNCERTAIN(params)

params["use_adaptive"] = USE_ADAPTIVE
params["Gamma_L"] = np.diag([1,1]) * 1e-4 # adaptive gain matrix for CRaCLF
# True a: -np.array([[0.5], [5.0], [1.0]]) / 2000 * 1.5
params["a_hat_norm_max"] = np.linalg.norm(np.array([[3.0], [3.0]]), 2) # max norm of a_hat
params["a_0"] = np.array([[0.0], [0.0]])# initial guess for a_hat
params["epsilon"] = 1e-3 # small value for numerical stability of projection operator

ip_learned = IP_UNCERTAIN_SINDY(params)

# ====== Sample initial states from a level set of the CLF ======
# Create a grid of states
resolution = 200
x_ = np.linspace(-np.pi/2, np.pi/2, resolution)
y_ = np.linspace(-np.pi * 3, np.pi * 3, resolution)

state = np.zeros((resolution, resolution, 2))
state_norm_square = np.zeros((resolution, resolution))
V_ = np.zeros((resolution, resolution))
gradV_ = np.zeros((resolution, resolution, 2))

for i in range(resolution):
    for j in range(resolution):
        state_norm_square[i, j] = x_[i] ** 2 + y_[j] ** 2
        if USE_ADAPTIVE:
            V_[i, j] = ip_learned.aclf(np.array([x_[i], y_[j]]), np.array([0,0]))
            gradV_[i, j, :] = ip_learned.daclfdx(np.array([x_[i], y_[j]]), np.array([0,0])).reshape(-1)
            #TODO: check a_hat value in aclf and daclfdx
        else:
            V_[i, j] = ip_learned.clf(np.array([x_[i], y_[j]]))
            gradV_[i, j, :] = ip_learned.dclfdx(np.array([x_[i], y_[j]])).reshape(-1)

# Find ROA of CLF (min across domain edges)
clf_level = np.min(
    np.concatenate([V_[0, :], V_[-1, :], V_[:, 0], V_[:, -1]])
)
V0 = clf_level

# Collect states near the CLF level set as initial conditions
x0_list = []
for i in range(resolution):
    for j in range(resolution):
        if (V_[i, j] <= clf_level) and (V_[i, j] >= clf_level - 0.01):
            x0_list.append([x_[i], y_[j]])
if len(x0_list) > 0:
    x0_list = np.array(x0_list)  
else:
    raise ValueError("No initial states found near the CLF level set")

# Sample around the level set
N = 2  # number of trajectories to be simulated
N = min(N, x0_list.shape[0])
perm = np.random.permutation(x0_list.shape[0])
x0_list = x0_list[perm, :]
x0 = x0_list[:N, :]

# ========= Run simulation =========
dt = 0.01
T = 5
tt = np.arange(0, T + dt/2, dt)  # include T

# Time histories
xdim = ip_true.xdim
x_hist = np.zeros((N, len(tt), xdim))
x_norm_hist = np.zeros((N, len(tt)))
u_hist = np.zeros((N, len(tt)))
V_hist = np.zeros((N, len(tt)))
if USE_ADAPTIVE:
    a_hat = np.zeros((N, len(tt), 2))
slack_hist = np.zeros((N, len(tt)))

# Main simulation loops
for n in range(N):
    x = np.copy(x0[n, :])

    if USE_ADAPTIVE:
        # Reset a_hat for each new trajectory
        ip_learned.a_L_hat = np.copy(params["a_0"])

    for k in range(len(tt)):
        # Log time hisotry
        x_hist[n, k, :] = x
        x_hist[n, k, 0] = wrapToPi(x_hist[n, k, 0]) # Crucial
        x_norm_hist[n, k] = np.linalg.norm(x)

        # Control
        u_ref = ip_learned.ctrl_nominal(x)
        if USE_ADAPTIVE:
            a_hat[n, k, :] = ip_learned.a_L_hat[:,0]
            u, V, slack_val, feas = ip_learned.ctrl_cra_clf_qp(x, u_ref, cp_quantile, dt, with_slack=True)
        else:
            u, V, slack_val, feas = ip_learned.ctrl_clf_qp(x, u_ref, with_slack=True)

        if not feas:
            raise RuntimeError("controller infeasible")

        u_hist[n, k] = u.item()
        V_hist[n, k] = V
        if slack_val is None:
            slack_val = 0.0
        slack_hist[n, k] = slack_val

        # Propagate dynamics
        dx = ip_true.dynamics(x, u)
        x = x + dx.reshape(-1) * dt

# ========= Plots =========
# States
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(tt, x_hist[:, :, 0].T)
plt.ylabel(r"$\theta$ (rad)")
#plt.xlabel("Time (s)")
plt.grid(True)
#plt.gca().tick_params(labelsize=12)
plt.subplot(3, 1, 2)
plt.plot(tt, x_hist[:, :, 1].T)
#plt.xlabel("Time (s)")
plt.ylabel(r"$\dot{\theta}$ (rad/s)")
plt.grid(True)
#plt.gca().tick_params(labelsize=12)
plt.subplot(3, 1, 3)
plt.plot(tt, x_norm_hist.T)
plt.xlabel("Time (s)")
plt.ylabel("State norm: ||x||")
plt.grid(True)
#plt.gca().tick_params(labelsize=12)

# Trajectories over CLF field
# Plot the CLF field and the sampled initial states
plt.figure()
X, Y = np.meshgrid(x_, y_)
plt.contourf(X, Y, V_.T, levels=20)
plt.colorbar()
plt.scatter(x0_list[:, 0], x0_list[:, 1], s=10, c=[[0, 1, 0]], marker='o')
plt.scatter(x0[:, 0], x0[:, 1], s=10, c=[[1, 0, 0]], marker='o')
for n in range(N):
    plt.plot(x_hist[n, :, 0], x_hist[n, :, 1], linewidth=1.5)
plt.xlabel("theta (rad)")
plt.ylabel("theta dot (rad/s)")
plt.title("Samples from the ROA Level Set")

# Control and slack histories
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(tt, u_hist.T)
#plt.xlabel("Time (s)")
plt.ylabel("Control: ut")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(tt, slack_hist.T)
plt.xlabel("Time (s)")
plt.ylabel("QP slack")
plt.grid(True)

# V(x) over time with exponential bound
plt.figure()
plt.plot(tt, V_hist.T)
#plt.plot(tt, V0 * np.exp(-params['clf']['rate'] * tt), 'r--', linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("V(x_t)")
plt.grid(True)
plt.gca().tick_params(labelsize=18)

if USE_ADAPTIVE:
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(tt, a_hat[:, :, 0].T)
    plt.axhline(ip_learned.a_err_max[0,0], color='r', linewidth=2)
    plt.axhline(-ip_learned.a_err_max[0,0], color='r', linewidth=2)
    plt.ylabel("a0_hat")
    plt.grid(True)
    plt.subplot(3, 2, 3)
    plt.plot(tt, a_hat[:, :, 1].T)
    plt.axhline(ip_learned.a_err_max[1,0], color='r', linewidth=2)
    plt.axhline(-ip_learned.a_err_max[1,0], color='r', linewidth=2)
    plt.ylabel("a1_hat")
    plt.grid(True)
    plt.subplot(3, 2, 5)
    plt.plot(tt, np.linalg.norm(a_hat[:, :, :], axis=2, ord=2).T)
    plt.axhline(ip_learned.a_hat_norm_max + ip_learned.epsilon, color='r', linewidth=2)
    plt.ylabel("a_hat_norm")
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(tt, a_hat[:, :, 0].T - params["a_true"][0,0])
    plt.axhline(ip_learned.a_err_max[0,0], color='r', linewidth=2)
    plt.axhline(-ip_learned.a_err_max[0,0], color='r', linewidth=2)
    plt.ylabel("a0 error")
    plt.grid(True)
    plt.subplot(3, 2, 4)
    plt.plot(tt, a_hat[:, :, 1].T - params["a_true"][1,0])
    plt.axhline(ip_learned.a_err_max[1,0], color='r', linewidth=2)
    plt.axhline(-ip_learned.a_err_max[1,0], color='r', linewidth=2)
    plt.ylabel("a1 error")
    plt.grid(True)
    
plt.show()