import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dynsys.inverted_pendulum.ip import IP
from dynsys.inverted_pendulum.ip_sindy import IP_SINDY
from dynsys.utils import wrapToPi
import pickle

USE_CP = 1 # 1 or 0: whether to use conformal prediction

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
sim_T = 5
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

# Instantiate models (must be provided elsewhere)
ip_learned = IP_SINDY(params)

ip_true = IP(params)

# Sample initial states from a level set of the CLF
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
x0 = np.array(x0_list) if len(x0_list) > 0 else np.zeros((0, 2))

# Plot the CLF field and the sampled initial states
plt.figure()
X, Y = np.meshgrid(x_, y_)
plt.contourf(X, Y, V_.T, levels=20)  # MATLAB uses V_' (transpose)
plt.colorbar()
marker_size = 10
if x0.shape[0] > 0:
    plt.scatter(x0[:, 0], x0[:, 1], s=marker_size, c=[[1, 0, 0]], marker='o')
plt.xlabel("theta (rad)")
plt.ylabel("theta dot (rad/s)")

# Sample around the level set
N = 3  # number of paths
N = min(N, x0.shape[0])
perm = np.random.permutation(len(x0))
x0 = x0[perm, :]
x0 = x0[:N, :]

# ========= Run simulation =========
dt = 0.005
T = 5
tt = np.arange(0, T + dt/2, dt)  # include T

# Time histories
xdim = ip_true.xdim
x_hist = np.zeros((N, len(tt), xdim))
x_norm_hist = np.zeros((N, len(tt)))
u_hist = np.zeros((N, len(tt)))
V_hist = np.zeros((N, len(tt)))
slack_hist = np.zeros((N, len(tt)))

Sigma_score = 0  # violation score

# Main simulation loops
for n in range(N):
    x = np.copy(x0[n, :])

    for k in range(len(tt)):
        # Log time hisotry
        x_hist[n, k, :] = x
        x_hist[n, k, 0] = wrapToPi(x_hist[n, k, 0]) # Crucial
        x_norm_hist[n, k] = np.linalg.norm(x)

        # Control
        u_ref = ip_learned.ctrl_nominal(x)
        #u, V, slack_val, feas = ip_learned.ctrl_clf_qp(x, u_ref, with_slack=True)
        u, V, slack_val, feas = ip_learned.ctrl_cr_clf_qp(x, u_ref, cp_quantile, with_slack=True)

        if not feas:
            raise RuntimeError("controller infeasible")

        u_hist[n, k] = u.item()
        V_hist[n, k] = V
        if slack_val is None:
            slack_val = 0.0
        slack_hist[n, k] = slack_val

        # Compute Sigma_score
        if V > V0 * np.exp(-params['clf']['rate'] * tt[k]):
            Sigma_score += 1

        # Propagate dynamics
        dx = ip_true.dynamics(x, u)
        x = x + dx.reshape(-1) * dt

# Violation score
Sigma_score = Sigma_score / (N * len(tt)) * 100.0
print(f"Sigma_score = {Sigma_score:6.3f} percent")

# ========= Plots =========
# States
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(tt, x_hist[:, :, 0].T)
plt.ylabel(r"$\theta$ (rad)")
plt.xlabel("Time (s)")
plt.grid(True)
plt.gca().tick_params(labelsize=18)
plt.subplot(2, 1, 2)
plt.plot(tt, x_hist[:, :, 1].T)
plt.xlabel("Time (s)")
plt.ylabel(r"$\dot{\theta}$ (rad/s)")
plt.grid(True)
plt.gca().tick_params(labelsize=18)

# Trajectories over CLF field
plt.figure()
plt.contourf(X, Y, V_.T, levels=20)
for n in range(N):
    plt.plot(x_hist[n, :, 0], x_hist[n, :, 1], linewidth=1.5)
plt.xlabel("theta (rad)")
plt.ylabel("theta dot (rad/s)")

# Control and slack histories
plt.figure()
plt.plot(tt, u_hist.T)
plt.xlabel("Time (s)")
plt.ylabel("Control: ut")
plt.grid(True)

plt.figure()
plt.plot(tt, slack_hist.T)
plt.xlabel("Time (s)")
plt.ylabel("QP slack")
plt.grid(True)

# State norm
plt.figure()
plt.plot(tt, x_norm_hist.T)
plt.xlabel("Time (s)")
plt.ylabel("State norm: ||x||")
plt.grid(True)

# V(x) over time with exponential bound
plt.figure()
plt.plot(tt, V_hist.T)
plt.plot(tt, V0 * np.exp(-params['clf']['rate'] * tt), 'r--', linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("V(x_t)")
plt.grid(True)
plt.gca().tick_params(labelsize=18)

plt.show()