import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dynsys.Dubins_car.Dubins_uncertain import DUBINS_UNCERTAIN
from dynsys.Dubins_car.Dubins_uncertain_sindy import DUBINS_UNCERTAIN_SINDY
import pickle

USE_CP = 1 # 1 or 0: whether to use conformal prediction
USE_ADAPTIVE = 1 # 1 or 0: whether to use adaptive control

# Load the SINDy model ##########################################################################
with open('sindy_models/model_dubins_car_sindy', 'rb') as file:
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
sim_T = 10 # Simulation time
tt = np.arange(0, sim_T, dt)

# System parameters
params = {
    "v": 1.0,
    "target": {"x": 2, "y": 4},
    "obstacle": {"x": 5, "y": 4, "r": 2},
    "cbf": {"rate": 1, "gamma": 15},
    "weight": {"input": 10000},
    "Kp": 10,  # P gain for the nominal controller
}
params["a_true"] = np.array([
    [0.05],
    [0.05], 
    [-0.05]
]) # true a(Theta)

# True system
Dubins_true = DUBINS_UNCERTAIN(params)

params["feature_names"] = feature_names
params["coefficients"] = coefficients
params["idx_x"] = idx_x
params["idx_u"] = idx_u
params["use_adaptive"] = USE_ADAPTIVE
params["Gamma_b"] = np.eye(3) * 0.01 # adaptive gain matrix for CRaCBF
#params["Gamma_L"] = np.eye(3) * 0.01 # adaptive gain matrix for CRaCLF
params["a_hat_norm_max"] = np.linalg.norm(np.array([[0.05], [0.05], [0.05]]), 2) # max norm of a_hat
params["a_0"] = np.array([[0.0], [0.0], [0.0]]) # initial guess for a_hat
params["epsilon"] = 1e-3 # small value for numerical stability of projection operator

# Learned mode
Dubins_learned = DUBINS_UNCERTAIN_SINDY(params)

# Sample initial states outside the obstacle
N = 10  # number of trajectories
x0 = np.vstack([
    np.random.rand(N) * 3 + 7,                  # x in [3, 10]
    np.random.rand(N) * 6 + 1,                  # y in [1, 7]
    np.random.rand(N) * 2 * np.pi - np.pi       # theta in [-pi, pi]
])
# Filter out states inside the obstacle (distance > 1.1 * params["obstacle"]["r"])
xo = params["obstacle"]["x"]
yo = params["obstacle"]["y"]
r = params["obstacle"]["r"]
mask = (x0[0, :] - xo)**2 + (x0[1, :] - yo)**2 > (r * 1.2)**2
x0 = x0[:, mask]
N = x0.shape[1]

# History arrays
x_hist = np.zeros((N, len(tt), 3))
u_hist = np.zeros((N, len(tt)))
h_hist = np.zeros((N, len(tt)))
if USE_ADAPTIVE:
    a_hat = np.zeros((N, len(tt), 3))
Sigma_score = 0

# Check if Gamma_b is valid
if USE_ADAPTIVE:
    set_tightening = 0.5 * Dubins_learned.a_err_max.T @ np.linalg.inv(Dubins_learned.Gamma_b) @ Dubins_learned.a_err_max
    if np.min(np.linalg.eigvals(Dubins_learned.Gamma_b)) < np.linalg.norm(Dubins_learned.a_err_max, 2)**2 / (2 * set_tightening):
        raise RuntimeError("Gamma_b is not valid: minimal eigenvalue is too small")

for n in range(N):
    x = np.copy(x0[:, n])

    if USE_ADAPTIVE:
        # Reset a_hat for each new trajectory
        Dubins_learned.a_b_hat = np.copy(params["a_0"])

        if Dubins_learned.acbf(x0[:, n], Dubins_learned.a_b_hat) - set_tightening <= 0:
            raise RuntimeError("Initial condition unsafe: h(x0, a_hat_0) < 0")
    else:
        if Dubins_learned.cbf(x0[:, n]) <= 0:
            raise RuntimeError("Initial condition unsafe: h(x0) < 0")

    for k in range(len(tt)):
        t = tt[k]
        x_hist[n, k, :] = x

        # Control
        u_ref = Dubins_learned.ctrl_nominal(x)
        if USE_ADAPTIVE:
            a_hat[n, k, :] = Dubins_learned.a_b_hat[:,0]
            u, h, feas = Dubins_learned.ctrl_cra_cbf_qp(x, u_ref, cp_quantile, dt)
            if h - set_tightening < 0:
                Sigma_score += 1
        else:
            #u, h, feas = Dubins_learned.ctrl_cbf_qp(x, u_ref)
            u, h, feas = Dubins_learned.ctrl_cr_cbf_qp(x, u_ref, cp_quantile)
            if h < 0: #TODO:
                Sigma_score += 1

        if not feas:
            raise RuntimeError("controller infeasible")

        u_hist[n, k] = u.item()
        h_hist[n, k] = h

        # Propagate dynamics
        dx = Dubins_true.dynamics(x, u)
        x = x + dx.reshape(-1) * dt

# Violation score
Sigma_score = Sigma_score / (N * len(tt)) * 100
print(f"Sigma_score = {Sigma_score:.3f} percent")

# === Plots ===
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(tt, x_hist[:, :, 0].T)
plt.ylabel("p_x")
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(tt, x_hist[:, :, 1].T)
plt.ylabel("p_y")
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(tt, x_hist[:, :, 2].T)
plt.ylabel("theta")
plt.grid(True)

plt.figure()
circle = plt.Circle((params["obstacle"]["x"], params["obstacle"]["y"]), params["obstacle"]["r"], color='r', fill=False, linewidth=2)
plt.gca().add_patch(circle)
plt.plot(params["target"]["x"], params["target"]["y"], 'go', markersize=10)
for n in range(N):
    plt.plot(x_hist[n, :, 0], x_hist[n, :, 1])
plt.axis('equal')
plt.grid(True)

plt.figure()
plt.plot(tt, u_hist.T)
plt.ylabel("Control Input: u (N)")
plt.grid(True)

if USE_ADAPTIVE:
    plt.figure()
    plt.subplot(4, 2, 1)
    plt.plot(tt, a_hat[:, :, 0].T)
    plt.axhline(Dubins_learned.a_err_max[0,0], color='r', linewidth=2)
    plt.axhline(-Dubins_learned.a_err_max[0,0], color='r', linewidth=2)
    plt.ylabel("a0_hat")
    plt.grid(True)
    plt.subplot(4, 2, 3)
    plt.plot(tt, a_hat[:, :, 1].T)
    plt.axhline(Dubins_learned.a_err_max[1,0], color='r', linewidth=2)
    plt.axhline(-Dubins_learned.a_err_max[1,0], color='r', linewidth=2)
    plt.ylabel("a1_hat")
    plt.grid(True)
    plt.subplot(4, 2, 5)
    plt.plot(tt, a_hat[:, :, 2].T)
    plt.axhline(Dubins_learned.a_err_max[2,0], color='r', linewidth=2)
    plt.axhline(-Dubins_learned.a_err_max[2,0], color='r', linewidth=2)
    plt.ylabel("a2_hat")
    plt.grid(True)
    plt.subplot(4, 2, 7)
    plt.plot(tt, np.linalg.norm(a_hat[:, :, :], axis=2, ord=2).T)
    plt.axhline(Dubins_learned.a_hat_norm_max + Dubins_learned.epsilon, color='r', linewidth=2)
    plt.ylabel("a_hat_norm")
    plt.grid(True)

    #plt.figure()
    plt.subplot(4, 2, 2)
    plt.plot(tt, a_hat[:, :, 0].T - params["a_true"][0,0])
    plt.axhline(Dubins_learned.a_err_max[0,0], color='r', linewidth=2)
    plt.axhline(-Dubins_learned.a_err_max[0,0], color='r', linewidth=2)
    plt.ylabel("a0 error")
    plt.grid(True)
    plt.subplot(4, 2, 4)
    plt.plot(tt, a_hat[:, :, 1].T - params["a_true"][1,0])
    plt.axhline(Dubins_learned.a_err_max[1,0], color='r', linewidth=2)
    plt.axhline(-Dubins_learned.a_err_max[1,0], color='r', linewidth=2)
    plt.ylabel("a1 error")
    plt.grid(True)
    plt.subplot(4, 2, 6)
    plt.plot(tt, a_hat[:, :, 2].T - params["a_true"][2,0])
    plt.axhline(Dubins_learned.a_err_max[2,0], color='r', linewidth=2)
    plt.axhline(-Dubins_learned.a_err_max[2,0], color='r', linewidth=2)
    plt.ylabel("a2 error")
    plt.grid(True)

plt.figure()
for n in range(N):
    h_plot = plt.plot(tt, h_hist[n, :], alpha=0.9)
if USE_ADAPTIVE:
    plt.axhline(set_tightening, color='r', linewidth=2)
else:
    plt.axhline(0, color='r', linewidth=2)
plt.ylabel("h(x_t)")
plt.xlabel("Time (s)")
plt.grid(True)
plt.show()