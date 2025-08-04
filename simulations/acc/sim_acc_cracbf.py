import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dynsys.acc.acc_uncertain import ACC_UNCERTAIN
from dynsys.acc.acc_uncertain_sindy import ACC_UNCERTAIN_SINDY
import pickle

USE_CP = 1 # 1 or 0: whether to use conformal prediction
USE_ADAPTIVE = 1 # 1 or 0: whether to use adaptive control

# Load the SINDy model ##########################################################################
with open('sindy_models/model_acc_traj_sindy', 'rb') as file:
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
sim_T = 5 # Simulation time
tt = np.arange(0, sim_T, dt)

# System parameters
params = {
    "v0": 15,
    "vd": 20,
    "m": 2000,
    "f0": 0.5,
    "f1": 5.0,
    "f2": 1.0,
    "T": 1.0,
    "cbf": {"rate": 20},
    "weight": {"input": 2 / (2000**2)},#2 / (2000**2)
    "Kp": 10,  # P gain for the nominal controller
}

# True system
acc_true = ACC_UNCERTAIN(params)

params["feature_names"] = feature_names
params["coefficients"] = coefficients
params["idx_x"] = idx_x
params["idx_u"] = idx_u
params["use_adaptive"] = USE_ADAPTIVE
params["Gamma_b"] = np.diag([1,1,1]) * 1e-4 # adaptive gain matrix for CRaCBF
params["Gamma_L"] = np.eye(3) * 0.01 # adaptive gain matrix for CRaCLF
# True a: -np.array([[0.5], [5.0], [1.0]]) / 2000 * 1.5
params["a_hat_norm_max"] = np.linalg.norm(np.array([[0.5], [5.0], [1.0]])/2000 * 2.0, 2) # max norm of a_hat
params["a_0"] = np.array([[0.0], [0.0], [0.0]])# initial guess for a_hat
params["epsilon"] = 1e-3 # small value for numerical stability of projection operator

# Learned mode
acc_learned = ACC_UNCERTAIN_SINDY(params)

# Initial conditions
N = 1
rand_temp = np.random.rand(N)
x0 = np.vstack([
    rand_temp * 0,
    rand_temp * 21 + params["vd"],
    params["T"] * (rand_temp * 21 + params["vd"]) + np.random.rand(N) * 15
])

# History arrays
x_hist = np.zeros((N, len(tt), 3))
u_hist = np.zeros((N, len(tt)-1))
h_hist = np.zeros((N, len(tt)-1))
if USE_ADAPTIVE:
    a_hat = np.zeros((N, len(tt), 3))
Sigma_score = 0

# Check if Gamma_b is valid
if USE_ADAPTIVE:
    set_tightening = 0.5 * acc_learned.a_err_max.T @ np.linalg.inv(acc_learned.Gamma_b) @ acc_learned.a_err_max
    if np.min(np.linalg.eigvals(acc_learned.Gamma_b)) < np.linalg.norm(acc_learned.a_err_max, 2)**2 / (2 * set_tightening):
        raise RuntimeError("Gamma_b is not valid: minimal eigenvalue is too small")

for n in range(N):
    if USE_ADAPTIVE:
        if acc_learned.acbf(x0[:, n], acc_learned.a_b_hat) - set_tightening <= 0:
            raise RuntimeError("Initial condition unsafe: h(x0, a_hat_0) < 0")
    else:
        if acc_learned.cbf(x0[:, n]) <= 0:
            raise RuntimeError("Initial condition unsafe: h(x0) < 0")

    for k in range(len(tt)-1):
        if k == 0:
            x_hist[n, 0, :] = x0[:, n]

        t = tt[k]
        x = x_hist[n, k, :]

        # Control
        u_ref = acc_learned.ctrl_nominal(x)
        if USE_ADAPTIVE:
            a_hat[n, k, :] = acc_learned.a_b_hat[:,0]
            u, h, feas = acc_learned.ctrl_cra_cbf_qp(x, u_ref, cp_quantile, dt)
            if h - set_tightening < 0:
                Sigma_score += 1
        else:
            #u, h, feas = acc_learned.ctrl_cbf_qp(x, u_ref)
            u, h, feas = acc_learned.ctrl_cr_cbf_qp(x, u_ref, cp_quantile)
            if h < 0: #TODO:
                Sigma_score += 1

        if not feas:
            raise RuntimeError("controller infeasible")

        u_hist[n, k] = u.item()
        h_hist[n, k] = h

        # Propagate dynamics
        dx = acc_true.dynamics(x, u)
        x_hist[n, k+1, :] = x + dx.reshape(-1) * dt

# Violation score
Sigma_score = Sigma_score / (N * len(tt) - 1) * 100
print(f"Sigma_score = {Sigma_score:.3f} percent")

# Save results
"""
saved_trajectories = {
    "time": tt,
    "dt": dt,
    "x_hist": x_hist,
    "u_hist": u_hist,
    "h_hist": h_hist,
}
filename = "simulations/acc/acc_crcbf_trajectories" if USE_CP else "simulations/acc/acc_cbf_trajectories"
with open(filename, "wb") as f:
    pickle.dump(saved_trajectories, f)
"""

# === Plots ===
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(tt, x_hist[:, :, 1].T)
plt.axhline(params["vd"], color='k', linestyle='--')
plt.axhline(params["v0"], color='b', linestyle='--')
plt.ylabel("v (m/s)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(tt, x_hist[:, :, 2].T)
plt.ylabel("z (m)")
plt.grid(True)

plt.figure()
plt.plot(tt[:-1], u_hist.T)
plt.ylabel("Control Input: u (N)")
plt.grid(True)

if USE_ADAPTIVE:
    plt.figure()
    plt.subplot(4, 2, 1)
    plt.plot(tt, a_hat[:, :, 0].T)
    plt.axhline(acc_learned.a_err_max[0,0], color='r', linewidth=2)
    plt.axhline(-acc_learned.a_err_max[0,0], color='r', linewidth=2)
    plt.ylabel("a0_hat")
    plt.grid(True)
    plt.subplot(4, 2, 3)
    plt.plot(tt, a_hat[:, :, 1].T)
    plt.axhline(acc_learned.a_err_max[1,0], color='r', linewidth=2)
    plt.axhline(-acc_learned.a_err_max[1,0], color='r', linewidth=2)
    plt.ylabel("a1_hat")
    plt.grid(True)
    plt.subplot(4, 2, 5)
    plt.plot(tt, a_hat[:, :, 2].T)
    plt.axhline(acc_learned.a_err_max[2,0], color='r', linewidth=2)
    plt.axhline(-acc_learned.a_err_max[2,0], color='r', linewidth=2)
    plt.ylabel("a2_hat")
    plt.grid(True)
    plt.subplot(4, 2, 7)
    plt.plot(tt, np.linalg.norm(a_hat[:, :, :], axis=2, ord=2).T)
    plt.axhline(acc_learned.a_hat_norm_max + acc_learned.epsilon, color='r', linewidth=2)
    plt.ylabel("a_hat_norm")
    plt.grid(True)

    #plt.figure()
    plt.subplot(4, 2, 2)
    plt.plot(tt, a_hat[:, :, 0].T + params["f0"]/params["m"])
    plt.axhline(acc_learned.a_err_max[0,0], color='r', linewidth=2)
    plt.axhline(-acc_learned.a_err_max[0,0], color='r', linewidth=2)
    plt.ylabel("a0 error")
    plt.grid(True)
    plt.subplot(4, 2, 4)
    plt.plot(tt, a_hat[:, :, 1].T + params["f1"]/params["m"])
    plt.axhline(acc_learned.a_err_max[1,0], color='r', linewidth=2)
    plt.axhline(-acc_learned.a_err_max[1,0], color='r', linewidth=2)
    plt.ylabel("a1 error")
    plt.grid(True)
    plt.subplot(4, 2, 6)
    plt.plot(tt, a_hat[:, :, 2].T + params["f2"]/params["m"])
    plt.axhline(acc_learned.a_err_max[2,0], color='r', linewidth=2)
    plt.axhline(-acc_learned.a_err_max[2,0], color='r', linewidth=2)
    plt.ylabel("a2 error")
    plt.grid(True)

plt.figure()
for n in range(N):
    h_plot = plt.plot(tt[:-1], h_hist[n, :], alpha=0.9)
if USE_ADAPTIVE:
    plt.axhline(set_tightening, color='r', linewidth=2)
else:
    plt.axhline(0, color='r', linewidth=2)
plt.ylabel("h(x_t)")
plt.xlabel("Time (s)")
plt.grid(True)
plt.show()