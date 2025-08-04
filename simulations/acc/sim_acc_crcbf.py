import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dynsys.acc.acc import ACC
from dynsys.acc.acc_sindy import ACC_SINDY
import pickle

USE_CP = 1 # 1 or 0: whether to use conformal prediction

# Load the SINDy model ##########################################################################
with open('sindy_models/model_acc_traj_sindy', 'rb') as file:
    model = pickle.load(file)

feature_names = model["feature_names"]
n_features = len(feature_names)
for i in range(n_features):
    feature_names[i] = feature_names[i].replace(" ", "*")
    feature_names[i] = feature_names[i].replace("^", "**")
    feature_names[i] = feature_names[i].replace("sin", "torch.sin")
    feature_names[i] = feature_names[i].replace("cos", "torch.cos")

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

# System parameters
params = {
    "v0": 15,
    "vd": 20,
    "m": 2000,
    "f0": 0.5,
    "f1": 5.0,
    "f2": 1.0,
    "T": 1.0,
    "cbf": {"rate": 2},
    "weight": {"input": 2 / (2000**2)},
    "feature_names": feature_names,
    "coefficients": coefficients,
    "idx_x": idx_x,
    "idx_u": idx_u,
}

params["Kp"] = 100  # P gain for the nominal controller

# Learned mode
acc_learned = ACC_SINDY(params)

# True system
acc_true = ACC(params)

# Initial conditions
N = 5
rand_temp = np.random.rand(N)
x0 = np.vstack([
    rand_temp * 0,
    rand_temp * 20.8 + params["vd"],
    params["T"] * (rand_temp * 20.8 + params["vd"]) + np.random.rand(N) * 0.3
])

# History arrays
x_hist = np.zeros((N, len(tt), 3))
u_hist = np.zeros((N, len(tt)-1))
h_hist = np.zeros((N, len(tt)-1))

Sigma_score = 0

for n in range(N):
    for k in range(len(tt)-1):
        if k == 0:
            x_hist[n, 0, :] = x0[:, n]

        t = tt[k]
        x = x_hist[n, k, :]

        # Control
        u_ref = acc_learned.ctrl_nominal(x)
        #u, h, feas = acc_learned.ctrl_cbf_qp(x, u_ref)
        u, h, feas = acc_learned.ctrl_cr_cbf_qp(x, u_ref, cp_quantile)

        if not feas:
            raise RuntimeError("controller infeasible")

        u_hist[n, k] = u.item()
        h_hist[n, k] = h

        if h < 0:
            Sigma_score += 1

        # Propagate dynamics
        dx = acc_true.dynamics(x, u)
        x_hist[n, k+1, :] = x + dx.reshape(-1) * dt

# Violation score
Sigma_score = Sigma_score / (N * len(tt) - 1) * 100
print(f"Sigma_score = {Sigma_score:.3f} percent")

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

plt.figure()
for n in range(N):
    h_plot = plt.plot(tt[:-1], h_hist[n, :], alpha=0.9)
plt.axhline(0, color='r', linewidth=2)
plt.ylabel("h(x_t)")
plt.xlabel("Time (s)")
plt.ylim([-0.05, None])
plt.grid(True)
plt.show()