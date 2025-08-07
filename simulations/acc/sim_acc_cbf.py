import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dynsys.acc.acc import ACC

# === Parameters and Initialization ===
dt = 0.01
sim_T = 10
tt = np.arange(0, sim_T, dt)

params = {
    "v0": 14,
    "vd": 24,
    "m": 1650,
    "g": 9.81,
    "f0": 0.1,
    "f1": 5,
    "f2": 0.25,
    "ca": 0.3,
    "cd": 0.3,
    "T": 1.8,
    "clf": {"rate": 5},
    "cbf": {"rate": 5}
}

params["u_max"] = params["ca"] * params["m"] * params["g"]
params["u_min"] = -params["cd"] * params["m"] * params["g"]

params["weight"] = {
    "input": 2 / params["m"]**2,
    "slack": 2e-2
}

params["Kp"] = 100  # P gain for the nominal controller

# === ACC System Setup ===
acc_sys = ACC(params)

# === Simulation ===
x0 = np.array([0, 20, 100])
x = x0.copy()

xs = np.zeros((len(tt), 3))
us = np.zeros(len(tt))
hs = np.zeros(len(tt))

for k in range(len(tt)):
    t = tt[k]
    xs[k, :] = x

    u_ref = acc_sys.ctrl_nominal(x)
    u, h,_ = acc_sys.ctrl_cbf_qp(x, u_ref)
    us[k] = u.item()
    hs[k] = h

    dx = acc_sys.dynamics(x, u)
    x = x + dx.reshape(-1) * dt

# === Plotting Results ===
def plot_results(ts, xs, us, hs, params):
    plt.figure(figsize=(10, 15))
    plt.subplot(4, 1, 1)
    plt.plot(ts, xs[:, 1], color='blue', linewidth=1.5)
    plt.plot(ts, np.ones_like(ts) * params['vd'], 'k--')
    plt.ylabel("v (m/s)")
    plt.title("State - Velocity")
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(ts, xs[:, 2], color='magenta', linewidth=1.5)
    plt.ylabel("z (m)")
    plt.title("State - Distance to lead vehicle")
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(ts, us, color='orange', linewidth=1.5)
    plt.plot(ts, np.ones_like(us) * params['u_max'], 'k--')
    plt.plot(ts, np.ones_like(us) * params['u_min'], 'k--')
    plt.ylabel("u(N)")
    plt.title("Control Input - Wheel Force")
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(ts, hs, color='navy', linewidth=1.5)
    plt.ylabel("CBF (h(x))")
    plt.title("CBF")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_results(tt, xs, us, hs, params)
