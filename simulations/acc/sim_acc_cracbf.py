import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dynsys.acc.acc import ACC

USE_CP = False # whether to use conformal prediction
USE_ADAPTIVE = False # whether to use adaptive control

# Parameters and Initialization
dt = 0.01
sim_T = 10
tt = np.arange(0, sim_T, dt)

params = {
    "v0": 14,
    "vd": 24,
    "m": 1650,
    "g": 9.81,
    "ca": 0.3,
    "cd": 0.3,
    "Kp": 100, # P gain for the nominal controller
    "T": 1.8,
    "cbf": {"rate": 5},
    "dt": dt
}
params["u_max"] = params["ca"] * params["m"] * params["g"] *1000
params["u_min"] = -params["cd"] * params["m"] * params["g"] *1000
params["use_adaptive"] = USE_ADAPTIVE
params["use_cp"] = USE_CP
params["Gamma_cbf"] = np.diag(np.array([0.5, 0.5, 0.5, 0.5])) # adaptive gain matrix for CRaCCM
params["a_true"] = np.array([[0.1], [5.0], [0.25], [0.0]]) # true parameters [theta1, theta2, theta3]
params["a_hat_norm_max"] = np.linalg.norm(params["a_true"], 2) # TODO: check this
params["a_0"] = np.array([[0.1], [5.0], [0.25], [0.0]]) # initial guess for a_hat
params["epsilon"] = 1e-2 # small value for numerical stability of projection operator
params["eta_cbf"] = 5.0
params["rho_cbf"] = 0.0

# Construct the system
acc_sys = ACC(params)

# Simulation
x0 = np.array([0.0, 20.0, 100.0])
x = x0.copy()

xs = np.zeros((len(tt), 3))
us = np.zeros(len(tt))
hs = np.zeros(len(tt))

for k in range(len(tt)):
    t = tt[k]
    xs[k, :] = x

    u_ref = acc_sys.ctrl_nominal(x)
    u = acc_sys.ctrl_cracbf(x, u_ref)
    us[k] = u.item()
    hs[k] = acc_sys.cbf(x, acc_sys.a_hat_cbf)

    dx = acc_sys.dynamics(x, u)
    x = x + dx.reshape(-1) * dt

# Plotting
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
