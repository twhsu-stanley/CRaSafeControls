import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dynsys.inverted_pendulum.inverted_pendulum import INVERTED_PENDULUM

# === Parameters and Initialization ===
dt = 0.01
sim_T = 5
tt = np.arange(0, sim_T, dt)

params = {
    "l": 1.0,    # length of pendulum [m]
    "m": 1.0,    # mass [kg]
    "g": 9.81,   # gravity [m/s^2]
    "b": 0.01,   # friction [s*Nm/rad]
    "u_max": 7.0,
    "u_min": -7.0,
    "Kp": 8,
    "Kd": 5,
    "clf": {"rate": 0.5},
    "weight": {"slack": 1e5}
}
params["I"] = params["m"] * params["l"]**2 / 3

# === Inverted Pendulum System Setup ===
ip_sys = INVERTED_PENDULUM(params)

# === Simulation ===
total_steps = int(np.ceil(sim_T / dt))
x0 = np.array([0.76, 0.05])
x = x0.copy()
t = 0.0

xs = np.zeros((total_steps, 2))
us = np.zeros(total_steps - 1)
Vs = np.zeros(total_steps - 1)

xs[0, :] = x0

for k in range(total_steps - 1):
    u, V, slack_val, feas = ip_sys.ctrl_clf_qp(x, u_ref=None, with_slack=True)

    if not feas:
        print(f"Controller infeasible at step {k}, time {t:.2f}s")
        u = ip_sys.ctrl_nominal(x)
        #raise RuntimeError("controller infeasible")
    
    us[k] = u.item()
    Vs[k] = V

    dx = ip_sys.dynamics(x, u)
    x = x + dx.reshape(-1) * dt # WARNING: must not use x += dx.reshape(-1) * dt
    xs[k + 1, :] = x

# === Plotting Results ===
plt.figure(figsize=(10, 12))
plt.subplot(2, 1, 1)
plt.plot(tt, 180 * xs[:, 0] / np.pi, linewidth=1.5)
plt.ylabel("theta (deg)")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(tt, 180 * xs[:, 1] / np.pi, linewidth=1.5)
plt.ylabel("theta dot (deg/s)")
plt.xlabel("Time (s)")
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(tt[:-1], Vs)
plt.ylabel("V (CLF)")
plt.xlabel("Time (s)")
plt.title("CLF Value")
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(tt[:-1], us, linewidth=1.5)
plt.plot(tt[:-1], np.ones_like(us) * params["u_max"], 'k--')
plt.plot(tt[:-1], np.ones_like(us) * params["u_min"], 'k--')
plt.ylabel("u (N.m)")
plt.title("Control Input")
plt.xlabel("Time (s)")
plt.grid(True)

plt.tight_layout()
plt.show()
