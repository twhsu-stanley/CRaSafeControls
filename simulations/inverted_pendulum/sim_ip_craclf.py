import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.integrate import solve_ivp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dynsys.inverted_pendulum.ip import IP

USE_CP = False # whether to use conformal prediction
USE_ADAPTIVE = True # whether to use adaptive control

# Parameters and Initialization
dt = 0.01
sim_T = 25
tt = np.arange(0, sim_T, dt)

params = {
    "l": 1.0,    # length of pendulum [m]
    "m": 1.0,    # mass [kg]
    "grav": 9.81,   # gravity [m/s^2]
    "b": 0.01,   # friction [s*Nm/rad]
    "u_max": 7.0,
    "u_min": -7.0,
    "Kp": 8.0,
    "Kd": 5.0,
    "clf": {"rate": 0.8},
    "weight_slack": 100,
}
params["use_adaptive"] = USE_ADAPTIVE
params["use_cp"] = USE_CP
params["Gamma_clf"] = np.diag(np.array([4.0, 4.0, 4.0])) # adaptive gain matrix for CRaCCM
params["a_true"] = np.array([[0.4], [0.1], [0.1]]) # true parameters
params["a_hat_norm_max"] = np.linalg.norm(params["a_true"], 2) # TODO: check this
params["epsilon"] = 1e-2 # small value for numerical stability of projection operator
params["eta_clf"] = 10.0

# Construct the system
ip = IP(params)

# Disturbance (non-parametric uncertainty)
Delta = lambda t: np.array([
    0.0,#0.04 + 0.05 * np.sin(2 * np.pi / 0.3 * t),
    0.0,#0.1 + 0.05 * np.cos(2 * np.pi / 0.5 * t + 0.03),
], dtype = float)

# Compute upper bound of Delta
Delta_max = np.max([np.linalg.norm(Delta(t), 2) for t in np.arange(0.0, sim_T, 0.01)])
ip.cp_quantile = Delta_max * 0.95

# Simulation
x0 = np.array([0.05, 0.0])
x = x0.copy()
a_hat_clf = np.array([[0.0], [0.0], [0.0]]) # initial guess for a_hat
rho_clf = 0.0
x_ext = np.hstack((x, a_hat_clf.ravel(), rho_clf)) # extended state with a_hat and rho

x_hist = np.zeros((len(tt), 2))
u_hist = np.zeros(len(tt))
slack_hist = np.zeros(len(tt))
Vc_hist = np.zeros(len(tt))
V_hist = np.zeros(len(tt))
a_hat_clf_hist = np.zeros((ip.adim, len(tt)))
a_true_hist = np.zeros((ip.adim, len(tt)))
nu_clf_hist = np.zeros((len(tt),))
rho_clf_hist = np.zeros((len(tt),))

for k in range(len(tt)):
    t = tt[k]
    print("Time: ", t)

    x_hist[k, :] = x

    # Store adaptation parameters
    a_hat_clf_hist[:, k] = a_hat_clf.ravel()
    a_true_hist[:, k] = ip.a_true.ravel()
    nu_clf_hist[k] = ip.nu_clf(rho_clf)
    rho_clf_hist[k] = rho_clf

    u_ref = ip.ctrl_nominal(x)
    u, slack = ip.ctrl_craclf(x, a_hat_clf, u_ref, use_slack=False)
    
    u_hist[k] = u.item()
    slack_hist[k] = slack
    Vc_hist[k] = ip.clf(x, a_hat_clf).item()
    a_tilde = a_hat_clf - ip.a_true
    V_hist[k] = (ip.nu_clf(rho_clf) * (ip.clf(x, a_hat_clf).item() + ip.eta_clf) + 0.5 * a_tilde.T @ np.linalg.inv(ip.Gamma_clf) @ a_tilde).item()

    # Propagate with zero-order hold on control and disturbance
    if k < len(tt) - 1:
        t_span = (tt[k], tt[k + 1])
        
        sol = solve_ivp(
            #lambda t, y: ip.dynamics(y, u),
            lambda t, y: ip.dynamics_extended(y, u),
            t_span,
            x_ext,
            method = "BDF", #"LSODA", #"Radau",  # stiff solver
            rtol = 1e-6, # 1e-6
            atol = 1e-6, # 1e-6
            t_eval = [tt[k + 1]],
        )
        try:
            x_ext = sol.y[:, -1]
        except Exception as e:
            raise ValueError("Error occurred while solving IVP:", e)
        
        x = x_ext[0:ip.xdim]
        a_hat_clf = x_ext[ip.xdim:(ip.xdim+ip.adim)].reshape(-1,1)
        rho_clf = x_ext[(ip.xdim+ip.adim)]

# Plotting
plt.figure()
plt.subplot(2,1,1)
plt.plot(tt, 180 * x_hist[:, 0] / np.pi, linewidth=1.5)
plt.ylabel("theta (deg)")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(tt, 180 * x_hist[:, 1] / np.pi, linewidth=1.5)
plt.ylabel("theta dot (deg/s)")
plt.xlabel("Time (s)")
plt.grid(True)

plt.figure()
plt.subplot(2,1,1)
plt.plot(tt, Vc_hist)
plt.ylabel("Vc")
plt.title("Lyapunov Functions")
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(tt, V_hist)
plt.ylabel("V (composite)")
plt.xlabel("Time (s)")
plt.grid(True)

plt.figure()
plt.subplot(2,1,1)
plt.plot(tt, u_hist, linewidth=1.5)
plt.plot(tt, np.ones_like(u_hist) * params["u_max"], 'k--')
plt.plot(tt, np.ones_like(u_hist) * params["u_min"], 'k--')
plt.grid(True)
plt.ylabel("u (N.m)")
plt.title("Control Input")
plt.subplot(2,1,2)
plt.plot(tt, slack_hist, linewidth=1.5)
plt.ylabel("slack")
plt.xlabel("Time (s)")
plt.grid(True)

# Uncertainty parameters
fig, axs = plt.subplots(ip.adim, 1)
axs = axs.flatten()
for i in range(ip.adim):
    axs[i].plot(tt, a_hat_clf_hist[i, :], label='a_hat')
    axs[i].plot(tt, a_true_hist[i, :], label='a_true')
    axs[i].legend()
    axs[i].grid(True)
    axs[i].set_ylabel(f"a{i}")
axs[ip.adim-1].set_xlabel('Time (s)')

# Scaling function and parameter
fig, axs = plt.subplots(2, 1)
axs = axs.flatten()
axs[0].plot(tt, nu_clf_hist, lw=1)
axs[0].set_ylabel('nu_clf')
axs[0].grid(True)
axs[1].plot(tt, rho_clf_hist, lw=1)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('rho_clf')
axs[1].grid(True)

plt.tight_layout()
plt.show()
