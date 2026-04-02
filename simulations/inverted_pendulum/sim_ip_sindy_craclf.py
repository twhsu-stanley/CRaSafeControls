import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.integrate import solve_ivp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from systems.inverted_pendulum.ip_sindy import IP_SINDY
from utils import wrapToPi
import pickle

USE_CP = False # whether to use conformal prediction
USE_ADAPTIVE = True # whether to use adaptive control

# Load the SINDy model #######################################################
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
###############################################################################

# Time setup
dt = 0.01
sim_T = 10
tt = np.arange(0, sim_T, dt)

# Prior knowledge of the uncertainty parameter
a_true = np.array([0.4, 0.1, 0.1]) # unknown to the controller
a_ub = np.array([0.8, 0.8, 0.8])
a_lb = np.array([-0.8, -0.8, -0.8])

# System parameters
params = {
    "l": 1.0,     # length of pendulum [m]
    "m": 1.0,     # mass [kg]
    "grav": 9.81, # gravity [m/s^2]
    "b": 0.01,    # friction [s*Nm/rad]
    "u_max": 7.0,
    "u_min": -7.0,
    "Kp": 8.0,
    "Kd": 5.0,
    "clf": {"rate": 0.8},
    "weight_slack": 100,
}
params["use_adaptive"] = USE_ADAPTIVE
params["use_cp"] = USE_CP
params["cp_quantile"] = cp_quantile
params["Gamma_clf"] = np.diag(np.array([4.0, 4.0, 4.0]))
params["a_true"] = a_true
params["a_ub"] = a_ub
params["a_lb"] = a_lb
params["a_hat_norm_max"] = 0.5 * np.linalg.norm(a_ub - a_lb, ord=2) * 1.5
params["epsilon"] = 1e-2 # small value for numerical stability of projection operator
params["eta_clf"] = 10.0

# For SINDy model
params["feature_names"] = feature_names
params["coefficients"] = coefficients
params["idx_x"] = idx_x
params["idx_u"] = idx_u

ip_sindy = IP_SINDY(params)

# Simulation
x0 = np.array([0.05, 0.0])
x = x0.copy()
a_hat_clf = np.array([0.0, 0.0, 0.0]) # initial guess for a_hat
rho_clf = 0.0
x_ext = np.hstack((x, a_hat_clf, rho_clf)) # extended state with a_hat and rho

x_hist = np.zeros((len(tt), 2))
u_hist = np.zeros(len(tt))
slack_hist = np.zeros(len(tt))
Vc_hist = np.zeros(len(tt))
V_hist = np.zeros(len(tt))
a_hat_clf_hist = np.zeros((ip_sindy.adim, len(tt)))
a_true_hist = np.zeros((ip_sindy.adim, len(tt)))
nu_clf_hist = np.zeros((len(tt),))
rho_clf_hist = np.zeros((len(tt),))

for k in range(len(tt)):
    t = tt[k]
    print("Time: ", t)

    x_hist[k, :] = x

    # Store adaptation parameters
    a_hat_clf_hist[:, k] = a_hat_clf
    a_true_hist[:, k] = ip_sindy.a_true
    nu_clf_hist[k] = ip_sindy.nu_clf(rho_clf)
    rho_clf_hist[k] = rho_clf

    u_ref = ip_sindy.ctrl_nominal(x)
    u, slack = ip_sindy.ctrl_craclf(x, a_hat_clf, u_ref, use_slack=False)
    
    u_hist[k] = u.item()
    slack_hist[k] = slack
    Vc_hist[k] = ip_sindy.clf(x, a_hat_clf).item()
    a_tilde = a_hat_clf - ip_sindy.a_true
    V_hist[k] = (ip_sindy.nu_clf(rho_clf) * (ip_sindy.clf(x, a_hat_clf).item() + ip_sindy.eta_clf) + 0.5 * a_tilde.T @ np.linalg.inv(ip_sindy.Gamma_clf) @ a_tilde).item()

    # Propagate with zero-order hold on control and disturbance
    if k < len(tt) - 1:
        t_span = (tt[k], tt[k + 1])
        
        sol = solve_ivp(
            #lambda t, y: ip_sindy.dynamics(y, u),
            lambda t, y: ip_sindy.dynamics_extended(y, u),
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
        
        x = x_ext[0:ip_sindy.xdim]
        x[0] = wrapToPi(x[0])  # wrap angle to [-pi, pi]
        a_hat_clf = x_ext[ip_sindy.xdim:(ip_sindy.xdim+ip_sindy.adim)]
        rho_clf = x_ext[(ip_sindy.xdim+ip_sindy.adim)]

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
fig, axs = plt.subplots(ip_sindy.adim, 1)
axs = axs.flatten()
for i in range(ip_sindy.adim):
    axs[i].plot(tt, a_hat_clf_hist[i, :], label='a_hat')
    axs[i].plot(tt, a_true_hist[i, :], label='a_true')
    axs[i].legend()
    axs[i].grid(True)
    axs[i].set_ylabel(f"a{i}")
axs[ip_sindy.adim-1].set_xlabel('Time (s)')

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
