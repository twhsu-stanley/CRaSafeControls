import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.integrate import solve_ivp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dynsys.inverted_pendulum.ip import IP_UNCERTAIN
from dynsys.inverted_pendulum.ip_sindy import IP_UNCERTAIN_SINDY
from dynsys.utils import wrapToPi
import pickle

USE_CP = True # whether to use conformal prediction
USE_ADAPTIVE = True # whether to use adaptive control

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
    "Kp": 8,
    "Kd": 5,
    "clf": {"rate": 0.5},
    "weight": {"slack": 500},
    "feature_names": feature_names,
    "coefficients": coefficients,
    "idx_x": idx_x,
    "idx_u": idx_u,
    "use_adaptive": USE_ADAPTIVE,
    "use_cp": USE_CP,
    "cp_quantile": cp_quantile,
}
params["I"] = params["m"] * params["l"]**2 / 3

params["a_true"] = np.array([
    [(params["m"]*params['g']*params["l"]/2/params["I"])*0.5],
    [-params['b']/params["I"]*0.5]
]) # true a(Theta)

ip_true = IP_UNCERTAIN(params)

params["Gamma_clf"] = np.diag([1, 1]) * 1e-4 # adaptive gain matrix for CRaCLF
params["a_hat_norm_max"] = np.linalg.norm(np.array([[3.0], [3.0]]), 2) # max norm of a_hat
params["epsilon"] = 1e-3 # small value for numerical stability of projection operator
params["eta_clf"] = 1e-2 # small value for numerical stability of scaling function

ip_learned = IP_UNCERTAIN_SINDY(params)

# ========= Run simulation =========
# Choose initial condition
x0 = np.array([0.5, 0.5])
x = x0.copy()

a_hat_clf = np.array([[0.0], [0.0]])  # initial guess for a_hat
rho_clf = 0.0
x_ext = np.hstack((x, a_hat_clf.ravel(), rho_clf))  # extended state with a_hat and rho

# Check if Gamma_clf is valid
if USE_ADAPTIVE:
    if np.min(np.linalg.eigvals(ip_learned.Gamma_clf)) < np.linalg.norm(ip_learned.a_err_max, 2)**2 / (2 * ip_learned.nu_clf(rho_clf) * (ip_learned.clf(x, a_hat_clf).item() + ip_learned.eta_clf)):
        print("Warning: Gamma_clf might not be valid, but continuing.")

# Time histories
x_hist = np.zeros((len(tt), 2))
u_hist = np.zeros(len(tt))
V_hist = np.zeros(len(tt))
a_hat_clf_hist = np.zeros((ip_learned.adim, len(tt)))
a_true_hist = np.zeros((ip_learned.adim, len(tt)))
nu_clf_hist = np.zeros((len(tt),))
rho_clf_hist = np.zeros((len(tt),))
slack_hist = np.zeros(len(tt))

for k in range(len(tt)):
    t = tt[k]
    print("Time: ", t)

    x_hist[k, :] = x

    # Store adaptation parameters
    a_hat_clf_hist[:, k] = a_hat_clf.ravel()
    a_true_hist[:, k] = ip_learned.a_true.ravel()
    nu_clf_hist[k] = ip_learned.nu_clf(rho_clf)
    rho_clf_hist[k] = rho_clf

    u_ref = ip_learned.ctrl_nominal(x)
    u, slack = ip_learned.ctrl_craclf(x, a_hat_clf, u_ref, use_slack=True)
    u_hist[k] = u.item()
    V_hist[k] = ip_learned.clf(x, a_hat_clf).item()
    slack_hist[k] = slack

    # Propagate with zero-order hold on control and disturbance
    if k < len(tt) - 1:
        t_span = (tt[k], tt[k + 1])
        
        sol = solve_ivp(
            lambda t, y: ip_learned.dynamics_extended(y, u),
            t_span,
            x_ext,
            method="BDF",
            rtol=1e-6,
            atol=1e-6,
            t_eval=[tt[k + 1]],
        )
        try:
            x_ext = sol.y[:, -1]
        except Exception as e:
            raise ValueError("Error occurred while solving IVP:", e)
        
        x = x_ext[0:ip_learned.xdim]
        x[0] = wrapToPi(x[0])  # wrap angle to [-pi, pi]
        a_hat_clf = x_ext[ip_learned.xdim:(ip_learned.xdim+ip_learned.adim)].reshape(-1,1)
        rho_clf = x_ext[(ip_learned.xdim+ip_learned.adim)]



# ========= Plots =========
# States
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(tt, x_hist[:, 0], color='blue', linewidth=1.5)
plt.ylabel(r"$\theta$ (rad)")
plt.title("State - Angle")
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(tt, x_hist[:, 1], color='magenta', linewidth=1.5)
plt.ylabel(r"$\dot{\theta}$ (rad/s)")
plt.title("State - Angular Velocity")
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(tt, u_hist, color='orange', linewidth=1.5)
plt.ylabel("u (Nm)")
plt.title("Control Input - Torque")
plt.grid(True)
plt.tight_layout()

# CLF over time
plt.figure()
plt.plot(tt, V_hist, color='navy', linewidth=1.5)
plt.ylabel("V(x)")
plt.xlabel("Time (s)")
plt.title("Control Lyapunov Function")
plt.grid(True)

# Uncertainty parameters
fig, axs = plt.subplots(ip_learned.adim, 1)
axs = axs.flatten()
for i in range(ip_learned.adim):
    axs[i].plot(tt, a_hat_clf_hist[i, :], label='a_hat')
    axs[i].plot(tt, a_true_hist[i, :], label='a_true')
    axs[i].legend()
    axs[i].grid(True)
    axs[i].set_ylabel(f"a{i}")
axs[ip_learned.adim-1].set_xlabel('Time (s)')

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

# Control and slack histories
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(tt, u_hist)
plt.ylabel("Control: u")
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(tt, slack_hist)
plt.xlabel("Time (s)")
plt.ylabel("QP slack")
plt.grid(True)

plt.show()