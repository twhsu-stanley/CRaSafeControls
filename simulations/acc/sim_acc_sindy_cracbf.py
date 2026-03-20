import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.integrate import solve_ivp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dynsys.acc.acc_sindy import ACC_SINDY
import pickle

USE_CP = True # whether to use conformal prediction
USE_ADAPTIVE = True # whether to use adaptive control

# Load the SINDy model ###################################################
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
##########################################################################

# Time setup
dt = 0.01
sim_T = 5 # Simulation time
tt = np.arange(0, sim_T, dt)

# System parameters
params = {
    "v0": 15.0,
    "vd": 20.0,
    "m": 2000.0,
    "grav": 9.81,
    "ca": 0.3,
    "cd": 0.3,
    "Kp": 100.0, # P gain for the nominal controller
    "T": 1.0, # look-ahead time
    "cbf": {"rate": 2.0},
    "dt": dt
}
#params["u_max"] = params["ca"] * params["m"] * params["grav"]
#params["u_min"] = -params["cd"] * params["m"] * params["grav"]
params["use_adaptive"] = USE_ADAPTIVE
params["use_cp"] = USE_CP
params["cp_quantile"] = cp_quantile
params["Gamma_cbf"] = np.diag(np.array([100.0, 100.0, 10.0, 10.0])) # adaptive gain matrix for CRaCCM
params["a_true"] = np.array([[-0.1], [-2.0], [-0.1], [-4.0]]) # true parameters
params["a_hat_norm_max"] = np.linalg.norm(params["a_true"], 2) # TODO: check this
params["epsilon"] = 1e-2 # small value for numerical stability of projection operator
params["eta_cbf"] = 10.0

params["feature_names"] = feature_names
params["coefficients"] = coefficients
params["idx_x"] = idx_x
params["idx_u"] = idx_u

# Learned model
acc_sindy = ACC_SINDY(params)

# Initial conditions
x0 = np.array([0.0, 30.0, 50.0])
x = x0.copy()
a_hat_cbf = np.array([[0.0], [0.0], [0.0], [0.0]]) # initial guess for a_hat
rho_cbf = 0.0
x_ext = np.hstack((x, a_hat_cbf.ravel(), rho_cbf)) # extended state with a_hat and rho

# Check if Gamma_cbf is valid
if USE_ADAPTIVE:
    if np.min(np.linalg.eigvals(acc_sindy.Gamma_cbf)) < np.linalg.norm(acc_sindy.a_err_max, 2)**2 / (2 * acc_sindy.nu_cbf(rho_cbf) * acc_sindy.cbf(x, a_hat_cbf).item()):
        raise RuntimeError("Gamma_cbf is not valid: minimal eigenvalue is too small")

# Check if the initial state is in the safe set
if (acc_sindy.cbf(x, a_hat_cbf).item() - 0.5 * acc_sindy.a_err_max.T @ np.linalg.inv(acc_sindy.Gamma_cbf) @ acc_sindy.a_err_max) < 1e-3:
    raise ValueError("Initial condition unsafe")

x_hist = np.zeros((len(tt), 3))
u_hist = np.zeros(len(tt))
h_hist = np.zeros(len(tt))
a_hat_cbf_hist = np.zeros((acc_sindy.adim, len(tt)))
a_true_hist = np.zeros((acc_sindy.adim, len(tt)))
nu_cbf_hist = np.zeros((len(tt),))
rho_cbf_hist = np.zeros((len(tt),))

for k in range(len(tt)):
    t = tt[k]
    print("Time: ", t)

    x_hist[k, :] = x

    # Store adaptation parameters
    a_hat_cbf_hist[:, k] = a_hat_cbf.ravel()
    a_true_hist[:, k] = acc_sindy.a_true.ravel()
    nu_cbf_hist[k] = acc_sindy.nu_cbf(rho_cbf)
    rho_cbf_hist[k] = rho_cbf

    u_ref = acc_sindy.ctrl_nominal(x)
    u = acc_sindy.ctrl_cracbf(x, a_hat_cbf, u_ref)
    u_hist[k] = u.item()
    h_hist[k] = acc_sindy.cbf(x, a_hat_cbf).item()

    # Propagate with zero-order hold on control and disturbance
    if k < len(tt) - 1:
        t_span = (tt[k], tt[k + 1])
        
        sol = solve_ivp(
            #lambda t, y: acc_sindy.dynamics(y, u),
            lambda t, y: acc_sindy.dynamics_extended(y, u),
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
        
        x = x_ext[0:acc_sindy.xdim]
        a_hat_cbf = x_ext[acc_sindy.xdim:(acc_sindy.xdim+acc_sindy.adim)].reshape(-1,1)
        rho_cbf = x_ext[(acc_sindy.xdim+acc_sindy.adim)]

# Plotting
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(tt, x_hist[:, 1], color='blue', linewidth=1.5)
plt.plot(tt, np.ones_like(tt) * params['vd'], 'k--')
plt.ylabel("v (m/s)")
plt.title("State - Velocity")
plt.grid(True)
plt.subplot(4, 1, 2)
plt.plot(tt, x_hist[:, 2], color='magenta', linewidth=1.5)
plt.ylabel("z (m)")
plt.title("State - Distance to lead vehicle")
plt.grid(True)
plt.subplot(4, 1, 3)
plt.plot(tt, u_hist, color='orange', linewidth=1.5)
#plt.plot(tt, np.ones_like(u_hist) * params['u_max'], 'k--')
#plt.plot(tt, np.ones_like(u_hist) * params['u_min'], 'k--')
plt.ylabel("u(N)")
plt.title("Control Input - Wheel Force")
plt.grid(True)
plt.subplot(4, 1, 4)
plt.plot(tt, h_hist, color='navy', linewidth=1.5)
plt.plot(tt, np.ones_like(tt) * 0.5 * (acc_sindy.a_err_max.T @ np.linalg.inv(acc_sindy.Gamma_cbf) @ acc_sindy.a_err_max).item(), 'r--')
plt.ylabel("CBF (h(x))")
plt.title("CBF")
plt.grid(True)
plt.tight_layout()

# Uncertainty parameters
fig, axs = plt.subplots(acc_sindy.adim, 1)
axs = axs.flatten()
for i in range(acc_sindy.adim):
    axs[i].plot(tt, a_hat_cbf_hist[i, :], label='a_hat')
    axs[i].plot(tt, a_true_hist[i, :], label='a_true')
    axs[i].legend()
    axs[i].grid(True)
    axs[i].set_ylabel(f"a{i}")
axs[acc_sindy.adim-1].set_xlabel('Time (s)')

# Scaling function and parameter
fig, axs = plt.subplots(2, 1)
axs = axs.flatten()
axs[0].plot(tt, nu_cbf_hist, lw=1)
axs[0].set_ylabel('nu_cbf')
axs[0].grid(True)
axs[1].plot(tt, rho_cbf_hist, lw=1)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('rho_cbf')
axs[1].grid(True)

plt.show()
