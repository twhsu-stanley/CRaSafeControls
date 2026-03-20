import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.integrate import solve_ivp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dynsys.acc.acc import ACC

USE_CP = False # whether to use conformal prediction
USE_ADAPTIVE = True # whether to use adaptive control

# Parameters and Initialization
dt = 0.01
sim_T = 10
tt = np.arange(0, sim_T, dt)

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
params["Gamma_cbf"] = np.diag(np.array([100.0, 100.0, 10.0, 10.0])) # adaptive gain matrix for CRaCCM
params["a_true"] = np.array([[0.5], [5.0], [1.0], [-4.0]]) # true parameters
params["a_hat_norm_max"] = np.linalg.norm(params["a_true"], 2) # TODO: check this
params["epsilon"] = 1e-2 # small value for numerical stability of projection operator
params["eta_cbf"] = 10.0

# Construct the system
acc = ACC(params)

# Disturbance (non-parametric uncertainty)
Delta = lambda t: np.array([
    0.0,#0.04 + 0.05 * np.sin(2 * np.pi / 0.3 * t),
    0.0,#0.1 + 0.05 * np.cos(2 * np.pi / 0.5 * t + 0.03),
    0.0,#-0.1 + 0.2 * np.sin(2 * np.pi / 2 * t + 0.01),
], dtype = float)

# Compute upper bound of Delta
Delta_max = np.max([np.linalg.norm(Delta(t), 2) for t in np.arange(0.0, sim_T, 0.01)])
acc.cp_quantile = Delta_max * 0.95

# Simulation
x0 = np.array([0.0, 30.0, 50.0])
x = x0.copy()
a_hat_cbf = np.array([[0.25], [4.0], [0.0], [0.0]]) # initial guess for a_hat
rho_cbf = 0.0
x_ext = np.hstack((x, a_hat_cbf.ravel(), rho_cbf)) # extended state with a_hat and rho

# Check if Gamma_cbf is valid
if USE_ADAPTIVE:
    if np.min(np.linalg.eigvals(acc.Gamma_cbf)) < np.linalg.norm(acc.a_err_max, 2)**2 / (2 * acc.nu_cbf(rho_cbf) * acc.cbf(x, a_hat_cbf).item()):
        raise RuntimeError("Gamma_cbf is not valid: minimal eigenvalue is too small")
    
# Check if the initial state is in the safe set
if (acc.cbf(x, a_hat_cbf).item() - 0.5 * acc.a_err_max.T @ np.linalg.inv(acc.Gamma_cbf) @ acc.a_err_max) < 1e-3:
    raise ValueError("Initial condition unsafe")

x_hist = np.zeros((len(tt), 3))
u_hist = np.zeros(len(tt))
h_hist = np.zeros(len(tt))
a_hat_cbf_hist = np.zeros((acc.adim, len(tt)))
a_true_hist = np.zeros((acc.adim, len(tt)))
nu_cbf_hist = np.zeros((len(tt),))
rho_cbf_hist = np.zeros((len(tt),))

for k in range(len(tt)):
    t = tt[k]
    print("Time: ", t)

    x_hist[k, :] = x

    # Store adaptation parameters
    a_hat_cbf_hist[:, k] = a_hat_cbf.ravel()
    a_true_hist[:, k] = acc.a_true.ravel()
    nu_cbf_hist[k] = acc.nu_cbf(rho_cbf)
    rho_cbf_hist[k] = rho_cbf

    u_ref = acc.ctrl_nominal(x)
    u = acc.ctrl_cracbf(x, a_hat_cbf, u_ref)
    u_hist[k] = u.item()
    h_hist[k] = acc.cbf(x, a_hat_cbf).item()

    # Propagate with zero-order hold on control and disturbance
    if k < len(tt) - 1:
        t_span = (tt[k], tt[k + 1])
        
        sol = solve_ivp(
            #lambda t, y: acc.dynamics(y, u),
            lambda t, y: acc.dynamics_extended(y, u),
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
        
        x = x_ext[0:acc.xdim]
        a_hat_cbf = x_ext[acc.xdim:(acc.xdim+acc.adim)].reshape(-1,1)
        rho_cbf = x_ext[(acc.xdim+acc.adim)]

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
plt.plot(tt, np.ones_like(tt) * 0.5 * (acc.a_err_max.T @ np.linalg.inv(acc.Gamma_cbf) @ acc.a_err_max).item(), 'r--')
plt.ylabel("CBF (h(x))")
plt.title("CBF")
plt.grid(True)
plt.tight_layout()

# Uncertainty parameters
fig, axs = plt.subplots(acc.adim, 1)
axs = axs.flatten()
for i in range(acc.adim):
    axs[i].plot(tt, a_hat_cbf_hist[i, :], label='a_hat')
    axs[i].plot(tt, a_true_hist[i, :], label='a_true')
    axs[i].legend()
    axs[i].grid(True)
    axs[i].set_ylabel(f"a{i}")
axs[acc.adim-1].set_xlabel('Time (s)')

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
