import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from scipy.integrate import solve_ivp
from dynsys.plannar_quad.plannar_quad_uncertain import PLANNAR_QUAD_UNCERTAIN
from dynsys.plannar_quad.plannar_quad import PLANNAR_QUAD
from dynsys.geodesic_solver import GeodesicSolver
from scipy.io import loadmat
from scipy.interpolate import interp1d

USE_CP = 0 # 1 or 0: whether to use conformal prediction
USE_ADAPTIVE = 0 # 1 or 0: whether to use adaptive control

weight_slack = 50.0 if USE_CP else 1000.0

# Load the desired trajectory
# TODO: load a motion planner
# x_d_fcn and u_d_fcn: functions of time
data = loadmat("simulations/plannar_quad/nomTraj.mat")
t_d_data = data["soln"]["grid"][0][0][0][0][0][0,:]
x_d_data = data["soln"]["grid"][0][0][0][0][1][:]
u_d_data = data["soln"]["grid"][0][0][0][0][2][:]
# interpolation functions
interp_x = interp1d(
    t_d_data, x_d_data, kind='linear', axis=1,
    bounds_error=False, fill_value='extrapolate'
)
interp_u = interp1d(
    t_d_data, u_d_data, kind='linear', axis=1,
    bounds_error=False, fill_value='extrapolate'
)
x_d_fcn = lambda t: interp_x(t)
u_d_fcn = lambda t: interp_u(t)

# Disturbance
dist_config = {}
dist_config["include_dist"] = False
dist_config["gen_dist"] = lambda t: np.array([
    0.0 + 0.1 * np.sin(2 * np.pi / 1 * t),
    0.0 + 0.1 * np.cos(2 * np.pi / 1 * t + 0.03),
    0.0 + 0.1 * np.sin(2 * np.pi / 2 * t + 0.01),
    0.0 + 0.1 * np.cos(2 * np.pi / 2 * t + 0.04),
    0.0 + 0.15 * np.sin(2 * np.pi / 2 * t + 0.01),
    0.0 + 0.15 * np.cos(2 * np.pi / 2 * t + 0.05),
], dtype=float)#.reshape((6,1))

# compute w_max
w_norms = [np.linalg.norm(dist_config["gen_dist"](t)) for t in np.arange(0.0, 10.0, 0.01)]
w_max = np.max(w_norms)

# Time setup
dt = 0.005
sim_T = np.floor(t_d_data[-1]) # Simulation time
tt = np.arange(0, sim_T, dt)
T_steps = len(tt)

# System parameters
params = {
    "l": 0.25,
    "m": 0.486,
    "g": 9.81,
    "J": 0.00383,
    "ccm": {"rate": 0.8, "weight_slack": weight_slack},
    "geodesic": {"N": 8, "D": 2},
}
params["use_adaptive"] = USE_ADAPTIVE
params["use_cp"] = USE_CP
params["cp_quantile"] = w_max * 0.95 if USE_CP else 0.0
params["Gamma_ccm"] = np.eye(1) # adaptive gain matrix for CRaCCM
params["a_true"] = np.array([[0.0]])  # true a
params["a_hat_norm_max"] = np.linalg.norm(np.array([[0.6]]), 2) # max norm of a_hat
params["a_0"] = np.array([[0.0]]) # initial guess for a_hat
params["epsilon"] = 1e-2 # small value for numerical stability of projection operator

params["eta_ccm"] = 1000.0

plannar_quad = PLANNAR_QUAD_UNCERTAIN(params)
#plannar_quad = PLANNAR_QUAD(params)

x_hist = np.zeros((plannar_quad.xdim, T_steps))
u_hist = np.zeros((plannar_quad.udim, T_steps))
Erem_hist = np.zeros((T_steps,))
slack_hist = np.zeros((T_steps,))
a_hat_ccm_hist = np.zeros((plannar_quad.adim, T_steps))
a_true_hist = np.zeros((plannar_quad.adim, T_steps))
nu_ccm_hist = np.zeros((T_steps,))
rho_ccm_hist = np.zeros((T_steps,))
x_d_hist = np.zeros((plannar_quad.xdim, T_steps))
u_d_hist = np.zeros((plannar_quad.udim, T_steps))

# Initial state
x0 = np.zeros(plannar_quad.xdim)
x = x0.copy() + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # initial condition perturbation

# Initialize geodesic solver
N = plannar_quad.params["geodesic"]["N"]
D = plannar_quad.params["geodesic"]["D"]
n = plannar_quad.xdim
geodesic_solver = GeodesicSolver(n, D, N, plannar_quad.W_fcn, plannar_quad.dW_dxi_fcn, plannar_quad.dW_dai_fcn)

# Main simulation loop
for i in range(T_steps):
    t = tt[i]
    print("Time: ", t)
    
    # store current state
    x_hist[:, i] = x

    # nominal logs
    x_d = x_d_fcn(t)
    u_d = u_d_fcn(t)
    x_d_hist[:, i] = x_d
    u_d_hist[:, i] = u_d

    a_hat_ccm_hist[:, i] = plannar_quad.a_hat_ccm.ravel()
    a_true_hist[:,i] = plannar_quad.a_true.ravel() # TODO: just to verify that a_true is constant; can remove later
    nu_ccm_hist[i] = plannar_quad.nu_ccm() if USE_ADAPTIVE else 0.0
    rho_ccm_hist[i] = plannar_quad.rho_ccm if USE_ADAPTIVE else 0.0

    # Disturbance
    if dist_config["include_dist"]:
        wt = dist_config["gen_dist"](t)
    else:
        wt = np.zeros((x.shape[0],))

    # Precompute geodesic
    plannar_quad.calc_geodesic(geodesic_solver, x, x_d)
    Erem_hist[i] = plannar_quad.Erem

    # Log controller inputs
    uc, slack = plannar_quad.ctrl_cra_ccm(x, x_d, u_d)
    u_hist[:, i] = uc.ravel()
    slack_hist[i] = slack

    # Propagate with zero-order hold on control and disturbance
    if i < T_steps - 1:
        t_span = (tt[i], tt[i + 1])
        
        sol = solve_ivp(
            lambda t, y: plannar_quad.dynamics(y, uc) + wt,
            t_span,
            x,
            method = "BDF", #"LSODA", #"Radau",  # stiff solver
            rtol = 1e-6, # 1e-6
            atol = 1e-6, # 1e-6
            t_eval = [tt[i + 1]],
        )
        try:
            x = sol.y[:, -1]
        except Exception as e:
            print("Error occurred while solving IVP:", e)

    # Update adaptive parameter
    if USE_ADAPTIVE:
        plannar_quad.adaptation_cra_ccm(x, x_d, dt)

# Plot results
# x vs. x_d
fig, axs = plt.subplots(3, 2)
axs = axs.flatten()
axs[0].plot(tt, x_d_hist[0, :], '--', label='Nominal')
axs[0].plot(tt, x_hist[0, :], '-', label='CCM')
axs[0].set_xlabel('Time (s)'); axs[0].set_ylabel('x (m)'); axs[0].legend()
axs[0].grid(True)
axs[1].plot(tt, x_d_hist[1, :], '--')
axs[1].plot(tt, x_hist[1, :], '-')
axs[1].set_xlabel('Time (s)'); axs[1].set_ylabel('z (m)')
axs[1].grid(True)
axs[2].plot(tt, x_d_hist[2, :] * 180.0/np.pi, '--')
axs[2].plot(tt, x_hist[2, :] * 180.0/np.pi)
axs[2].set_xlabel('Time (s)'); axs[2].set_ylabel('Phi (deg)')
axs[2].grid(True)
axs[3].plot(tt, x_d_hist[3, :], '--')
axs[3].plot(tt, x_hist[3, :], '-')
axs[3].set_xlabel('Time (s)'); axs[3].set_ylabel('vx (m/s)')
axs[3].grid(True)
axs[4].plot(tt, x_d_hist[4, :], '--')
axs[4].plot(tt, x_hist[4, :], '-')
axs[4].set_xlabel('Time (s)'); axs[4].set_ylabel('vz (m/s)')
axs[4].grid(True)
axs[5].plot(tt, x_d_hist[5, :] * 180.0/np.pi, '--')
axs[5].plot(tt, x_hist[5, :] * 180.0/np.pi)
axs[5].set_xlabel('Time (s)'); axs[5].set_ylabel('Phi rate (deg/s)')
axs[5].grid(True)
plt.suptitle('State: CCM vs Nominal')

# u vs u_d
fig, axs = plt.subplots(2, 1)
axs = axs.flatten()
axs[0].plot(tt, u_d_hist[0, :], 'k--', label='u_d_1')
axs[1].plot(tt, u_d_hist[1, :], 'k--', label='u_d_2')
axs[0].plot(tt, u_hist[0, :], 'r-', label='u_1: CCM')
axs[1].plot(tt, u_hist[1, :], 'b-', label='u_2: CCM')
axs[0].set_ylabel('u0 (N)')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('u1 (N)')
axs[0].legend()
axs[1].legend()
axs[0].grid(True)
axs[1].grid(True)

# Slack
plt.figure(figsize=(8,3))
plt.plot(tt, slack_hist, lw=1)
plt.xlabel('Time (s)')
plt.ylabel('QP slack')
plt.grid(True)

# Uncertainty parameter
plt.figure(figsize=(8,3))
plt.plot(tt, a_hat_ccm_hist[0, :], label='a0')
plt.plot(tt, a_true_hist[0, :], label='a true')
plt.xlabel('Time (s)')
plt.ylabel('a_hat_ccm')
plt.legend()
plt.grid(True)

# nu_ccm
fig, axs = plt.subplots(2, 1)
axs = axs.flatten()
axs[0].plot(tt, nu_ccm_hist, lw=1)
axs[0].set_ylabel('nu_ccm')
axs[0].grid(True)
axs[1].plot(tt, rho_ccm_hist, lw=1)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('rho_ccm')
axs[1].grid(True)

# Tracking error norm
plt.figure()
err_norm = np.linalg.norm(x_d_hist - x_hist, ord=2, axis=0)  # column-wise 2-norm
plt.plot(tt, err_norm)
plt.xlabel('Time (s)')
plt.ylabel('||x-x_d||_2')
plt.grid(True)

# Riemannian energy
plt.figure()
plt.plot(tt, np.sqrt(np.maximum(Erem_hist, 0.0)))
plt.ylim([0, None])
plt.xlabel('Time (s)')
plt.ylabel('Riemann distance: sqrt E(t)')
plt.grid(True)

plt.show()