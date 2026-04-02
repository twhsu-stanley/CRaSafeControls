import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from scipy.integrate import solve_ivp
from systems.nonlinear_toy.nonlinear_toy import NONLINEAR_TOY
from geodesic_solver import GeodesicSolver
from acp import ACP
from motion_planner import MotionPlanner
from scipy.interpolate import interp1d

USE_CP = True # whether to use conformal prediction
USE_ADAPTIVE = True # whether to use adaptive control

I_length = 200 # number of time steps in I_k

VERIFY_GEODESIC = False
USE_QPSOLVERS = True

weight_slack = 1000.0

# Time setup
dt = 0.01
sim_T = 6.0 # Simulation time
tt = np.arange(0, sim_T, dt)
T_steps = len(tt)

# Prior knowledge of the uncertainty parameter
a_true = np.array([-1.0, -0.5, -1.5]) # unknown to the controller
a_ub = np.array([0.5, 0.5, 0.5])
a_lb = np.array([-1.5, -1.5, -3.5])

# System parameters
params = {
    "ccm": {"rate": 0.8},
    "geodesic": {"N": 8, "D": 2}, # N = D + a
    "weight_slack": weight_slack,
    "dt": dt,
}
params["use_adaptive"] = USE_ADAPTIVE
params["use_cp"] = USE_CP
params["Gamma_ccm"] = np.diag(np.array([2.0, 2.0, 2.0]))
params["a_true"] = a_true
params["a_ub"] = a_ub
params["a_lb"] = a_lb
params["a_hat_norm_max"] = 0.5 * np.linalg.norm(a_ub - a_lb, ord=2) * 1.2
params["epsilon"] = 1e-3 # small value for numerical stability of projection operator
params["eta_ccm"] = 5.0

# Construct the system 
toy = NONLINEAR_TOY(params)

# Motion planner: plan x_d and u_d
planner = MotionPlanner(
    system = toy,
    dt = dt,
    Q = np.eye(toy.xdim),
    R = np.eye(toy.udim) * 0.1,
    Q_f = np.eye(toy.xdim) * 10.0
)

x_init = np.array([1.0, -1.8, -1.2])
x_goal = np.array([2.0, 1.0, 1.5])
t0 = 0.0
horizon_steps = T_steps
x_guess = np.tile(x_init.reshape(-1,1), (1, horizon_steps + 1))
u_guess = np.zeros((toy.udim, horizon_steps))
x_d_planned, u_d_planned = planner.plan(x_init, x_goal, horizon_steps, x_guess, u_guess)
t_planned = t0 + dt * np.arange(horizon_steps + 1)

interp_x_d = interp1d(
    t_planned, x_d_planned, kind='linear', axis=1,
    bounds_error=False, fill_value='extrapolate'
)
interp_u_d = interp1d(
    t_planned[:-1], u_d_planned, kind='linear', axis=1,
    bounds_error=False, fill_value='extrapolate'
)

fig, axs = plt.subplots(3, 1)
axs = axs.flatten()
axs[0].plot(t_planned, x_d_planned[0, :], '--')
axs[0].set_ylabel('x1'); 
axs[0].grid(True)
axs[1].plot(t_planned, x_d_planned[1, :], '--')
axs[1].set_ylabel('x2')
axs[1].grid(True)
axs[2].plot(t_planned, x_d_planned[2, :], '--')
axs[2].set_ylabel('x3')
axs[2].set_xlabel('Time (s)')
axs[2].grid(True)
plt.suptitle('Nominal Trajectory: x_d')

plt.figure()
plt.plot(t_planned[:-1], u_d_planned[0, :], '--')
plt.ylabel('u'); 
plt.grid(True)
plt.xlabel('Time (s)')
plt.title('Nominal Control: u_d')
plt.show()

# Disturbance (non-parametric uncertainty)
Delta = lambda t: np.array([
    0.01 + 0.5 * np.sin(2 * np.pi / 0.1 * t + 0.3),
    0.02 - 0.5 * np.sin(2 * np.pi / 0.08 * t + 0.1),
    -0.01 + 0.8 * np.sin(2 * np.pi / 0.15 * t + 0.2),
], dtype=float)

# Compute the initial calibration set for ACP
Delta_data = [np.linalg.norm(Delta(t), 2) for t in np.arange(0.0, sim_T*20, 0.01)] 
S_cal_init = []
for k in range(0, len(Delta_data), I_length): 
    S_cal_init.append(np.max(Delta_data[k:k + I_length]))

# Initialzie the ACP object
acp = ACP(S_cal_init,
          N_cal = 200,
          lr = 0.05, # learning rate
          delta_target = 0.05,
          delta_init = 0.2,
          score_max = max(S_cal_init) * 2, # max possible score
          score_min = 0.0, # min possible score
          buffer_maxlen = 800
          )

toy.cp_quantile = acp.Q_k

# Time hisotry of logged data
x_hist = np.zeros((toy.xdim, T_steps))
u_hist = np.zeros((toy.udim, T_steps))
Erem_hist = np.zeros((T_steps,))
Erem_dot_hist = np.zeros((T_steps,))
V1_hist = np.zeros((T_steps,))
V2_hist = np.zeros((T_steps,))
slack_hist = np.zeros((T_steps,))
a_hat_ccm_hist = np.zeros((toy.adim, T_steps))
a_true_hist = np.zeros((toy.adim, T_steps))
nu_ccm_hist = np.zeros((T_steps,))
rho_ccm_hist = np.zeros((T_steps,))
# for acp debugging
a_k_hist = np.zeros((toy.adim, T_steps))
s_k_hist = np.zeros((T_steps,))
Q_k_hist = np.zeros((T_steps,))
delta_k_hist = np.zeros((T_steps,))

# Initial state
x = interp_x_d(0).copy() #+ np.array([0.7, 0.5, -0.2])  # initial condition + perturbation
a_hat_ccm = np.array([0.0, 0.0, 0.0]) # initial guess for a_hat
rho_ccm = 0.0
x_ext = np.hstack((x, a_hat_ccm, rho_ccm)) # extended state with a_hat and rho

# Initialize geodesic solver
N = toy.params["geodesic"]["N"]
D = toy.params["geodesic"]["D"]
n = toy.xdim
geodesic_solver = GeodesicSolver(n, D, N, toy.W_fcn, toy.dW_dxi_fcn, toy.dW_dai_fcn)

# Main simulation loop
for i in range(T_steps):
    t = tt[i]
    print("Time: ", t)

    # Store current state
    x_hist[:, i] = x

    # Nominal trajectory
    x_d = interp_x_d(t)
    u_d = interp_u_d(t)

    # Store adaptation parameters
    a_hat_ccm_hist[:, i] = a_hat_ccm
    a_true_hist[:, i] = toy.a_true
    nu_ccm_hist[i] = toy.nu_ccm(rho_ccm)
    rho_ccm_hist[i] = rho_ccm

    # Implement ccm control law
    uc, slack = toy.ctrl_craccm(x, a_hat_ccm, x_d, u_d, geodesic_solver, use_qpsolvers=USE_QPSOLVERS)

    u_hist[:, i] = uc.ravel()
    slack_hist[i] = slack
    Erem_hist[i] = toy.Erem

    # Adaptive conformal prediction
    Q_k_hist[i] = toy.cp_quantile
    delta_k_hist[i] = acp.delta
    acp.add_data_to_buffers(x, toy.dynamics_nominal(x,uc), toy.Y(x))
    if (i+1) % I_length == 0:
        acp.estimate_uncertainty(dt)
        s_k = acp.compute_score(toy.a_ub, toy.a_lb,) # acp.a_k is updated here
        acp.update_delta(s_k)
        acp.S_cal.append(s_k)
        toy.cp_quantile = acp.compute_quantile()

        a_k_hist[:,(i-I_length+1):(i+1)] = acp.a_k.reshape(-1,1)
        s_k_hist[(i-I_length+1):(i+1)] = s_k

    # For debugging ###########################################
    # Lyapunov function
    V1 = toy.nu_ccm(rho_ccm) * (toy.Erem + toy.eta_ccm)
    a_tilde = a_hat_ccm - toy.a_true
    V2 = a_tilde.T @ np.linalg.inv(toy.Gamma_ccm) @ a_tilde
    V1_hist[i] = V1.item()
    V2_hist[i] = V2.item()
    ###########################################################

    # Propagate with zero-order hold on control
    if i < T_steps - 1:
        t_span = (tt[i], tt[i + 1])
        sol = solve_ivp(
            #lambda t, y: toy.dynamics(y, uc) + Delta(t),
            lambda t, y: toy.dynamics_extended(y, x_d, uc, geodesic_solver) + np.concatenate([Delta(t), np.zeros(4)]) ,
            t_span,
            x_ext,
            method="BDF",
            rtol=1e-8,
            atol=1e-8,
            t_eval=[tt[i + 1]],
        )
        try:
            x_ext = sol.y[:, -1]
        except Exception as e:
            raise ValueError("Error occurred while solving IVP:", e)
        
        x = x_ext[0:toy.xdim]
        a_hat_ccm = x_ext[toy.xdim:(toy.xdim+toy.adim)]
        rho_ccm = x_ext[(toy.xdim+toy.adim)]

# Plot results
# x vs. x_d
fig, axs = plt.subplots(3, 1)
axs = axs.flatten()
axs[0].plot(t_planned, x_d_planned[0, :], '--', label='Nominal')
axs[0].plot(tt, x_hist[0, :], '-', label='CCM')
axs[0].set_ylabel('x1'); 
axs[0].legend()
axs[0].grid(True)
axs[1].plot(t_planned, x_d_planned[1, :], '--')
axs[1].plot(tt, x_hist[1, :], '-')
axs[1].set_ylabel('x2')
axs[1].grid(True)
axs[2].plot(t_planned, x_d_planned[2, :], '--')
axs[2].plot(tt, x_hist[2, :], '-') 
axs[2].set_ylabel('x3')
axs[2].set_xlabel('Time (s)')
axs[2].grid(True)
plt.suptitle('State: Actual vs. Nominal')

# Controls and slack
fig, axs = plt.subplots(2, 1)
axs = axs.flatten()
axs[0].plot(tt, u_hist[0, :], 'r-', label='u: CCM')
axs[0].plot(t_planned[:-1], u_d_planned[0, :], 'k--', label='u_d')
axs[0].set_ylabel('Control u')
axs[0].legend()
axs[0].grid(True)
axs[1].plot(tt, slack_hist)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('QP slack')
axs[1].grid(True)

# Uncertainty parameters
fig, axs = plt.subplots(toy.adim, 1)
axs = axs.flatten()
for i in range(toy.adim):
    axs[i].plot(tt, a_hat_ccm_hist[i, :], label='a_hat')
    axs[i].plot(tt, a_true_hist[i, :], label='a_true')
    axs[i].plot(tt, a_k_hist[i, :], label='a_k of ACP', linestyle='--')
    axs[i].legend()
    axs[i].grid(True)
    axs[i].set_ylabel(f"a{i}")
axs[toy.adim-1].set_xlabel('Time (s)')

# Scaling function and parameter
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
fig, axs = plt.subplots(3, 1)
axs = axs.flatten()
err_norm = np.linalg.norm(x_d_planned[:,:-1] - x_hist, ord=2, axis=0)  # column-wise 2-norm
axs[0].plot(tt, err_norm)
axs[0].set_ylabel('||x-x_d||_2')
axs[0].grid(True)
# Erem
axs[1].plot(tt, Erem_hist)
axs[1].set_ylabel('Riemann Energy: Erem(t)')
axs[1].grid(True)
# Lyapunov
axs[2].plot(tt, V1_hist, label='V1 = nu(rho)(E+eta)')
axs[2].plot(tt, V2_hist, label='V2 = a~^T Gamma^-1 a~')
axs[2].plot(tt, V1_hist + V2_hist, label='V = V1+V2')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Lyapunov: V(t)')
axs[2].legend()
axs[2].grid(True)

# Adaptive Quantile
fig, axs = plt.subplots(2, 1)
axs = axs.flatten()
axs[0].plot(tt, Q_k_hist, lw=1, label='Q_k')
axs[0].scatter(tt, s_k_hist, color='k', s=5, label='scores')
axs[0].legend()
axs[0].set_ylabel('Q_k')
axs[0].grid(True)
axs[1].plot(tt, delta_k_hist, lw=1)
axs[1].plot(tt, -np.ones_like(tt) * acp.lr, 'r--')
axs[1].plot(tt, np.ones_like(tt) * (1+acp.lr), 'r--')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('delta_k')
axs[1].grid(True)

plt.show()