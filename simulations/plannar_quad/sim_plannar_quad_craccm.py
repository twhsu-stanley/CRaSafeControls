import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from scipy.integrate import solve_ivp
from dynsys.plannar_quad.plannar_quad_uncertain import PLANNAR_QUAD_UNCERTAIN
from scipy.io import loadmat
from scipy.interpolate import interp1d

USE_CP = 1 # 1 or 0: whether to use conformal prediction
USE_ADAPTIVE = 1 # 1 or 0: whether to use adaptive control

# TODO: load the desired trajectory: desired_traj
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

# Time setup
dt = 0.001
sim_T = 3#np.floor(t_d_data[-1]) # Simulation time
tt = np.arange(0, sim_T, dt)
T_steps = len(tt)

# System parameters
params = {
    "l": 0.25,
    "m": 0.486,
    "g": 9.81,
    "J": 0.00383,
    "ccm": {"rate": 0.8},
    "geodesic": {"N": 8, "D": 2},
}
params["use_adaptive"] = USE_ADAPTIVE
params["Gamma_ccm"] = np.eye(3) * 50 # adaptive gain matrix for CRaCCM
params["a_true"] = np.array([[-0.02], [-0.02], [-0.02]]) # true a(Theta)
params["a_hat_norm_max"] = np.linalg.norm(np.array([[0.02], [0.02], [0.02]]), 2) # max norm of a_hat
params["a_0"] = np.array([[0.0], [0.0], [0.0]]) # initial guess for a_hat
params["epsilon"] = 1e-2 # small value for numerical stability of projection operator

plannar_quad = PLANNAR_QUAD_UNCERTAIN(params)

# Disturbance
dist_config = {}
dist_config["include_dist"] = False
dist_config["gen_dist"] = lambda t: np.array([
    -0.5 + 0.8 * np.sin(2 * np.pi / 1 * t),
     0.5 + 0.8 * np.cos(2 * np.pi / 2 * t + 0.03),
     0.5 + 0.4 * np.sin(2 * np.pi / 3 * t + 0.01),
     0.5 + 0.2 * np.cos(2 * np.pi / 1 * t + 0.04),
     0.7 + 0.1 * np.sin(2 * np.pi / 4 * t + 0.01),
     0.2 + 0.5 * np.cos(2 * np.pi / 5 * t + 0.05),
], dtype=float).reshape((6,1))

# compute w_max
w_norms = [np.linalg.norm(dist_config["gen_dist"](t)) for t in np.arange(1.0, 10.0, 0.01)]
w_max = np.max(w_norms)

x_hist = np.zeros((plannar_quad.xdim, T_steps))
u_hist = np.zeros((plannar_quad.udim, T_steps))
energy_hist = np.zeros((T_steps,))
slack_hist = np.zeros((T_steps,))
x_d_hist = np.zeros((plannar_quad.xdim, T_steps))
u_d_hist = np.zeros((plannar_quad.udim, T_steps))

# initial state
x0 = np.zeros(plannar_quad.xdim)
x = x0.copy() + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # initial condition perturbation

# Main simulation loop
for i in range(T_steps):
    t = tt[i]
    # store current state
    x_hist[:, i] = x

    # nominal logs
    x_d = x_d_fcn(t)
    u_d = u_d_fcn(t)
    x_d_hist[:, i] = x_d
    u_d_hist[:, i] = u_d

    # Precompute geodesic
    plannar_quad.calc_geodesic(x, x_d)  
    energy_hist[i] = plannar_quad.Erem

    # Log controller inputs
    uc, slack = plannar_quad.ctrl_cra_ccm(x, x_d, u_d)
    u_hist[:, i] = uc.ravel()
    slack_hist[i] = slack

    # propagate with zero-order hold on control
    if i < T_steps - 1:
        t_span = (tt[i], tt[i + 1])
        
        # TODO: consider all cases of USE_ADAPTIVE and USE_CP
        sol = solve_ivp(
            lambda t, y: plannar_quad.cra_ccm_closed_loop_dyn(t, y, x_d_fcn, u_d_fcn, dist_config),
            t_span,
            x,
            method = "Radau",  # stiff solver comparable to ode23s
            rtol = 1e-3,
            atol = 1e-6,
            t_eval = [tt[i + 1]],
        )
        x = sol.y[:, -1]
    
    # Update adaptive parameter
    #if USE_ADAPTIVE:
    plannar_quad.update_a_ccm_hat(x, dt)

# Plot results
# x vs. x_d
fig, axs = plt.subplots(3, 2, figsize=(10, 8), constrained_layout=True)
axs = axs.flatten()

axs[0].plot(tt, x_d_hist[0, :], '--', label='Nominal')
axs[0].plot(tt, x_hist[0, :], '-', label='CCM')
axs[0].set_xlabel('Time (s)'); axs[0].set_ylabel('x (m)'); axs[0].legend()

axs[1].plot(tt, x_d_hist[1, :], '--')
axs[1].plot(tt, x_hist[1, :], '-')
axs[1].set_xlabel('Time (s)'); axs[1].set_ylabel('z (m)')

axs[2].plot(tt, x_d_hist[2, :] * 180.0/np.pi, '--')
axs[2].plot(tt, x_hist[2, :] * 180.0/np.pi)
axs[2].set_xlabel('Time (s)'); axs[2].set_ylabel('Phi (deg)')

axs[3].plot(tt, x_d_hist[3, :], '--')
axs[3].plot(tt, x_hist[3, :], '-')
axs[3].set_xlabel('Time (s)'); axs[3].set_ylabel('vx (m/s)')

axs[4].plot(tt, x_d_hist[4, :], '--')
axs[4].plot(tt, x_hist[4, :], '-')
axs[4].set_xlabel('Time (s)'); axs[4].set_ylabel('vz (m/s)')

axs[5].plot(tt, x_d_hist[5, :] * 180.0/np.pi, '--')
axs[5].plot(tt, x_hist[5, :] * 180.0/np.pi)
axs[5].set_xlabel('Time (s)'); axs[5].set_ylabel('Phi rate (deg/s)')

plt.suptitle('State: CCM vs Nominal')

# u vs u_d
plt.figure(figsize=(8,4))
plt.plot(tt, u_d_hist[0, :], 'b--', label='u_d_1')
plt.plot(tt, u_d_hist[1, :], 'r--', label='u_d_2')
plt.plot(tt, u_hist[0, :], 'b-', label='u_1: CCM')
plt.plot(tt, u_hist[1, :], 'r-', label='u_2: CCM')
plt.xlabel('Time (s)')
plt.ylabel('u (N)')
plt.legend()
plt.grid(True)

# Slack
plt.figure(figsize=(8,3))
plt.plot(tt, slack_hist, lw=1)
plt.xlabel('Time (s)')
plt.ylabel('QP slack')
plt.grid(True)

# Tracking error norm
plt.figure()
err_norm = np.linalg.norm(x_d_hist - x_hist, ord=2, axis=0)  # column-wise 2-norm
plt.plot(tt, err_norm)
plt.xlabel('Time (s)')
plt.ylabel('||x-x_d||_2')
plt.grid(True)

# Riemannian energy
plt.figure()
plt.plot(tt, np.sqrt(np.maximum(energy_hist, 0.0)))
plt.ylim([0, None])
plt.xlabel('Time (s)')
plt.ylabel('Riemann distance: sqrt E(t)')
plt.grid(True)

plt.show()