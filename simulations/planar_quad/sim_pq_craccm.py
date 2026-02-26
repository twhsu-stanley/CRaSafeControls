import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from scipy.integrate import solve_ivp
from dynsys.planar_quad.planar_quad_uncertain import PLANAR_QUAD_UNCERTAIN
from dynsys.planar_quad.planar_quad import PLANAR_QUAD
from dynsys.geodesic_solver import GeodesicSolver
from scipy.io import loadmat
from scipy.interpolate import interp1d

USE_CP = 0 # 1 or 0: whether to use conformal prediction
USE_ADAPTIVE = 1 # 1 or 0: whether to use adaptive control

weight_slack = 5.0 if USE_CP else 10000.0

# Load the desired trajectory
# TODO: load a motion planner
# x_d_fcn and u_d_fcn: functions of time
data = loadmat("simulations/planar_quad/nomTraj.mat")
t_d_data = data["soln"]["grid"][0][0][0][0][0][0,:]
x_d_data = data["soln"]["grid"][0][0][0][0][1][:]
u_d_data = data["soln"]["grid"][0][0][0][0][2][:]
# interpolation functions
interp_x = interp1d(
    t_d_data, x_d_data, kind='linear', axis=1,
    bounds_error=False, fill_value='extrapolate'
)
interp_u = interp1d(
    t_d_data, u_d_data.reshape(2,1,-1), kind='linear', axis=2,
    bounds_error=False, fill_value='extrapolate'
)
x_d_fcn = lambda t: interp_x(t)
u_d_fcn = lambda t: interp_u(t)

# Disturbance (non-parametric uncertainty)
Delta = lambda t: np.array([
    0.0,#0.04 + 0.05 * np.sin(2 * np.pi / 0.3 * t),
    0.0,#0.1 + 0.05 * np.cos(2 * np.pi / 0.5 * t + 0.03),
    0.0,#-0.1 + 0.2 * np.sin(2 * np.pi / 2 * t + 0.01),
    0.0,#0.1 + 0.1 * np.cos(2 * np.pi / 1 * t + 0.04),
    0.0,#0.05 + 0.12 * np.sin(2 * np.pi / 0.2 * t + 0.01),
    0.0,#0.0 + 0.2 * np.cos(2 * np.pi / 1.6 * t + 0.05),
], dtype = float)#.reshape((6,1))

# Time setup
dt = 0.01
sim_T = np.floor(t_d_data[-1]) # Simulation time
tt = np.arange(0, sim_T, dt)
T_steps = len(tt)

# Compute upper bound of Delta
Delta_max = np.max([np.linalg.norm(Delta(t), 2) for t in np.arange(0.0, sim_T, 0.01)])

# System parameters
params = {
    "l": 0.25,
    "m": 0.486,
    "g": 9.81,
    "J": 0.00383,
    "ccm": {"rate": 0.8, "weight_slack": weight_slack},
    "geodesic": {"N": 6, "D": 2},
    "dt": dt,
}
params["use_adaptive"] = USE_ADAPTIVE
params["use_cp"] = USE_CP
params["cp_quantile"] = Delta_max * 0.8 if USE_CP else 0.0
params["Gamma_ccm"] = np.diag(np.array([1, 1, 1, 1])) # adaptive gain matrix for CRaCCM
params["a_true"] = np.array([[0.2], [0.01], [-0.2], [0.01]]) # true parameters
params["a_hat_norm_max"] = np.linalg.norm(np.array([[0.4], [0.1], [0.4], [0.1]]), 2) # max norm of a_hat
params["a_0"] = np.array([[-0.1], [0.02], [0.05], [0.02]]) # initial guess for a_hat
params["epsilon"] = 1e-2 # small value for numerical stability of projection operator

params["eta_ccm"] = 5
params["rho_ccm"] = 0.0

# Construct the system
pq = PLANAR_QUAD_UNCERTAIN(params)
#pq = PLANAR_QUAD(params)

# Time hisotry of logged data
x_hist = np.zeros((pq.xdim, T_steps))
u_hist = np.zeros((pq.udim, T_steps))
Erem_hist = np.zeros((T_steps,))
Erem_dot_err_hist = np.zeros((T_steps,))
V1_hist = np.zeros((T_steps,))
V2_hist = np.zeros((T_steps,))
U1_hist = np.zeros((T_steps,))
U2_hist = np.zeros((T_steps,))
slack_hist = np.zeros((T_steps,))
a_hat_ccm_hist = np.zeros((pq.adim, T_steps))
a_true_hist = np.zeros((pq.adim, T_steps))
nu_ccm_hist = np.zeros((T_steps,))
rho_ccm_hist = np.zeros((T_steps,))
x_d_hist = np.zeros((pq.xdim, T_steps))
u_d_hist = np.zeros((pq.udim, T_steps))

# Initial state
x0 = x_d_fcn(0)
x = x0.copy() #+ np.array([0.5, 0.2, np.pi/4, 0.3, 0.2, np.pi/3])  # initial condition perturbation

# Initialize geodesic solver
N = pq.params["geodesic"]["N"]
D = pq.params["geodesic"]["D"]
n = pq.xdim
geodesic_solver = GeodesicSolver(n, D, N, pq.W_fcn, pq.dW_dxi_fcn, pq.dW_dai_fcn)

# Compute initial Erem (for debugging)
pq.calc_geodesic(geodesic_solver, x, x_d_fcn(0))

# Main simulation loop
for i in range(T_steps):
    t = tt[i]
    print("Time: ", t)

    # Store current state
    x_hist[:, i] = x

    # Nominal trajectory
    x_d = x_d_fcn(t)
    u_d = u_d_fcn(t)
    x_d_hist[:, i] = x_d
    u_d_hist[:, i] = u_d.ravel()

    # Store adaptation parameters
    a_hat_ccm_hist[:, i] = pq.a_hat_ccm.ravel()
    a_true_hist[:,i] = pq.a_true.ravel() # TODO: just to verify that a_true is constant; can remove later
    nu_ccm_hist[i] = pq.nu_ccm()
    rho_ccm_hist[i] = pq.rho_ccm

    # Store previous Erem (for debugging)
    Erem_prev = pq.Erem

    # Compute geodesic
    pq.calc_geodesic(geodesic_solver, x, x_d)

    # Implement ccm control law
    uc, slack = pq.ctrl_cra_ccm(x, x_d, u_d)

    u_hist[:, i] = uc.ravel()
    slack_hist[i] = slack
    Erem_hist[i] = pq.Erem

    # Lyapunov function (for debugging)
    V1 = pq.nu_ccm() * (pq.Erem + pq.eta_ccm)
    a_tilde = pq.a_hat_ccm - pq.a_true
    V2 = a_tilde.T @ pq.Gamma_ccm @ a_tilde
    V1_hist[i] = V1.item()
    V2_hist[i] = V2.item()

    # Uncertanity-induced term to be cancelled by the adaptation law (for debugging)
    a_hat_dot = pq.a_hat_dot if USE_ADAPTIVE else np.zeros((pq.adim,1))
    rho_dot = pq.rho_dot if USE_ADAPTIVE else 0.0
    dErem_dai = pq.dErem_dai if USE_ADAPTIVE else np.zeros(pq.adim)
    U1_hist[i] = ( 2 * a_tilde.T @ (-pq.nu_ccm() * pq.Y(x).T @ pq.gamma_s1_M_x.T + pq.Gamma_ccm @ a_hat_dot) ).item()
    U2_hist[i] = ( pq.nu_ccm() * (2 * pq.gamma_s0_M_d @ pq.Y(x_d) @ pq.a_hat_ccm +  dErem_dai @ a_hat_dot) + pq.dnu_drho_ccm() * rho_dot * (pq.Erem + pq.eta_ccm) ).item()

    # Edot error (for debugging)
    Erem_dot_fixa = (pq.gamma_s1_M_x @ (pq.f(x) + pq.g(x) @ uc + pq.Y(x) @ pq.a_true)
                   - pq.gamma_s0_M_d @ (pq.f(x_d) + pq.g(x_d) @ u_d))
    Erem_dot_err_hist[i] = ((Erem_dot_fixa + dErem_dai @ a_hat_dot) - (pq.Erem - Erem_prev)/dt).item()

    # Propagate with zero-order hold on control and disturbance
    if i < T_steps - 1:
        t_span = (tt[i], tt[i + 1])
        
        sol = solve_ivp(
            lambda t, y: pq.dynamics(y, uc) + Delta(t),
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

# Plot results
# x vs. x_d
fig, axs = plt.subplots(3, 2)
axs = axs.flatten()
axs[0].plot(tt, x_d_hist[0, :], '--', label='Nominal')
axs[0].plot(tt, x_hist[0, :], '-', label='CCM')
axs[0].set_ylabel('x (m)'); 
axs[0].legend()
axs[0].grid(True)
axs[1].plot(tt, x_d_hist[1, :], '--')
axs[1].plot(tt, x_hist[1, :], '-')
axs[1].set_ylabel('z (m)')
axs[1].grid(True)
axs[2].plot(tt, x_d_hist[2, :] * 180.0/np.pi, '--')
axs[2].plot(tt, x_hist[2, :] * 180.0/np.pi) 
axs[2].set_ylabel('Phi (deg)')
axs[2].grid(True)
axs[3].plot(tt, x_d_hist[3, :], '--')
axs[3].plot(tt, x_hist[3, :], '-')
axs[3].set_ylabel('vx (m/s)')
axs[3].grid(True)
axs[4].plot(tt, x_d_hist[4, :], '--')
axs[4].plot(tt, x_hist[4, :], '-')
axs[4].set_xlabel('Time (s)'); 
axs[4].set_ylabel('vz (m/s)')
axs[4].grid(True)
axs[5].plot(tt, x_d_hist[5, :] * 180.0/np.pi, '--')
axs[5].plot(tt, x_hist[5, :] * 180.0/np.pi)
axs[5].set_xlabel('Time (s)'); 
axs[5].set_ylabel('Phi rate (deg/s)')
axs[5].grid(True)
plt.suptitle('State: Actual vs. Nominal')

# Controls and slack
fig, axs = plt.subplots(3, 1)
axs = axs.flatten()
axs[0].plot(tt, u_hist[0, :], 'r-', label='u_1: CCM')
axs[0].plot(tt, u_d_hist[0, :], 'k--', label='u_d_1')
axs[0].set_ylabel('u0 (N)')
axs[0].legend()
axs[0].grid(True)
axs[1].plot(tt, u_hist[1, :], 'b-', label='u_2: CCM')
axs[1].plot(tt, u_d_hist[1, :], 'k--', label='u_d_2')
axs[1].set_ylabel('u1 (N)')
axs[1].legend()
axs[1].grid(True)
axs[2].plot(tt, slack_hist)
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('QP slack')
axs[2].grid(True)

# Erem dot error
plt.figure()
plt.plot(tt, Erem_dot_err_hist, lw=1)
plt.xlabel('Time (s)')
plt.ylabel('E dot error')
plt.grid(True)

# Uncertainty parameters
fig, axs = plt.subplots(pq.adim, 1)
axs = axs.flatten()
for i in range(pq.adim):
    axs[i].plot(tt, a_hat_ccm_hist[i, :], label='a hat')
    axs[i].plot(tt, a_true_hist[i, :], label='a true')
    axs[i].legend()
    axs[i].grid(True)
    axs[i].set_ylabel(f"a{i}_hat_ccm")
axs[pq.adim-1].set_xlabel('Time (s)')

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
err_norm = np.linalg.norm(x_d_hist - x_hist, ord=2, axis=0)  # column-wise 2-norm
axs[0].plot(tt, err_norm)
axs[0].set_ylabel('||x-x_d||_2')
axs[0].grid(True)
# Erem
axs[1].plot(tt, Erem_hist)
axs[1].set_ylabel('Riemann Energy: Erem(t)')
axs[1].grid(True)
# Lyapunov
axs[2].plot(tt, V1_hist, label='V1 = nu(rho)(E+eta)')
axs[2].plot(tt, V2_hist, label='V2 = a~^T Gamma a~')
axs[2].plot(tt, V1_hist + V2_hist, label='V = V1+V2')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Lyapunov: V(t)')
axs[2].legend()
axs[2].grid(True)

# Uncertainty-induced terms
plt.figure()
plt.plot(tt, U1_hist, label='U1')
plt.plot(tt, U2_hist, label='U2')
plt.plot(tt, U1_hist + U2_hist, label='U1+U2')
plt.xlabel('Time (s)')
plt.legend()
plt.ylabel('Uncertainty-induced term: U(t)')
plt.grid(True)

plt.show()