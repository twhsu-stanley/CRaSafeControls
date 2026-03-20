import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from scipy.integrate import solve_ivp
from dynsys.planar_quad.planar_quad import PLANAR_QUAD
from dynsys.geodesic_solver import GeodesicSolver
from scipy.io import loadmat
from scipy.interpolate import interp1d

USE_CP = False # whether to use conformal prediction
USE_ADAPTIVE = False # whether to use adaptive control

VERIFY_GEODESIC = False
USE_QPSOLVERS = False
USE_SLACK = True

weight_slack = 10.0 if USE_CP else 1000.0

# Load the desired trajectory from MATLAB code
data = loadmat("simulations/planar_quad/nomTraj.mat")
t_d_data = data["soln"]["grid"][0][0][0][0][0][0,:]
x_d_data = data["soln"]["grid"][0][0][0][0][1][:]
u_d_data = data["soln"]["grid"][0][0][0][0][2][:]
interp_x_d = interp1d(
    t_d_data, x_d_data, kind='cubic', axis=1,
    bounds_error=False, fill_value='extrapolate'
)
interp_u_d = interp1d(
    t_d_data, u_d_data.reshape(2,1,-1), kind='cubic', axis=2,
    bounds_error=False, fill_value='extrapolate'
)

# Time setup
dt = 0.01
sim_T = t_d_data[-1] # Simulation time
tt = np.arange(0, sim_T, dt)
T_steps = len(tt)

# System parameters
params = {
    "l": 0.25,
    "m": 0.486,
    "g": 9.81,
    "J": 0.00383,
    "ccm": {"rate": 0.8},
    "geodesic": {"N": 8, "D": 2}, # N = D + a
    "weight_slack": weight_slack,
    "dt": dt,
}
params["use_adaptive"] = USE_ADAPTIVE
params["use_cp"] = USE_CP
params["Gamma_ccm"] = np.diag(np.array([15, 15])) # adaptive gain matrix for CRaCCM
params["a_true"] = np.array([[-0.0], [-0.6]]) # true parameters
params["a_hat_norm_max"] = np.linalg.norm(np.array([[1.0], [0.6]]), 2) # max norm of a_hat
params["epsilon"] = 1e-3 # small value for numerical stability of projection operator
params["eta_ccm"] = 5.0

# Construct the system
pq = PLANAR_QUAD(params)

# Disturbance (non-parametric uncertainty)
Delta = lambda t: np.array([
    0.0,#0.04 + 0.05 * np.sin(2 * np.pi / 0.3 * t),
    0.0,#0.1 + 0.05 * np.cos(2 * np.pi / 0.5 * t + 0.03),
    0.0,#-0.1 + 0.2 * np.sin(2 * np.pi / 2 * t + 0.01),
    0.0,#0.1 + 0.1 * np.cos(2 * np.pi / 1 * t + 0.04),
    0.0,#0.05 + 0.12 * np.sin(2 * np.pi / 0.2 * t + 0.01),
    0.0,#0.0 + 0.2 * np.cos(2 * np.pi / 1.6 * t + 0.05),
], dtype = float)#.reshape((6,1))

# Compute upper bound of Delta
Delta_max = np.max([np.linalg.norm(Delta(t), 2) for t in np.arange(0.0, sim_T, 0.01)])
pq.cp_quantile = Delta_max * 0.95

# Time hisotry of logged data
x_hist = np.zeros((pq.xdim, T_steps))
u_hist = np.zeros((pq.udim, T_steps))
Erem_hist = np.zeros((T_steps,))
Erem_dot_hist = np.zeros((T_steps,))
V1_hist = np.zeros((T_steps,))
V2_hist = np.zeros((T_steps,))
slack_hist = np.zeros((T_steps,))
a_hat_ccm_hist = np.zeros((pq.adim, T_steps))
a_true_hist = np.zeros((pq.adim, T_steps))
nu_ccm_hist = np.zeros((T_steps,))
rho_ccm_hist = np.zeros((T_steps,))

# Initial state
x = interp_x_d(0).copy() + np.array([-1.0, -0.0, 0.0, -1.2, -0.5, 0.0])  # initial condition + perturbation
a_hat_ccm = np.array([[0.0], [0.0]]) # initial guess for a_hat
rho_ccm = 0.0
x_ext = np.hstack((x, a_hat_ccm.ravel(), rho_ccm)) # extended state with a_hat and rho

# Initialize geodesic solver
N = pq.params["geodesic"]["N"]
D = pq.params["geodesic"]["D"]
n = pq.xdim
geodesic_solver = GeodesicSolver(n, D, N, pq.W_fcn, pq.dW_dxi_fcn, pq.dW_dai_fcn)

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
    a_hat_ccm_hist[:, i] = a_hat_ccm.ravel()
    a_true_hist[:,i] = pq.a_true.ravel() # TODO: just to verify that a_true is constant; can remove later
    nu_ccm_hist[i] = pq.nu_ccm(rho_ccm)
    rho_ccm_hist[i] = rho_ccm

    # Implement ccm control law
    uc, slack = pq.ctrl_craccm(x, a_hat_ccm, x_d, u_d, geodesic_solver, use_qpsolvers=USE_QPSOLVERS)
    #uc = u_d + u_qp
    u_hist[:,i] = uc.ravel()
    slack_hist[i] = slack
    Erem_hist[i] = pq.Erem

    # For debugging #######################################################################
    # Lyapunov function
    V1 = pq.nu_ccm(rho_ccm) * (pq.Erem + pq.eta_ccm)
    a_tilde = a_hat_ccm - pq.a_true
    V2 = a_tilde.T @ pq.Gamma_ccm @ a_tilde
    V1_hist[i] = V1.item()
    V2_hist[i] = V2.item()
    #######################################################################################

    # Propagate with zero-order hold on control
    if i < T_steps - 1:
        t_span = (tt[i], tt[i + 1])
        
        sol = solve_ivp(
            #lambda t, y: pq.dynamics(y, uc) + Delta(t),
            lambda t, y: pq.dynamics_extended(y, x_d, uc, geodesic_solver),
            t_span,
            x_ext,
            method = "BDF", #"LSODA", #"Radau",  # stiff solver
            rtol = 1e-6, # 1e-6
            atol = 1e-6, # 1e-6
            t_eval = [tt[i + 1]],
        )
        try:
            x_ext = sol.y[:, -1]
        except Exception as e:
            raise ValueError("Error occurred while solving IVP:", e)
        
        x = x_ext[0:pq.xdim]
        a_hat_ccm = x_ext[pq.xdim:(pq.xdim+pq.adim)].reshape(-1,1)
        rho_ccm = x_ext[(pq.xdim+pq.adim)]

# Plot results
# x vs. x_d
fig, axs = plt.subplots(3, 2)
axs = axs.flatten()
axs[0].plot(tt, interp_x_d(tt)[0, :], '--', label='Nominal')
axs[0].plot(tt, x_hist[0, :], '-', label='CCM')
axs[0].set_ylabel('x (m)'); 
axs[0].legend()
axs[0].grid(True)
axs[1].plot(tt, interp_x_d(tt)[1, :], '--')
axs[1].plot(tt, x_hist[1, :], '-')
axs[1].set_ylabel('z (m)')
axs[1].grid(True)
axs[2].plot(tt, interp_x_d(tt)[2, :] * 180.0/np.pi, '--')
axs[2].plot(tt, x_hist[2, :] * 180.0/np.pi) 
axs[2].set_ylabel('Phi (deg)')
axs[2].grid(True)
axs[3].plot(tt, interp_x_d(tt)[3, :], '--')
axs[3].plot(tt, x_hist[3, :], '-')
axs[3].set_ylabel('vx (m/s)')
axs[3].grid(True)
axs[4].plot(tt, interp_x_d(tt)[4, :], '--')
axs[4].plot(tt, x_hist[4, :], '-')
axs[4].set_xlabel('Time (s)'); 
axs[4].set_ylabel('vz (m/s)')
axs[4].grid(True)
axs[5].plot(tt, interp_x_d(tt)[5, :] * 180.0/np.pi, '--')
axs[5].plot(tt, x_hist[5, :] * 180.0/np.pi)
axs[5].set_xlabel('Time (s)'); 
axs[5].set_ylabel('Phi rate (deg/s)')
axs[5].grid(True)
plt.suptitle('State: Actual vs. Nominal')

# Controls and slack
fig, axs = plt.subplots(3, 1)
axs = axs.flatten()
axs[0].plot(tt, u_hist[0, :], 'r-', label='u_1: CCM')
axs[0].plot(tt, interp_u_d(tt)[0,0,:], 'k--', label='u_d_1')
axs[0].set_ylabel('u0 (N)')
axs[0].legend()
axs[0].grid(True)
axs[1].plot(tt, u_hist[1, :], 'b-', label='u_2: CCM')
axs[1].plot(tt, interp_u_d(tt)[1,0,:], 'k--', label='u_d_2')
axs[1].set_ylabel('u1 (N)')
axs[1].legend()
axs[1].grid(True)
axs[2].plot(tt, slack_hist)
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('QP slack')
axs[2].grid(True)

# Uncertainty parameters
fig, axs = plt.subplots(pq.adim, 1, squeeze=False)
#axs = axs.flatten()
for i in range(pq.adim):
    axs[i,0].plot(tt, a_hat_ccm_hist[i, :], label='a hat')
    axs[i,0].plot(tt, a_true_hist[i, :], label='a true')
    axs[i,0].legend()
    axs[i,0].grid(True)
    axs[i,0].set_ylabel(f"a{i}_hat_ccm")
axs[pq.adim-1,0].set_xlabel('Time (s)')

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
fig, axs = plt.subplots(2, 1)
axs = axs.flatten()
# Erem
axs[0].plot(tt, Erem_hist)
axs[0].set_ylabel('Riemann Energy: Erem(t)')
axs[0].grid(True)
# Lyapunov
axs[1].plot(tt, V1_hist, label='V1 = nu(rho)(E+eta)')
axs[1].plot(tt, V2_hist, label='V2 = a~^T Gamma a~')
axs[1].plot(tt, V1_hist + V2_hist, label='V = V1+V2')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Lyapunov: V(t)')
axs[1].legend()
axs[1].grid(True)

plt.show()