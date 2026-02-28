import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from scipy.integrate import solve_ivp
from dynsys.nonlinear_toy.nonlinear_toy import NONLINEAR_TOY
from dynsys.geodesic_solver import GeodesicSolver
from scipy.interpolate import interp1d

USE_CP = 0 # 1 or 0: whether to use conformal prediction
USE_ADAPTIVE = 0 # 1 or 0: whether to use adaptive control

VERIFY_GEODESIC = False
USE_QPSOLVERS = True

weight_slack = 5.0 if USE_CP else 1000.0

# Time setup
dt = 0.01
#sim_T = np.floor(t_d_data[-1]) # Simulation time
sim_T = 5.0 # Simulation time
tt = np.arange(0, sim_T, dt)
T_steps = len(tt)

# System parameters
params = {
    "ccm": {"rate": 0.8, "weight_slack": weight_slack},
    "geodesic": {"N": 8, "D": 2},
    "dt": dt,
}
params["use_adaptive"] = USE_ADAPTIVE
params["use_cp"] = USE_CP
params["Gamma_ccm"] = np.diag(np.array([0.5, 0.5, 0.5])) # adaptive gain matrix for CRaCCM
params["a_true"] = np.array([[-1.0], [-0.5], [-1.5]]) * 1 # true parameters [theta1, theta2, theta3]
params["a_hat_norm_max"] = np.linalg.norm(np.array([[1.0], [0.5], [1.5]]), 2) # max norm of a_hat
params["a_0"] = np.array([[1.0], [0.0], [-0.5]]) * 0.0# initial guess for a_hat
params["epsilon"] = 1e-2 # small value for numerical stability of projection operator
params["eta_ccm"] = 5.0
params["rho_ccm"] = 0.0

# Construct the system
toy = NONLINEAR_TOY(params)

# Compute x_d using the nominal dynamics with u_d to make sure the trajectory follows the nominal dynamics
def u_d_fcn(t):
    # Simple reference control: sinusoidal input to x3 equation

    u_d = np.array([[0.1 * np.sin(2 * np.pi / 2.0 * t)]])  # 2-second period

    return u_d

x_d_data = np.zeros((toy.xdim, T_steps))
x_d = np.array([0.0, 0.0, 0.0]) # initial state [x1, x2, x3]
for i in range(T_steps):
    t = tt[i]
    print("Time: ", t)
    u_d = u_d_fcn(t)
    x_d_data[:, i] = x_d
    # Propagate with zero-order hold on control
    if i < T_steps - 1:
        t_span = (tt[i], tt[i + 1])
        sol = solve_ivp(
            lambda t, y: toy.dynamics_nominal(y, u_d),
            t_span,
            x_d,
            method="BDF",
            rtol=1e-9,
            atol=1e-9,
            t_eval=[tt[i + 1]],
        )
        x_d = sol.y[:, -1]
interp_x = interp1d(
    tt, x_d_data, kind='linear', axis=1,
    bounds_error=False, fill_value='extrapolate'
)
x_d_fcn = lambda t: interp_x(t)

fig, axs = plt.subplots(3, 1)
axs = axs.flatten()
axs[0].plot(tt, x_d_data[0, :], '--')
axs[0].set_ylabel('x1'); 
axs[0].grid(True)
axs[1].plot(tt, x_d_data[1, :], '--')
axs[1].set_ylabel('x2')
axs[1].grid(True)
axs[2].plot(tt, x_d_data[2, :], '--')
axs[2].set_ylabel('x3')
axs[2].set_xlabel('Time (s)')
axs[2].grid(True)
plt.suptitle('Nominal Trajectory: x_d')
plt.show()

# Disturbance (non-parametric uncertainty)
Delta = lambda t: np.array([
    0.0,
    0.0,
    0.0,
], dtype=float)

# Compute upper bound of Delta
Delta_max = np.max([np.linalg.norm(Delta(t), 2) for t in np.arange(0.0, sim_T, 0.01)])
toy.cp_quantile = Delta_max * 0.8

# Time hisotry of logged data
x_hist = np.zeros((toy.xdim, T_steps))
u_hist = np.zeros((toy.udim, T_steps))
Erem_hist = np.zeros((T_steps,))
Erem_dot_hist = np.zeros((T_steps,))
V1_hist = np.zeros((T_steps,))
V2_hist = np.zeros((T_steps,))
U1_hist = np.zeros((T_steps,))
U2_hist = np.zeros((T_steps,))
slack_hist = np.zeros((T_steps,))
a_hat_ccm_hist = np.zeros((toy.adim, T_steps))
a_true_hist = np.zeros((toy.adim, T_steps))
nu_ccm_hist = np.zeros((T_steps,))
rho_ccm_hist = np.zeros((T_steps,))
x_d_hist = np.zeros((toy.xdim, T_steps))
u_d_hist = np.zeros((toy.udim, T_steps))

# Initial state
x = x_d_fcn(0).copy() + np.array([0.7, 0.5, -0.1])  # initial condition perturbation

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
    x_d = x_d_data[:,i]#x_d_fcn(t)
    u_d = u_d_fcn(t)
    x_d_hist[:, i] = x_d
    u_d_hist[:, i] = u_d.ravel()

    # Store adaptation parameters
    a_hat_ccm_hist[:, i] = toy.a_hat_ccm.ravel()
    a_true_hist[:, i] = toy.a_true.ravel()
    nu_ccm_hist[i] = toy.nu_ccm()
    rho_ccm_hist[i] = toy.rho_ccm

    # Compute geodesic
    toy.calc_geodesic(geodesic_solver, x, x_d, verify_geodesic=VERIFY_GEODESIC)

    # Implement ccm control law
    uc, slack = toy.ctrl_cra_ccm(x, x_d, u_d, use_qpsolvers=USE_QPSOLVERS)

    u_hist[:, i] = uc.ravel()
    slack_hist[i] = slack
    Erem_hist[i] = toy.Erem

    # Lyapunov function (for debugging)
    V1 = toy.nu_ccm() * (toy.Erem + toy.eta_ccm)
    a_tilde = toy.a_hat_ccm - toy.a_true
    V2 = a_tilde.T @ toy.Gamma_ccm @ a_tilde
    V1_hist[i] = V1.item()
    V2_hist[i] = V2.item()

    # Uncertanity-induced term to be cancelled by the adaptation law (for debugging)
    a_hat_dot = toy.a_hat_dot if USE_ADAPTIVE else np.zeros((toy.adim, 1))
    rho_dot = toy.rho_dot if USE_ADAPTIVE else 0.0
    dErem_dai = toy.dErem_dai if USE_ADAPTIVE else np.zeros(toy.adim)
    U1_hist[i] = (2 * a_tilde.T @ (-toy.nu_ccm() * toy.Y(x).T @ toy.gamma_s1_M_x.T + toy.Gamma_ccm @ a_hat_dot)).item()
    U2_hist[i] = (toy.nu_ccm() * (2 * toy.gamma_s0_M_d @ toy.Y(x_d) @ toy.a_hat_ccm + dErem_dai @ a_hat_dot) + toy.dnu_drho_ccm() * rho_dot * (toy.Erem + toy.eta_ccm)).item()

    # Edot error (for debugging)
    Erem_dot_fixa = (toy.gamma_s1_M_x @ (toy.f(x) + toy.g(x) @ uc + toy.Y(x) @ toy.a_true)
                   - toy.gamma_s0_M_d @ (toy.f(x_d) + toy.g(x_d) @ u_d))
    Erem_dot_hist[i] = (Erem_dot_fixa + dErem_dai @ a_hat_dot).item()

    # Propagate with zero-order hold on control and disturbance
    if i < T_steps - 1:
        t_span = (tt[i], tt[i + 1])
        sol = solve_ivp(
            lambda t, y: toy.dynamics(y, uc) + Delta(t),
            t_span,
            x,
            method="BDF",
            rtol=1e-6,
            atol=1e-6,
            t_eval=[tt[i + 1]],
        )
        try:
            x = sol.y[:, -1]
        except Exception as e:
            print("Error occurred while solving IVP:", e)

# Plot results
# x vs. x_d
fig, axs = plt.subplots(3, 1)
axs = axs.flatten()
axs[0].plot(tt, x_d_hist[0, :], '--', label='Nominal')
axs[0].plot(tt, x_hist[0, :], '-', label='CCM')
axs[0].set_ylabel('x1'); 
axs[0].legend()
axs[0].grid(True)
axs[1].plot(tt, x_d_hist[1, :], '--')
axs[1].plot(tt, x_hist[1, :], '-')
axs[1].set_ylabel('x2')
axs[1].grid(True)
axs[2].plot(tt, x_d_hist[2, :], '--')
axs[2].plot(tt, x_hist[2, :], '-') 
axs[2].set_ylabel('x3')
axs[2].set_xlabel('Time (s)')
axs[2].grid(True)
plt.suptitle('State: Actual vs. Nominal')

# Controls and slack
fig, axs = plt.subplots(2, 1)
axs = axs.flatten()
axs[0].plot(tt, u_hist[0, :], 'r-', label='u: CCM')
axs[0].plot(tt, u_d_hist[0, :], 'k--', label='u_d')
axs[0].set_ylabel('Control u')
axs[0].legend()
axs[0].grid(True)
axs[1].plot(tt, slack_hist)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('QP slack')
axs[1].grid(True)

# Erem dot error
Erem_dot_num = np.gradient(Erem_hist, tt)
fig, axs = plt.subplots(2, 1)
axs[0].plot(tt, Erem_dot_hist, label='analytic: computed using geodesics')
axs[0].plot(tt, Erem_dot_num, label='numeric: np.gradient of Erem')
axs[0].legend()
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Erem dot')
axs[0].grid(True)
axs[1].plot(tt, Erem_dot_hist - Erem_dot_num)
axs[1].legend()
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Erem dot error')
axs[1].grid(True)


# Uncertainty parameters
fig, axs = plt.subplots(toy.adim, 1)
axs = axs.flatten()
for i in range(toy.adim):
    axs[i].plot(tt, a_hat_ccm_hist[i, :], label='a_hat')
    axs[i].plot(tt, a_true_hist[i, :], label='a_true')
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