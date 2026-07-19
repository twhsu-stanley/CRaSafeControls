import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import lsq_linear

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from olacp import OLACP
from systems.inverted_pendulum.ip import IP


USE_CP = True
USE_ADAPTIVE = True

# Algorithm 1 setup.
K = 20  # total number of time intervals I_k
B = 2   # update the representation every B intervals

# Time setup for each interval I_k.
dt = 0.01
interval_duration = 2.5
I_length = int(round(interval_duration / dt))
if I_length < 2 or not np.isclose(I_length * dt, interval_duration):
    raise ValueError("interval_duration must be an integer multiple of dt")
sim_T = K * interval_duration
tt = np.arange(0.0, sim_T, dt)
if len(tt) != K * I_length:
    raise ValueError("K, interval_duration, and dt define inconsistent sample counts")


def wind_velocity(t):
    """Piecewise-constant 2-D wind, changed at each interval boundary."""
    interval_index = int(np.floor(max(float(t), 0.0) / interval_duration))
    phase = 2.0 * np.pi * (interval_index % K) / K
    return np.array(
        [0.05 * np.cos(phase), 0.02 * np.sin(phase + 0.2)]
    )

# Bounds surround a rank-two factorization of the total physical uncertainty.
# The first latent direction captures the fixed plant mismatch and the second
# captures the wind-dependent variation, with margin for representation error.
theta_lb = 10 * np.array(
    [
        [-5.0, -5.0, -0.0, -5.0, -5.0],
        [-5.0, -5.0, -0.0, -5.0, -5.0],
        [-5.0, -5.0, -0.0, -5.0, -5.0],
        [-5.0, -5.0, -5.0, -5.0, -5.0],
        [-5.0, -5.0, -0.0, -5.0, -5.0],
        [-5.0, -5.0, -0.0, -5.0, -5.0],
    ]
)
theta_ub = 10 * np.array(
    [
        [5.0, 5.0, 0.0, 5.0, 5.0],
        [5.0, 5.0, 0.0, 5.0, 5.0],
        [5.0, 5.0, 0.0, 5.0, 5.0],
        [5.0, 5.0, 0.1, 5.0, 5.0],
        [5.0, 5.0, 0.0, 5.0, 5.0],
        [5.0, 5.0, 0.0, 5.0, 5.0],
    ]
)
theta_rng = np.random.default_rng(11)
Theta_init = theta_rng.uniform(theta_lb, theta_ub)
# These latent coordinates are related to, but are not equal to, [w_x, w_z].
a_lb = np.array([-0.5, -0.5, -0.5, -0.5, -0.5])
a_ub = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
a_center = 0.5 * (a_lb + a_ub)
projection_epsilon = 0.01
a_radius = 0.5 * np.linalg.norm(a_ub - a_lb, ord=2) + projection_epsilon

params = {
    "length": 1.0,       # pendulum length [m]
    "mass": 1.0,       # nominal mass [kg]
    #"inertia": 0.5,    # nominal inertia [kg m^2]
    "grav": 9.81,   # gravitational acceleration [m/s^2]
    "damping": 0.04,      # nominal damping [N m s/rad]
    "T_a": 0.01,    # actuator time constant [s]
    "c_w": 0.01,    # wind drag coefficient
    "true_damping": 0.2, # true damping, unknown to the controller
    "true_mass": 3.0, # true mass, unknown to the controller
    #"true_inertia": 0.50, # true inertia, unknown to the controller
    "wind_velocity": wind_velocity,
    "Theta_init": Theta_init,
    "use_adaptive": USE_ADAPTIVE,
    "use_cp": USE_CP,
    "Gamma_clf": np.diag([0.01, 0.01, 0.01, 0.01, 0.01]) * 1e-6,
    "a_ub": a_ub,
    "a_lb": a_lb,
    "a_hat_norm_max": a_radius,
    "epsilon": projection_epsilon,
    "eta_clf": 10.0,
    # The backstepping construction establishes rate 4. A smaller QP rate
    # leaves control authority for the conformal tightening.
    "clf_rate": 1.0,
    "weight_slack": 1e4,
}

# Construct the actuator-augmented inverted pendulum.
system = IP(params)

# Initial calibration set. Each historical interval follows lines 8 and 11 of
# Algorithm 1: fit a bounded a_k and retain the largest residual norm.
rng = np.random.default_rng(7)
N_cal = 100
S_cal_init = []
for k in range(N_cal):
    calibration_states = np.column_stack(
        (
            rng.uniform(-1.25, 1.25, I_length),
            rng.uniform(-1.5, 1.5, I_length),
            rng.uniform(-1.0, 1.0, I_length),
        )
    )
    Y_cal = []
    w_cal = []
    for x_cal in calibration_states:
        Y_cal.append(system.Y_theta(x_cal, Theta_init))
        w_cal.append(
            system.true_uncertainty(x_cal, k * interval_duration)
        )

    Y_stack = np.vstack(Y_cal)
    w_stack = np.hstack(w_cal)
    a_cal = lsq_linear(Y_stack, w_stack, bounds=(a_lb, a_ub)).x
    S_cal_init.append(
        max(
            np.linalg.norm(Y_t @ a_cal - w_t, ord=2)
            for Y_t, w_t in zip(Y_cal, w_cal)
        )
    )
S_cal_init = np.asarray(S_cal_init)

# Adaptive conformal prediction and Algorithm 1 representation learning.
olacp = OLACP(
    S_cal_init,
    N_cal=N_cal,
    acp_lr=0.02,
    delta_target=0.05,
    delta_init=0.05,
    buffer_maxlen=I_length,
    theta_init=Theta_init,
    representation_period=B,
    representation_lr=2e-3, #lambda j: 2e-3 / j,
    theta_lb=theta_lb,
    theta_ub=theta_ub,
    Y_theta=system.Y_theta,
    representation_loss_gradient=system.representation_loss_gradient,
)
system.set_representation(olacp.Theta)
system.cp_quantile = olacp.Q_k

# Simulation initialization.
x = np.array([1.2, 0.6, 0.0])
a_hat_clf = a_center.copy()
rho_clf = 0.0
x_ext = np.hstack((x, a_hat_clf, rho_clf))

x_hist = np.zeros((len(tt), system.xdim))
u_hist = np.zeros(len(tt))
V_hist = np.zeros(len(tt))
a_hat_clf_hist = np.zeros((len(tt), system.adim))
a_k_hist = np.full((len(tt), system.adim), np.nan)
rho_clf_hist = np.zeros(len(tt))
nu_clf_hist = np.zeros(len(tt))
Q_k_hist = np.zeros(len(tt))
Theta_hist = np.zeros((len(tt), Theta_init.size))
prediction_error_hist = np.full(len(tt), np.nan)

interval_times = []
s_k_hist = []
delta_k_hist = []
e_k_hist = []

# Main simulation loop.
for i, t in enumerate(tt):
    x_hist[i] = x
    a_hat_clf_hist[i] = a_hat_clf
    rho_clf_hist[i] = rho_clf
    nu_clf_hist[i] = system.nu_clf(rho_clf)
    Q_k_hist[i] = system.cp_quantile
    Theta_hist[i] = system.Theta_hat.reshape(-1)
    V_hist[i] = float(np.asarray(system.clf(x, a_hat_clf)).item())

    u_ref = system.ctrl_nominal(x)
    u, _ = system.ctrl_craclf(x, a_hat_clf, u_ref, use_slack=False)
    u_hist[i] = u.item()

    # Store the data required by line 6 of Algorithm 1.
    olacp.add_data_to_buffers(
        x,
        system.dynamics_nominal(x, u),
        xdot=system.dynamics(x, u, t),
    )

    # Propagate with zero-order hold on the control input.
    if i < len(tt) - 1:
        sol = solve_ivp(
            lambda tau, y: system.dynamics_extended(y, u, tau),
            (tt[i], tt[i + 1]),
            x_ext,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
            t_eval=[tt[i + 1]],
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        x_ext = sol.y[:, -1]
        x = x_ext[: system.xdim]
        a_hat_clf = x_ext[system.xdim : system.xdim + system.adim]
        rho_clf = x_ext[system.xdim + system.adim]

    # Complete lines 7--23 of Algorithm 1 at the interval boundary.
    if (i + 1) % I_length == 0:
        interval_index = (i + 1) // I_length - 1
        olacp.estimate_uncertainty(dt)
        s_k = olacp.compute_score(system.a_ub, system.a_lb)
        interval_prediction_error = np.array(
            [
                np.linalg.norm(Y_t @ olacp.a_k - w_t, ord=2) ** 2
                for Y_t, w_t in zip(olacp._Y_buffer, olacp._w_buffer)
            ]
        )
        e_k = olacp.update_delta(s_k)
        olacp.append_score(s_k)
        representation_update = olacp.update_representation()

        interval_start = i - I_length + 1
        a_k_hist[interval_start : i + 1] = olacp.a_k
        prediction_error_hist[interval_start : i + 1] = (
            interval_prediction_error
        )
        interval_times.append(t)
        s_k_hist.append(s_k)
        delta_k_hist.append(olacp.delta)
        e_k_hist.append(e_k)

        if representation_update is not None:
            system.set_representation(representation_update["Theta"])

        system.cp_quantile = olacp.compute_quantile()
        olacp.clear_buffers()

        print(
            f"interval={interval_index + 1:02d}, "
            f"score={s_k:.3e}, Q={Q_k_hist[i]:.3e}, "
            f"delta_next={olacp.delta:.3f}, miscoverage={e_k}"
        )

interval_times = np.asarray(interval_times)
s_k_hist = np.asarray(s_k_hist)
delta_k_hist = np.asarray(delta_k_hist)
e_k_hist = np.asarray(e_k_hist)

# Plot pendulum and actuator states.
fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(tt, np.rad2deg(x_hist[:, 0]), linewidth=1.4)
axs[0].set_ylabel(r"$\phi$ (deg)")
axs[1].plot(tt, np.rad2deg(x_hist[:, 1]), linewidth=1.4)
axs[1].set_ylabel(r"$\dot{\phi}$ (deg/s)")
axs[2].plot(tt, x_hist[:, 2], linewidth=1.4)
axs[2].set_ylabel(r"$\tau_a$ (N m)")
axs[2].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
fig.suptitle("Actuator-augmented inverted-pendulum states")

# Plot the two-dimensional wind disturbance.
wind_hist = np.asarray([wind_velocity(t) for t in tt])
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(tt, wind_hist[:, 0], linewidth=1.4)
axs[0].set_ylabel(r"$w_x$ (m/s)")
axs[1].plot(tt, wind_hist[:, 1], linewidth=1.4)
axs[1].set_ylabel(r"$w_z$ (m/s)")
axs[1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
fig.suptitle("Wind disturbance")

# Plot the commanded torque and CRaCLF.
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(tt, u_hist, linewidth=1.2)
axs[0].set_ylabel("u (N m)")
axs[1].plot(tt, np.maximum(V_hist, 1e-12), linewidth=1.2)
axs[1].set_ylabel(r"$V_r$")
axs[1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
fig.suptitle("CRaCLF-QP")

# Plot OLACP variables.
fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].step(interval_times, s_k_hist, where="post", label=r"$s_k$")
axs[0].step(tt, Q_k_hist, where="post", label=r"$Q_k$")
axs[0].set_ylabel("score")
axs[0].legend()
axs[1].step(interval_times, delta_k_hist, where="post")
axs[1].set_ylabel(r"$\delta_k$")
axs[2].step(interval_times, e_k_hist, where="post")
axs[2].set_ylabel(r"$e_k$")
axs[2].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
    for interval_index in range(1, K):
        ax.axvline(
            interval_index * interval_duration,
            color="0.7",
            linestyle=":",
            linewidth=0.8,
        )
fig.suptitle("Adaptive conformal prediction")

# Plot the pointwise loss minimized by Algorithm 1. Each interval uses the
# representation that was installed while that interval's data were sampled.
fig, ax = plt.subplots(figsize=(7, 4))
#ax.semilogy(
ax.plot(
    tt,
    np.maximum(prediction_error_hist, 1e-16),
    label=r"$\|Y_{\Theta_j}(x_t)a_k-w_t\|_2^2$",
)
for update_index in range(B, K, B):
    ax.axvline(
        update_index * interval_duration,
        color="k",
        linestyle=":",
        alpha=0.6,
        label="new representation active" if update_index == B else None,
    )
ax.set_ylabel("squared prediction error")
ax.set_xlabel("Time (s)")
ax.grid(True)
ax.legend()
fig.suptitle("Uncertainty-prediction error")

# Plot adaptive and interval-fitted environmental parameters.
fig, axs = plt.subplots(system.adim, 1, sharex=True)
for i in range(system.adim):
    axs[i].plot(tt, a_hat_clf_hist[:, i], label=r"$\hat{a}$")
    axs[i].plot(tt, a_k_hist[:, i], "--", label=r"$a_k$")
    axs[i].set_ylabel(f"a{i + 1}")
    axs[i].grid(True)
    axs[i].legend()
axs[-1].set_xlabel("Time (s)")
for ax in axs:
    for interval_index in range(1, K):
        ax.axvline(
            interval_index * interval_duration,
            color="0.7",
            linestyle=":",
            linewidth=0.8,
        )
fig.suptitle("Adaptive and interval-fitted parameters")

# Plot the learned 6-by-2 representation matrix.
fig, axs = plt.subplots(
    *Theta_init.shape, sharex=True, figsize=(11, 11), squeeze=False
)
for row in range(Theta_init.shape[0]):
    for column in range(Theta_init.shape[1]):
        theta_index = row * Theta_init.shape[1] + column
        axs[row, column].plot(tt, Theta_hist[:, theta_index])
        axs[row, column].set_ylabel(rf"$\Theta_{{{row + 1},{column + 1}}}$")
        axs[row, column].grid(True)
for ax in axs[-1, :]:
    ax.set_xlabel("Time (s)")
fig.suptitle("Learned representation")

# Plot scaling state and scaling function.
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(tt, nu_clf_hist)
axs[0].set_ylabel(r"$\nu_{\mathrm{clf}}$")
axs[1].plot(tt, rho_clf_hist)
axs[1].set_ylabel(r"$\rho_{\mathrm{clf}}$")
axs[1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)

plt.tight_layout()
plt.show()
