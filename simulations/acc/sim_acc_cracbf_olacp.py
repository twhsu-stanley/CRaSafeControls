"""Algorithm 1 and CRaCBF simulation for adaptive cruise control."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import lsq_linear

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from olacp import OLACP
from systems.acc.acc import ACC


USE_CP = True
USE_ADAPTIVE = True

# Algorithm 1 setup.
K = 12  # total number of time intervals I_k
B = 4   # update the representation every B intervals

# Time setup for each interval I_k.
dt = 0.02
interval_duration = 2.0
I_length = int(round(interval_duration / dt))
if I_length < 10 or not np.isclose(I_length * dt, interval_duration):
    raise ValueError(
        "interval_duration must be an integer multiple of dt and contain "
        "at least 10 samples"
    )
sim_T = K * interval_duration
tt = np.arange(0.0, sim_T, dt)
if len(tt) != K * I_length:
    raise ValueError("K, interval_duration, and dt define inconsistent sample counts")

N_cal = 200
if N_cal < 100:
    raise ValueError("N_cal must be at least 100")

# Only theta_3 has a physical true value in this parameterization. Theta_1 and
# theta_2 regulate the ranges of the abstract interval parameters.
WAKE_DECAY_TRUE = 0.045
THETA_INIT = np.array([0.0025, 0.0002, 0.08])
THETA_LB = np.array([0.002, 0.00015, 0.005])
THETA_UB = np.array([0.004, 0.00030, 0.12])

# Physical and controller parameters.
mass = 1650.0
wind_speed = 5.0
gravity = 9.81

# In practice, the latent-parameter box is only approximately known. These
# rounded values are engineering guesses for the intended ACC regime.
a_lb = np.array([-0.20, -2.0, -2.666667, 0.0, -1.0, 0.0, 21.5])
a_ub = np.array([-0.09, 0.5, 0.0, 0.005, 0.0, 1.333333, 24.5])
a_center = 0.5 * (a_lb + a_ub)

# Generate the piecewise-constant physical environment directly. The first K
# intervals are simulated, while all N_cal intervals are used for calibration.
environment_interval_count = max(K, N_cal)
environment_phase = (
    2.0 * np.pi * np.arange(environment_interval_count) / 13.0
)
b0_schedule = 220.0 + 35.0 * np.sin(environment_phase + 0.15)
b1_schedule = 6.0 + 1.1 * np.cos(0.8 * environment_phase + 0.35)
b2_schedule = 0.375 + 0.065 * np.sin(1.3 * environment_phase + 0.7)
b3_schedule = 0.35 + 0.09 * np.cos(1.1 * environment_phase - 0.25)
lead_velocity_schedule = 23.0 + 1.15 * np.sin(
    0.9 * environment_phase + 0.55
)

# Coefficients of the physical drag expansion. The corresponding abstract
# parameters a_2, a_3, a_5, and a_6 also depend on the installed Theta.
c1_schedule = -(b0_schedule + b2_schedule * wind_speed**2) / mass
c2_schedule = (-b1_schedule + 2.0 * b2_schedule * wind_speed) / mass
c3_schedule = -b2_schedule / mass
c4_schedule = b2_schedule * b3_schedule * wind_speed**2 / mass
c5_schedule = -2.0 * b2_schedule * b3_schedule * wind_speed / mass
c6_schedule = b2_schedule * b3_schedule / mass

a_true_initial = np.array(
    [
        c1_schedule[0],
        c2_schedule[0] / THETA_INIT[0],
        c3_schedule[0] / THETA_INIT[1],
        c4_schedule[0],
        c5_schedule[0] / THETA_INIT[0],
        c6_schedule[0] / THETA_INIT[1],
        lead_velocity_schedule[0],
    ]
)


def true_uncertainty(x, t):
    """Evaluate the unknown physical uncertainty w(x, t)."""
    interval_index = int(np.floor(max(float(t), 0.0) / interval_duration))
    v = float(np.asarray(x, dtype=float).reshape(3)[1])
    z = float(np.asarray(x, dtype=float).reshape(3)[2])
    relative_air_speed = v - wind_speed
    drag = (
        b0_schedule[interval_index]
        + b1_schedule[interval_index] * v
        + b2_schedule[interval_index]
        * relative_air_speed**2
        * (
            1.0
            - b3_schedule[interval_index]
            * np.exp(-WAKE_DECAY_TRUE * z)
        )
    )
    return np.array(
        [0.0, -drag / mass, lead_velocity_schedule[interval_index]]
    )


# The projection ball contains the guessed box A inside its epsilon-interior.
projection_epsilon = 0.01
a_hat_norm_max = 0.5 * np.linalg.norm(a_ub - a_lb, ord=2) + projection_epsilon
Gamma_cbf = 1.0 * np.eye(7)
Gamma_cbf_inv = np.linalg.inv(Gamma_cbf)

params = {
    "xdim": 3,
    "udim": 1,
    "adim": 7,
    "Theta_init": THETA_INIT.copy(),
    "true_uncertainty": true_uncertainty,
    "m": mass,
    "grav": gravity,
    "vd": 30.0,
    "Kp": 800.0,
    "z_min": 5.0,
    "T_h": 1.5,
    "cbf_smoothing_epsilon": 0.1,
    "use_adaptive": USE_ADAPTIVE,
    "use_cp": USE_CP,
    "Gamma_cbf": Gamma_cbf,
    "a_true": a_true_initial,
    "a_ub": a_ub,
    "a_lb": a_lb,
    "a_hat_norm_max": a_hat_norm_max,
    "epsilon": projection_epsilon,
    "eta_cbf": 5.0,
    "cbf_rate": 0.5,
    "u_max": 0.3 * mass * gravity,
    "u_min": -0.3 * mass * gravity,
    "dt": dt,
}

# Construct the ACC system.
system = ACC(params)

# Initial calibration set. Each historical interval follows lines 8 and 11 of
# Algorithm 1: fit a bounded a_k and retain the largest residual norm.
rng = np.random.default_rng(7)
S_cal_init = []
for k in range(N_cal):
    positions = rng.uniform(0.0, 500.0, I_length)
    velocities = rng.uniform(19.0, 28.0, I_length)
    distances = rng.uniform(12.0, 65.0, I_length)
    states_cal = np.column_stack((positions, velocities, distances))
    times_cal = (
        k * interval_duration
        + (np.arange(I_length) + 0.5) * interval_duration / I_length
    )

    Y_cal = []
    w_cal = []
    for x_cal, t_cal in zip(states_cal, times_cal):
        Y_cal.append(system.Y_theta(x_cal, THETA_INIT))
        w_cal.append(true_uncertainty(x_cal, t_cal))

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
    delta_target=0.1,
    delta_init=0.1,
    buffer_maxlen=I_length,
    theta_init=THETA_INIT,
    representation_period=B,
    representation_lr=lambda j: np.array([100.0, 0.1, 1e5]) / j,# Coordinate-wise rates account for the different Theta scales
    theta_lb=THETA_LB,
    theta_ub=THETA_UB,
    Y_theta=system.Y_theta,
    representation_loss_gradient=system.representation_loss_gradient,
)
system.set_representation(olacp.Theta)
system.cp_quantile = olacp.Q_k

# Simulation initialization.
x = np.array([0.0, 24.0, 24.0])
a_hat_cbf = a_center.copy()
rho_cbf = 0.0
x_ext = np.hstack((x, a_hat_cbf, rho_cbf))

x_hist = np.zeros((len(tt), system.xdim))
u_hist = np.zeros(len(tt))
h_hist = np.zeros(len(tt))
physical_safety_hist = np.zeros(len(tt))
tightened_cbf_margin_hist = np.zeros(len(tt))
true_parameter_barrier_hist = np.zeros(len(tt))
a_hat_cbf_hist = np.zeros((len(tt), system.adim))
a_k_hist = np.full((len(tt), system.adim), np.nan)
a_true_hist = np.zeros((len(tt), system.adim))
lead_velocity_hist = np.zeros(len(tt))
rho_cbf_hist = np.zeros(len(tt))
nu_cbf_hist = np.zeros(len(tt))
Q_k_hist = np.zeros(len(tt))
Theta_hist = np.zeros((len(tt), THETA_INIT.size))

interval_times = []
s_k_hist = []
delta_k_hist = []
e_k_hist = []

# Main simulation loop.
for i, t in enumerate(tt):
    interval_index = i // I_length
    theta_1, theta_2, _ = system.Theta_hat
    a_true = np.array(
        [
            c1_schedule[interval_index],
            c2_schedule[interval_index] / theta_1,
            c3_schedule[interval_index] / theta_2,
            c4_schedule[interval_index],
            c5_schedule[interval_index] / theta_1,
            c6_schedule[interval_index] / theta_2,
            lead_velocity_schedule[interval_index],
        ]
    )

    x_hist[i] = x
    a_hat_cbf_hist[i] = a_hat_cbf
    a_true_hist[i] = a_true
    lead_velocity_hist[i] = lead_velocity_schedule[interval_index]
    rho_cbf_hist[i] = rho_cbf
    nu_cbf_hist[i] = system.nu_cbf(rho_cbf)
    Q_k_hist[i] = system.cp_quantile
    Theta_hist[i] = system.Theta_hat.reshape(-1)
    h_hist[i] = float(np.asarray(system.cbf(x, a_hat_cbf)).item())
    physical_safety_hist[i] = x[2] - params["z_min"]
    tightened_cbf_margin_hist[i] = (
        h_hist[i]
        - 0.5 / nu_cbf_hist[i] * system.safe_set_tightening
    )
    parameter_error = a_hat_cbf - a_true
    true_parameter_barrier_hist[i] = (
        nu_cbf_hist[i] * h_hist[i]
        - 0.5 * parameter_error @ Gamma_cbf_inv @ parameter_error
    )

    if i == 0 and h_hist[i] < 0.0:
        raise ValueError(
            f"Initial condition is outside the CBF set: h={h_hist[i]:.3f}"
        )
    if (
        i % I_length == 0
        and USE_ADAPTIVE
        and tightened_cbf_margin_hist[i] < 0.0
    ):
        raise ValueError(
            "Interval-start state violates equation (37): "
            f"interval={interval_index + 1}, "
            f"margin={tightened_cbf_margin_hist[i]:.3f}"
        )

    u_ref = system.ctrl_nominal(x)
    u = system.ctrl_cracbf(x, a_hat_cbf, u_ref, rho_cbf)
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
            method="BDF",
            rtol=1e-7,
            atol=1e-9,
            t_eval=[tt[i + 1]],
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        x_ext = sol.y[:, -1]
        if not np.all(np.isfinite(x_ext)):
            raise RuntimeError("The extended ACC state became non-finite")
        x = x_ext[: system.xdim]
        a_hat_cbf = x_ext[system.xdim : system.xdim + system.adim]
        rho_cbf = float(x_ext[system.xdim + system.adim])

    # Complete lines 7--23 of Algorithm 1 at the interval boundary.
    if (i + 1) % I_length == 0:
        olacp.estimate_uncertainty(dt)
        s_k = float(olacp.compute_score(system.a_ub, system.a_lb))
        e_k = int(olacp.update_delta(s_k))
        olacp.append_score(s_k)
        representation_update = olacp.update_representation()

        interval_start = i - I_length + 1
        a_k_hist[interval_start : i + 1] = olacp.a_k
        interval_parameter_error = (
            a_hat_cbf_hist[interval_start : i + 1] - olacp.a_k
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
            f"score={s_k:.5f}, Q={Q_k_hist[i]:.5f}, "
            f"delta_next={olacp.delta:.3f}, miscoverage={e_k}, "
            "Theta="
            + np.array2string(
                system.Theta_hat, precision=6, separator=","
            )
        )

interval_times = np.asarray(interval_times)
s_k_hist = np.asarray(s_k_hist)
delta_k_hist = np.asarray(delta_k_hist)
e_k_hist = np.asarray(e_k_hist)

# Plot states, input, and safety margins.
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 11))
axs[0].plot(tt, x_hist[:, 1], label="ego velocity")
axs[0].plot(tt, lead_velocity_hist, "--", label="lead velocity")
axs[0].axhline(params["vd"], color="k", linestyle=":", label="desired")
axs[0].set_ylabel("v (m/s)")
axs[0].legend(ncol=3)
axs[1].plot(tt, x_hist[:, 2], label=r"$z(t)$")
axs[1].axhline(params["z_min"], color="r", linestyle="--", label=r"$z_{\min}$")
axs[1].set_ylabel("z (m)")
axs[1].legend()
axs[2].plot(tt, u_hist, label="u")
axs[2].axhline(params["u_max"], color="k", linestyle=":", label="input bounds")
axs[2].axhline(params["u_min"], color="k", linestyle=":")
axs[2].set_ylabel("u (N)")
axs[2].legend()
for ax in axs:
    ax.grid(True)
fig.suptitle("Adaptive cruise control with a CRaCBF-QP")

# Plot safety margins.
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(tt, h_hist, label=r"$h(x,\hat{a})$")
axs[0].plot(tt, tightened_cbf_margin_hist, ":", label="tightened h margin")
axs[0].axhline(0.0, color="r", linestyle="--", label=r"$h=0$")
axs[0].set_ylabel(r"$h(x,\hat{a})$")
axs[0].legend()
axs[1].plot(tt, physical_safety_hist, label=r"$z-z_{\min}$")
axs[1].axhline(0.0, color="r", linestyle="--", label="safety boundary")
axs[1].set_ylabel("safety margin")
axs[1].set_xlabel("Time (s)")
axs[1].legend()
for ax in axs:
    ax.grid(True)
fig.suptitle("Safety margins")

# Plot OLACP variables.
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
axs[0].step(interval_times, s_k_hist, where="post", label="s_k")
axs[0].step(tt, Q_k_hist, where="post", label="Q_k")
axs[0].set_ylabel("score")
axs[0].legend()
axs[1].step(interval_times, delta_k_hist, where="post")
axs[1].set_ylabel("delta_k")
axs[2].step(interval_times, e_k_hist, where="post")
axs[2].set_ylabel("e_k")
axs[2].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
fig.suptitle("Adaptive conformal prediction")

# Plot the learned representation.
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
for j in range(THETA_INIT.size):
    axs[j].plot(tt, Theta_hist[:, j], label=rf"$\hat{{\theta}}_{j + 1}$")
    axs[j].axhline(THETA_LB[j], color="0.5", linestyle=":")
    axs[j].axhline(THETA_UB[j], color="0.5", linestyle=":")
    axs[j].set_ylabel(rf"$\theta_{j + 1}$")
    axs[j].legend()
axs[2].axhline(
    WAKE_DECAY_TRUE,
    color="k",
    linestyle="--",
    label=r"true $\theta_3$",
)
axs[2].legend()
axs[-1].set_xlabel("Time (s)")
fig.suptitle("Learned Representation (θ)")

# Plot adaptive, interval-fitted, and physical latent parameters.
fig, axs = plt.subplots(system.adim, 1, sharex=True, figsize=(8, 12))
for i in range(system.adim):
    axs[i].plot(tt, a_hat_cbf_hist[:, i], label="a_hat")
    axs[i].plot(tt, a_k_hist[:, i], "--", label="a_k")
    axs[i].plot(tt, a_true_hist[:, i], ":", label="physical a")
    axs[i].set_ylabel(f"a{i + 1}")
    axs[i].grid(True)
axs[0].legend(ncol=3)
axs[-1].set_xlabel("Time (s)")
fig.suptitle("Adaptive, fitted, and physical latent parameters")

# Plot the CRaCBF scaling variables.
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 10))
axs[0].plot(tt, nu_cbf_hist)
axs[0].set_ylabel("nu(rho)")
axs[1].plot(tt, rho_cbf_hist)
axs[1].set_ylabel("rho")
axs[1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
fig.suptitle("Learned representation and CRaCBF scaling")

plt.tight_layout()
plt.show()
