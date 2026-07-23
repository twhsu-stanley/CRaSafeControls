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


USE_CP = True#False#
USE_ADAPTIVE = True#False#

# Algorithm 1 setup.
K = 10 # total number of time intervals I_k
B = 2   # update the representation every B intervals

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

# Only theta_4 has a physical true value in this parameterization. Theta_1,
# theta_2, and theta_3 regulate the ranges of the abstract interval parameters.
WAKE_DECAY_TRUE = 0.045

Theta_lb = np.array([0.08, 0.002, 0.00015, 0.005])
Theta_ub = np.array([0.15, 0.004, 0.00030, 0.12])
theta_rng = np.random.default_rng(11)
Theta_init = theta_rng.uniform(Theta_lb, Theta_ub)

# Physical and controller parameters.
mass = 1650.0
wind_speed = 5.0
gravity = 9.81
alpha_b = 1.5

# In practice, the latent-parameter box is only approximately known. These
# rounded values are engineering guesses for the intended ACC regime.
a_lb = np.array([-1.8, -1.1, -1.5, 0.005, -0.55, 0.15, 2.15])
a_ub = np.array([-1.0, 0.0, -0.8, 0.04, -0.1, 0.7, 2.45])
a_center = 0.5 * (a_lb + a_ub)

# Generate the piecewise-constant physical environment directly. The first K
# intervals are simulated, while all N_cal intervals are used for calibration.
environment_interval_count = max(K, N_cal)
environment_phase = (
    2.0 * np.pi * np.arange(environment_interval_count) / 20.0
)
b0_schedule = 220.0 + 35.0 * np.sin(environment_phase + 0.15)
b1_schedule = 6.0 + 1.1 * np.cos(0.8 * environment_phase + 0.35)
b2_schedule = 0.375 + 0.065 * np.sin(1.3 * environment_phase + 0.7)
b3_schedule = 0.35 + 0.09 * np.cos(1.1 * environment_phase - 0.25)
lead_velocity_schedule = 24.0 - 15.0 * np.sin(0.9 * environment_phase)

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
        * (1.0 - b3_schedule[interval_index] * np.exp(-WAKE_DECAY_TRUE * z))
    )
    return np.array(
        [0.0, -drag / mass, lead_velocity_schedule[interval_index]]
    )


# Use the circumscribed-ball radius for the conservative error bound, while
# projecting the estimate coordinate-wise onto the guessed parameter box.
projection_epsilon = 0.01
a_hat_norm_max = 0.5 * np.linalg.norm(a_ub - a_lb, ord=2) + projection_epsilon
# Reduce the seventh-coordinate gain to offset its larger fixed regressor.
Gamma_cbf = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25])
Gamma_cbf_inv = np.linalg.inv(Gamma_cbf)

params = {
    "xdim": 3,
    "udim": 1,
    "adim": 7,
    "Theta_init": Theta_init.copy(),
    "true_uncertainty": true_uncertainty,
    "m": mass,
    "grav": gravity,
    "vd": 30.0,
    "Kp": 500.0,
    "z_min": 1.0,
    "T_h": 1.0,
    "cbf_smoothing_epsilon": 0.01,
    # The seventh column of Y is fixed at 10
    "lead_velocity_regressor": 10.0,
    "use_adaptive": USE_ADAPTIVE,
    "use_cp": USE_CP,
    "Gamma_cbf": Gamma_cbf,
    "a_ub": a_ub,
    "a_lb": a_lb,
    "a_hat_norm_max": a_hat_norm_max,
    "epsilon": projection_epsilon,
    #"projection_geometry": "box",
    "eta_cbf": 10.0,
    "cbf_rate": alpha_b,
    "u_max": 1000 * mass * gravity,
    "u_min": -1000 * mass * gravity,
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
        Y_cal.append(system.Y_Theta(x_cal, Theta_init))
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
    delta_target=0.05,
    delta_init=0.05,
    buffer_maxlen=I_length,
    Theta_init=Theta_init,
    representation_period=B,
    # Coordinate-wise rates account for the different Theta scales.
    representation_lr=lambda j: np.array([1e3, 1e4, 1e1, 1e6]) / j,
    Theta_lb=Theta_lb,
    Theta_ub=Theta_ub,
    Y_Theta=system.Y_Theta,
    representation_loss_gradient=system.representation_loss_gradient,
)
system.set_representation(olacp.Theta)
system.cp_quantile = olacp.Q_k

# Simulation initialization.
x = np.array([0.0, 24.0, 30.0])
a_hat_cbf = a_center.copy()
rho_cbf = 0.0
x_ext = np.hstack((x, a_hat_cbf, rho_cbf))

x_hist = np.zeros((len(tt), system.xdim))
u_hist = np.zeros(len(tt))
h_hist = np.zeros(len(tt))
physical_safety_hist = np.zeros(len(tt))
tightened_cbf_margin_hist = np.zeros(len(tt))
z_b_hist = np.zeros(len(tt))
z_b_exponential_bound_hist = np.zeros(len(tt))
a_hat_cbf_hist = np.zeros((len(tt), system.adim))
a_k_hist = np.full((len(tt), system.adim), np.nan)
lead_velocity_hist = np.zeros(len(tt))
rho_cbf_hist = np.zeros(len(tt))
nu_cbf_hist = np.zeros(len(tt))
Q_k_hist = np.zeros(len(tt))
Theta_hist = np.zeros((len(tt), Theta_init.size))
prediction_error_hist = np.full(len(tt), np.nan)

interval_times = []
s_k_hist = []
delta_k_hist = []
e_k_hist = []

# Main simulation loop.
for i, t in enumerate(tt):
    interval_index = i // I_length

    x_hist[i] = x
    a_hat_cbf_hist[i] = a_hat_cbf
    lead_velocity_hist[i] = lead_velocity_schedule[interval_index]
    rho_cbf_hist[i] = rho_cbf
    nu_cbf_hist[i] = system.nu_cbf(rho_cbf)
    Q_k_hist[i] = system.cp_quantile
    Theta_hist[i] = system.Theta_hat.reshape(-1)
    h_hist[i] = float(np.asarray(system.cbf(x, a_hat_cbf)).item())
    physical_safety_hist[i] = x[2] - params["z_min"]
    tightened_cbf_margin_hist[i] = h_hist[i] - 0.5 / nu_cbf_hist[i] * system.safe_set_tightening
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

    # Propagate with zero-order hold on the control input
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

    # Complete lines 7--23 of Algorithm 1 at the interval boundary
    if (i + 1) % I_length == 0:
        olacp.estimate_uncertainty(dt)
        s_k = float(olacp.compute_score(system.a_ub, system.a_lb))
        interval_prediction_error = np.array(
            [
                np.linalg.norm(Y_t @ olacp.a_k - w_t, ord=2) ** 2
                for Y_t, w_t in zip(olacp._Y_buffer, olacp._w_buffer)
            ]
        )
        e_k = int(olacp.update_delta(s_k))
        olacp.append_score(s_k)
        representation_update = olacp.update_representation()

        interval_start = i - I_length + 1
        # Algorithm 1 obtains a_k only after I_k is complete. Treat it
        # retrospectively as the fictitious true parameter for that same
        # interval; never carry it forward as the plant parameter of I_{k+1}.
        system.a_true = olacp.a_k.copy()
        a_k_hist[interval_start : i + 1] = olacp.a_k
        prediction_error_hist[interval_start : i + 1] = interval_prediction_error
        interval_z_b = np.empty(i - interval_start + 1)
        for interval_sample_index, history_index in enumerate(
            range(interval_start, i + 1)
        ):
            a_tilde = a_hat_cbf_hist[history_index] - system.a_true
            interval_z_b[interval_sample_index] = (
                nu_cbf_hist[history_index] * h_hist[history_index]
                - 0.5 * a_tilde @ Gamma_cbf_inv @ a_tilde
            )
        z_b_hist[interval_start : i + 1] = interval_z_b
        interval_time = tt[interval_start : i + 1]
        z_b_exponential_bound_hist[interval_start : i + 1] = (
            interval_z_b[0] * np.exp(-alpha_b * (interval_time - interval_time[0]))
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

# Plot states and input.
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
fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(tt, h_hist, label=r"$h(x,\hat{a})$")
axs[0].plot(tt, tightened_cbf_margin_hist, ":", label="tightened h margin")
axs[0].axhline(0.0, color="r", linestyle="--", label=r"$h=0$")
axs[0].set_ylabel(r"$h(x,\hat{a})$")
axs[0].legend()
axs[1].plot(tt, physical_safety_hist, label=r"$z-z_{\min}$")
axs[1].axhline(0.0, color="r", linestyle="--", label="safety boundary")
axs[1].set_ylabel("safety margin")
axs[1].legend()
# Verify the within-interval comparison result used in Theorem 3:
# z_b(t) >= z_b(tau_k) exp(-alpha_b (t - tau_k)), t in I_k.
z_b_bound_gap_hist = z_b_hist - z_b_exponential_bound_hist
axs[2].plot(tt, z_b_hist, label=r"$z_b(t)$")
axs[2].plot(tt, z_b_exponential_bound_hist, "--", label=r"$z_b(\tau_k)e^{-\alpha_b(t-\tau_k)}$")
axs[2].set_ylabel(r"$z_b$")
axs[2].set_xlabel("Time (s)")
axs[2].legend()
for ax in axs:
    ax.grid(True)
    for interval_index in range(1, K):
        ax.axvline(
            interval_index * interval_duration,
            color="0.7",
            linestyle=":",
            linewidth=1.0,
        )
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
    for interval_index in range(1, K):
        ax.axvline(
            interval_index * interval_duration,
            color="0.7",
            linestyle=":",
            linewidth=1.0,
        )
fig.suptitle("Adaptive conformal prediction")

# Plot the pointwise loss minimized by Algorithm 1. Each interval uses the
# representation that was installed while that interval's data were sampled.
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(tt, np.maximum(prediction_error_hist, 1e-16), label=r"$\|Y_{\Theta_j}(x_t)a_k-w_t\|_2^2$")
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

# Plot the learned representation.
fig, axs = plt.subplots(Theta_init.size, 1, sharex=True, figsize=(8, 11))
for j in range(Theta_init.size):
    axs[j].plot(tt, Theta_hist[:, j], label=rf"$\hat{{\theta}}_{j + 1}$")
    axs[j].axhline(Theta_lb[j], color="0.5", linestyle=":")
    axs[j].axhline(Theta_ub[j], color="0.5", linestyle=":")
    axs[j].set_ylabel(rf"$\theta_{j + 1}$")
    axs[j].legend()
axs[3].axhline(
    WAKE_DECAY_TRUE,
    color="k",
    linestyle="--",
    label=r"true $\theta_4$",
)
axs[3].legend()
axs[-1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
    for interval_index in range(1, K):
        ax.axvline(
            interval_index * interval_duration,
            color="0.7",
            linestyle=":",
            linewidth=1.0,
        )
fig.suptitle("Learned Representation (θ)")

# Plot adaptive, interval-fitted, and physical latent parameters.
fig, axs = plt.subplots(system.adim, 1, sharex=True, figsize=(8, 12))
for i in range(system.adim):
    axs[i].plot(tt, a_hat_cbf_hist[:, i], label="a_hat")
    axs[i].plot(tt, a_k_hist[:, i], "--", label="a_k")
    axs[i].set_ylabel(f"a{i + 1}")
    axs[i].grid(True)
axs[0].legend(ncol=3)
axs[-1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
    for interval_index in range(1, K):
        ax.axvline(
            interval_index * interval_duration,
            color="0.7",
            linestyle=":",
            linewidth=1.0,
        )
fig.suptitle("Adaptive and interval-fitted latent parameters")

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
