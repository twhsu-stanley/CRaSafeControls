"""Algorithm 1 and CRaCBF simulation for adaptive cruise control."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )
)

from olacp import OLACP
from systems.acc.acc import ACC


USE_CP = True
USE_ADAPTIVE = True

# Algorithm 1 setup. Pretraining only supplies the initial calibration
# window; the same OLACP object then continues through the main simulation.
K_pre = 32
N_cal = 30
K = 12
B = 4
if K_pre < N_cal:
    raise ValueError("K_pre must be at least as large as N_cal")
if K_pre % B != 0:
    raise ValueError("K_pre must be an integer multiple of B")

# Each Algorithm 1 interval contains I_length samples.
dt = 0.01
interval_duration = 2.0
I_length = int(round(interval_duration / dt))
if I_length < 10 or not np.isclose(
    I_length * dt,
    interval_duration,
):
    raise ValueError(
        "interval_duration must be an integer multiple of dt and "
        "contain at least 10 samples"
    )
tt_pre = np.arange(K_pre * I_length, dtype=float) * dt
tt = np.arange(K * I_length, dtype=float) * dt

# Physical ACC parameters. The signs make d(t, x) a signed resistive force.
mass = 1000.0
gravity = 9.81
beta_1 = -10.0
beta_2 = -0.75
beta_3 = 0.2
beta_4 = 0.02
desired_velocity = 26.0
nominal_lead_velocity = 23.0
lead_velocity_scale = 10.0

# A fixed, physically scaled starting representation. At the reference state,
# its three columns evaluate to approximately [5, 5, 5] m/s^2. The fitted
# coordinates can therefore remain small while representing the ACC drag.
v_reference = desired_velocity
z_reference = 35.0
psi_reference = np.kron(
    np.array([1.0, v_reference, v_reference**2]),
    np.array([1.0, z_reference, z_reference**2]),
)
Theta_init = np.zeros(ACC.theta_shape)
Theta_init[0, 0] = 5.0
Theta_init[3, 1] = 5.0 / v_reference
Theta_init[6, 2] = 5.0 / v_reference**2

theta_margin = 1.0 / psi_reference
Theta_lb = Theta_init - theta_margin[:, None]
Theta_ub = Theta_init + theta_margin[:, None]
theta_rng = np.random.default_rng(11)
Theta_init = theta_rng.uniform(Theta_lb, Theta_ub)

#############################################################
Theta_scale = 1 / psi_reference
Theta_lb = -5.0 * Theta_scale[:, None] * np.ones((1, 3))
Theta_ub = 5.0 * Theta_scale[:, None] * np.ones((1, 3))
Theta_init = theta_rng.uniform(
    -Theta_scale[:, None],
    Theta_scale[:, None],
    size=ACC.theta_shape,
)
#############################################################


# a_4 represents Delta v_l / 10. The online Delta v_l range is +/-1 m/s,
# hence |a_4| <= 0.1. Keeping every coordinate on a comparable scale also
# prevents the spherical projection from producing unphysical latent values.
a_lb = -0.2 * np.ones(ACC.adim)
a_ub = 0.2 * np.ones(ACC.adim)
a_center = 0.5 * (a_lb + a_ub)

projection_epsilon = 0.02
a_hat_norm_max = (
    0.5 * np.linalg.norm(a_ub - a_lb, ord=2)
    + projection_epsilon
)
Gamma_cbf = 0.05 * np.eye(ACC.adim)
Gamma_cbf_inv = np.linalg.inv(Gamma_cbf)

# Pretraining uses the same CRaCBF and extended dynamics as the main loop.
# Its physical lead velocity is always above v_d; this phase is used only to
# initialize the calibration window and the representation.
pretrain_phase = 2.0 * np.pi * np.arange(K_pre) / K_pre
pretrain_d0_schedule = (
    -120.0 + 80.0 * np.sin(pretrain_phase + 0.2)
)
pretrain_wind_velocity_schedule = (
    2.0 + 3.0 * np.sin(0.7 * pretrain_phase - 0.1)
)
pretrain_delta_lead_velocity_schedule = (
    0.5 * np.sin(0.9 * pretrain_phase)
)


def schedule_index(t, interval_count):
    """Return the piecewise-constant Algorithm 1 interval index."""
    return min(
        int(
            np.floor(
                max(float(t), 0.0) / interval_duration
            )
        ),
        interval_count - 1,
    )


def drag_force(x, d0, wind_velocity):
    """Evaluate the signed force d(t, x) in the updated draft."""
    x = np.asarray(x, dtype=float).reshape(ACC.xdim)
    v, z = x[1], x[2]
    return (
        d0
        + beta_1 * v
        + beta_2
        * (v - wind_velocity) ** 2
        * (1.0 - beta_3 * np.exp(-beta_4 * z))
    )


def pretrain_true_uncertainty(x, t):
    """Return [0, d/m, Delta v_l] during calibration."""
    interval_index = schedule_index(t, K_pre)
    return np.array(
        [
            0.0,
            drag_force(
                x,
                pretrain_d0_schedule[interval_index],
                pretrain_wind_velocity_schedule[interval_index],
            )
            / mass,
            pretrain_delta_lead_velocity_schedule[
                interval_index
            ],
        ]
    )


u_min = -0.5 * mass * gravity
u_max = 0.5 * mass * gravity
params = {
    "Theta_init": Theta_init.copy(),
    "true_uncertainty": pretrain_true_uncertainty,
    "m": mass,
    "vd": desired_velocity,
    "Kp": 400.0,
    "nominal_lead_velocity": nominal_lead_velocity,
    "lead_velocity_scale": lead_velocity_scale,
    "cbf_smoothing_epsilon": 0.2,
    "z_min": 10.0,
    "T_h": 1.0,
    "use_adaptive": USE_ADAPTIVE,
    "use_cp": USE_CP,
    "Gamma_cbf": Gamma_cbf,
    "a_ub": a_ub,
    "a_lb": a_lb,
    "a_hat_norm_max": a_hat_norm_max,
    "epsilon": projection_epsilon,
    "eta_cbf": 0.1,
    "cbf_rate": 0.5,
    "u_max": u_max,
    "u_min": u_min,
    "dt": dt,
}
system = ACC(params)

# One OLACP object is used throughout pretraining and online execution.
olacp = OLACP(
    [],
    N_cal=N_cal,
    acp_lr=0.02,
    delta_target=0.05,
    delta_init=0.05,
    buffer_maxlen=I_length,
    Theta_init=Theta_init,
    representation_period=B,
    # update_representation sums B*I_length sample gradients.
    representation_lr=lambda j: (
        0.1
        / (B * I_length)
        / psi_reference[:, None]
        / np.sqrt(j)
    ),
    Theta_lb=Theta_lb,
    Theta_ub=Theta_ub,
    Y_Theta=system.Y_Theta,
    representation_loss_gradient=(
        system.representation_loss_gradient
    ),
)
system.set_representation(olacp.Theta)

# -------------------------------------------------------------------------
# Initial calibration and representation pretraining.
# -------------------------------------------------------------------------
x_pre = np.array([0.0, 24.0, 35.0])
a_hat_pre = a_center.copy()
rho_pre = 0.0
x_pre_ext = np.hstack((x_pre, a_hat_pre, rho_pre))
x_pre_hist = np.zeros((len(tt_pre), system.xdim))
a_pre_hist = np.full(
    (len(tt_pre), system.adim),
    np.nan,
)
Theta_pre_hist = np.zeros(
    (K_pre + 1,) + Theta_init.shape
)
Theta_pre_hist[0] = olacp.Theta
s_pre_hist = np.zeros(K_pre)
pretrain_prediction_error_hist = np.full(
    len(tt_pre),
    np.nan,
)

for i_pre, t_pre in enumerate(tt_pre):
    pretrain_interval_index = i_pre // I_length
    x_pre_hist[i_pre] = x_pre

    excitation = (
        250.0 * np.sin(2.0 * np.pi * 0.31 * t_pre)
        + 100.0 * np.sin(2.0 * np.pi * 0.73 * t_pre)
    )
    u_ref_pre = np.array(
        [
            np.clip(
                system.ctrl_nominal(x_pre).item() + excitation,
                u_min,
                u_max,
            )
        ]
    )
    try:
        u_pre = system.ctrl_cracbf(
            x_pre,
            a_hat_pre,
            u_ref_pre,
            rho_pre,
        )
    except ValueError as error:
        raise RuntimeError(
            "Pretraining CRaCBF QP failed at "
            f"t={t_pre:.3f}, x={x_pre}, "
            f"a_hat={a_hat_pre}, rho={rho_pre:.3e}"
        ) from error
    u_pre = float(u_pre.item())
    olacp.add_data_to_buffers(
        x_pre,
        system.dynamics_nominal(x_pre, u_pre),
        xdot=system.dynamics(x_pre, u_pre, t_pre),
    )

    if i_pre < len(tt_pre) - 1:
        sol = solve_ivp(
            lambda tau, state: system.dynamics_extended(
                state,
                u_pre,
                tau,
            ),
            (tt_pre[i_pre], tt_pre[i_pre + 1]),
            x_pre_ext,
            method="BDF",
            rtol=1e-7,
            atol=1e-9,
            t_eval=[tt_pre[i_pre + 1]],
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        x_pre_ext = sol.y[:, -1]
        if not np.all(np.isfinite(x_pre_ext)):
            raise RuntimeError(
                "The extended ACC pretraining state became non-finite"
            )
        x_pre = x_pre_ext[: system.xdim]
        a_hat_pre = x_pre_ext[
            system.xdim : system.xdim + system.adim
        ]
        rho_pre = float(
            x_pre_ext[system.xdim + system.adim]
        )

    if (i_pre + 1) % I_length == 0:
        olacp.estimate_uncertainty(dt)
        s_pre = float(
            olacp.compute_score(system.a_ub, system.a_lb)
        )
        interval_prediction_error = np.array(
            [
                np.linalg.norm(
                    Y_t @ olacp.a_k - w_t,
                    ord=2,
                )
                ** 2
                for Y_t, w_t in zip(
                    olacp._Y_buffer,
                    olacp._w_buffer,
                )
            ]
        )
        olacp.append_score(s_pre)
        representation_update = (
            olacp.update_representation()
        )
        if representation_update is not None:
            system.set_representation(
                representation_update["Theta"]
            )

        interval_start = i_pre - I_length + 1
        a_pre_hist[
            interval_start : i_pre + 1
        ] = olacp.a_k
        pretrain_prediction_error_hist[
            interval_start : i_pre + 1
        ] = interval_prediction_error
        s_pre_hist[pretrain_interval_index] = s_pre
        Theta_pre_hist[
            pretrain_interval_index + 1
        ] = olacp.Theta
        olacp.clear_buffers()

if len(olacp.S_cal) != N_cal:
    raise RuntimeError(
        "Pretraining did not fill the calibration window"
    )

# -------------------------------------------------------------------------
# Main CRaCBF and Algorithm 1 simulation.
# -------------------------------------------------------------------------
environment_phase = 2.0 * np.pi * np.arange(K) / K
d0_schedule = (
    -150.0 + 500.0 * np.sin(environment_phase + 0.15)
)
wind_velocity_schedule = (
    2.0 + 4.0 * np.sin(0.7 * environment_phase - 0.2)
)
delta_lead_velocity_schedule = (
    -2.0 * np.sin(0.5 * environment_phase)
)
if np.any(
    delta_lead_velocity_schedule / lead_velocity_scale
    < a_lb[3]
) or np.any(
    delta_lead_velocity_schedule / lead_velocity_scale
    > a_ub[3]
):
    raise ValueError(
        "The online Delta v_l schedule is outside the a_4 bounds"
    )


def true_uncertainty(x, t):
    """Return the draft uncertainty [0, d/m, Delta v_l]."""
    interval_index = schedule_index(t, K)
    return np.array(
        [
            0.0,
            drag_force(
                x,
                d0_schedule[interval_index],
                wind_velocity_schedule[interval_index],
            )
            / mass,
            delta_lead_velocity_schedule[interval_index],
        ]
    )


system.true_uncertainty_fcn = true_uncertainty
system.set_representation(olacp.Theta)
system.cp_quantile = olacp.compute_quantile()

x = np.array([0.0, 24.0, 20.0])
a_hat_cbf = a_center.copy()
rho_cbf = 0.0
x_ext = np.hstack((x, a_hat_cbf, rho_cbf))

x_hist = np.zeros((len(tt), system.xdim))
u_hist = np.zeros(len(tt))
u_ref_hist = np.zeros(len(tt))
h_hist = np.zeros(len(tt))
physical_safety_hist = np.zeros(len(tt))
tightened_cbf_margin_hist = np.zeros(len(tt))
z_b_hist = np.full(len(tt), np.nan)
z_b_exponential_bound_hist = np.full(len(tt), np.nan)
a_hat_cbf_hist = np.zeros(
    (len(tt), system.adim)
)
a_k_hist = np.full(
    (len(tt), system.adim),
    np.nan,
)
lead_velocity_hist = np.zeros(len(tt))
rho_cbf_hist = np.zeros(len(tt))
nu_cbf_hist = np.zeros(len(tt))
Q_k_hist = np.zeros(len(tt))
Theta_hist = np.zeros(
    (len(tt),) + Theta_init.shape
)
prediction_error_hist = np.full(len(tt), np.nan)

interval_times = []
s_k_hist = []
delta_k_hist = []
e_k_hist = []

for i, t in enumerate(tt):
    interval_index = i // I_length

    x_hist[i] = x
    a_hat_cbf_hist[i] = a_hat_cbf
    lead_velocity_hist[i] = (
        nominal_lead_velocity
        + delta_lead_velocity_schedule[interval_index]
    )
    rho_cbf_hist[i] = rho_cbf
    nu_cbf_hist[i] = system.nu_cbf(rho_cbf)
    Q_k_hist[i] = system.cp_quantile
    Theta_hist[i] = system.Theta_hat
    h_hist[i] = float(system.cbf(x, a_hat_cbf))
    physical_safety_hist[i] = x[2] - system.z_min
    tightened_cbf_margin_hist[i] = (
        h_hist[i]
        - 0.5
        / nu_cbf_hist[i]
        * system.safe_set_tightening
    )
    if i == 0 and h_hist[i] < 0.0:
        raise ValueError(
            "Initial state is outside the certificate set"
        )
    if (
        i % I_length == 0
        and USE_ADAPTIVE
        and tightened_cbf_margin_hist[i] < -1e-8
    ):
        raise ValueError(
            "Interval-start state violates the tightened "
            f"CRaCBF set at interval {interval_index + 1}: "
            f"{tightened_cbf_margin_hist[i]:.3e}"
        )

    u_ref = system.ctrl_nominal(x)
    try:
        u = system.ctrl_cracbf(
            x,
            a_hat_cbf,
            u_ref,
            rho_cbf,
        )
    except ValueError as error:
        raise RuntimeError(
            "CRaCBF QP failed at "
            f"t={t:.3f}, x={x}, "
            f"a_hat={a_hat_cbf}, rho={rho_cbf:.3e}"
        ) from error
    u_ref_hist[i] = u_ref.item()
    u_hist[i] = u.item()

    # Line 6 of Algorithm 1: collect x, nominal xdot, and measured xdot.
    olacp.add_data_to_buffers(
        x,
        system.dynamics_nominal(x, u),
        xdot=system.dynamics(x, u, t),
    )

    # Zero-order hold on u while propagating x, a_hat, and rho.
    if i < len(tt) - 1:
        sol = solve_ivp(
            lambda tau, state: system.dynamics_extended(
                state,
                u,
                tau,
            ),
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
            raise RuntimeError(
                "The extended ACC state became non-finite"
            )
        x = x_ext[: system.xdim]
        a_hat_cbf = x_ext[
            system.xdim : system.xdim + system.adim
        ]
        rho_cbf = float(
            x_ext[system.xdim + system.adim]
        )

    # Lines 7--23 of Algorithm 1 at the end of I_k.
    if (i + 1) % I_length == 0:
        olacp.estimate_uncertainty(dt)
        s_k = float(
            olacp.compute_score(system.a_ub, system.a_lb)
        )
        interval_prediction_error = np.array(
            [
                np.linalg.norm(
                    Y_t @ olacp.a_k - w_t,
                    ord=2,
                )
                ** 2
                for Y_t, w_t in zip(
                    olacp._Y_buffer,
                    olacp._w_buffer,
                )
            ]
        )

        # Compare s_k with the Q_k used on I_k before changing either
        # the calibration window or the adaptive failure probability.
        e_k = int(olacp.update_delta(s_k))
        olacp.append_score(s_k)
        representation_update = (
            olacp.update_representation()
        )

        interval_start = i - I_length + 1
        system.a_true = olacp.a_k.copy()
        a_k_hist[
            interval_start : i + 1
        ] = olacp.a_k
        prediction_error_hist[
            interval_start : i + 1
        ] = interval_prediction_error

        # Retrospective diagnostic using the fitted a_k for I_k.
        interval_z_b = np.empty(I_length)
        for local_index, history_index in enumerate(
            range(interval_start, i + 1)
        ):
            a_tilde = (
                a_hat_cbf_hist[history_index]
                - system.a_true
            )
            interval_z_b[local_index] = (
                nu_cbf_hist[history_index]
                * h_hist[history_index]
                - 0.5
                * a_tilde
                @ Gamma_cbf_inv
                @ a_tilde
            )
        z_b_hist[
            interval_start : i + 1
        ] = interval_z_b
        interval_time = tt[interval_start : i + 1]
        z_b_exponential_bound_hist[
            interval_start : i + 1
        ] = interval_z_b[0] * np.exp(
            -system.cbf_rate
            * (interval_time - interval_time[0])
        )

        interval_times.append(t)
        s_k_hist.append(s_k)
        delta_k_hist.append(olacp.delta)
        e_k_hist.append(e_k)

        if representation_update is not None:
            system.set_representation(
                representation_update["Theta"]
            )
        system.cp_quantile = olacp.compute_quantile()
        olacp.clear_buffers()

        print(
            f"interval={interval_index + 1:02d}, "
            f"score={s_k:.3e}, "
            f"Q_used={Q_k_hist[i]:.3e}, "
            f"delta_next={olacp.delta:.3f}, "
            f"miscoverage={e_k}"
        )

interval_times = np.asarray(interval_times)
s_k_hist = np.asarray(s_k_hist)
delta_k_hist = np.asarray(delta_k_hist)
e_k_hist = np.asarray(e_k_hist)

if np.min(physical_safety_hist) < -1e-6:
    raise RuntimeError(
        "The physical collision-avoidance set was violated"
    )
if np.min(h_hist) < -1e-6:
    raise RuntimeError("The CRaCBF certificate set was violated")
if np.any(~np.isfinite(u_hist)):
    raise RuntimeError("The CRaCBF input became non-finite")
if np.min(u_hist) < u_min - 1e-6 or np.max(u_hist) > u_max + 1e-6:
    raise RuntimeError("The CRaCBF input bounds were violated")
if np.max(
    np.linalg.norm(
        a_hat_cbf_hist - a_center,
        axis=1,
    )
) > a_hat_norm_max + 1e-6:
    raise RuntimeError(
        "The CRaCBF parameter projection set was violated"
    )
if len(s_k_hist) != K:
    raise RuntimeError("Algorithm 1 did not complete every online interval")
if not np.allclose(system.Theta_hat, olacp.Theta):
    raise RuntimeError(
        "The learned representation was not installed in the ACC system"
    )

# -------------------------------------------------------------------------
# Diagnostics.
# -------------------------------------------------------------------------
fig, axs = plt.subplots(
    3,
    1,
    sharex=True,
    figsize=(8, 10),
)
axs[0].plot(tt, x_hist[:, 1], label="ego velocity")
axs[0].plot(
    tt,
    lead_velocity_hist,
    "--",
    label="lead velocity",
)
axs[0].axhline(
    desired_velocity,
    color="k",
    linestyle=":",
    label="desired velocity",
)
axs[0].set_ylabel("velocity (m/s)")
axs[0].legend()
axs[1].plot(tt, x_hist[:, 2], label="distance")
axs[1].axhline(
    system.z_min,
    color="r",
    linestyle="--",
    label="z_min",
)
axs[1].set_ylabel("z (m)")
axs[1].legend()
axs[2].plot(tt, u_hist, label="CRaCBF input")
axs[2].plot(
    tt,
    u_ref_hist,
    ":",
    label="nominal input",
)
axs[2].axhline(u_max, color="k", linestyle="--")
axs[2].axhline(u_min, color="k", linestyle="--")
axs[2].set_ylabel("force (N)")
axs[2].set_xlabel("time (s)")
axs[2].legend()
for ax in axs:
    ax.grid(True)
fig.suptitle("ACC with CRaCBF control")

fig, axs = plt.subplots(
    3,
    1,
    sharex=True,
    figsize=(8, 8),
)
axs[0].plot(tt, h_hist, label="h")
axs[0].plot(
    tt,
    tightened_cbf_margin_hist,
    ":",
    label="tightened h margin",
)
axs[0].axhline(0.0, color="r", linestyle="--")
axs[0].set_ylabel("certificate")
axs[0].legend()
axs[1].plot(
    tt,
    physical_safety_hist,
    label="z - z_min",
)
axs[1].axhline(0.0, color="r", linestyle="--")
axs[1].set_ylabel("physical margin")
axs[1].legend()
axs[2].plot(tt, z_b_hist, label="z_b")
axs[2].plot(
    tt,
    z_b_exponential_bound_hist,
    "--",
    label="comparison bound",
)
axs[2].set_ylabel("z_b")
axs[2].set_xlabel("time (s)")
axs[2].legend()
for ax in axs:
    ax.grid(True)
fig.suptitle("CRaCBF safety diagnostics")

fig, axs = plt.subplots(
    3,
    1,
    sharex=True,
    figsize=(8, 7),
)
axs[0].step(
    interval_times,
    s_k_hist,
    where="post",
    label="s_k",
)
axs[0].step(
    tt,
    Q_k_hist,
    where="post",
    label="Q_k",
)
axs[0].set_ylabel("score")
axs[0].legend()
axs[1].step(
    interval_times,
    delta_k_hist,
    where="post",
)
axs[1].set_ylabel("delta_k")
axs[2].step(
    interval_times,
    e_k_hist,
    where="post",
)
axs[2].set_ylabel("miscoverage")
axs[2].set_xlabel("time (s)")
for ax in axs:
    ax.grid(True)
fig.suptitle("Algorithm 1 and adaptive conformal prediction")

# Plot the CRaCBF safety state and the Algorithm 1 interval fit. Only the
# fourth physical latent coordinate is known in this simulation.
fig, axs = plt.subplots(system.adim, 1, sharex=True, figsize=(8, 12))
for i in range(system.adim):
    axs[i].plot(
        tt,
        a_hat_cbf_hist[:, i],
        label=r"CRaCBF safety state $\hat a$",
    )
    axs[i].plot(
        tt,
        a_k_hist[:, i],
        "--",
        label=r"OLACP interval fit $a_k$",
    )
    axs[i].set_ylabel(f"a{i + 1}")
    axs[i].grid(True)
axs[0].legend()
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
fig.suptitle("CRaCBF safety adaptation versus interval identification")

# Plot the CRaCBF scaling and spherical-projection variables.
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 10))
axs[0].plot(tt, nu_cbf_hist)
axs[0].set_ylabel("nu(rho)")
axs[1].plot(tt, rho_cbf_hist)
axs[1].set_ylabel("rho")
axs[1].set_xlabel("Time (s)")
for ax in axs:
    ax.grid(True)
fig.suptitle("CRaCBF scaling and parameter projection")

fig, axs = plt.subplots(
    2,
    1,
    figsize=(8, 7),
)
axs[0].semilogy(
    tt_pre,
    np.maximum(
        pretrain_prediction_error_hist,
        1e-16,
    ),
    label="pretraining",
)
axs[1].semilogy(
    tt,
    np.maximum(prediction_error_hist, 1e-16),
    label="online",
)
for ax in axs:
    ax.set_ylabel("squared prediction error")
    ax.grid(True)
    ax.legend()
axs[1].set_xlabel("time (s)")
fig.suptitle("Uncertainty-model prediction loss")

plt.show()
