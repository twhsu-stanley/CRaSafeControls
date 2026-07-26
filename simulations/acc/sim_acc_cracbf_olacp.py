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
desired_velocity = 28.0
nominal_lead_velocity = 25.0
lead_velocity_scale = 10.0

# A fixed, physically scaled starting representation. At the reference state,
# its three columns evaluate to [20, 20, 20] m/s^2. This scaling lets the
# longitudinal fitted coordinates use tight bounds without reducing the
# physical uncertainty range represented by Y_Theta(x) a.
v_reference = desired_velocity
z_reference = 35.0
psi_reference = np.kron(
    np.array([1.0, v_reference, v_reference**2]),
    np.array([1.0, z_reference, z_reference**2]),
)

###############################################################
#Theta_init = np.zeros(ACC.theta_shape)
#Theta_init[0, 0] = 20.0
#Theta_init[3, 1] = 20.0 / v_reference
#Theta_init[6, 2] = 20.0 / v_reference**2
#theta_margin = 0.05 / psi_reference
#Theta_lb = Theta_init - theta_margin[:, None]
#Theta_ub = Theta_init + theta_margin[:, None]
#############################################################
Theta_scale = 1 / psi_reference
Theta_lb = -20.0 * Theta_scale[:, None] * np.ones((1, 3))
Theta_ub = 20.0 * Theta_scale[:, None] * np.ones((1, 3))
theta_rng = np.random.default_rng(11)
Theta_init = theta_rng.uniform(
    -Theta_scale[:, None],
    Theta_scale[:, None],
    size=ACC.theta_shape,
)
#############################################################

# The first three coordinates multiply the scaled longitudinal columns.
# The fourth coordinate is Delta v_l / 10 and must contain [-0.4, 0].
a_lb = np.array([-0.05, -0.05, -0.05, -0.42])
a_ub = np.array([0.05, 0.05, 0.05, 0.42])
a_center = 0.5 * (a_lb + a_ub)

projection_epsilon = 0.01
a_hat_norm_max = 0.5 * np.linalg.norm(a_ub - a_lb, ord=2) + projection_epsilon

Gamma_cbf = np.eye(ACC.adim)
Gamma_cbf_inv = np.linalg.inv(Gamma_cbf)

# Pretraining uses an expert controller and the same extended dynamics as the
# main loop. It initializes the calibration window and representation.
pretrain_phase = 2.0 * np.pi * np.arange(K_pre) / K_pre
pretrain_d0_schedule = -120.0 + 80.0 * np.sin(pretrain_phase + 0.2)
pretrain_wind_velocity_schedule = 2.0 + 3.0 * np.sin(0.7 * pretrain_phase - 0.1)
pretrain_delta_lead_velocity_schedule = -2.0 + 0.5 * np.sin(0.9 * pretrain_phase)


def schedule_index(t, interval_count):
    """Return the piecewise-constant Algorithm 1 interval index."""
    return min(
        int(np.floor(max(float(t), 0.0) / interval_duration)),
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

# Expert controller for pretraining. The controller uses only the
# measured velocity and gap, a clock, and these fixed design constants. It has
# no access to the simulated drag or the lead-velocity schedule.
pretrain_z_lower = 10.0
pretrain_z_upper = 50.0
pretrain_v_lower = 10.0
pretrain_v_upper = 30.0
pretrain_gap_reference_center = 37.0
pretrain_gap_reference_amplitude = 11.0
pretrain_gap_reference_period = 20.0
pretrain_gap_gain = 0.5
pretrain_velocity_gain = 1.6


def expert_pretrain_control(x, t):
    """Return an expert input and its two references."""
    x = np.asarray(x, dtype=float).reshape(ACC.xdim)
    phase = (float(t) / pretrain_gap_reference_period + 0.25) % 1.0
    triangular_wave = 1.0 - 4.0 * abs(phase - 0.5)
    gap_reference = (
        pretrain_gap_reference_center
        + pretrain_gap_reference_amplitude * triangular_wave
    )

    # The desired ego velocity provides excitation. Gap feedback slows the
    # ego vehicle when it gets too close and accelerates it when it falls too
    # far behind, without measuring or estimating the lead velocity.
    velocity_command = (
        desired_velocity
        + pretrain_gap_gain * (x[2] - gap_reference)
    )
    velocity_command = np.clip(velocity_command, pretrain_v_lower + 1.0, pretrain_v_upper - 1.0)

    # Plain proportional velocity tracking: deliberately no drag
    # cancellation or use of any environment schedule.
    input_command = mass * pretrain_velocity_gain * (velocity_command - x[1])
    input_command = np.clip(input_command, u_min, u_max)

    return input_command, velocity_command, gap_reference


params = {
    "Theta_init": Theta_init.copy(),
    "true_uncertainty": pretrain_true_uncertainty,
    "m": mass,
    "vd": desired_velocity,
    "Kp": 2000.0,
    "nominal_lead_velocity": nominal_lead_velocity,
    "lead_velocity_scale": lead_velocity_scale,
    "cbf_smoothing_epsilon": 0.02,
    "z_min": 10.0,
    "T_h": 0.3,
    "use_adaptive": USE_ADAPTIVE,
    "use_cp": USE_CP,
    "Gamma_cbf": Gamma_cbf,
    "a_ub": a_ub,
    "a_lb": a_lb,
    "a_hat_norm_max": a_hat_norm_max,
    "epsilon": projection_epsilon,
    "eta_cbf": 0.005,
    "cbf_rate": 2.5,
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
x_pre_hist = np.zeros((len(tt_pre), system.xdim))
u_pre_hist = np.zeros(len(tt_pre))
expert_velocity_command_pre_hist = np.zeros(len(tt_pre))
gap_reference_pre_hist = np.zeros(len(tt_pre))
lead_velocity_pre_hist = np.zeros(len(tt_pre))
a_pre_hist = np.full((len(tt_pre), system.adim), np.nan)
Theta_pre_hist = np.zeros((K_pre + 1,) + Theta_init.shape)
Theta_pre_hist[0] = olacp.Theta
s_pre_hist = np.zeros(K_pre)
pretrain_prediction_error_hist = np.full(len(tt_pre), np.nan)
true_uncertainty_pre_hist = np.full((len(tt_pre), system.xdim), np.nan)
fitted_uncertainty_pre_hist = np.full((len(tt_pre), system.xdim), np.nan)

for i_pre, t_pre in enumerate(tt_pre):
    pretrain_interval_index = i_pre // I_length
    x_pre_hist[i_pre] = x_pre

    (
        u_pre,
        expert_velocity_command_pre_hist[i_pre],
        gap_reference_pre_hist[i_pre],
    ) = expert_pretrain_control(x_pre, t_pre)
    u_pre_hist[i_pre] = u_pre
    # Truth is recorded only for diagnostics after the expert has selected
    # its input; the lead-velocity schedule is not available to the expert.
    lead_velocity_pre_hist[i_pre] = (
        nominal_lead_velocity
        + pretrain_delta_lead_velocity_schedule[
            pretrain_interval_index
        ]
    )
    olacp.add_data_to_buffers(
        x_pre,
        system.dynamics_nominal(x_pre, u_pre),
        xdot=system.dynamics(x_pre, u_pre, t_pre),
    )

    if i_pre < len(tt_pre) - 1:
        sol = solve_ivp(
            lambda tau, state: system.dynamics(state, u_pre, tau),
            (tt_pre[i_pre], tt_pre[i_pre + 1]),
            x_pre,
            method="BDF",
            rtol=1e-7,
            atol=1e-9,
            t_eval=[tt_pre[i_pre + 1]],
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        x_pre = sol.y[:, -1]
        if not np.all(np.isfinite(x_pre)):
            raise RuntimeError("The extended ACC pretraining state became non-finite")

    if (i_pre + 1) % I_length == 0:
        olacp.estimate_uncertainty(dt)
        s_pre = olacp.compute_score(system.a_ub, system.a_lb)
        
        interval_true_uncertainty = np.asarray(olacp._w_buffer,dtype=float)
        interval_fitted_uncertainty = np.asarray(
            [Y_t @ olacp.a_k for Y_t in olacp._Y_buffer],
            dtype=float,
        )
        interval_prediction_error = np.sum(
            (interval_fitted_uncertainty - interval_true_uncertainty)** 2,
            axis=1,
        )
        olacp.append_score(s_pre)
        representation_update = olacp.update_representation()
        
        if representation_update is not None:
            system.set_representation(representation_update["Theta"])

        interval_start = i_pre - I_length + 1
        a_pre_hist[interval_start : i_pre + 1] = olacp.a_k
        pretrain_prediction_error_hist[interval_start : i_pre + 1] = interval_prediction_error
        true_uncertainty_pre_hist[interval_start : i_pre + 1] = interval_true_uncertainty
        fitted_uncertainty_pre_hist[interval_start : i_pre + 1] = interval_fitted_uncertainty
        s_pre_hist[pretrain_interval_index] = s_pre
        Theta_pre_hist[pretrain_interval_index + 1] = olacp.Theta
        olacp.clear_buffers()

if len(olacp.S_cal) != N_cal:
    raise RuntimeError(
        "Pretraining did not fill the calibration window"
    )
if (
    np.min(x_pre_hist[:, 1]) < pretrain_v_lower - 1e-6
    or np.max(x_pre_hist[:, 1]) > pretrain_v_upper + 1e-6
):
    raise RuntimeError(
        "The expert pretraining velocity left [10, 30] m/s"
    )
if (
    np.min(x_pre_hist[:, 2]) < pretrain_z_lower - 1e-6
    or np.max(x_pre_hist[:, 2]) > pretrain_z_upper + 1e-6
):
    raise RuntimeError(
        "The expert pretraining distance left [10, 50] m"
    )
if np.ptp(x_pre_hist[:, 1]) < 1.0 or np.ptp(x_pre_hist[:, 2]) < 5.0:
    raise RuntimeError(
        "The expert pretraining trajectory does not vary enough"
    )
if np.any(~np.isfinite(u_pre_hist)):
    raise RuntimeError("The expert pretraining input became non-finite")
if (
    np.min(u_pre_hist) < u_min - 1e-6
    or np.max(u_pre_hist) > u_max + 1e-6
):
    raise RuntimeError("The expert pretraining input bounds were violated")

print(
    "expert pretraining: "
    f"v=[{np.min(x_pre_hist[:, 1]):.3f}, "
    f"{np.max(x_pre_hist[:, 1]):.3f}] m/s, "
    f"z=[{np.min(x_pre_hist[:, 2]):.3f}, "
    f"{np.max(x_pre_hist[:, 2]):.3f}] m"
)


# -------------------------------------------------------------------------
# Main CRaCBF and Algorithm 1 simulation.
# -------------------------------------------------------------------------
environment_phase = 2.0 * np.pi * np.arange(K) / K
d0_schedule = -100.0 + 300.0 * np.sin(environment_phase + 0.15)
wind_velocity_schedule = 10 * np.sin(0.7 * environment_phase - 0.2)
delta_lead_velocity_schedule = np.linspace(0.0, -4.0, K)

if np.any(np.diff(delta_lead_velocity_schedule) > 0.0):
    raise ValueError("The main lead vehicle must continue slowing down")

if np.any(
    delta_lead_velocity_schedule / lead_velocity_scale < a_lb[3]
) or np.any(
    delta_lead_velocity_schedule / lead_velocity_scale > a_ub[3]
):
    raise ValueError("The online Delta v_l schedule is outside the a_4 bounds")


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

x = np.array([0.0, 26.0, 40.0])
a_hat_cbf = a_center.copy()
rho_cbf = 0.0
x_ext = np.hstack((x, a_hat_cbf, rho_cbf))

x_hist = np.zeros((len(tt), system.xdim))
u_hist = np.full(len(tt), np.nan)
u_ref_hist = np.full(len(tt), np.nan)
h_hist = np.zeros(len(tt))
physical_safety_hist = np.zeros(len(tt))
tightened_cbf_margin_hist = np.zeros(len(tt))
z_b_hist = np.full(len(tt), np.nan)
z_b_exponential_bound_hist = np.full(len(tt), np.nan)
a_hat_cbf_hist = np.zeros((len(tt), system.adim))
a_k_hist = np.full((len(tt), system.adim), np.nan)
lead_velocity_hist = np.zeros(len(tt))
rho_cbf_hist = np.zeros(len(tt))
nu_cbf_hist = np.zeros(len(tt))
Q_k_hist = np.zeros(len(tt))
Theta_hist = np.zeros((len(tt),) + Theta_init.shape)
prediction_error_hist = np.full(len(tt), np.nan)
true_uncertainty_hist = np.full((len(tt), system.xdim), np.nan)
fitted_uncertainty_hist = np.full((len(tt), system.xdim), np.nan)

interval_times = []
s_k_hist = []
delta_k_hist = []
e_k_hist = []
safety_violation_index = None
last_simulation_index = -1

for i, t in enumerate(tt):
    interval_index = i // I_length
    last_simulation_index = i

    x_hist[i] = x
    a_hat_cbf_hist[i] = a_hat_cbf
    lead_velocity_hist[i] = nominal_lead_velocity + delta_lead_velocity_schedule[interval_index]
    rho_cbf_hist[i] = rho_cbf
    nu_cbf_hist[i] = system.nu_cbf(rho_cbf)
    Q_k_hist[i] = system.cp_quantile
    Theta_hist[i] = system.Theta_hat
    h_hist[i] = system.cbf(x, a_hat_cbf)
    physical_safety_hist[i] = x[2] - system.z_min
    tightened_cbf_margin_hist[i] = h_hist[i] - 0.5 / nu_cbf_hist[i] * system.safe_set_tightening
    
    # A negative CBF value is a safety violating event. Terminate the simulation loop.
    # Save this state in the histories, but do not run the CRaCBF QP.
    if h_hist[i] < 0.0:
        safety_violation_index = i
        estimated_lead_velocity = nominal_lead_velocity + lead_velocity_scale * a_hat_cbf[3]
        print(
            "SAFETY VIOLATION: "
            f"h={h_hist[i]:.3e} at t={t:.3f} s; "
            f"actual v_l={lead_velocity_hist[i]:.3f} m/s, "
            f"CBF estimate={estimated_lead_velocity:.3f} m/s; "
            "terminating the main CRaCBF loop"
        )
        break

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
        u = system.ctrl_cracbf(x, a_hat_cbf, u_ref, rho_cbf)
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
            lambda tau, state: system.dynamics_extended(state, u, tau),
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
        rho_cbf = x_ext[system.xdim + system.adim]
        

    # Lines 7--23 of Algorithm 1 at the end of I_k.
    if (i + 1) % I_length == 0:
        olacp.estimate_uncertainty(dt)
        s_k = float(olacp.compute_score(system.a_ub, system.a_lb))
        interval_true_uncertainty = np.asarray(olacp._w_buffer,dtype=float)
        interval_fitted_uncertainty = np.asarray(
            [
                Y_t @ olacp.a_k
                for Y_t in olacp._Y_buffer
            ],
            dtype=float,
        )
        interval_prediction_error = np.sum(
            (interval_fitted_uncertainty - interval_true_uncertainty) ** 2,
            axis=1,
        )

        # Compare s_k with the Q_k used on I_k before changing either
        # the calibration window or the adaptive failure probability.
        e_k = int(olacp.update_delta(s_k))
        olacp.append_score(s_k)
        representation_update = olacp.update_representation()
        
        interval_start = i - I_length + 1
        system.a_true = olacp.a_k.copy()
        a_k_hist[interval_start : i + 1] = olacp.a_k
        prediction_error_hist[interval_start : i + 1] = interval_prediction_error
        true_uncertainty_hist[interval_start : i + 1] = interval_true_uncertainty
        fitted_uncertainty_hist[interval_start : i + 1] = interval_fitted_uncertainty

        # Retrospective diagnostic using the fitted a_k for I_k.
        interval_z_b = np.empty(I_length)
        for local_index, history_index in enumerate(
            range(interval_start, i + 1)
        ):
            a_tilde = a_hat_cbf_hist[history_index] - system.a_true
            
            interval_z_b[local_index] = (
                nu_cbf_hist[history_index] * h_hist[history_index]
                - 0.5 * a_tilde @ Gamma_cbf_inv @ a_tilde
            )
        z_b_hist[interval_start : i + 1] = interval_z_b
        interval_time = tt[interval_start : i + 1]
        z_b_exponential_bound_hist[interval_start : i + 1] = interval_z_b[0] * np.exp(
            -system.cbf_rate * (interval_time - interval_time[0])
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

# If termination occurs during an interval, use the most recently available a_k to finish
# the diagnostic histories. Algorithm 1 itself is not updated on this incomplete interval.
if safety_violation_index is not None:
    partial_interval_start = (safety_violation_index // I_length) * I_length
    buffered_sample_count = safety_violation_index - partial_interval_start
    partial_buffer_lengths = (
        len(olacp._x_buffer),
        len(olacp._xdot_buffer),
        len(olacp._xdot_nom_buffer),
        len(olacp._Y_buffer),
    )
    if any(length != buffered_sample_count for length in partial_buffer_lengths):
        raise RuntimeError(
            "The partial Algorithm 1 buffers do not match the "
            "safety-termination time: "
            f"expected {buffered_sample_count}, got "
            f"{partial_buffer_lengths}"
        )

    partial_true_uncertainty = [
        np.asarray(xdot_t, dtype=float) - np.asarray(xdot_nom_t, dtype=float)
        for xdot_t, xdot_nom_t in zip(olacp._xdot_buffer, olacp._xdot_nom_buffer)
    ]
    partial_Y = [np.asarray(Y_t, dtype=float) for Y_t in olacp._Y_buffer]
    violation_time = tt[safety_violation_index]
    violation_state = x_hist[safety_violation_index]
    partial_true_uncertainty.append(true_uncertainty(violation_state, violation_time))
    partial_Y.append(system.Y(violation_state))

    partial_fitted_a = olacp.a_k.copy()
    partial_fitted_uncertainty = np.asarray(
        [Y_t @ partial_fitted_a for Y_t in partial_Y],
        dtype=float,
    )
    partial_prediction_error = np.sum(
        (partial_fitted_uncertainty - partial_true_uncertainty) ** 2,
        axis=1,
    )
    partial_interval_stop = safety_violation_index + 1
    partial_slice = slice(partial_interval_start, partial_interval_stop)
    a_k_hist[partial_slice] = partial_fitted_a
    true_uncertainty_hist[partial_slice] = partial_true_uncertainty
    fitted_uncertainty_hist[partial_slice] = partial_fitted_uncertainty
    prediction_error_hist[partial_slice] = partial_prediction_error
    
    partial_z_b = np.empty(partial_interval_stop - partial_interval_start)
    for local_index, history_index in enumerate(range(partial_interval_start, partial_interval_stop)):
        a_tilde = a_hat_cbf_hist[history_index] - partial_fitted_a
        partial_z_b[local_index] = (
            nu_cbf_hist[history_index] * h_hist[history_index] 
            - 0.5 * a_tilde @ Gamma_cbf_inv @ a_tilde
        )
    z_b_hist[partial_slice] = partial_z_b
    partial_time = tt[partial_slice]
    z_b_exponential_bound_hist[partial_slice] = (
        partial_z_b[0] * np.exp(-system.cbf_rate * (partial_time - partial_time[0]))
    )

# All subsequent checks and plots use only the states actually simulated,
# including the first state for which h < 0.
main_sample_count = last_simulation_index + 1
tt = tt[:main_sample_count]
x_hist = x_hist[:main_sample_count]
u_hist = u_hist[:main_sample_count]
u_ref_hist = u_ref_hist[:main_sample_count]
h_hist = h_hist[:main_sample_count]
physical_safety_hist = physical_safety_hist[:main_sample_count]
tightened_cbf_margin_hist = tightened_cbf_margin_hist[:main_sample_count]
z_b_hist = z_b_hist[:main_sample_count]
z_b_exponential_bound_hist = z_b_exponential_bound_hist[:main_sample_count]
a_hat_cbf_hist = a_hat_cbf_hist[:main_sample_count]
a_k_hist = a_k_hist[:main_sample_count]
lead_velocity_hist = lead_velocity_hist[:main_sample_count]
rho_cbf_hist = rho_cbf_hist[:main_sample_count]
nu_cbf_hist = nu_cbf_hist[:main_sample_count]
Q_k_hist = Q_k_hist[:main_sample_count]
Theta_hist = Theta_hist[:main_sample_count]
prediction_error_hist = prediction_error_hist[:main_sample_count]
true_uncertainty_hist = true_uncertainty_hist[:main_sample_count]
fitted_uncertainty_hist = fitted_uncertainty_hist[:main_sample_count]

interval_times = np.asarray(interval_times)
s_k_hist = np.asarray(s_k_hist)
delta_k_hist = np.asarray(delta_k_hist)
e_k_hist = np.asarray(e_k_hist)

safety_violated = safety_violation_index is not None
if safety_violated:
    if h_hist[-1] >= 0.0 or np.any(h_hist[:-1] < 0.0):
        raise RuntimeError("The recorded safety-termination index is inconsistent")
else:
    if np.min(physical_safety_hist) < -1e-6:
        raise RuntimeError("The physical collision-avoidance set was violated")
    if np.min(h_hist) < -1e-6:
        raise RuntimeError("The CRaCBF certificate set was violated")

expected_control_mask = np.ones(len(tt), dtype=bool)
if safety_violated:
    expected_control_mask[-1] = False
if not np.array_equal(np.isfinite(u_hist), expected_control_mask):
    raise RuntimeError("The CRaCBF input became non-finite")
if not np.array_equal(np.isfinite(u_ref_hist), expected_control_mask):
    raise RuntimeError("The nominal input history became non-finite")
issued_u_hist = u_hist[expected_control_mask]
if (
    issued_u_hist.size > 0
    and (
        np.min(issued_u_hist) < u_min - 1e-6
        or np.max(issued_u_hist) > u_max + 1e-6
    )
):
    raise RuntimeError("The CRaCBF input bounds were violated")
if np.max(np.linalg.norm(a_hat_cbf_hist - a_center, axis=1)) > a_hat_norm_max + 1e-6:
    raise RuntimeError("The CRaCBF parameter projection set was violated")
expected_completed_intervals = (
    safety_violation_index // I_length
    if safety_violated
    else K
)
if len(s_k_hist) != expected_completed_intervals:
    raise RuntimeError("Algorithm 1 did not complete every online interval")
if not np.allclose(system.Theta_hat, olacp.Theta):
    raise RuntimeError("The learned representation was not installed in the ACC system")
if (
    not safety_violated
    and USE_ADAPTIVE
    and np.min(tightened_cbf_margin_hist) < -1e-6
):
    raise RuntimeError("The tightened CRaCBF set was violated")

# Verify the requested traffic scenario, in addition to basic safety.
controller_activation_threshold = 1e-3
cracbf_active_hist = np.clip(u_ref_hist, u_min, u_max) - u_hist > controller_activation_threshold
cracbf_active_indices = np.flatnonzero(cracbf_active_hist)
if cracbf_active_indices.size > 0:
    cracbf_activation_index = int(cracbf_active_indices[0])
    cracbf_activation_time = tt[cracbf_activation_index]
else:
    cracbf_activation_index = None
    cracbf_activation_time = None

ego_peak_index = int(np.argmax(x_hist[:, 1]))
minimum_gap_index = int(np.argmin(x_hist[:, 2]))
minimum_gap_margin = x_hist[minimum_gap_index, 2] - system.z_min

activation_summary = (
    f"CRaCBF active at t={cracbf_activation_time:.3f} s"
    if cracbf_activation_time is not None
    else "CRaCBF never active"
)
safety_summary = (
    f"terminated at h={h_hist[-1]:.3e}"
    if safety_violated
    else "safety maintained"
)
print(
    "scenario: lead velocity "
    f"{lead_velocity_hist[0]:.3f} -> "
    f"{lead_velocity_hist[-1]:.3f} m/s, "
    f"{activation_summary}, "
    f"ego peak={x_hist[ego_peak_index, 1]:.3f} m/s, "
    f"minimum z={x_hist[minimum_gap_index, 2]:.3f} m, "
    f"{safety_summary}"
)


# -------------------------------------------------------------------------
# Diagnostics.
# -------------------------------------------------------------------------
main_time_limits = (float(tt[0]), float(tt[-1]))

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
axs[0].plot(tt_pre, x_pre_hist[:, 1], label="ego velocity")
axs[0].plot(
    tt_pre,
    lead_velocity_pre_hist,
    "--",
    label="lead velocity",
)
axs[0].plot(
    tt_pre,
    expert_velocity_command_pre_hist,
    ":",
    label="expert velocity command",
)
axs[0].axhline(
    pretrain_v_lower,
    color="r",
    linestyle="--",
    label="velocity bounds",
)
axs[0].axhline(pretrain_v_upper, color="r", linestyle="--")
axs[0].set_ylabel("velocity (m/s)")
axs[0].legend()
axs[1].plot(tt_pre, x_pre_hist[:, 2], label="distance")
axs[1].plot(
    tt_pre,
    gap_reference_pre_hist,
    ":",
    label="expert gap reference",
)
axs[1].axhline(
    pretrain_z_lower,
    color="r",
    linestyle="--",
    label="distance bounds",
)
axs[1].axhline(pretrain_z_upper, color="r", linestyle="--")
axs[1].set_ylabel("z (m)")
axs[1].legend()
axs[2].plot(tt_pre, u_pre_hist, label="expert input")
axs[2].axhline(u_max, color="k", linestyle="--")
axs[2].axhline(u_min, color="k", linestyle="--")
axs[2].set_ylabel("force (N)")
axs[2].set_xlabel("time (s)")
axs[2].legend()
for ax in axs:
    ax.grid(True)
fig.suptitle("ACC pretraining with expert control")

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
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
axs[2].plot(tt, u_ref_hist, ":", label="nominal input")
axs[2].axhline(u_max, color="k", linestyle="--")
axs[2].axhline(u_min, color="k", linestyle="--")
axs[2].set_ylabel("force (N)")
axs[2].set_xlabel("time (s)")
axs[2].legend()
if cracbf_activation_time is not None:
    for ax in axs[1:]:
        ax.axvline(
            cracbf_activation_time,
            color="tab:purple",
            linestyle="-.",
            linewidth=1.0,
        )
    axs[0].axvline(
        cracbf_activation_time,
        color="tab:purple",
        linestyle="-.",
        linewidth=1.0,
        label="first CRaCBF intervention",
    )
for ax in axs:
    ax.set_xlim(*main_time_limits)
    ax.legend()
    ax.grid(True)
fig.suptitle("ACC with CRaCBF control")

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
axs[0].plot(tt, h_hist, label="h")
axs[0].plot(tt, tightened_cbf_margin_hist, ":", label="tightened h margin")
axs[0].axhline(0.0, color="r", linestyle="--")
axs[0].set_ylabel("certificate")
axs[1].plot(tt, physical_safety_hist, label="z - z_min")
axs[1].axhline(0.0, color="r", linestyle="--")
axs[1].set_ylabel("physical margin")
axs[2].plot(tt, z_b_hist, label="z_b")
axs[2].plot(tt, z_b_exponential_bound_hist, "--", label="comparison bound")
axs[2].set_ylabel("z_b")
axs[2].set_xlabel("time (s)")
for ax in axs:
    ax.set_xlim(*main_time_limits)
    ax.grid(True)
    ax.legend()
fig.suptitle("CRaCBF safety diagnostics")

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
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
    ax.set_xlim(*main_time_limits)
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
    if i == 3:
        axs[i].step(
            tt, (lead_velocity_hist - nominal_lead_velocity) / lead_velocity_scale,
            where="post",
            linestyle=":",
            linewidth=2.0,
            label=r"true $\Delta v_l/10$",
        )
    axs[i].set_ylabel(f"a{i + 1}")
    axs[i].grid(True)
axs[0].legend()
axs[3].legend()
axs[-1].set_xlabel("Time (s)")
for ax in axs:
    ax.set_xlim(*main_time_limits)
    ax.grid(True)
    for interval_index in range(
        1,
        int(np.floor(tt[-1] / interval_duration)) + 1,
    ):
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
    ax.set_xlim(*main_time_limits)
    ax.grid(True)
fig.suptitle("CRaCBF scaling and parameter projection")

# Compare Y_Theta(x) a_k with the true uncertainty during pretraining.
uncertainty_components = (
    (1, r"$w_2=d/m$"),
    (2, r"$w_3=\Delta v_l$"),
)
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
for ax, (component, component_label) in zip(axs, uncertainty_components):
    ax.plot(
        tt_pre,
        true_uncertainty_pre_hist[:, component],
        label="true uncertainty",
        linewidth=2.0,
    )
    ax.plot(
        tt_pre,
        fitted_uncertainty_pre_hist[:, component],
        "--",
        label=r"$Y_\Theta(x)a_k$",
    )
    ax.set_ylabel(component_label)
    ax.grid(True)
    ax.legend()
axs[-1].set_xlabel("time (s)")
fig.suptitle(r"Pretraining: $Y_\Theta(x)a_k$ versus true uncertainty")

# Compare Y_Theta(x) a_k with the true uncertainty in the main CRaCBF loop.
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
for ax, (component, component_label) in zip(axs, uncertainty_components):
    ax.plot(
        tt,
        true_uncertainty_hist[:, component],
        label="true uncertainty",
        linewidth=2.0,
    )
    ax.plot(
        tt,
        fitted_uncertainty_hist[:, component],
        "--",
        label=r"$Y_\Theta(x)a_k$",
    )
    ax.set_ylabel(component_label)
    ax.set_xlim(*main_time_limits)
    ax.grid(True)
    ax.legend()
axs[-1].set_xlabel("time (s)")
fig.suptitle(
    r"Main CRaCBF: $Y_\Theta(x)a_k$ versus true uncertainty"
)

fig, axs = plt.subplots(2, 1, figsize=(8, 7))
axs[0].semilogy(
    tt_pre,
    np.maximum(pretrain_prediction_error_hist, 1e-16),
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
axs[1].set_xlim(*main_time_limits)
fig.suptitle("Uncertainty-model prediction loss")

plt.show()
