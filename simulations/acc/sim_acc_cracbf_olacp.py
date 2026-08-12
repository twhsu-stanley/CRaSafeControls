"""Algorithm 1 and CRaCBF for adaptive cruise control (acc)"""

import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from olacp import OLACP
from systems.acc.acc import ACC


def run_pretraining(system, olacp, config, plot=True):
    """Pre-train the shared representation and fill the calibration dataset using OLACP"""
    K_pre = config["K_pre"]
    I_length = config["I_length"]
    dt = config["dt"]
    interval_duration = config["interval_duration"]
    mass = config["mass"]
    beta_1 = config["beta_1"]
    beta_2 = config["beta_2"]
    beta_3 = config["beta_3"]
    beta_4 = config["beta_4"]
    desired_velocity = config["desired_velocity"]
    nominal_lead_velocity = config["nominal_lead_velocity"]
    u_min = config["u_min"]
    u_max = config["u_max"]
    t_pre = np.arange(K_pre * I_length, dtype=float) * dt

    pretrain_phase = 2.0 * np.pi * np.arange(K_pre) / K_pre
    d0_schedule = mass * config["gravity"] * np.sin(
        np.deg2rad(5.0 * np.sin(pretrain_phase + 0.2))
    )
    wind_schedule = 2.0 + 3.0 * np.sin(0.7 * pretrain_phase - 0.1)
    delta_lead_schedule = -2.0 + 0.5 * np.sin(0.9 * pretrain_phase)

    def schedule_index(t):
        return min(int(np.floor(max(float(t), 0.0) / interval_duration)), K_pre - 1)

    def drag_force(x, d0, wind_velocity):
        v, z = np.asarray(x, dtype=float).reshape(ACC.xdim)[1:]
        wake = 1.0 - beta_3 * np.exp(-beta_4 * z)
        return d0 + beta_1 * v + beta_2 * (v - wind_velocity) ** 2 * wake

    def true_uncertainty(x, t):
        index = schedule_index(t)
        drag = drag_force(x, d0_schedule[index], wind_schedule[index])
        return np.array([0.0, drag / mass, delta_lead_schedule[index]])

    def expert_control(x, t):
        x = np.asarray(x, dtype=float).reshape(ACC.xdim)
        period = config["pretrain_gap_reference_period"]
        phase = (float(t) / period + 0.25) % 1.0
        triangular_wave = 1.0 - 4.0 * abs(phase - 0.5)
        gap_reference = config["pretrain_gap_reference_center"]
        gap_reference += config["pretrain_gap_reference_amplitude"] * triangular_wave
        velocity_command = desired_velocity
        velocity_command += config["pretrain_gap_gain"] * (x[2] - gap_reference)
        velocity_command = np.clip(
            velocity_command, config["pretrain_v_lower"] + 1.0, config["pretrain_v_upper"] - 1.0
        )
        input_command = mass * config["pretrain_velocity_gain"] * (velocity_command - x[1])
        input_command = float(np.clip(input_command, u_min, u_max))
        return input_command, float(velocity_command), float(gap_reference)

    system.true_uncertainty_fcn = true_uncertainty
    system.set_representation(olacp.Theta)
    x = config["pretrain_initial_state"].copy()
    x_hist = np.zeros((len(t_pre), system.xdim))
    u_hist = np.zeros(len(t_pre))
    velocity_command_hist = np.zeros(len(t_pre))
    gap_reference_hist = np.zeros(len(t_pre))
    lead_velocity_hist = np.zeros(len(t_pre))
    a_k_hist = np.full((len(t_pre), system.adim), np.nan)
    theta_hist = np.zeros((K_pre + 1,) + olacp.Theta.shape)
    theta_hist[0] = olacp.Theta
    score_hist = np.zeros(K_pre)
    prediction_error_hist = np.full(len(t_pre), np.nan)
    true_uncertainty_hist = np.full((len(t_pre), system.xdim), np.nan)
    fitted_uncertainty_hist = np.full((len(t_pre), system.xdim), np.nan)

    for sample_index, t in enumerate(t_pre):
        interval_index = sample_index // I_length
        x_hist[sample_index] = x
        u, velocity_command, gap_reference = expert_control(x, t)
        u_hist[sample_index] = u
        velocity_command_hist[sample_index] = velocity_command
        gap_reference_hist[sample_index] = gap_reference
        lead_velocity_hist[sample_index] = (
            nominal_lead_velocity + delta_lead_schedule[interval_index]
        )
        olacp.add_data_to_buffers(
            x, system.dynamics_nominal(x, u), xdot=system.dynamics(x, u, t)
        )

        if sample_index < len(t_pre) - 1:
            solution = solve_ivp(
                lambda tau, state: system.dynamics(state, u, tau),
                (t_pre[sample_index], t_pre[sample_index + 1]),
                x,
                method="BDF",
                rtol=1e-7,
                atol=1e-9,
                t_eval=[t_pre[sample_index + 1]],
            )
            if not solution.success:
                raise RuntimeError(solution.message)
            x = solution.y[:, -1]
            if not np.all(np.isfinite(x)):
                raise RuntimeError("The ACC pretraining state became non-finite")

        if (sample_index + 1) % I_length == 0:
            olacp.estimate_uncertainty(dt)
            score = float(olacp.compute_score(system.a_ub, system.a_lb))
            interval_true = np.asarray(olacp._w_buffer, dtype=float)
            interval_fitted = np.asarray([Y_t @ olacp.a_k for Y_t in olacp._Y_buffer])
            interval_error = np.sum((interval_fitted - interval_true) ** 2, axis=1)
            olacp.append_score(score)
            representation_update = olacp.update_representation()
            if representation_update is not None:
                system.set_representation(representation_update["Theta"])
            interval_start = sample_index - I_length + 1
            interval_slice = slice(interval_start, sample_index + 1)
            a_k_hist[interval_slice] = olacp.a_k
            prediction_error_hist[interval_slice] = interval_error
            true_uncertainty_hist[interval_slice] = interval_true
            fitted_uncertainty_hist[interval_slice] = interval_fitted
            score_hist[interval_index] = score
            theta_hist[interval_index + 1] = olacp.Theta
            olacp.clear_buffers()

    if len(olacp.S_cal) != config["N_cal"]:
        raise RuntimeError("Pretraining did not fill the calibration window")
    if any(len(buffer) != 0 for buffer in (
        olacp._x_buffer,
        olacp._xdot_buffer,
        olacp._xdot_nom_buffer,
        olacp._Y_buffer,
        olacp._w_buffer,
    )):
        raise RuntimeError("Pretraining left nonempty sample buffers")
    if olacp._representation_intervals:
        raise RuntimeError("Pretraining left an incomplete representation block")
    if not np.allclose(system.Theta_hat, olacp.Theta):
        raise RuntimeError("The trained representation was not installed in the ACC system")
    if np.min(x_hist[:, 1]) < config["pretrain_v_lower"] - 1e-6:
        raise RuntimeError("The expert pretraining velocity fell below 10 m/s")
    if np.max(x_hist[:, 1]) > config["pretrain_v_upper"] + 1e-6:
        raise RuntimeError("The expert pretraining velocity exceeded 30 m/s")
    if np.min(x_hist[:, 2]) < config["pretrain_z_lower"] - 1e-6:
        raise RuntimeError("The expert pretraining distance fell below 10 m")
    if np.max(x_hist[:, 2]) > config["pretrain_z_upper"] + 1e-6:
        raise RuntimeError("The expert pretraining distance exceeded 50 m")
    if np.ptp(x_hist[:, 1]) < 1.0 or np.ptp(x_hist[:, 2]) < 5.0:
        raise RuntimeError("The expert pretraining trajectory does not vary enough")
    if np.any(~np.isfinite(u_hist)) or np.min(u_hist) < u_min - 1e-6:
        raise RuntimeError("The expert pretraining input is invalid")
    if np.max(u_hist) > u_max + 1e-6:
        raise RuntimeError("The expert pretraining input exceeded its upper bound")

    history = {
        "t": t_pre,
        "x_hist": x_hist,
        "u_hist": u_hist,
        "velocity_command_hist": velocity_command_hist,
        "gap_reference_hist": gap_reference_hist,
        "lead_velocity_hist": lead_velocity_hist,
        "a_k_hist": a_k_hist,
        "theta_hist": theta_hist,
        "score_hist": score_hist,
        "prediction_error_hist": prediction_error_hist,
        "true_uncertainty_hist": true_uncertainty_hist,
        "fitted_uncertainty_hist": fitted_uncertainty_hist,
        "d0_schedule": d0_schedule,
        "wind_schedule": wind_schedule,
        "delta_lead_schedule": delta_lead_schedule,
    }

    print(
        f"expert pretraining: v=[{np.min(x_hist[:, 1]):.3f}, {np.max(x_hist[:, 1]):.3f}] m/s, "
        f"z=[{np.min(x_hist[:, 2]):.3f}, {np.max(x_hist[:, 2]):.3f}] m"
    )

    if plot:
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
        axs[0].plot(t_pre, x_hist[:, 1], label="ego velocity")
        axs[0].plot(t_pre, lead_velocity_hist, "--", label="lead velocity")
        axs[0].plot(t_pre, velocity_command_hist, ":", label="expert velocity command")
        axs[0].axhline(config["pretrain_v_lower"], color="r", linestyle="--")
        axs[0].axhline(config["pretrain_v_upper"], color="r", linestyle="--")
        axs[0].set_ylabel("velocity (m/s)")
        axs[1].plot(t_pre, x_hist[:, 2], label="distance")
        axs[1].plot(t_pre, gap_reference_hist, ":", label="expert gap reference")
        axs[1].axhline(config["pretrain_z_lower"], color="r", linestyle="--")
        axs[1].axhline(config["pretrain_z_upper"], color="r", linestyle="--")
        axs[1].set_ylabel("z (m)")
        axs[2].plot(t_pre, u_hist, label="expert input")
        axs[2].axhline(u_max, color="k", linestyle="--")
        axs[2].axhline(u_min, color="k", linestyle="--")
        axs[2].set_ylabel("force (N)")
        axs[2].set_xlabel("time (s)")
        for ax in axs:
            ax.grid(True)
            ax.legend()
        fig.suptitle("Pretraining: expert control")

        components = ((1, r"$w_2=d/m$"), (2, r"$w_3=\Delta v_l$"))
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
        for ax, (component, component_label) in zip(axs, components):
            ax.plot(t_pre, true_uncertainty_hist[:, component], label="true uncertainty")
            ax.plot(t_pre, fitted_uncertainty_hist[:, component], "--", label=r"$Y_\Theta(x)a_k$")
            ax.set_ylabel(component_label)
            ax.grid(True)
            ax.legend()
        axs[-1].set_xlabel("time (s)")
        fig.suptitle(r"Pretraining: $Y_\Theta(x)a_k$ versus true uncertainty")

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.semilogy(t_pre, np.maximum(prediction_error_hist, 1e-16), label="pretraining")
        ax.set_ylabel("squared prediction error")
        ax.set_xlabel("time (s)")
        ax.grid(True)
        ax.legend()
        fig.suptitle("Pretraining: uncertainty-model prediction loss")

        plt.show()

    return olacp, history


def run_cracbf_simulation(
    system, online_olacp, config, use_cp, use_adaptive, plot=False, label=None
):
    """Run one main CRaCBF experiment"""
    if bool(system.use_cp) != bool(use_cp) or bool(system.use_adaptive) != bool(use_adaptive):
        raise ValueError(
            "The ACC object must be constructed with the requested CP and adaptation flags"
        )

    K = config["K"]
    I_length = config["I_length"]
    dt = config["dt"]
    interval_duration = config["interval_duration"]
    mass = config["mass"]
    beta_1 = config["beta_1"]
    beta_2 = config["beta_2"]
    beta_3 = config["beta_3"]
    beta_4 = config["beta_4"]
    nominal_lead_velocity = config["nominal_lead_velocity"]
    lead_velocity_scale = config["lead_velocity_scale"]
    u_min = config["u_min"]
    u_max = config["u_max"]
    gamma_cbf_inv = config["Gamma_cbf_inv"]
    t_full = np.arange(K * I_length, dtype=float) * dt
    run_label = label or f"CP={bool(use_cp)}, adaptive={bool(use_adaptive)}"

    environment_phase = 2.0 * np.pi * np.arange(K) / K / 2.0
    decline_angle_schedule = np.zeros(K)
    decline_angle_schedule[8:] = 10.0
    d0_schedule = mass * config["gravity"] * np.sin(np.deg2rad(decline_angle_schedule))
    wind_schedule = 10.0 * np.sin(0.7 * environment_phase - 0.2)
    wind_schedule[8:] = 21.0
    delta_lead_schedule = np.hstack((np.linspace(0.0, -4.199, 8), -4.199 * np.ones(K - 8)))

    if np.any(np.diff(delta_lead_schedule) > 0.0):
        raise ValueError("The main lead vehicle must continue slowing down")
    true_a4_schedule = delta_lead_schedule / lead_velocity_scale
    if np.any(true_a4_schedule < system.a_lb[3]) or np.any(true_a4_schedule > system.a_ub[3]):
        raise ValueError("The online Delta v_l schedule is outside the a_4 bounds")

    def schedule_index(t):
        return min(int(np.floor(max(float(t), 0.0) / interval_duration)), K - 1)

    def drag_force(x, d0, wind_velocity):
        v, z = np.asarray(x, dtype=float).reshape(ACC.xdim)[1:]
        wake = 1.0 - beta_3 * np.exp(-beta_4 * z)
        return d0 + beta_1 * v + beta_2 * (v - wind_velocity) ** 2 * wake

    def true_uncertainty(x, t):
        index = schedule_index(t)
        drag = drag_force(x, d0_schedule[index], wind_schedule[index])
        return np.array([0.0, drag / mass, delta_lead_schedule[index]])

    online_olacp.clear_buffers()
    if online_olacp._representation_intervals:
        raise RuntimeError(
            "The pretrained OLACP snapshot contains an incomplete representation block"
        )
    online_olacp.Y_Theta = system.Y_Theta
    online_olacp.representation_loss_gradient = system.representation_loss_gradient
    if online_olacp.Y_Theta.__self__ is not system:
        raise RuntimeError("The online representation callback is bound to the wrong ACC object")
    if online_olacp.representation_loss_gradient.__self__ is not system:
        raise RuntimeError("The online loss-gradient callback is bound to the wrong ACC object")
    system.true_uncertainty_fcn = true_uncertainty
    system.set_representation(online_olacp.Theta)
    system.cp_quantile = online_olacp.compute_quantile()
    if np.shares_memory(system.Theta_hat, online_olacp.Theta):
        raise RuntimeError("The ACC and OLACP representation arrays must be independent")

    x = config["main_initial_state"].copy()
    initial_state = x.copy()
    a_hat = system.a_center.copy()
    rho = 0.0
    x_ext = np.hstack((x, a_hat, rho))
    x_hist = np.zeros((len(t_full), system.xdim))
    u_hist = np.full(len(t_full), np.nan)
    u_ref_hist = np.full(len(t_full), np.nan)
    h_hist = np.zeros(len(t_full))
    physical_margin_hist = np.zeros(len(t_full))
    tightened_margin_hist = np.zeros(len(t_full))
    z_b_hist = np.full(len(t_full), np.nan)
    z_b_bound_hist = np.full(len(t_full), np.nan)
    a_hat_hist = np.zeros((len(t_full), system.adim))
    a_k_hist = np.full((len(t_full), system.adim), np.nan)
    lead_velocity_hist = np.zeros(len(t_full))
    rho_hist = np.zeros(len(t_full))
    nu_hist = np.zeros(len(t_full))
    quantile_hist = np.zeros(len(t_full))
    theta_hist = np.zeros((len(t_full),) + online_olacp.Theta.shape)
    prediction_error_hist = np.full(len(t_full), np.nan)
    true_uncertainty_hist = np.full((len(t_full), system.xdim), np.nan)
    fitted_uncertainty_hist = np.full((len(t_full), system.xdim), np.nan)
    interval_times = []
    score_hist = []
    delta_hist = []
    miscoverage_hist = []
    safety_violation_index = None
    last_sample_index = -1

    for sample_index, t in enumerate(t_full):
        interval_index = sample_index // I_length
        last_sample_index = sample_index
        x_hist[sample_index] = x
        a_hat_hist[sample_index] = a_hat
        lead_velocity_hist[sample_index] = (
            nominal_lead_velocity + delta_lead_schedule[interval_index]
        )
        rho_hist[sample_index] = rho
        nu_hist[sample_index] = system.nu_cbf(rho)
        quantile_hist[sample_index] = system.cp_quantile
        theta_hist[sample_index] = system.Theta_hat
        h_hist[sample_index] = float(system.cbf(x, a_hat))
        physical_margin_hist[sample_index] = x[2] - system.z_min
        tightening = 0.5 / nu_hist[sample_index] * system.safe_set_tightening
        tightened_margin_hist[sample_index] = h_hist[sample_index] - tightening

        if h_hist[sample_index] < 0.0:
            safety_violation_index = sample_index
            estimated_lead_velocity = nominal_lead_velocity + lead_velocity_scale * a_hat[3]
            print(
                f"{run_label}: SAFETY VIOLATION h={h_hist[sample_index]:.3e} at t={t:.3f} s, "
                f"actual v_l={lead_velocity_hist[sample_index]:.3f} m/s, "
                f"CBF estimate={estimated_lead_velocity:.3f} m/s"
            )
            break

        # The unsafe baseline may leave the tightened set before its certificate crosses zero.
        if sample_index % I_length == 0 and use_adaptive and use_cp:
            if tightened_margin_hist[sample_index] < -1e-8:
                raise ValueError(
                    f"{run_label}: interval {interval_index + 1} starts outside the tightened set"
                )

        u_ref = system.ctrl_nominal(x)
        try:
            u = system.ctrl_cracbf(x, a_hat, u_ref, rho)
        except ValueError as error:
            raise RuntimeError(
                f"{run_label}: CRaCBF QP failed at t={t:.3f}, x={x}, a_hat={a_hat}, rho={rho:.3e}"
            ) from error
        u_ref_hist[sample_index] = u_ref.item()
        u_hist[sample_index] = u.item()
        online_olacp.add_data_to_buffers(
            x, system.dynamics_nominal(x, u), xdot=system.dynamics(x, u, t)
        )

        if sample_index < len(t_full) - 1:
            solution = solve_ivp(
                lambda tau, state: system.dynamics_extended(state, u, tau),
                (t_full[sample_index], t_full[sample_index + 1]),
                x_ext,
                method="BDF",
                rtol=1e-7,
                atol=1e-9,
                t_eval=[t_full[sample_index + 1]],
            )
            if not solution.success:
                raise RuntimeError(solution.message)
            x_ext = solution.y[:, -1]
            if not np.all(np.isfinite(x_ext)):
                raise RuntimeError(f"{run_label}: the extended ACC state became non-finite")
            x = x_ext[: system.xdim]
            a_hat = x_ext[system.xdim : system.xdim + system.adim]
            rho = float(x_ext[system.xdim + system.adim])

        if (sample_index + 1) % I_length == 0:
            online_olacp.estimate_uncertainty(dt)
            score = float(online_olacp.compute_score(system.a_ub, system.a_lb))
            interval_true = np.asarray(online_olacp._w_buffer, dtype=float)
            interval_fitted = np.asarray([Y_t @ online_olacp.a_k for Y_t in online_olacp._Y_buffer])
            interval_error = np.sum((interval_fitted - interval_true) ** 2, axis=1)
            miscoverage = int(online_olacp.update_delta(score))
            online_olacp.append_score(score)
            representation_update = online_olacp.update_representation()
            interval_start = sample_index - I_length + 1
            interval_slice = slice(interval_start, sample_index + 1)
            system.a_true = online_olacp.a_k.copy()
            a_k_hist[interval_slice] = online_olacp.a_k
            prediction_error_hist[interval_slice] = interval_error
            true_uncertainty_hist[interval_slice] = interval_true
            fitted_uncertainty_hist[interval_slice] = interval_fitted
            interval_z_b = np.empty(I_length)
            for local_index, history_index in enumerate(range(interval_start, sample_index + 1)):
                a_tilde = a_hat_hist[history_index] - system.a_true
                interval_z_b[local_index] = nu_hist[history_index] * h_hist[history_index]
                interval_z_b[local_index] -= 0.5 * a_tilde @ gamma_cbf_inv @ a_tilde
            z_b_hist[interval_slice] = interval_z_b
            interval_time = t_full[interval_slice]
            z_b_bound_hist[interval_slice] = interval_z_b[0] * np.exp(
                -system.cbf_rate * (interval_time - interval_time[0])
            )
            interval_times.append(t)
            score_hist.append(score)
            delta_hist.append(online_olacp.delta)
            miscoverage_hist.append(miscoverage)
            if representation_update is not None:
                system.set_representation(representation_update["Theta"])
            system.cp_quantile = online_olacp.compute_quantile()
            online_olacp.clear_buffers()
            print(
                f"{run_label}: interval={interval_index + 1:02d}, score={score:.3e}, "
                f"Q_used={quantile_hist[sample_index]:.3e}, "
                f"delta_next={online_olacp.delta:.3f}, miscoverage={miscoverage}"
            )

    if safety_violation_index is not None:
        partial_start = safety_violation_index // I_length * I_length
        buffered_count = safety_violation_index - partial_start
        buffer_lengths = (
            len(online_olacp._x_buffer),
            len(online_olacp._xdot_buffer),
            len(online_olacp._xdot_nom_buffer),
            len(online_olacp._Y_buffer),
        )
        if any(length != buffered_count for length in buffer_lengths):
            raise RuntimeError(
                f"{run_label}: partial Algorithm 1 buffers have lengths {buffer_lengths}"
            )
        partial_true = [
            np.asarray(xdot) - np.asarray(xdot_nom)
            for xdot, xdot_nom in zip(
                online_olacp._xdot_buffer, online_olacp._xdot_nom_buffer
            )
        ]
        partial_Y = [np.asarray(Y_t) for Y_t in online_olacp._Y_buffer]
        violation_state = x_hist[safety_violation_index]
        violation_time = t_full[safety_violation_index]
        partial_true.append(true_uncertainty(violation_state, violation_time))
        partial_Y.append(system.Y(violation_state))
        partial_true = np.asarray(partial_true)
        partial_fitted_a = online_olacp.a_k.copy()
        partial_fitted = np.asarray([Y_t @ partial_fitted_a for Y_t in partial_Y])
        partial_error = np.sum((partial_fitted - partial_true) ** 2, axis=1)
        partial_stop = safety_violation_index + 1
        partial_slice = slice(partial_start, partial_stop)
        a_k_hist[partial_slice] = partial_fitted_a
        true_uncertainty_hist[partial_slice] = partial_true
        fitted_uncertainty_hist[partial_slice] = partial_fitted
        prediction_error_hist[partial_slice] = partial_error
        partial_z_b = np.empty(partial_stop - partial_start)
        for local_index, history_index in enumerate(range(partial_start, partial_stop)):
            a_tilde = a_hat_hist[history_index] - partial_fitted_a
            partial_z_b[local_index] = nu_hist[history_index] * h_hist[history_index]
            partial_z_b[local_index] -= 0.5 * a_tilde @ gamma_cbf_inv @ a_tilde
        z_b_hist[partial_slice] = partial_z_b
        partial_time = t_full[partial_slice]
        z_b_bound_hist[partial_slice] = partial_z_b[0] * np.exp(
            -system.cbf_rate * (partial_time - partial_time[0])
        )

    sample_count = last_sample_index + 1
    t = t_full[:sample_count]
    x_hist = x_hist[:sample_count]
    u_hist = u_hist[:sample_count]
    u_ref_hist = u_ref_hist[:sample_count]
    h_hist = h_hist[:sample_count]
    physical_margin_hist = physical_margin_hist[:sample_count]
    tightened_margin_hist = tightened_margin_hist[:sample_count]
    z_b_hist = z_b_hist[:sample_count]
    z_b_bound_hist = z_b_bound_hist[:sample_count]
    a_hat_hist = a_hat_hist[:sample_count]
    a_k_hist = a_k_hist[:sample_count]
    lead_velocity_hist = lead_velocity_hist[:sample_count]
    rho_hist = rho_hist[:sample_count]
    nu_hist = nu_hist[:sample_count]
    quantile_hist = quantile_hist[:sample_count]
    theta_hist = theta_hist[:sample_count]
    prediction_error_hist = prediction_error_hist[:sample_count]
    true_uncertainty_hist = true_uncertainty_hist[:sample_count]
    fitted_uncertainty_hist = fitted_uncertainty_hist[:sample_count]
    interval_times = np.asarray(interval_times)
    score_hist = np.asarray(score_hist)
    delta_hist = np.asarray(delta_hist)
    miscoverage_hist = np.asarray(miscoverage_hist)
    safety_violated = safety_violation_index is not None

    expected_control_mask = np.ones(len(t), dtype=bool)
    if safety_violated:
        expected_control_mask[-1] = False
        if h_hist[-1] >= 0.0 or np.any(h_hist[:-1] < 0.0):
            raise RuntimeError(f"{run_label}: inconsistent safety-termination index")
    elif np.min(h_hist) < -1e-6 or np.min(physical_margin_hist) < -1e-6:
        raise RuntimeError(f"{run_label}: safety was violated without terminating")
    if not np.array_equal(np.isfinite(u_hist), expected_control_mask):
        raise RuntimeError(f"{run_label}: invalid CRaCBF input history")
    if not np.array_equal(np.isfinite(u_ref_hist), expected_control_mask):
        raise RuntimeError(f"{run_label}: invalid nominal input history")
    issued_u = u_hist[expected_control_mask]
    if issued_u.size and (np.min(issued_u) < u_min - 1e-6 or np.max(issued_u) > u_max + 1e-6):
        raise RuntimeError(f"{run_label}: CRaCBF input bounds were violated")
    projection_radius = np.max(np.linalg.norm(a_hat_hist - system.a_center, axis=1))
    if projection_radius > system.a_hat_norm_max + 1e-6:
        raise RuntimeError(f"{run_label}: CRaCBF parameter projection was violated")
    expected_intervals = safety_violation_index // I_length if safety_violated else K
    if len(score_hist) != expected_intervals:
        raise RuntimeError(f"{run_label}: unexpected number of completed Algorithm 1 intervals")
    if not np.allclose(system.Theta_hat, online_olacp.Theta):
        raise RuntimeError(f"{run_label}: OLACP representation was not propagated to the ACC")
    for history in (prediction_error_hist, true_uncertainty_hist, fitted_uncertainty_hist):
        if np.any(~np.isfinite(history)):
            raise RuntimeError(f"{run_label}: an uncertainty history is incomplete")

    activation = np.clip(u_ref_hist, u_min, u_max) - u_hist > 1e-3
    activation_indices = np.flatnonzero(activation)
    activation_time = float(t[activation_indices[0]]) if activation_indices.size else None
    ego_peak_index = int(np.argmax(x_hist[:, 1]))
    minimum_gap_index = int(np.argmin(x_hist[:, 2]))
    if not safety_violated:
        if activation_time is None:
            raise RuntimeError(f"{run_label}: the CRaCBF never modifies the nominal input")
        if x_hist[ego_peak_index, 1] <= x_hist[0, 1] + 1.0:
            raise RuntimeError(f"{run_label}: the ego vehicle did not initially accelerate")
        if abs(t[ego_peak_index] - activation_time) > 0.25:
            raise RuntimeError(f"{run_label}: CRaCBF intervention did not cause the velocity peak")
        if x_hist[-1, 1] >= x_hist[ego_peak_index, 1] - 1.0:
            raise RuntimeError(f"{run_label}: the ego vehicle did not slow after intervention")
        if use_adaptive and use_cp and np.min(tightened_margin_hist) < -1e-6:
            raise RuntimeError(f"{run_label}: the tightened CRaCBF set was violated")

    true_a4_hist = (lead_velocity_hist - nominal_lead_velocity) / lead_velocity_scale
    status = f"terminated at h={h_hist[-1]:.3e}" if safety_violated else "safety maintained"
    activation_text = (
        f"active at t={activation_time:.3f} s"
        if activation_time is not None
        else "never active"
    )
    print(
        f"{run_label}: lead={lead_velocity_hist[0]:.3f}->{lead_velocity_hist[-1]:.3f} m/s, "
        f"CRaCBF {activation_text}, peak v={x_hist[ego_peak_index, 1]:.3f} m/s, "
        f"minimum z={x_hist[minimum_gap_index, 2]:.3f} m, {status}"
    )

    result = {
        "label": run_label,
        "use_cp": bool(use_cp),
        "use_adaptive": bool(use_adaptive),
        "initial_state": initial_state,
        "t": t,
        "x_hist": x_hist,
        "u_hist": u_hist,
        "u_ref_hist": u_ref_hist,
        "h_hist": h_hist,
        "physical_margin_hist": physical_margin_hist,
        "tightened_margin_hist": tightened_margin_hist,
        "z_b_hist": z_b_hist,
        "z_b_bound_hist": z_b_bound_hist,
        "a_hat_hist": a_hat_hist,
        "a_k_hist": a_k_hist,
        "lead_velocity_hist": lead_velocity_hist,
        "rho_hist": rho_hist,
        "nu_hist": nu_hist,
        "quantile_hist": quantile_hist,
        "theta_hist": theta_hist,
        "prediction_error_hist": prediction_error_hist,
        "true_uncertainty_hist": true_uncertainty_hist,
        "fitted_uncertainty_hist": fitted_uncertainty_hist,
        "interval_times": interval_times,
        "score_hist": score_hist,
        "delta_hist": delta_hist,
        "miscoverage_hist": miscoverage_hist,
        "true_a4_hist": true_a4_hist,
        "safety_violated": safety_violated,
        "safety_violation_time": float(t[-1]) if safety_violated else None,
        "activation_time": activation_time,
        "d0_schedule": d0_schedule,
        "wind_schedule": wind_schedule,
        "delta_lead_schedule": delta_lead_schedule,
        "desired_velocity": config["desired_velocity"],
        "z_min": system.z_min,
        "u_min": u_min,
        "u_max": u_max,
        "interval_duration": interval_duration,
        "final_theta": online_olacp.Theta.copy(),
    }
    if plot:
        plot_cracbf_results({run_label: result})
    return result


def plot_cracbf_results(results):
    """Plot and compare one or more main CRaCBF simulation results"""
    items = (
        list(results.items())
        if isinstance(results, dict)
        else [(result["label"], result) for result in results]
    )
    if not items:
        raise ValueError("At least one CRaCBF result is required")

    reference = max((result for _, result in items), key=lambda result: len(result["t"]))
    for label, result in items[1:]:
        if not np.array_equal(result["initial_state"], reference["initial_state"]):
            raise RuntimeError(f"{label}: comparison initial state does not match")
        for key in ("d0_schedule", "wind_schedule", "delta_lead_schedule"):
            if not np.array_equal(result[key], reference[key]):
                raise RuntimeError(f"{label}: comparison disturbance schedule {key} does not match")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {label: colors[index % len(colors)] for index, (label, _) in enumerate(items)}
    maximum_time = max(float(result["t"][-1]) for _, result in items)
    figures = []

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    figures.append(fig)
    axs[0].plot(reference["t"], reference["lead_velocity_hist"], "k--", label="lead velocity")
    axs[0].axhline(
        reference["desired_velocity"], color="k", linestyle=":", label="desired velocity"
    )
    axs[1].axhline(reference["z_min"], color="r", linestyle="--", label="z_min")
    for label, result in items:
        color = color_map[label]
        t = result["t"]
        axs[0].plot(t, result["x_hist"][:, 1], color=color, label=f"ego: {label}")
        axs[1].plot(t, result["x_hist"][:, 2], color=color, label=label)
        if result["safety_violated"]:
            axs[0].plot(t[-1], result["x_hist"][-1, 1], "x", color=color, markersize=9)
            axs[1].plot(t[-1], result["x_hist"][-1, 2], "x", color=color, markersize=9)
    axs[0].set_ylabel("velocity (m/s)")
    axs[1].set_ylabel("z (m)")
    for ax in axs:
        ax.set_xlim(0.0, maximum_time)
        ax.grid(True)
        ax.legend()
    fig.suptitle("ACC state comparison")

    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(8, 7))
    figures.append(fig)
    axs.axhline(reference["u_max"], color="k", linestyle="--")
    axs.axhline(reference["u_min"], color="k", linestyle="--")
    for label, result in items:
        color = color_map[label]
        t = result["t"]
        axs.plot(t, result["u_hist"], color=color, label=f"control: {label}")
        axs.plot(t, result["u_ref_hist"], color=color, linestyle=":", label=f"nominal: {label}")
    axs.set_ylabel("force (N)")
    axs.set_xlabel("time (s)")
    axs.set_xlim(0.0, maximum_time)
    axs.grid(True)
    axs.legend()
    fig.suptitle("ACC control comparison")    

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    figures.append(fig)
    for label, result in items:
        color = color_map[label]
        t = result["t"]
        axs[0].plot(t, result["h_hist"], color=color, label=f"h: {label}")
        axs[0].plot(
            t,
            result["tightened_margin_hist"],
            color=color,
            linestyle=":",
            label=f"tightened: {label}",
        )
        axs[1].plot(t, result["physical_margin_hist"], color=color, label=label)
        axs[2].plot(t, result["z_b_hist"], color=color, label=f"z_b: {label}")
        axs[2].plot(
            t, result["z_b_bound_hist"], color=color, linestyle="--", label=f"bound: {label}"
        )
        if result["safety_violated"]:
            axs[0].plot(t[-1], result["h_hist"][-1], "x", color=color, markersize=10)
    axs[0].axhline(0.0, color="r", linestyle="--")
    axs[1].axhline(0.0, color="r", linestyle="--")
    axs[0].set_ylabel("certificate")
    axs[1].set_ylabel("z - z_min")
    axs[2].set_ylabel("z_b")
    axs[2].set_xlabel("time (s)")
    for ax in axs:
        ax.set_xlim(0.0, maximum_time)
        ax.grid(True)
        ax.legend()
    fig.suptitle("CRaCBF safety comparison")

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    figures.append(fig)
    for label, result in items:
        color = color_map[label]
        axs[0].step(
            result["interval_times"],
            result["score_hist"],
            where="post",
            color=color,
            label=f"s_k: {label}",
        )
        axs[0].step(
            result["t"],
            result["quantile_hist"],
            where="post",
            color=color,
            linestyle=":",
            label=f"Q_k: {label}",
        )
        axs[1].step(
            result["interval_times"], result["delta_hist"], where="post", color=color, label=label
        )
        axs[2].step(
            result["interval_times"],
            result["miscoverage_hist"],
            where="post",
            color=color,
            label=label,
        )
    axs[0].set_ylabel("score")
    axs[1].set_ylabel("delta_k")
    axs[2].set_ylabel("miscoverage")
    axs[2].set_xlabel("time (s)")
    for ax in axs:
        ax.set_xlim(0.0, maximum_time)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Algorithm 1 comparison")

    fig, axs = plt.subplots(ACC.adim, 1, sharex=True, figsize=(8, 7))
    figures.append(fig)
    for component in range(ACC.adim):
        for label, result in items:
            color = color_map[label]
            axs[component].plot(
                result["t"],
                result["a_hat_hist"][:, component],
                color=color,
                label=f"hat a: {label}",
            )
            axs[component].plot(
                result["t"],
                result["a_k_hist"][:, component],
                color=color,
                linestyle="--",
                label=f"a_k: {label}",
            )
        if component == 3:
            axs[component].step(
                reference["t"],
                reference["true_a4_hist"],
                where="post",
                color="k",
                linestyle=":",
                label="true a4",
            )
        axs[component].set_ylabel(f"a{component + 1}")
        axs[component].set_xlim(0.0, maximum_time)
        axs[component].grid(True)
    axs[0].legend()
    axs[3].legend()
    axs[-1].set_xlabel("time (s)")
    fig.suptitle("CRaCBF adaptation and OLACP identification comparison")

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    figures.append(fig)
    for label, result in items:
        color = color_map[label]
        axs[0].plot(result["t"], result["nu_hist"], color=color, label=label)
        axs[1].plot(result["t"], result["rho_hist"], color=color, label=label)
        axs[2].semilogy(
            result["t"],
            np.maximum(result["prediction_error_hist"], 1e-16),
            color=color,
            label=label,
        )
    axs[0].set_ylabel("nu(rho)")
    axs[1].set_ylabel("rho")
    axs[2].set_ylabel("squared prediction error")
    axs[2].set_xlabel("time (s)")
    for ax in axs:
        ax.set_xlim(0.0, maximum_time)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Scaling, projection, and prediction-loss comparison")

    components = ((1, r"$w_2=d/m$"), (2, r"$w_3=\Delta v_l$"))
    fig, axs = plt.subplots(2, len(items), squeeze=False, figsize=(6 * len(items), 7))
    figures.append(fig)
    for column, (label, result) in enumerate(items):
        for row, (component, component_label) in enumerate(components):
            ax = axs[row, column]
            ax.plot(result["t"], result["true_uncertainty_hist"][:, component], label="true")
            ax.plot(
                result["t"],
                result["fitted_uncertainty_hist"][:, component],
                "--",
                label=r"$Y_\Theta(x)a_k$",
            )
            ax.set_xlim(0.0, float(result["t"][-1]))
            ax.set_ylabel(component_label)
            ax.set_title(label)
            ax.grid(True)
            ax.legend()
        axs[-1, column].set_xlabel("time (s)")
    fig.suptitle(r"Main CRaCBF uncertainty-model comparison")
    return figures


def main():
    """Build the shared experiment, run the controller settings, and compare them"""
    K_pre = 32
    N_cal = 30
    K = 12
    B = 4
    dt = 0.01
    interval_duration = 2.0
    I_length = int(round(interval_duration / dt))
    if K_pre < N_cal:
        raise ValueError("K_pre must be at least as large as N_cal")
    if K_pre % B != 0:
        raise ValueError("K_pre must be an integer multiple of B")
    if I_length < 10 or not np.isclose(I_length * dt, interval_duration):
        raise ValueError("interval_duration must be an integer multiple of dt")

    mass = 1000.0
    gravity = 9.81
    beta_1 = -10.0
    beta_2 = -0.75
    beta_3 = 0.2
    beta_4 = 0.02
    desired_velocity = 28.0
    nominal_lead_velocity = 25.0
    lead_velocity_scale = 10.0
    u_min = -1.0 * mass * gravity
    u_max = 0.5 * mass * gravity
    v_reference = desired_velocity
    z_reference = 35.0
    psi_reference = np.kron(
        np.array([1.0, v_reference, v_reference**2]),
        np.array([1.0, z_reference, z_reference**2]),
    )
    theta_scale = 1.0 / psi_reference
    theta_lb = -1.0 * theta_scale[:, None] * np.ones((1, 3))
    theta_ub = 1.0 * theta_scale[:, None] * np.ones((1, 3))
    theta_rng = np.random.default_rng(11)
    theta_init = theta_rng.uniform(
        -theta_scale[:, None], theta_scale[:, None], size=ACC.theta_shape
    )

    a_lb = np.array([-0.1, -0.1, -0.1, -0.42])
    a_ub = np.array([0.1, 0.1, 0.1, 0.42])
    projection_epsilon = 0.01
    a_hat_norm_max = 0.5 * np.linalg.norm(a_ub - a_lb, ord=2) + projection_epsilon
    gamma_cbf = 10.0 * np.eye(ACC.adim)

    config = {
        "K_pre": K_pre,
        "N_cal": N_cal,
        "K": K,
        "B": B,
        "dt": dt,
        "interval_duration": interval_duration,
        "I_length": I_length,
        "mass": mass,
        "gravity": gravity,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "beta_3": beta_3,
        "beta_4": beta_4,
        "desired_velocity": desired_velocity,
        "nominal_lead_velocity": nominal_lead_velocity,
        "lead_velocity_scale": lead_velocity_scale,
        "u_min": u_min,
        "u_max": u_max,
        "Gamma_cbf": gamma_cbf,
        "Gamma_cbf_inv": np.linalg.inv(gamma_cbf),
        "pretrain_initial_state": np.array([0.0, 24.0, 35.0]),
        "main_initial_state": np.array([0.0, 26.0, 40.0]),
        "pretrain_z_lower": 10.0,
        "pretrain_z_upper": 50.0,
        "pretrain_v_lower": 10.0,
        "pretrain_v_upper": 30.0,
        "pretrain_gap_reference_center": 37.0,
        "pretrain_gap_reference_amplitude": 11.0,
        "pretrain_gap_reference_period": 20.0,
        "pretrain_gap_gain": 0.5,
        "pretrain_velocity_gain": 1.6,
    }
    base_acc_params = {
        "Theta_init": theta_init.copy(),
        "true_uncertainty": lambda x, t: np.zeros(ACC.xdim),
        "m": mass,
        "vd": desired_velocity,
        "Kp": 2000.0,
        "nominal_lead_velocity": nominal_lead_velocity,
        "lead_velocity_scale": lead_velocity_scale,
        "cbf_smoothing_epsilon": 0.02,
        "z_min": 10.0,
        "T_h": 0.45,
        "Gamma_cbf": gamma_cbf,
        "a_ub": a_ub,
        "a_lb": a_lb,
        "a_hat_norm_max": a_hat_norm_max,
        "epsilon": projection_epsilon,
        "eta_cbf": 1.0,
        "cbf_rate": 7.0,
        "u_max": u_max,
        "u_min": u_min,
        "dt": dt,
    }

    pretrain_params = dict(base_acc_params)
    pretrain_params.update({"use_cp": True, "use_adaptive": True})
    pretrain_system = ACC(pretrain_params)
    representation_rate = (
        lambda update_index: 0.1
        / (B * I_length)
        / psi_reference[:, None]
        / np.sqrt(update_index)
    )
    olacp = OLACP(
        [],
        N_cal=N_cal,
        acp_lr=0.02,
        delta_target=0.1,
        delta_init=0.1,
        buffer_maxlen=I_length,
        Theta_init=theta_init,
        representation_period=B,
        representation_lr=representation_rate,
        Theta_lb=theta_lb,
        Theta_ub=theta_ub,
        Y_Theta=pretrain_system.Y_Theta,
        representation_loss_gradient=pretrain_system.representation_loss_gradient,
    )
    trained_olacp, pretraining_history = run_pretraining(
        pretrain_system, olacp, config, plot=True
    )
    canonical_theta = trained_olacp.Theta.copy()
    canonical_a_k = trained_olacp.a_k.copy()
    canonical_scores = np.asarray(trained_olacp.S_cal).copy()
    canonical_delta = float(trained_olacp.delta)
    canonical_interval_index = int(trained_olacp.interval_index)
    canonical_update_index = int(trained_olacp.representation_update_index)

    settings = (
        ("CP + adaptive", True, True),
        ("No CP + adaptive", False, True),
        ("No CP + nonadaptive", False, False),
    )
    results = {}
    for label, use_cp, use_adaptive in settings:
        run_params = dict(base_acc_params)
        run_params["Theta_init"] = canonical_theta.copy()
        run_params["use_cp"] = use_cp
        run_params["use_adaptive"] = use_adaptive
        run_system = ACC(run_params)
        run_olacp = copy.deepcopy(trained_olacp)
        if np.shares_memory(run_olacp.Theta, trained_olacp.Theta):
            raise RuntimeError(f"{label}: online and canonical OLACP objects share Theta memory")
        results[label] = run_cracbf_simulation(
            run_system,
            run_olacp,
            config,
            use_cp,
            use_adaptive,
            plot=False,
            label=label,
        )

    if not np.array_equal(trained_olacp.Theta, canonical_theta):
        raise RuntimeError("A main simulation mutated the canonical pretrained Theta")
    if not np.array_equal(trained_olacp.a_k, canonical_a_k):
        raise RuntimeError("A main simulation mutated the canonical pretrained a_k")
    if not np.array_equal(np.asarray(trained_olacp.S_cal), canonical_scores):
        raise RuntimeError("A main simulation mutated the canonical calibration set")
    if trained_olacp.delta != canonical_delta:
        raise RuntimeError("A main simulation mutated the canonical adaptive delta")
    if trained_olacp.interval_index != canonical_interval_index:
        raise RuntimeError("A main simulation mutated the canonical interval index")
    if trained_olacp.representation_update_index != canonical_update_index:
        raise RuntimeError("A main simulation mutated the canonical representation-update index")
    if results["CP + adaptive"]["safety_violated"]:
        raise RuntimeError("The CP + adaptive run unexpectedly violated safety")
    if not results["No CP + nonadaptive"]["safety_violated"]:
        raise RuntimeError("The no-CP nonadaptive run unexpectedly maintained safety")

    plot_cracbf_results(results)
    plt.show()
    return pretraining_history, results


if __name__ == "__main__":
    main()
