"""Algorithm 1 and CRaCCM for the nonlinear toy system"""

import copy
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from geodesic_solver import GeodesicSolver
from motion_planner import MotionPlanner
from olacp import OLACP
from systems.nonlinear_toy.nonlinear_toy import NONLINEAR_TOY


def noise_fcn(t):
    """Add noise to the true uncertainty"""
    return np.array(
        [
            0.20 * np.sin(2.0 * np.pi * 0.67 * t + 0.3),
            0.50 * np.sin(2.0 * np.pi * 0.1 * t),
            0.30 * np.cos(2.0 * np.pi * 0.87 * t + 0.1),
        ]
    )


def true_uncertainty_fcn(x, t, schedule):
    """True uncertainty: Y_Theta(x)a + noise, where a is scheduled"""
    x1, _, x3 = np.asarray(x, dtype=float).reshape(NONLINEAR_TOY.xdim)
    schedule = np.asarray(schedule, dtype=float).reshape(6)
    noise = noise_fcn(t)
    if not np.all(np.isfinite(schedule)) or not np.all(np.isfinite(noise)):
        raise ValueError("uncertainty schedules and noise must be finite")

    w1 = (0.8 * schedule[0] + 1.3 * schedule[1]) * x1 + noise[0]
    w2 = 0.0 + noise[1]
    w3 = (0.23 * schedule[2] + 0.87 * schedule[3]) * x3
    w3 += (0.23 * schedule[4] + 0.87 * schedule[5]) * x1**2 + noise[2]
    return np.array([w1, w2, w3])


def pretraining_control(x, t, config):
    """A stabilizing + persistently exciting controller for pretraining"""
    x = np.asarray(x, dtype=float).reshape(3)
    amplitude = config["pretrain_excitation_amplitude"]
    omega = 2.0 * np.pi * config["pretrain_excitation_frequency"]
    phase_1 = omega * t
    phase_2 = 2.3 * omega * t + 0.4
    x1_reference = 0.4 + amplitude * (
        0.75 * np.sin(phase_1) + 0.35 * np.sin(phase_2)
    )
    x3_reference = amplitude * omega * (
        0.75 * np.cos(phase_1) + 0.805 * np.cos(phase_2)
    )
    x3_reference_dot = -amplitude * omega**2 * (
        0.75 * np.sin(phase_1) + 1.8515 * np.sin(phase_2)
    )
    feedback = x3_reference_dot - np.tanh(x[1])
    feedback += 6.0 * (x1_reference - x[0]) + 4.0 * (x3_reference - x[2])
    return np.array([feedback])


def run_pretraining(system, olacp, config, plot=True):
    """Pre-train the shared representation and fill the calibration dataset using OLACP"""
    K_pre = config["K_pre"]
    I_length = config["I_length"]
    dt = config["dt"]
    interval_duration = config["interval_duration"]
    t_pre = np.arange(K_pre * I_length, dtype=float) * dt

    interval_indices = np.arange(K_pre, dtype=float)
    latent_1 = -0.65 + 0.30 * np.sin(2.0 * np.pi * interval_indices / 9.0)
    latent_2 = 0.65 * np.cos(2.0 * np.pi * interval_indices / 7.0)
    schedule_values = np.vstack(
        (
            latent_1,
            latent_2,
            latent_1 + 0.08 * np.sin(2.0 * np.pi * interval_indices / 5.0 + 0.3),
            latent_2 + 0.07 * np.cos(2.0 * np.pi * interval_indices / 6.0 - 0.2),
            latent_1 + 0.06 * np.cos(2.0 * np.pi * interval_indices / 8.0 + 0.4),
            latent_2 + 0.09 * np.sin(2.0 * np.pi * interval_indices / 10.0 + 0.6),
        )
    )

    def schedule_index(t):
        return min(int(np.floor(max(float(t), 0.0) / interval_duration)), K_pre - 1)

    def true_uncertainty(x, t):
        return true_uncertainty_fcn(x, t, schedule_values[:, schedule_index(t)])

    system.true_uncertainty_fcn = true_uncertainty
    system.set_representation(olacp.Theta)
    x = config["pretrain_initial_state"].copy()
    x_hist = np.zeros((len(t_pre), system.xdim))
    u_hist = np.zeros(len(t_pre))
    a_k_hist = np.full((len(t_pre), system.adim), np.nan)
    theta_hist = np.zeros((K_pre + 1,) + olacp.Theta.shape)
    theta_hist[0] = olacp.Theta
    score_hist = np.zeros(K_pre)
    prediction_error_hist = np.full(len(t_pre), np.nan)
    true_uncertainty_hist = np.full((len(t_pre), system.xdim), np.nan)
    fitted_uncertainty_hist = np.full((len(t_pre), system.xdim), np.nan)

    for interval_index in range(K_pre):
        for interval_sample_index in range(I_length):
            sample_index = interval_index * I_length + interval_sample_index
            t = t_pre[sample_index]
            u = pretraining_control(x, t, config)
            u = np.clip(u, system.params["u_min"], system.params["u_max"])

            x_hist[sample_index] = x
            u_hist[sample_index] = float(u.item())
            xdot = system.dynamics(x, u, t)
            olacp.add_data_to_buffers(x, system.dynamics_nominal(x, u), xdot=xdot)

            if sample_index < len(t_pre) - 1:
                solution = solve_ivp(
                    lambda tau, state: system.dynamics(state, u, tau),
                    (t, t + dt),
                    x,
                    method="RK45",
                    rtol=1e-7,
                    atol=1e-9,
                    t_eval=[t + dt],
                )
                if not solution.success:
                    raise RuntimeError(solution.message)
                x = solution.y[:, -1]

        olacp.estimate_uncertainty(dt)
        score = float(olacp.compute_score(system.a_ub, system.a_lb))
        interval_true = np.asarray(olacp._w_buffer, dtype=float)
        interval_fitted = np.asarray([Y_t @ olacp.a_k for Y_t in olacp._Y_buffer])
        interval_error = np.sum((interval_fitted - interval_true) ** 2, axis=1)
        olacp.append_score(score)
        representation_update = olacp.update_representation()
        if representation_update is not None:
            system.set_representation(representation_update["Theta"])

        interval_start = interval_index * I_length
        interval_slice = slice(interval_start, interval_start + I_length)
        a_k_hist[interval_slice] = olacp.a_k
        prediction_error_hist[interval_slice] = interval_error
        true_uncertainty_hist[interval_slice] = interval_true
        fitted_uncertainty_hist[interval_slice] = interval_fitted
        score_hist[interval_index] = score
        theta_hist[interval_index + 1] = olacp.Theta
        olacp.clear_buffers()

    if len(olacp.S_cal) != config["N_cal"]:
        raise RuntimeError("Pretraining did not fill the calibration window")
    if olacp._representation_intervals:
        raise RuntimeError("Pretraining left an incomplete representation block")
    if not np.allclose(system.Theta_hat, olacp.Theta):
        raise RuntimeError("The trained representation was not installed in the toy system")

    quantile = olacp.compute_quantile()
    history = {
        "t": t_pre,
        "x_hist": x_hist,
        "u_hist": u_hist,
        "a_k_hist": a_k_hist,
        "theta_hist": theta_hist,
        "score_hist": score_hist,
        "prediction_error_hist": prediction_error_hist,
        "true_uncertainty_hist": true_uncertainty_hist,
        "fitted_uncertainty_hist": fitted_uncertainty_hist,
        "uncertainty_schedule_values": schedule_values,
        "quantile": quantile,
    }

    print(
        f"Pretraining complete: Q_0={quantile:.3e}, a_last={olacp.a_k}, "
        f"score_first={score_hist[0]:.3e}, score_last={score_hist[-1]:.3e}, "
        f"theta_change={np.linalg.norm(theta_hist[-1] - theta_hist[0]):.3e}, "
        f"max_state_norm={np.max(np.linalg.norm(x_hist, axis=1)):.3e}, "
        f"max|u|={np.max(np.abs(u_hist)):.3e}"
    )

    if plot:
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
        for state_index, ax in enumerate(axs):
            ax.plot(t_pre, x_hist[:, state_index])
            ax.set_ylabel(rf"$x_{state_index + 1}$")
            ax.grid(True)
        axs[-1].set_xlabel("time (s)")
        fig.suptitle("Pretraining: states")

        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 7))
        ax.plot(t_pre, u_hist)
        ax.set_ylabel("input")
        ax.set_xlabel("time (s)")
        ax.grid(True)
        fig.suptitle("Pretraining: control input")

        components = ((0, r"$w_1$"), (1, r"$w_2$"), (2, r"$w_3$"))
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
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


def plan_nominal_trajectory(system, config):
    """Plan a nominal trajectory"""
    horizon_steps = config["K"] * config["I_length"]
    planner = MotionPlanner(
        system=system,
        dt=config["dt"],
        Q=config["motion_planner_Q"],
        R=config["motion_planner_R"],
        Q_f=config["motion_planner_Q_f"],
        u_min=np.array([system.params["u_min"]]),
        u_max=np.array([system.params["u_max"]]),
    )
    alpha = np.linspace(0.0, 1.0, horizon_steps + 1)
    x_guess = config["nominal_initial_state"][:, None]
    x_guess = x_guess + (
        config["nominal_goal_state"] - config["nominal_initial_state"]
    )[:, None] * alpha
    u_guess = np.zeros((system.udim, horizon_steps))
    x_d, u_d = planner.plan(
        config["nominal_initial_state"],
        config["nominal_goal_state"],
        horizon_steps,
        x_guess,
        u_guess,
    )
    t_x = config["dt"] * np.arange(horizon_steps + 1)
    t_u = t_x[:-1]
    regressor_norm = np.asarray(
        [
            np.linalg.norm(system.Y(x_d[:, index]), ord="fro")
            for index in range(horizon_steps + 1)
        ]
    )
    maximum_regressor_norm = float(np.max(regressor_norm))
    print(
        "Nominal plan complete: "
        f"max ||Y_Theta(x_d)||_F={maximum_regressor_norm:.3e}, "
    )
    return {
        "t_x": t_x,
        "t_u": t_u,
        "x_d": x_d,
        "u_d": u_d,
        "interp_x_d": interp1d(t_x, x_d, axis=1, bounds_error=False, fill_value="extrapolate"),
        "interp_u_d": interp1d(t_u, u_d, axis=1, bounds_error=False, fill_value="extrapolate"),
    }


def run_craccm_simulation(
    system,
    online_olacp,
    desired_trajectory,
    config,
    use_cp,
    use_adaptive,
    label=None,
):
    """Run one CRaCCM experiment"""
    if bool(system.use_cp) != bool(use_cp) or bool(system.use_adaptive) != bool(use_adaptive):
        raise ValueError("The toy system must be constructed with the requested controller flags")

    K = config["K"]
    interval_duration = config["interval_duration"]

    interval_times = interval_duration * np.arange(K, dtype=float)
    alternating = np.cos(np.pi * interval_times / interval_duration)
    latent_1 = 0.80 + 0.40 * np.sin(2.0 * np.pi * interval_times / (7.0 * interval_duration) + 0.2)
    latent_2 = 0.35 * np.cos(2.0 * np.pi * interval_times / (5.0 * interval_duration) - 0.3)
    schedule_values = np.vstack(
        (
            latent_1 + 0.10 * alternating,
            latent_2 + 0.010 * np.sin(np.pi * interval_times / interval_duration + 0.4),
            latent_1 + 0.012 * np.sin(0.40 * interval_times + 0.3),
            latent_2 - 0.06 * alternating,
            latent_1 + 0.010 * np.cos(0.35 * interval_times + 0.2),
            latent_2 + 0.06 * alternating,
        )
    )

    def schedule_index(t):
        return min(int(np.floor(max(float(t), 0.0) / interval_duration)), K - 1)

    def true_uncertainty(x, t):
        return true_uncertainty_fcn(x, t, schedule_values[:, schedule_index(t)])

    system.true_uncertainty_fcn = true_uncertainty
    online_olacp.clear_buffers()
    if online_olacp._representation_intervals:
        raise RuntimeError("The pretrained OLACP contains an incomplete representation block")
    online_olacp.Y_Theta = system.Y_Theta
    online_olacp.representation_loss_gradient = system.representation_loss_gradient
    system.set_representation(online_olacp.Theta)
    system.cp_quantile = online_olacp.compute_quantile()
    if np.shares_memory(system.Theta_hat, online_olacp.Theta):
        raise RuntimeError("The toy system and OLACP must own independent Theta arrays")

    I_length = config["I_length"]
    dt = config["dt"]
    t_full = np.arange(K * I_length, dtype=float) * dt
    t_hist = t_full.copy()
    run_label = label or f"CP={bool(use_cp)}, adaptive={bool(use_adaptive)}"
    rho_divergence_threshold = config["rho_divergence_threshold"]
    if not np.isfinite(rho_divergence_threshold) or rho_divergence_threshold <= 0.0:
        raise ValueError("rho_divergence_threshold must be finite and positive")

    def rho_divergence_event(_time, state):
        return rho_divergence_threshold - abs(float(state[-1]))

    rho_divergence_event.terminal = True
    rho_divergence_event.direction = -1.0

    x = desired_trajectory["x_d"][:, 0] + config["tracking_initial_offset"]
    a_hat = system.a_center.copy()
    rho = 0.0
    x_ext = np.hstack((x, a_hat, rho))

    geodesic_solver = GeodesicSolver(
        system.xdim,
        config["geodesic_degree"],
        config["geodesic_nodes"],
        system.W_fcn,
        system.dW_dxi_fcn,
        system.dW_dai_fcn,
    )

    x_hist = np.zeros((len(t_full), system.xdim))
    x_d_hist = np.zeros_like(x_hist)
    u_hist = np.full(len(t_full), np.nan)
    u_d_hist = np.full(len(t_full), np.nan)
    energy_hist = np.full(len(t_full), np.nan)
    slack_hist = np.full(len(t_full), np.nan)
    a_hat_hist = np.zeros((len(t_full), system.adim))
    a_k_hist = np.full((len(t_full), system.adim), np.nan)
    rho_hist = np.zeros(len(t_full))
    nu_hist = np.zeros(len(t_full))
    quantile_hist = np.zeros(len(t_full))
    theta_hist = np.zeros((len(t_full),) + online_olacp.Theta.shape)
    true_uncertainty_hist = np.full((len(t_full), system.xdim), np.nan)
    fitted_uncertainty_hist = np.full((len(t_full), system.xdim), np.nan)
    interval_times = []
    score_hist = []
    delta_hist = []
    miscoverage_hist = []
    status = "completed"
    failure_reason = None
    failure_time = None
    failure_rho = None
    last_sample_index = -1

    def record_terminal_sample(index, sample_time, state, held_u, held_u_d, held_slack):
        terminal_x = state[: system.xdim]
        terminal_a_hat = state[system.xdim : system.xdim + system.adim]
        terminal_rho = float(state[-1])
        terminal_x_d = np.asarray(
            desired_trajectory["interp_x_d"](sample_time), dtype=float
        ).reshape(system.xdim)
        t_hist[index] = sample_time
        x_hist[index] = terminal_x
        x_d_hist[index] = terminal_x_d
        u_hist[index] = float(held_u.item())
        u_d_hist[index] = float(held_u_d.item())
        energy_hist[index] = float(system.Erem)
        slack_hist[index] = float(held_slack)
        a_hat_hist[index] = terminal_a_hat
        rho_hist[index] = terminal_rho
        nu_hist[index] = system.nu_ccm(terminal_rho)
        quantile_hist[index] = system.cp_quantile
        theta_hist[index] = system.Theta_hat
        if np.all(np.isfinite(terminal_x)):
            true_uncertainty_hist[index] = system.true_uncertainty(terminal_x, sample_time)
        return terminal_x, terminal_a_hat, terminal_rho

    for sample_index, t in enumerate(t_full):
        last_sample_index = sample_index
        x_d = np.asarray(desired_trajectory["interp_x_d"](t), dtype=float).reshape(system.xdim)
        u_d = np.asarray(desired_trajectory["interp_u_d"](t), dtype=float).reshape(system.udim)

        x_hist[sample_index] = x
        x_d_hist[sample_index] = x_d
        a_hat_hist[sample_index] = a_hat
        rho_hist[sample_index] = rho
        nu_hist[sample_index] = system.nu_ccm(rho)
        quantile_hist[sample_index] = system.cp_quantile
        theta_hist[sample_index] = system.Theta_hat

        u, slack = system.ctrl_craccm(
            x,
            a_hat,
            x_d,
            u_d,
            geodesic_solver,
            use_qpsolvers=config["use_qpsolvers"],
            use_slack=True,
            verify_geodesic=config["verify_geodesic"],
        )
        u_hist[sample_index] = float(u.item())
        u_d_hist[sample_index] = float(u_d.item())
        energy_hist[sample_index] = system.Erem
        slack_hist[sample_index] = float(slack)

        xdot = system.dynamics(x, u, t)
        online_olacp.add_data_to_buffers(x, system.dynamics_nominal(x, u), xdot=xdot)

        if sample_index < len(t_full) - 1:
            solution = solve_ivp(
                lambda tau, state: system.dynamics_extended(
                    state, x_d, u, geodesic_solver, tau
                ),
                (t_full[sample_index], t_full[sample_index + 1]),
                x_ext,
                method="RK45",
                rtol=1e-7,
                atol=1e-9,
                t_eval=[t_full[sample_index + 1]],
                events=rho_divergence_event if use_adaptive else None,
            )
            if use_adaptive and solution.t_events[0].size:
                event_state = solution.y_events[0][-1]
                status = "diverged"
                failure_reason = "rho_divergence"
                failure_time = float(solution.t_events[0][-1])
                failure_rho = float(event_state[-1])
                event_index = sample_index + 1
                last_sample_index = event_index
                x_ext = event_state
                x, a_hat, rho = record_terminal_sample(
                    event_index, failure_time, x_ext, u, u_d, slack
                )
                break
            if not solution.success:
                raise RuntimeError(
                    f"{run_label}: extended dynamics failed at t={t:.3f}: {solution.message}"
                )
            x_ext = solution.y[:, -1]
            x = x_ext[: system.xdim]
            a_hat = x_ext[system.xdim : system.xdim + system.adim]
            rho = float(x_ext[-1])

        rho_diverged = not np.isfinite(rho) or abs(rho) >= rho_divergence_threshold
        state_is_finite = np.all(np.isfinite(x_ext))
        state_norm_diverged = np.linalg.norm(x) > config["x_norm_divergence_threshold"]
        if rho_diverged or not state_is_finite or state_norm_diverged:
            status = "diverged"
            failure_time = float(t_full[min(sample_index + 1, len(t_full) - 1)])
            failure_rho = rho
            if rho_diverged:
                failure_reason = "rho_divergence"
            elif not state_is_finite:
                failure_reason = "nonfinite_extended_state"
            else:
                failure_reason = "state_norm"
            if sample_index < len(t_full) - 1:
                terminal_index = sample_index + 1
                last_sample_index = terminal_index
                x, a_hat, rho = record_terminal_sample(
                    terminal_index, failure_time, x_ext, u, u_d, slack
                )
            break

        if (sample_index + 1) % I_length == 0:
            online_olacp.estimate_uncertainty(dt)
            score = float(online_olacp.compute_score(system.a_ub, system.a_lb))
            interval_true = np.asarray(online_olacp._w_buffer, dtype=float)
            interval_fitted = np.asarray(
                [Y_t @ online_olacp.a_k for Y_t in online_olacp._Y_buffer]
            )
            miscoverage = online_olacp.update_delta(score)
            online_olacp.append_score(score)
            representation_update = online_olacp.update_representation()

            interval_start = sample_index - I_length + 1
            interval_slice = slice(interval_start, sample_index + 1)
            a_k_hist[interval_slice] = online_olacp.a_k
            true_uncertainty_hist[interval_slice] = interval_true
            fitted_uncertainty_hist[interval_slice] = interval_fitted
            interval_times.append(t)
            score_hist.append(score)
            delta_hist.append(online_olacp.delta)
            miscoverage_hist.append(miscoverage)

            if representation_update is not None:
                system.set_representation(representation_update["Theta"])
            system.cp_quantile = online_olacp.compute_quantile()
            online_olacp.clear_buffers()

    used_slice = slice(0, last_sample_index + 1)
    t = t_hist[used_slice]
    x_hist = x_hist[used_slice]
    x_d_hist = x_d_hist[used_slice]
    u_hist = u_hist[used_slice]
    u_d_hist = u_d_hist[used_slice]
    energy_hist = energy_hist[used_slice]
    slack_hist = slack_hist[used_slice]
    a_hat_hist = a_hat_hist[used_slice]
    a_k_hist = a_k_hist[used_slice]
    rho_hist = rho_hist[used_slice]
    nu_hist = nu_hist[used_slice]
    quantile_hist = quantile_hist[used_slice]
    theta_hist = theta_hist[used_slice]
    true_uncertainty_hist = true_uncertainty_hist[used_slice]
    fitted_uncertainty_hist = fitted_uncertainty_hist[used_slice]

    tracking_error = np.linalg.norm(x_hist - x_d_hist, axis=1)
    metrics = {
        "final_error": float(tracking_error[-1]),
        "rms_error": float(np.sqrt(np.mean(tracking_error ** 2))),
        "max_error": float(np.max(tracking_error)),
        "max_control": float(np.nanmax(np.abs(u_hist))),
        "max_slack": float(np.nanmax(slack_hist)),
        "a_hat_change": float(np.linalg.norm(a_hat_hist[-1] - a_hat_hist[0])),
        "rho_range": float(np.ptp(rho_hist)),
    }
    result = {
        "label": run_label,
        "use_cp": bool(use_cp),
        "use_adaptive": bool(use_adaptive),
        "t": t,
        "x_hist": x_hist,
        "x_d_hist": x_d_hist,
        "tracking_error": tracking_error,
        "u_hist": u_hist,
        "u_d_hist": u_d_hist,
        "energy_hist": energy_hist,
        "slack_hist": slack_hist,
        "a_hat_hist": a_hat_hist,
        "a_k_hist": a_k_hist,
        "rho_hist": rho_hist,
        "nu_hist": nu_hist,
        "quantile_hist": quantile_hist,
        "theta_hist": theta_hist,
        "true_uncertainty_hist": true_uncertainty_hist,
        "fitted_uncertainty_hist": fitted_uncertainty_hist,
        "interval_times": np.asarray(interval_times),
        "score_hist": np.asarray(score_hist),
        "delta_hist": np.asarray(delta_hist),
        "miscoverage_hist": np.asarray(miscoverage_hist),
        "uncertainty_schedule_values": schedule_values,
        "status": status,
        "failure_reason": failure_reason,
        "failure_time": failure_time,
        "failure_rho": failure_rho,
        "rho_divergence_threshold": rho_divergence_threshold,
        "metrics": metrics,
        "final_theta": online_olacp.Theta.copy(),
    }

    failure_summary = ""
    if failure_reason is not None:
        failure_summary = (
            f", failure_reason={failure_reason}, failure_time={failure_time:.6f}, "
            f"failure_rho={failure_rho:.3e}"
        )
    
    print(
        f"{run_label}: status={status}{failure_summary}, "
        f"final_error={metrics['final_error']:.3e}, "
        f"rms_error={metrics['rms_error']:.3e}, max_error={metrics['max_error']:.3e}, "
        f"max|u|={metrics['max_control']:.3e}, max_slack={metrics['max_slack']:.3e}"
    )
    return result


def plot_craccm_results(results, desired_trajectory):
    """Plot and compare one or more main CRaCCM simulation results"""
    items = (
        list(results.items())
        if isinstance(results, dict)
        else [(result["label"], result) for result in results]
    )
    if not items:
        raise ValueError("At least one CRaCCM result is required")

    reference = max((result for _, result in items), key=lambda result: len(result["t"]))
    for label, result in items:
        if result is reference:
            continue
        if not np.array_equal(result["x_hist"][0], reference["x_hist"][0]):
            raise RuntimeError(f"{label}: comparison initial state does not match")
        if not np.array_equal(
            result["uncertainty_schedule_values"],
            reference["uncertainty_schedule_values"],
        ):
            raise RuntimeError(f"{label}: comparison uncertainty schedule does not match")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {label: colors[index % len(colors)] for index, (label, _) in enumerate(items)}
    maximum_time = max(float(result["t"][-1]) for _, result in items)
    figures = []

    state_labels = (r"$x_1$", r"$x_2$", r"$x_3$")
    fig, axs = plt.subplots(NONLINEAR_TOY.xdim, 1, sharex=True, figsize=(8, 7))
    figures.append(fig)
    for state_index, ax in enumerate(axs):
        ax.plot(
            desired_trajectory["t_x"],
            desired_trajectory["x_d"][state_index],
            "k--",
            label="desired" if state_index == 0 else None,
        )
        for label, result in items:
            ax.plot(
                result["t"],
                result["x_hist"][:, state_index],
                color=color_map[label],
                label=label,
            )
            if result["status"] != "completed":
                ax.plot(
                    result["t"][-1],
                    result["x_hist"][-1, state_index],
                    "x",
                    color=color_map[label],
                    markersize=9,
                )
        ax.set_ylabel(state_labels[state_index])
        ax.set_xlim(0.0, maximum_time)
        ax.grid(True)
    axs[0].legend()
    axs[-1].set_xlabel("time (s)")
    fig.suptitle("Nonlinear-toy state tracking comparison")

    for label, result in items:
        color = color_map[label]
        result_end_time = float(result["t"][-1])
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
        figures.append(fig)
        axs[0].plot(result["t"], result["u_hist"], color=color, label=r"$u$")
        axs[0].plot(
            result["t"],
            result["u_d_hist"],
            "k--",
            label=r"$u_d$",
        )
        axs[1].semilogy(
            result["t"],
            np.maximum(result["slack_hist"], 1e-12),
            color=color,
            label="QP slack",
        )
        axs[0].set_ylabel("input")
        axs[1].set_ylabel("QP slack")
        axs[1].set_xlabel("time (s)")
        for ax in axs:
            ax.set_xlim(0.0, result_end_time)
            ax.grid(True)
            ax.legend()
        fig.suptitle(f"{label}: control and QP slack")

    for label, result in items:
        color = color_map[label]
        result_end_time = float(result["t"][-1])
        fig, axs = plt.subplots(
            NONLINEAR_TOY.adim + 2, 1, sharex=True, figsize=(8, 9)
        )
        figures.append(fig)
        for parameter_index, ax in enumerate(axs[: NONLINEAR_TOY.adim]):
            ax.plot(
                result["t"],
                result["a_hat_hist"][:, parameter_index],
                color=color,
                label=rf"$\hat a_{parameter_index + 1}$",
            )
            ax.plot(
                result["t"],
                result["a_k_hist"][:, parameter_index],
                "--",
                color=color,
                label=rf"$a_{{k,{parameter_index + 1}}}$",
            )
            ax.set_ylabel(rf"$a_{parameter_index + 1}$")
            ax.set_xlim(0.0, result_end_time)
            ax.grid(True)
            ax.legend()
        axs[-2].plot(result["t"], result["rho_hist"], color=color, label=r"$\rho$")
        axs[-1].plot(
            result["t"], result["nu_hist"], color=color, label=r"$\nu(\rho)$"
        )
        axs[-2].set_ylabel(r"$\rho$")
        axs[-1].set_ylabel(r"$\nu$")
        for ax in axs[-2:]:
            ax.set_xlim(0.0, result_end_time)
            ax.grid(True)
            ax.legend()
        axs[-1].set_xlabel("time (s)")
        fig.suptitle(f"{label}: CRaCCM adaptation and OLACP identification")

    for label, result in items:
        color = color_map[label]
        result_end_time = float(result["t"][-1])
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
        figures.append(fig)
        axs[0].step(
            result["interval_times"],
            result["score_hist"],
            where="post",
            color=color,
            label=r"$s_k$",
        )
        axs[0].step(
            result["t"],
            result["quantile_hist"],
            where="post",
            color=color,
            linestyle=":",
            label=r"$Q_k$",
        )
        axs[1].step(
            result["interval_times"],
            result["delta_hist"],
            where="post",
            color=color,
            label=r"$\delta_k$",
        )
        axs[2].step(
            result["interval_times"],
            result["miscoverage_hist"],
            where="post",
            color=color,
            label="miscoverage",
        )
        axs[0].set_ylabel("score")
        axs[1].set_ylabel(r"$\delta_k$")
        axs[2].set_ylabel("miscoverage")
        axs[2].set_xlabel("time (s)")
        for ax in axs:
            ax.set_xlim(0.0, result_end_time)
            ax.grid(True)
            ax.legend()
        fig.suptitle(f"{label}: Algorithm 1")

    coefficient_labels = (
        r"$c_{w_1,x_1}$",
        r"$c_{w_3,x_3}$",
        r"$c_{w_3,x_1^2}$",
    )
    for label, result in items:
        schedule_values = np.asarray(result["uncertainty_schedule_values"], dtype=float)
        if schedule_values.ndim != 2 or schedule_values.shape[0] != 6:
            raise ValueError(f"{label}: expected a 6-by-K uncertainty schedule")
        number_of_intervals = schedule_values.shape[1]
        schedule_times = np.linspace(
            0.0, float(desired_trajectory["t_x"][-1]), number_of_intervals + 1
        )
        coefficient_values = np.vstack(
            (
                0.8 * schedule_values[0] + 1.3 * schedule_values[1],
                0.23 * schedule_values[2] + 0.87 * schedule_values[3],
                0.23 * schedule_values[4] + 0.87 * schedule_values[5],
            )
        )
        coefficient_values = np.column_stack(
            (coefficient_values, coefficient_values[:, -1])
        )

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
        figures.append(fig)
        for coefficient_index, ax in enumerate(axs):
            ax.step(
                schedule_times,
                coefficient_values[coefficient_index],
                where="post",
                color=color_map[label],
                label=coefficient_labels[coefficient_index],
            )
            ax.set_ylabel(coefficient_labels[coefficient_index])
            ax.set_xlim(0.0, float(result["t"][-1]))
            ax.grid(True)
            ax.legend()
        axs[-1].set_xlabel("time (s)")
        fig.suptitle(f"{label}: true-uncertainty coefficient schedule")

    components = ((0, r"$w_1$"), (2, r"$w_3$"))
    for label, result in items:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
        figures.append(fig)
        for ax, (component, component_label) in zip(axs, components):
            ax.plot(
                result["t"],
                result["true_uncertainty_hist"][:, component],
                label="true uncertainty",
            )
            ax.plot(
                result["t"],
                result["fitted_uncertainty_hist"][:, component],
                "--",
                label=r"$Y_\Theta(x)a_k$",
            )
            ax.set_xlim(0.0, float(result["t"][-1]))
            ax.set_ylabel(component_label)
            ax.grid(True)
            ax.legend()
        axs[-1].set_xlabel("time (s)")
        fig.suptitle(rf"{label}: $Y_\Theta(x)a_k$ versus true uncertainty")

    plt.tight_layout()
    plt.show()
    
    return figures


def main():
    """Pretrain Algorithm 1, plan once, and compare three controller settings"""
    K_pre = 100
    N_cal = 80
    K = 10
    B = 5
    dt = 0.01
    interval_duration = 2.0
    I_length = int(round(interval_duration / dt))
    if K_pre < N_cal or K_pre % B != 0:
        raise ValueError("K_pre must fill N_cal and contain complete representation blocks")
    if I_length < 10 or not np.isclose(I_length * dt, interval_duration):
        raise ValueError("interval_duration must be an integer multiple of dt")

    theta_lb = -1.25 * np.ones(NONLINEAR_TOY.theta_shape)
    theta_ub = 1.25 * np.ones(NONLINEAR_TOY.theta_shape)
    theta_rng = np.random.default_rng(42)
    theta_init = theta_rng.uniform(theta_lb, theta_ub, size=NONLINEAR_TOY.theta_shape)

    a_lb = -2.0 * np.ones(NONLINEAR_TOY.adim)
    a_ub = 2.0 * np.ones(NONLINEAR_TOY.adim)
    projection_epsilon = 0.01
    a_hat_norm_max = 0.5 * np.linalg.norm(a_ub - a_lb) + projection_epsilon

    config = {
        "K_pre": K_pre,
        "N_cal": N_cal,
        "K": K,
        "B": B,
        "dt": dt,
        "interval_duration": interval_duration,
        "I_length": I_length,
        "pretrain_excitation_amplitude": 0.75,
        "pretrain_excitation_frequency": 0.12,
        "pretrain_initial_state": np.array([0.5, -0.3, 0.0]),
        "nominal_initial_state": np.array([0.01, 5.0, -0.05]),
        "nominal_goal_state": np.array([-0.05, 0.0, 0.01]),
        "motion_planner_Q": np.diag([100.0, 1.0, 100.0]),
        "motion_planner_R": 1e-3 * np.eye(NONLINEAR_TOY.udim),
        "motion_planner_Q_f": np.diag([1000.0, 10.0, 1000.0]),
        "tracking_initial_offset": np.array([0.3, -0.2, 0.4]) * 0.0,
        "x_norm_divergence_threshold": 10.0,
        "rho_divergence_threshold": 1e10,
        "geodesic_degree": 2,
        "geodesic_nodes": 8,
        "use_qpsolvers": True,
        "verify_geodesic": False,
    }
    base_params = {
        "Theta_init": theta_init.copy(),
        "Gamma_ccm": 0.5 * np.eye(NONLINEAR_TOY.adim),
        "a_ub": a_ub,
        "a_lb": a_lb,
        "a_hat_norm_max": a_hat_norm_max,
        "epsilon": projection_epsilon,
        "eta_ccm": 50.0,
        "ccm_rate": 0.1,
        "weight_slack": 1e5,
        "u_min": -20.0,
        "u_max": 20.0,
        "dt": dt,
    }

    pretrain_params = dict(base_params)
    pretrain_params.update({"use_cp": False, "use_adaptive": False})
    pretrain_system = NONLINEAR_TOY(pretrain_params)

    def representation_rate(update_index):
        return 5e-3 / np.sqrt(update_index)

    olacp = OLACP(
        [],
        N_cal=N_cal,
        acp_lr=0.1,
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
    pretrained_olacp, pretraining_history = run_pretraining(
        pretrain_system, olacp, config, plot=True
    )
    desired_trajectory = plan_nominal_trajectory(pretrain_system, config)

    pretrained_theta = pretrained_olacp.Theta.copy()
    pretrained_a_k = pretrained_olacp.a_k.copy()
    pretrained_scores = np.asarray(pretrained_olacp.S_cal).copy()
    pretrained_delta = float(pretrained_olacp.delta)

    settings = (
        ("CP + adaptive", True, True),
        ("No CP + adaptive", False, True),
        ("No CP + nonadaptive", False, False),
    )
    results = {}
    for label, use_cp, use_adaptive in settings:
        run_params = dict(base_params)
        run_params["Theta_init"] = pretrained_theta.copy()
        run_params["use_cp"] = use_cp
        run_params["use_adaptive"] = use_adaptive
        run_system = NONLINEAR_TOY(run_params)
        run_olacp = copy.deepcopy(pretrained_olacp)
        results[label] = run_craccm_simulation(
            run_system,
            run_olacp,
            desired_trajectory,
            config,
            use_cp,
            use_adaptive,
            label=label,
        )

    if not np.array_equal(pretrained_olacp.Theta, pretrained_theta):
        raise RuntimeError("A main simulation mutated the pretrained Theta")
    if not np.array_equal(pretrained_olacp.a_k, pretrained_a_k):
        raise RuntimeError("A main simulation mutated the pretrained a_k")
    if not np.array_equal(np.asarray(pretrained_olacp.S_cal), pretrained_scores):
        raise RuntimeError("A main simulation mutated the pretrained calibration set")
    if pretrained_olacp.delta != pretrained_delta:
        raise RuntimeError("A main simulation mutated the pretrained adaptive delta")

    # Save results to a file
    timestamp = time.strftime("%Y_%m%d_%H%M")
    results_filename = f"./simulations/nonlinear_toy/sim_toy_craccm_results_{timestamp}.pkl"
    with open(results_filename, "wb") as f:
        pickle.dump(
            {
                "config": config,
                "base_params": base_params,
                "desired_trajectory": desired_trajectory,
                "results": results,
            },
            f,
        )

    plot_craccm_results(results, desired_trajectory)

    return pretraining_history, desired_trajectory, results


if __name__ == "__main__":
    main()
