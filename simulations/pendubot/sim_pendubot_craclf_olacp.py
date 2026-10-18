"""Algorithm 1 and CRaCLF for the Pendubot"""

import copy
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from olacp import OLACP
from systems.pendubot.pendubot import Pendubot


def run_pretraining(system, olacp, config, plot=True):
    """Pre-train the shared representation and fill the calibration dataset using OLACP"""
    K_pre = config["K_pre"]
    I_length = config["I_length"]
    dt = config["dt"]
    interval_duration = config["interval_duration"]
    t_pre = np.arange(K_pre * I_length, dtype=float) * dt

    wind_indices = np.arange(K_pre)
    horizontal_wind_schedule = 0.4 * np.sin(2.0 * np.pi * wind_indices / 5.0)
    vertical_wind_schedule = -8.0
    vertical_wind_schedule += 1.5 * np.sin(2.0 * np.pi * wind_indices / 7.0)
    vertical_wind_schedule += 0.5 * np.sin(2.0 * np.pi * wind_indices / 3.0)
    if np.ptp(vertical_wind_schedule) <= 0.0 or np.any(vertical_wind_schedule >= 0.0):
        raise ValueError("Pretraining requires a time-varying downward vertical wind")

    def schedule_index(t):
        return min(int(np.floor(max(float(t), 0.0) / interval_duration)), K_pre - 1)

    def wind_velocity(t):
        index = schedule_index(t)
        return np.array([horizontal_wind_schedule[index], vertical_wind_schedule[index]])

    system.wind_velocity_fcn = wind_velocity
    system.set_representation(olacp.Theta)
    x = config["main_initial_state"].copy()
    x_hist = np.zeros((len(t_pre), system.xdim))
    u_hist = np.zeros(len(t_pre))
    wind_hist = np.zeros((len(t_pre), 2))
    a_k_hist = np.full((len(t_pre), system.adim), np.nan)
    theta_hist = np.zeros((K_pre + 1,) + olacp.Theta.shape)
    theta_hist[0] = olacp.Theta
    score_hist = np.zeros(K_pre)
    prediction_error_hist = np.full(len(t_pre), np.nan)
    true_uncertainty_hist = np.full((len(t_pre), system.xdim), np.nan)
    fitted_uncertainty_hist = np.full((len(t_pre), system.xdim), np.nan)
    a_hat_0 = system.a_center.copy()

    for interval_index in range(K_pre):
        for interval_sample_index in range(I_length):
            sample_index = interval_index * I_length + interval_sample_index
            t = t_pre[sample_index]
            excitation = config["pretrain_excitation_amplitude"] * np.sin(
                2.0 * np.pi * config["pretrain_excitation_frequency"] * t
            ) #+ config["pretrain_excitation_amplitude"] / 2.0 * np.sin(
            #    2.0 * np.pi * config["pretrain_excitation_frequency"] * 2 * t
            #)
            u = system.local_lqr_control(x, a_hat_0) + excitation

            x_hist[sample_index] = x
            u_hist[sample_index] = float(u.item())
            wind_hist[sample_index] = wind_velocity(t)
            xdot = system.dynamics(x, u, t)
            olacp.add_data_to_buffers(x, system.dynamics_nominal(x, u), xdot=xdot)

            if sample_index < len(t_pre) - 1:
                solution = solve_ivp(
                    lambda tau, state: system.dynamics(state, u, tau),
                    (t, t + dt),
                    x,
                    method="RK45", #"LSODA", #
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
        raise RuntimeError("The trained representation was not installed in the Pendubot")
    if np.max(np.linalg.norm(x_hist, axis=1)) > config["pretrain_state_norm_max"]:
        raise RuntimeError("Pendubot pretraining left the prescribed local state region")

    quantile = olacp.compute_quantile()
    history = {
        "t": t_pre,
        "x_hist": x_hist,
        "u_hist": u_hist,
        "wind_hist": wind_hist,
        "a_k_hist": a_k_hist,
        "theta_hist": theta_hist,
        "score_hist": score_hist,
        "prediction_error_hist": prediction_error_hist,
        "true_uncertainty_hist": true_uncertainty_hist,
        "fitted_uncertainty_hist": fitted_uncertainty_hist,
        "horizontal_wind_schedule": horizontal_wind_schedule,
        "vertical_wind_schedule": vertical_wind_schedule,
        "quantile": quantile,
    }

    theta_change = np.linalg.norm(theta_hist[-1] - theta_hist[0])
    print(
        f"Pretraining complete: Q_0={quantile:.3e}, a_last={olacp.a_k}, "
        f"score_first={score_hist[0]:.3e}, score_last={score_hist[-1]:.3e}, "
        f"theta_change={theta_change:.3e}, "
        f"max_state_norm={np.max(np.linalg.norm(x_hist, axis=1)):.3e}, "
        f"max|u|={np.max(np.abs(u_hist)):.3e}"
    )

    if plot:
        state_labels = (
            r"$q_1$ (deg)",
            r"$q_2$ (deg)",
            r"$\dot q_1$ (deg/s)",
            r"$\dot q_2$ (deg/s)",
        )
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
        for state_index, ax in enumerate(axs):
            ax.plot(t_pre, np.rad2deg(x_hist[:, state_index]))
            ax.set_ylabel(state_labels[state_index])
            ax.grid(True)
        axs[-1].set_xlabel("time (s)")
        fig.suptitle("Pretraining: states")

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
        axs[0].plot(t_pre, u_hist)
        axs[0].set_ylabel("torque (N m)")
        axs[1].plot(t_pre, wind_hist[:, 0], label=r"$v_{w,x}$")
        axs[1].plot(t_pre, wind_hist[:, 1], label=r"$v_{w,z}$")
        axs[1].set_ylabel("wind (m/s)")
        axs[1].legend()
        axs[1].set_xlabel("time (s)")
        for ax in axs:
            ax.grid(True)
        fig.suptitle("Pretraining: torque and wind")

        components = ((2, r"$w_3$"), (3, r"$w_4$"))
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
        for ax, (component, component_label) in zip(axs, components):
            ax.plot(t_pre, true_uncertainty_hist[:, component], label="true uncertainty")
            ax.plot(
                t_pre,
                fitted_uncertainty_hist[:, component],
                "--",
                label=r"$Y_\Theta(x)a_k$",
            )
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


def run_craclf_simulation(
    system, online_olacp, config, use_cp, use_adaptive, plot=False, label=None
):
    """Run one main CRaCLF experiment"""
    if bool(system.use_cp) != bool(use_cp) or bool(system.use_adaptive) != bool(use_adaptive):
        raise ValueError(
            "The Pendubot object must be constructed with the requested controller flags"
        )

    K = config["K"]
    I_length = config["I_length"]
    dt = config["dt"]
    interval_duration = config["interval_duration"]
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

    # Wind velocities (m/s) over the six 2-second simulation intervals.
    # End horizontal gusts before the tail to preserve the upright equilibrium with u_ref=0.
    horizontal_wind_schedule = np.array([0.0, 1.0, -1.5, 1.0, 0.0, 0.0])
    vertical_wind_schedule = np.array(
        [0.0, -5.27114277397773, -17.1152930753884, -26.6135641506339, -6.0, -6.0]
    )

    def schedule_index(t):
        return min(int(np.floor(max(float(t), 0.0) / interval_duration)), K - 1)

    def wind_velocity(t):
        index = schedule_index(t)
        return np.array([horizontal_wind_schedule[index], vertical_wind_schedule[index]])

    online_olacp.clear_buffers()
    if online_olacp._representation_intervals:
        raise RuntimeError("The pretrained OLACP contains an incomplete representation block")
    online_olacp.Y_Theta = system.Y_Theta
    online_olacp.representation_loss_gradient = system.representation_loss_gradient
    system.wind_velocity_fcn = wind_velocity
    system.set_representation(online_olacp.Theta)
    system.cp_quantile = online_olacp.compute_quantile()
    if np.shares_memory(system.Theta_hat, online_olacp.Theta):
        raise RuntimeError("The Pendubot and OLACP representation arrays must be independent")

    x = config["main_initial_state"].copy()
    initial_state = x.copy()
    a_hat = system.a_center.copy()
    rho = 0.0
    x_ext = np.hstack((x, a_hat, rho))
    x_hist = np.zeros((len(t_full), system.xdim))
    u_hist = np.full(len(t_full), np.nan)
    u_ref_hist = np.zeros(len(t_full))
    slack_hist = np.full(len(t_full), np.nan)
    V_hist = np.full(len(t_full), np.nan)
    a_hat_hist = np.zeros((len(t_full), system.adim))
    a_k_hist = np.full((len(t_full), system.adim), np.nan)
    rho_hist = np.zeros(len(t_full))
    nu_hist = np.zeros(len(t_full))
    quantile_hist = np.zeros(len(t_full))
    theta_hist = np.zeros((len(t_full),) + online_olacp.Theta.shape)
    prediction_error_hist = np.full(len(t_full), np.nan)
    true_uncertainty_hist = np.full((len(t_full), system.xdim), np.nan)
    fitted_uncertainty_hist = np.full((len(t_full), system.xdim), np.nan)
    wind_hist = np.zeros((len(t_full), 2))

    def record_terminal_sample(index, sample_time, state, held_u, held_u_ref, held_slack):
        terminal_x = state[: system.xdim]
        terminal_a_hat = state[system.xdim : system.xdim + system.adim]
        terminal_rho = float(state[-1])
        t_hist[index] = sample_time
        x_hist[index] = terminal_x
        u_hist[index] = float(held_u.item())
        u_ref_hist[index] = float(held_u_ref.item())
        slack_hist[index] = float(held_slack)
        a_hat_hist[index] = terminal_a_hat
        rho_hist[index] = terminal_rho
        nu_hist[index] = system.nu_clf(terminal_rho)
        quantile_hist[index] = system.cp_quantile
        theta_hist[index] = system.Theta_hat
        wind_hist[index] = wind_velocity(sample_time)
        if np.all(np.isfinite(terminal_x)) and np.all(np.isfinite(terminal_a_hat)):
            V_hist[index] = float(system.clf(terminal_x, terminal_a_hat))
        return terminal_x, terminal_a_hat, terminal_rho

    interval_times = []
    score_hist = []
    delta_hist = []
    miscoverage_hist = []
    status = "completed"
    failure_reason = None
    failure_time = None
    failure_rho = None
    last_sample_index = -1

    for sample_index, t in enumerate(t_full):
        last_sample_index = sample_index
        x_hist[sample_index] = x
        a_hat_hist[sample_index] = a_hat
        rho_hist[sample_index] = rho
        nu_hist[sample_index] = system.nu_clf(rho)
        quantile_hist[sample_index] = system.cp_quantile
        theta_hist[sample_index] = system.Theta_hat
        wind_hist[sample_index] = wind_velocity(t)
        V_hist[sample_index] = float(system.clf(x, a_hat))

        u_ref = np.zeros(system.udim)
        u, slack = system.ctrl_craclf(x, a_hat, u_ref, use_slack=True)
        u_ref_hist[sample_index] = float(u_ref.item())
        u_hist[sample_index] = float(u.item())
        slack_hist[sample_index] = float(slack)
        xdot = system.dynamics(x, u, t)
        online_olacp.add_data_to_buffers(x, system.dynamics_nominal(x, u), xdot=xdot)

        if sample_index < len(t_full) - 1:
            solution = solve_ivp(
                lambda tau, state: system.dynamics_extended(state, u, tau),
                (t_full[sample_index], t_full[sample_index + 1]),
                x_ext,
                method="RK45",#"LSODA", #
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
                    event_index, failure_time, x_ext, u, u_ref, slack
                )
                break
            if not solution.success:
                raise RuntimeError(
                    f"{run_label}: extended dynamics failed at t={t:.3f}, x={x}, "
                    f"a_hat={a_hat}, rho={rho:.3e}: {solution.message}"
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
                    terminal_index, failure_time, x_ext, u, u_ref, slack
                )
            break

        if (sample_index + 1) % I_length == 0:
            online_olacp.estimate_uncertainty(dt)
            score = float(online_olacp.compute_score(system.a_ub, system.a_lb))
            interval_true = np.asarray(online_olacp._w_buffer, dtype=float)
            interval_fitted = np.asarray([Y_t @ online_olacp.a_k for Y_t in online_olacp._Y_buffer])
            interval_error = np.sum((interval_fitted - interval_true) ** 2, axis=1)
            miscoverage = online_olacp.update_delta(score)
            online_olacp.append_score(score)
            representation_update = online_olacp.update_representation()
            interval_start = sample_index - I_length + 1
            interval_slice = slice(interval_start, sample_index + 1)
            a_k_hist[interval_slice] = online_olacp.a_k
            prediction_error_hist[interval_slice] = interval_error
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
    u_hist = u_hist[used_slice]
    u_ref_hist = u_ref_hist[used_slice]
    slack_hist = slack_hist[used_slice]
    V_hist = V_hist[used_slice]
    a_hat_hist = a_hat_hist[used_slice]
    a_k_hist = a_k_hist[used_slice]
    rho_hist = rho_hist[used_slice]
    nu_hist = nu_hist[used_slice]
    quantile_hist = quantile_hist[used_slice]
    theta_hist = theta_hist[used_slice]
    prediction_error_hist = prediction_error_hist[used_slice]
    true_uncertainty_hist = true_uncertainty_hist[used_slice]
    fitted_uncertainty_hist = fitted_uncertainty_hist[used_slice]
    wind_hist = wind_hist[used_slice]

    state_norm = np.linalg.norm(x_hist, axis=1)
    tail_start = max(int(0.8 * len(state_norm)), 0)
    tail_norm = state_norm[tail_start:]
    if len(a_hat_hist) > 1:
        a_hat_rate = np.diff(a_hat_hist, axis=0) / np.diff(t)[:, None]
        max_a_hat_dot = float(np.max(np.linalg.norm(a_hat_rate, axis=1)))
    else:
        max_a_hat_dot = 0.0
    metrics = {
        "final_norm": float(state_norm[-1]),
        "tail_rms": float(np.sqrt(np.mean(tail_norm**2))),
        "tail_peak": float(np.max(tail_norm)),
        "max_norm": float(np.max(state_norm)),
        "V_ratio": float(V_hist[-1] / max(V_hist[0], np.finfo(float).eps)),
        "max_control": float(np.nanmax(np.abs(u_hist))),
        "max_slack": float(np.nanmax(slack_hist)),
        "a_hat_change": float(np.linalg.norm(a_hat_hist[-1] - a_hat_hist[0])),
        "a_hat_path_length": float(np.sum(np.linalg.norm(np.diff(a_hat_hist, axis=0), axis=1))),
        "max_a_hat_dot": max_a_hat_dot,
        "rho_change": float(rho_hist[-1] - rho_hist[0]),
        "rho_range": float(np.ptp(rho_hist)),
        "nu_range": float(np.ptp(nu_hist)),
    }
    result = {
        "label": run_label,
        "use_cp": bool(use_cp),
        "use_adaptive": bool(use_adaptive),
        "initial_state": initial_state,
        "t": t,
        "x_hist": x_hist,
        "u_hist": u_hist,
        "u_ref_hist": u_ref_hist,
        "slack_hist": slack_hist,
        "V_hist": V_hist,
        "a_hat_hist": a_hat_hist,
        "a_k_hist": a_k_hist,
        "rho_hist": rho_hist,
        "nu_hist": nu_hist,
        "quantile_hist": quantile_hist,
        "theta_hist": theta_hist,
        "prediction_error_hist": prediction_error_hist,
        "true_uncertainty_hist": true_uncertainty_hist,
        "fitted_uncertainty_hist": fitted_uncertainty_hist,
        "wind_hist": wind_hist,
        "interval_times": np.asarray(interval_times),
        "score_hist": np.asarray(score_hist),
        "delta_hist": np.asarray(delta_hist),
        "miscoverage_hist": np.asarray(miscoverage_hist),
        "horizontal_wind_schedule": horizontal_wind_schedule,
        "vertical_wind_schedule": vertical_wind_schedule,
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
        f"{run_label}: status={status}{failure_summary}, t_final={t[-1]:.3f}, "
        f"final_norm={metrics['final_norm']:.3e}, tail_rms={metrics['tail_rms']:.3e}, "
        f"max_norm={metrics['max_norm']:.3e}, V_final/V_initial={metrics['V_ratio']:.3e}, "
        f"max|u|={metrics['max_control']:.2f}, max_slack={metrics['max_slack']:.3e}, "
        f"Delta_a={metrics['a_hat_change']:.3e}, a_path={metrics['a_hat_path_length']:.3e}, "
        f"Delta_rho={metrics['rho_change']:.3e}, Delta_nu={metrics['nu_range']:.3e}"
    )

    if plot:
        plot_craclf_results({run_label: result})
    return result


def plot_craclf_results(results, file_format="png"):
    """Compare results and plot the longest recorded wind history.

    Save figures in this example's results folder as PNG (default) or PDF.
    """
    if file_format not in ("png", "pdf"):
        raise ValueError('file_format must be "png" or "pdf"')
    items = (
        list(results.items())
        if isinstance(results, dict)
        else [(result["label"], result) for result in results]
    )
    if not items:
        raise ValueError("At least one CRaCLF result is required")

    reference = max((result for _, result in items), key=lambda result: len(result["t"]))
    for label, result in items:
        if result is reference:
            continue
        if not np.array_equal(result["initial_state"], reference["initial_state"]):
            raise RuntimeError(f"{label}: comparison initial state does not match")
        for schedule_name in ("horizontal_wind_schedule", "vertical_wind_schedule"):
            if not np.array_equal(result[schedule_name], reference[schedule_name]):
                raise RuntimeError(f"{label}: comparison {schedule_name} does not match")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {label: colors[index % len(colors)] for index, (label, _) in enumerate(items)}
    maximum_time = max(float(result["t"][-1]) for _, result in items)
    figures = []

    state_labels = (
        r"$q_1$ (deg)",
        r"$q_2$ (deg)",
        r"$\dot q_1$ (deg/s)",
        r"$\dot q_2$ (deg/s)",
    )
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 7))
    figures.append(fig)
    for state_index, ax in enumerate(axs):
        for label, result in items:
            ax.plot(
                result["t"],
                np.rad2deg(result["x_hist"][:, state_index]),
                color=color_map[label],
                label=label,
            )
            if result["status"] != "completed":
                ax.plot(
                    result["t"][-1],
                    np.rad2deg(result["x_hist"][-1, state_index]),
                    "x",
                    color=color_map[label],
                    markersize=9,
                )
        ax.set_ylabel(state_labels[state_index])
        ax.set_xlim(0.0, maximum_time)
        ax.grid(True)
    axs[0].legend()
    axs[-1].set_xlabel("time (s)")
    fig.suptitle("Pendubot state comparison")
    fig.set_label("pendubot_states")

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    figures.append(fig)
    for label, result in items:
        color = color_map[label]
        axs[0].plot(result["t"], result["u_hist"], color=color, label=label)
        axs[1].semilogy(
            result["t"],
            np.maximum(result["slack_hist"], 1e-12),
            color=color,
            label=label,
        )
    axs[0].set_ylabel("torque (N m)")
    axs[1].set_ylabel("QP slack")
    axs[1].set_xlabel("time (s)")
    for ax in axs:
        ax.set_xlim(0.0, maximum_time)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Pendubot control and QP slack comparison")
    fig.set_label("pendubot_control_slack")

    for label, result in items:
        color = color_map[label]
        result_end_time = float(result["t"][-1])
        setting_name = "".join(c if c.isalnum() else " " for c in label.lower())
        setting_name = "_".join(setting_name.split())
        fig, axs = plt.subplots(Pendubot.adim + 2, 1, sharex=True, figsize=(8, 11))
        figures.append(fig)
        for parameter_index, ax in enumerate(axs[: Pendubot.adim]):
            ax.plot(
                result["t"],
                result["a_hat_hist"][:, parameter_index],
                color=color,
                label=r"$\hat a$",
            )
            ax.plot(
                result["t"],
                result["a_k_hist"][:, parameter_index],
                "--",
                color=color,
                label=r"$a_k$",
            )
            ax.set_ylabel(f"a{parameter_index + 1}")
            ax.set_xlim(0.0, result_end_time)
            ax.grid(True)
        axs[-2].plot(result["t"], result["rho_hist"], color=color, label=r"$\rho$")
        axs[-1].plot(result["t"], result["nu_hist"], color=color, label=r"$\nu(\rho)$")
        axs[-2].set_ylabel(r"$\rho$")
        axs[-1].set_ylabel(r"$\nu$")
        for ax in axs[-2:]:
            ax.set_xlim(0.0, result_end_time)
            ax.grid(True)
            ax.legend()
        axs[0].legend()
        axs[-1].set_xlabel("time (s)")
        fig.suptitle(f"{label}: CRaCLF adaptation and OLACP identification")
        fig.set_label(f"pendubot_adaptation_{setting_name}")

    for label, result in items:
        color = color_map[label]
        result_end_time = float(result["t"][-1])
        setting_name = "".join(c if c.isalnum() else " " for c in label.lower())
        setting_name = "_".join(setting_name.split())
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
        fig.set_label(f"pendubot_olacp_{setting_name}")

    fig, ax = plt.subplots(figsize=(8, 4))
    figures.append(fig)
    for component, wind_label in enumerate((r"$v_{w,x}$", r"$v_{w,z}$")):
        ax.step(reference["t"], reference["wind_hist"][:, component], where="post", label=wind_label)
    ax.set_xlim(0.0, maximum_time)
    ax.set_ylabel("wind velocity (m/s)")
    ax.set_xlabel("time (s)")
    ax.grid(True)
    ax.legend()
    fig.suptitle("Pendubot recorded wind history")
    fig.set_label("pendubot_wind")

    components = ((2, r"$w_3$"), (3, r"$w_4$"))
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    figures.append(fig)
    for ax, (component, component_label) in zip(axs, components):
        for label, result in items:
            ax.plot(
                result["t"],
                result["true_uncertainty_hist"][:, component],
                color=color_map[label],
                label=f"true: {label}",
            )
            ax.plot(
                result["t"],
                result["fitted_uncertainty_hist"][:, component],
                "--",
                color=color_map[label],
                label=rf"$Y_{{\Theta_j}}(x)a_k$: {label}",
            )
        ax.set_xlim(0.0, maximum_time)
        ax.set_ylabel(component_label)
        ax.grid(True)
        ax.legend(fontsize="small")
    axs[-1].set_xlabel("time (s)")
    fig.suptitle(r"Pendubot: $Y_{\Theta_j}(x)a_k$ versus true uncertainty")
    fig.set_label("pendubot_uncertainty_fit")

    plt.tight_layout()
    results_directory = Path(__file__).resolve().parent / "results"
    results_directory.mkdir(exist_ok=True)
    for figure in figures:
        filename = results_directory / f"{figure.get_label()}.{file_format}"
        figure.savefig(filename, format=file_format, dpi=300, bbox_inches="tight")
    print(f"Saved {len(figures)} {file_format.upper()} figures to {results_directory}")
    plt.show()
    
    return figures


def main(plot=False):
    """Run the three controller settings and save results, optionally displaying plots."""
    K_pre = 45
    N_cal = 30
    K = 6.0
    B = 5
    dt = 0.01
    interval_duration = 2.0
    I_length = int(round(interval_duration / dt))
    if K_pre < N_cal:
        raise ValueError("K_pre must be at least as large as N_cal")
    if K_pre % B != 0:
        raise ValueError("K_pre must be an integer multiple of B")
    if I_length < 10 or not np.isclose(I_length * dt, interval_duration):
        raise ValueError("interval_duration must be an integer multiple of dt")
    if interval_duration < 2.0:
        raise ValueError("interval_duration must be at least 2.0 seconds")

    m1 = 1.0
    m2 = 2.0
    L1 = 1.0
    L2 = 1.5
    nominal_gravity = 9.0
    true_gravity = 11.0
    
    theta_lb = -10.0 * np.ones(Pendubot.theta_shape)
    theta_ub = 10.0 * np.ones(Pendubot.theta_shape)
    theta_rng = np.random.default_rng(11)
    theta_init = theta_rng.uniform(theta_lb, theta_ub)

    a_lb = -1.5 * np.ones(Pendubot.adim)
    a_ub = 1.5 * np.ones(Pendubot.adim)
    projection_epsilon = 0.01
    a_hat_norm_max = 0.5 * np.linalg.norm(a_ub - a_lb, ord=2) + projection_epsilon
    # Normalize the CLF; the gain, eta and slack-weight formulas preserve the control law.
    clf_scale = 1.0
    effective_gamma_clf = 1e-4
    gamma_clf = (effective_gamma_clf / clf_scale) * np.eye(Pendubot.adim) / 5.0

    config = {
        "K_pre": K_pre,
        "N_cal": N_cal,
        "K": K,
        "B": B,
        "dt": dt,
        "interval_duration": interval_duration,
        "I_length": I_length,
        "pretrain_excitation_amplitude": 6.0,
        "pretrain_excitation_frequency": 0.1,
        "pretrain_state_norm_max": 5.0,
        "main_initial_state": np.array([0.18, -0.10, 0.0, 0.0]),
        "x_norm_divergence_threshold": 5.0,
        "rho_divergence_threshold": 1e6,
    }
    base_pendubot_params = {
        "m1": m1,
        "m2": m2,
        "L1": L1,
        "L2": L2,
        "r1": L1 / 2.0,
        "r2": L2 / 2.0,
        "I1": m1 * L1**2 / 12.0,
        "I2": m2 * L2**2 / 12.0,
        "grav": nominal_gravity,
        "true_grav": true_gravity,
        "damping": np.array([0.20, 0.20]),
        "true_damping": np.array([0.03, 0.03]),
        "L_w": 0.5,
        "c_w": 0.15,
        "Theta_init": theta_init.copy(),
        "lqr_Q": clf_scale * np.diag([20.0, 20.0, 2.0, 2.0]),
        "lqr_R": clf_scale * 0.1,
        "Gamma_clf": gamma_clf,
        "a_ub": a_ub,
        "a_lb": a_lb,
        "a_hat_norm_max": a_hat_norm_max,
        "epsilon": projection_epsilon,
        "eta_clf": clf_scale * 10.0,
        "clf_rate": 0.5,
        # Restore full tightening closer to upright to reduce the residual regulation error.
        "clf_goal_radius": 0.08,
        # Allow slack near low control authority rather than nearly enforcing a hard constraint.
        "weight_slack": 1e2 / clf_scale**2,
        "dt": dt,
    }

    pretrain_params = dict(base_pendubot_params)
    pretrain_params.update({"use_cp": False, "use_adaptive": False})
    pretrain_system = Pendubot(pretrain_params)
    def representation_rate(update_index):
        return 1e-2 / np.sqrt(update_index)

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
    pretrained_olacp, pretraining_history = run_pretraining(
        pretrain_system, olacp, config, plot=plot
    )
    pretrained_theta = pretrained_olacp.Theta.copy()
    pretrained_a_k = pretrained_olacp.a_k.copy()
    pretrained_scores = np.asarray(pretrained_olacp.S_cal).copy()
    pretrained_delta = float(pretrained_olacp.delta)
    pretrained_interval_index = int(pretrained_olacp.interval_index)
    pretrained_update_index = int(pretrained_olacp.representation_update_index)

    settings = (
        ("CP + adaptive", True, True),
        ("No CP + adaptive", False, True),
        ("No CP + nonadaptive", False, False),
    )
    results = {}
    for label, use_cp, use_adaptive in settings:
        run_params = dict(base_pendubot_params)
        run_params["Theta_init"] = pretrained_theta.copy()
        run_params["use_cp"] = use_cp
        run_params["use_adaptive"] = use_adaptive
        run_system = Pendubot(run_params)
        run_olacp = copy.deepcopy(pretrained_olacp)
        if np.shares_memory(run_olacp.Theta, pretrained_olacp.Theta):
            raise RuntimeError(f"{label}: online and pretrained OLACP objects share Theta memory")
        results[label] = run_craclf_simulation(
            run_system,
            run_olacp,
            config,
            use_cp,
            use_adaptive,
            plot=False,
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
    if pretrained_olacp.interval_index != pretrained_interval_index:
        raise RuntimeError("A main simulation mutated the pretrained interval index")
    if pretrained_olacp.representation_update_index != pretrained_update_index:
        raise RuntimeError("A main simulation mutated the pretrained representation-update index")

    timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S_%f")
    results_directory = Path(__file__).resolve().parent / "results"
    results_directory.mkdir(exist_ok=True)
    results_filename = results_directory / f"sim_pendubot_craclf_results_{timestamp}.pkl"
    with results_filename.open("xb") as stream:
        pickle.dump(
            {
                "config": config,
                "base_params": base_pendubot_params,
                "pretraining_history": pretraining_history,
                "results": results,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved results to {results_filename}")

    cp_adaptive_result = results["CP + adaptive"]
    cp_adaptive_metrics = cp_adaptive_result["metrics"]
    nonadaptive_result = results["No CP + nonadaptive"]
    nonadaptive_metrics = nonadaptive_result["metrics"]
    if cp_adaptive_result["status"] != "completed" or cp_adaptive_metrics["tail_rms"] >= 0.05:
        warnings.warn("The CP + adaptive run did not stabilize")
    # Keep the slack warning invariant to the CLF normalization.
    if cp_adaptive_metrics["max_slack"] / clf_scale >= 20.0:
        warnings.warn("The CP + adaptive QP slack / clf_scale exceeded 20")
    if cp_adaptive_metrics["a_hat_change"] <= 3e-2 or cp_adaptive_metrics["rho_range"] <= 0.5:
        warnings.warn("The CRaCLF adaptive states did not change materially")
    if cp_adaptive_metrics["nu_range"] <= 0.1:
        warnings.warn("rho did not materially change the adaptation scaling")
    nonadaptive_failed = (
        nonadaptive_result["status"] != "completed" or nonadaptive_metrics["tail_rms"] > 0.25
    )
    separated = nonadaptive_metrics["tail_rms"] > 5.0 * cp_adaptive_metrics["tail_rms"]
    if not nonadaptive_failed or not separated:
        warnings.warn("The no-CP nonadaptive run did not fail clearly")
    if nonadaptive_metrics["a_hat_change"] > 1e-12 or nonadaptive_metrics["rho_range"] > 1e-12:
        warnings.warn("The nonadaptive run unexpectedly changed its CRaCLF adaptive states")

    if plot:
        plot_craclf_results(results)

    return pretraining_history, results


if __name__ == "__main__":
    main()
