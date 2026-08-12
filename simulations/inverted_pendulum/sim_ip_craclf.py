"""Algorithm 1 and CRaCLF for the inverted pendulum"""

import copy
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from olacp import OLACP
from systems.inverted_pendulum.ip import IP


def run_pretraining(system, olacp, config, plot=True):
    """Pre-train the shared representation and fill the calibration dataset using OLACP"""
    K_pre = config["K_pre"]
    I_length = config["I_length"]
    dt = config["dt"]
    interval_duration = config["interval_duration"]
    t_pre = np.arange(K_pre * I_length, dtype=float) * dt

    wind_indices = np.arange(K_pre)
    vertical_wind_schedule = -3.0
    vertical_wind_schedule += 1.5 * np.sin(2.0 * np.pi * wind_indices / 7.0)
    vertical_wind_schedule += 0.4 * np.sin(2.0 * np.pi * wind_indices / 3.0)
    if np.ptp(vertical_wind_schedule) <= 0.0 or np.any(vertical_wind_schedule >= 0.0):
        raise ValueError("Pretraining requires a time-varying downward vertical wind")

    def schedule_index(t):
        interval_index = int(np.floor(max(float(t), 0.0) / interval_duration))
        return min(interval_index, K_pre - 1)

    def wind_velocity(t):
        return np.array([0.0, vertical_wind_schedule[schedule_index(t)]])

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
            )
            u = system.local_lqr_control(x, a_hat_0) + excitation
            u = np.clip(u, system.params["u_min"], system.params["u_max"])

            x_hist[sample_index] = x
            u_hist[sample_index] = float(u.item())
            wind_hist[sample_index] = wind_velocity(t)
            xdot = system.dynamics(x, u, t)
            nominal_xdot = system.dynamics_nominal(x, u)
            olacp.add_data_to_buffers(x, nominal_xdot, xdot=xdot)

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
        raise RuntimeError("The trained representation was not installed in the pendulum")
    if np.max(np.linalg.norm(x_hist, axis=1)) > config["pretrain_state_norm_max"]:
        raise RuntimeError("Pendulum pretraining left the prescribed local state region")
    if not np.all(wind_hist[:, 0] == 0.0):
        raise RuntimeError("Horizontal wind must remain zero during pretraining")

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
        state_labels = (r"$\phi$ (deg)", r"$\dot\phi$ (deg/s)")
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        for state_index, ax in enumerate(axs):
            ax.plot(t_pre, np.rad2deg(x_hist[:, state_index]))
            ax.set_ylabel(state_labels[state_index])
            ax.grid(True)
        axs[-1].set_xlabel("time (s)")
        fig.suptitle("Pretraining: states")

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
        axs[0].plot(t_pre, u_hist)
        axs[0].set_ylabel("torque (N m)")
        axs[1].plot(t_pre, wind_hist[:, 1])
        axs[1].set_ylabel(r"$w_z$ (m/s)")
        axs[1].set_xlabel("time (s)")
        for ax in axs:
            ax.grid(True)
        fig.suptitle("Pretraining: torque and vertical wind")

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(t_pre, true_uncertainty_hist[:, 1], label="true uncertainty")
        ax.plot(t_pre, fitted_uncertainty_hist[:, 1], "--", label=r"$Y_\Theta(x)a_k$")
        ax.set_ylabel(r"$w_2$")
        ax.set_xlabel("time (s)")
        ax.grid(True)
        ax.legend()
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
        raise ValueError("The pendulum must be constructed with the requested controller flags")

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

    wind_indices = np.arange(K)
    vertical_wind_schedule = -4.25
    vertical_wind_schedule -= 0.5 * np.cos(2.0 * np.pi * wind_indices / 3.0)

    def schedule_index(t):
        interval_index = int(np.floor(max(float(t), 0.0) / interval_duration))
        return min(interval_index, K - 1)

    def wind_velocity(t):
        return np.array([0.0, vertical_wind_schedule[schedule_index(t)]])

    online_olacp.clear_buffers()
    if online_olacp._representation_intervals:
        raise RuntimeError("The pretrained OLACP contains an incomplete representation block")
    online_olacp.Y_Theta = system.Y_Theta
    online_olacp.representation_loss_gradient = system.representation_loss_gradient
    system.wind_velocity_fcn = wind_velocity
    system.set_representation(online_olacp.Theta)
    system.cp_quantile = online_olacp.compute_quantile()
    if np.shares_memory(system.Theta_hat, online_olacp.Theta):
        raise RuntimeError("The pendulum and OLACP representation arrays must be independent")

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
        nominal_xdot = system.dynamics_nominal(x, u)
        online_olacp.add_data_to_buffers(x, nominal_xdot, xdot=xdot)

        if sample_index < len(t_full) - 1:
            solution = solve_ivp(
                lambda tau, state: system.dynamics_extended(state, u, tau),
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
            interval_fitted = np.asarray(
                [Y_t @ online_olacp.a_k for Y_t in online_olacp._Y_buffer]
            )
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

    if not np.all(wind_hist[:, 0] == 0.0):
        raise RuntimeError("Horizontal wind must remain zero during CRaCLF simulation")
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
        "a_hat_path_length": float(
            np.sum(np.linalg.norm(np.diff(a_hat_hist, axis=0), axis=1))
        ),
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
        f"Delta_a={metrics['a_hat_change']:.3e}, "
        f"a_path={metrics['a_hat_path_length']:.3e}, "
        f"Delta_rho={metrics['rho_change']:.3e}, Delta_nu={metrics['nu_range']:.3e}"
    )

    if plot:
        plot_craclf_results({run_label: result})
    return result


def plot_craclf_results(results):
    """Plot and compare one or more main CRaCLF simulation results."""
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
        if not np.array_equal(
            result["vertical_wind_schedule"], reference["vertical_wind_schedule"]
        ):
            raise RuntimeError(f"{label}: comparison wind schedule does not match")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        label: colors[index % len(colors)] for index, (label, _) in enumerate(items)
    }
    maximum_time = max(float(result["t"][-1]) for _, result in items)
    figures = []

    state_labels = (r"$\phi$ (deg)", r"$\dot\phi$ (deg/s)")
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
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
    fig.suptitle("Inverted-pendulum state comparison")

    for label, result in items:
        color = color_map[label]
        result_end_time = float(result["t"][-1])
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
        figures.append(fig)
        axs[0].plot(result["t"], result["u_hist"], color=color, label="control")
        axs[1].semilogy(
            result["t"],
            np.maximum(result["slack_hist"], 1e-12),
            color=color,
            label="QP slack",
        )
        axs[0].set_ylabel("torque (N m)")
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
        fig, axs = plt.subplots(IP.adim + 2, 1, sharex=True, figsize=(8, 11))
        figures.append(fig)
        for parameter_index, ax in enumerate(axs[: IP.adim]):
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

    for label, result in items:
        color = color_map[label]
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        figures.append(fig)
        ax.plot(result["t"], result["wind_hist"][:, 1], color=color, label=r"$w_z$")
        ax.set_xlim(0.0, float(result["t"][-1]))
        ax.set_ylabel(r"$w_z$ (m/s)")
        ax.set_xlabel("time (s)")
        ax.grid(True)
        ax.legend()
        fig.suptitle(f"{label}: vertical-wind history")

    for label, result in items:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        figures.append(fig)
        ax.plot(result["t"], result["true_uncertainty_hist"][:, 1], label="true uncertainty")
        ax.plot(
            result["t"],
            result["fitted_uncertainty_hist"][:, 1],
            "--",
            label=r"$Y_\Theta(x)a_k$",
        )
        ax.set_xlim(0.0, float(result["t"][-1]))
        ax.set_ylabel(r"$w_2$")
        ax.set_xlabel("time (s)")
        ax.grid(True)
        ax.legend()
        fig.suptitle(rf"{label}: $Y_\Theta(x)a_k$ versus true uncertainty")
    return figures


def main():
    """Build the shared experiment, run three controller settings, and compare them."""
    K_pre = 45
    N_cal = 30
    K = 6
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

    mass = 1.0
    length = 2.0
    inertia = mass * length**2 / 3.0
    nominal_gravity = 9.81
    true_gravity = 12.21

    # Limit representation capacity so angle-dependent error remains for CP calibration.
    theta_lb = -0.5 * np.ones(IP.theta_shape)
    theta_ub = 0.5 * np.ones(IP.theta_shape)
    theta_rng = np.random.default_rng(11)
    theta_init = theta_rng.uniform(theta_lb, theta_ub)

    a_lb = -1.5 * np.ones(IP.adim)
    a_ub = 1.5 * np.ones(IP.adim)
    projection_epsilon = 0.01
    a_hat_norm_max = 0.5 * np.linalg.norm(a_ub - a_lb, ord=2)
    a_hat_norm_max += projection_epsilon
    # Scaling Q and R together preserves the LQR gain. This normalization also
    # keeps the slack-penalty Hessian well conditioned for the QP solver.
    clf_scale = 10.0
    effective_gamma_clf = 7.5e-2
    gamma_scale = effective_gamma_clf / clf_scale / IP.adim
    gamma_clf = gamma_scale * np.eye(IP.adim)
    u_min = -12.0
    u_max = 12.0

    config = {
        "K_pre": K_pre,
        "N_cal": N_cal,
        "K": K,
        "B": B,
        "dt": dt,
        "interval_duration": interval_duration,
        "I_length": I_length,
        "pretrain_excitation_amplitude": 3.0,
        "pretrain_excitation_frequency": 0.1,
        "pretrain_state_norm_max": 3.0,
        "main_initial_state": np.array([0.18, 0.0]),
        "x_norm_divergence_threshold": 2.0,  # exit threshold for the local upright domain
        "rho_divergence_threshold": 1e6,
    }
    base_ip_params = {
        "mass": mass,
        "length": length,
        "inertia": inertia,
        "grav": nominal_gravity,
        "true_grav": true_gravity,
        "damping": 0.15,
        "true_damping": 0.02,
        "c_w": 0.1,
        "Theta_init": theta_init.copy(),
        "lqr_Q": clf_scale * np.diag([20.0, 2.0]),
        "lqr_R": clf_scale * 0.1,
        "Gamma_clf": gamma_clf,
        "a_ub": a_ub,
        "a_lb": a_lb,
        "a_hat_norm_max": a_hat_norm_max,
        "epsilon": projection_epsilon,
        "eta_clf": clf_scale * 10.0,
        "clf_rate": 0.5,
        "weight_slack": 1e5 / clf_scale**2,
        "u_min": u_min,
        "u_max": u_max,
        "dt": dt,
    }

    pretrain_params = dict(base_ip_params)
    pretrain_params.update({"use_cp": False, "use_adaptive": False})
    pretrain_system = IP(pretrain_params)

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
        run_params = dict(base_ip_params)
        run_params["Theta_init"] = canonical_theta.copy()
        run_params["use_cp"] = use_cp
        run_params["use_adaptive"] = use_adaptive
        run_system = IP(run_params)
        run_olacp = copy.deepcopy(trained_olacp)
        if np.shares_memory(run_olacp.Theta, trained_olacp.Theta):
            raise RuntimeError(f"{label}: online and canonical OLACP objects share Theta memory")
        results[label] = run_craclf_simulation(
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
        raise RuntimeError("A main simulation mutated the representation-update index")

    cp_adaptive_result = results["CP + adaptive"]
    cp_adaptive_metrics = cp_adaptive_result["metrics"]
    no_cp_adaptive_result = results["No CP + adaptive"]
    no_cp_adaptive_metrics = no_cp_adaptive_result["metrics"]
    nonadaptive_result = results["No CP + nonadaptive"]
    nonadaptive_metrics = nonadaptive_result["metrics"]
    if cp_adaptive_result["status"] != "completed" or cp_adaptive_metrics["tail_rms"] >= 0.05:
        warnings.warn("The CP + adaptive run did not stabilize")
    if cp_adaptive_metrics["max_slack"] / clf_scale >= 1e-1:
        warnings.warn("The normalized CP + adaptive QP slack exceeded 1e-1")
    if cp_adaptive_metrics["a_hat_change"] <= 1e-4:
        warnings.warn("The CRaCLF parameter estimate did not change materially")
    if cp_adaptive_metrics["rho_range"] <= 1e-6:
        warnings.warn("The CRaCLF scaling state did not change materially")
    if cp_adaptive_metrics["nu_range"] <= 1e-6:
        warnings.warn("rho did not materially change the adaptation scaling")
    no_cp_adaptive_failed = (
        no_cp_adaptive_result["status"] != "completed"
        or no_cp_adaptive_metrics["tail_rms"] > 0.25
    )
    if not no_cp_adaptive_failed:
        warnings.warn("The no-CP adaptive run unexpectedly stabilized")
    nonadaptive_failed = (
        nonadaptive_result["status"] != "completed" or nonadaptive_metrics["tail_rms"] > 0.25
    )
    if not nonadaptive_failed:
        warnings.warn("The no-CP nonadaptive run unexpectedly stabilized")
    if cp_adaptive_metrics["tail_rms"] >= nonadaptive_metrics["tail_rms"]:
        warnings.warn("The CP + adaptive run did not improve on the nonadaptive baseline")
    if nonadaptive_metrics["a_hat_change"] > 1e-12:
        warnings.warn("The nonadaptive run unexpectedly changed its parameter estimate")
    if nonadaptive_metrics["rho_range"] > 1e-12:
        warnings.warn("The nonadaptive run unexpectedly changed its scaling state")

    plot_craclf_results(results)
    plt.tight_layout()
    if os.environ.get("CRASAFE_NO_PLOTS", "0") == "1":
        plt.close("all")
    else:
        plt.show()
    return pretraining_history, results


if __name__ == "__main__":
    main()
