"""Algorithm 1 and CRaCCM for the planar quadrotor"""

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
from systems.planar_quad.planar_quad import PLANAR_QUAD

length = 0.25
mass = 0.486
grav = 9.81
inertia = 0.00383

def noise_fcn(t):
    """Return the nonparametric part of the physical uncertainty"""
    return np.array(
        [
            0.0,
            0.0,
            0.0,
            0.04 * np.sin(2.0 * np.pi * 0.67 * t + 0.3),
            0.03 * np.cos(2.0 * np.pi * 0.41 * t + 0.1),
            0.0,
        ]
    )


def true_uncertainty_fcn(x, t, schedule):
    """Return true uncertainty w(t,x): wind-drag uncertainty + noise"""
    _, _, phi, vx, vz, _ = np.asarray(x, dtype=float).reshape(PLANAR_QUAD.xdim)
    schedule = np.asarray(schedule, dtype=float).reshape(2)
    noise = noise_fcn(t)
    if not np.all(np.isfinite(schedule)) or not np.all(np.isfinite(noise)):
        raise ValueError("uncertainty schedules and noise must be finite")

    wind_x, wind_z = schedule
    beta_x = 0.015
    beta_z = 0.03

    v_rel_x = vx - wind_x * np.cos(phi) - wind_z * np.sin(phi)
    v_rel_z = vz + wind_x * np.sin(phi) - wind_z * np.cos(phi)
    force_w_x = -beta_x * v_rel_x * np.abs(v_rel_x)
    force_w_z = -beta_z * v_rel_z * np.abs(v_rel_z)

    return np.array(
        [
            0.0,
            0.0,
            0.0,
            force_w_x / mass + noise[3],
            force_w_z / mass + noise[4],
            0.0,
        ]
    )


def pretraining_control(x, t, config):
    """Return a stabilizing, persistently exciting two-rotor input"""
    px, pz, phi, vx, vz, phi_dot = np.asarray(x, dtype=float).reshape(6)
    amplitude = config["pretrain_excitation_amplitude"]
    omega = 2.0 * np.pi * config["pretrain_excitation_frequency"]
    phase_1 = omega * t
    phase_2 = 2.3 * omega * t + 0.4

    px_reference = amplitude * (0.75 * np.sin(phase_1) + 0.35 * np.sin(phase_2))
    px_reference_dot = amplitude * omega * (
        0.75 * np.cos(phase_1) + 0.805 * np.cos(phase_2)
    )
    px_reference_ddot = -amplitude * omega**2 * (
        0.75 * np.sin(phase_1) + 1.8515 * np.sin(phase_2)
    )
    altitude = config["pretrain_altitude"]
    pz_reference = altitude + 0.25 * amplitude * np.sin(0.7 * phase_1 + 0.2)
    pz_reference_dot = 0.175 * amplitude * omega * np.cos(0.7 * phase_1 + 0.2)
    pz_reference_ddot = -0.1225 * amplitude * omega**2 * np.sin(0.7 * phase_1 + 0.2)

    inertial_vx = vx * np.cos(phi) - vz * np.sin(phi)
    inertial_vz = vx * np.sin(phi) + vz * np.cos(phi)
    ax_command = px_reference_ddot + 2.5 * (px_reference - px)
    ax_command += 3.0 * (px_reference_dot - inertial_vx)
    az_command = pz_reference_ddot + 4.0 * (pz_reference - pz)
    az_command += 3.5 * (pz_reference_dot - inertial_vz)

    grav = config["grav"]
    phi_reference = -np.arctan2(ax_command, grav + az_command)
    thrust = config["mass"] * (
        -ax_command * np.sin(phi) + (grav + az_command) * np.cos(phi)
    )
    angular_acceleration = 10.0 * (phi_reference - phi) - 4.0 * phi_dot
    thrust_difference = config["inertia"] * angular_acceleration / config["length"]
    return 0.5 * np.array([thrust + thrust_difference, thrust - thrust_difference])


def run_pretraining(system, olacp, config, plot=True):
    """Pretrain the representation and fill the OLACP calibration window"""
    K_pre = config["K_pre"]
    I_length = config["I_length"]
    dt = config["dt"]
    interval_duration = config["interval_duration"]
    t_pre = np.arange(K_pre * I_length, dtype=float) * dt

    indices = np.arange(K_pre, dtype=float)
    wind_x = -0.45 + 2.0 * np.sin(2.0 * np.pi * indices / 9.0)
    wind_z = 1.0 * np.cos(2.0 * np.pi * indices / 8.0)
    schedule_values = np.vstack((wind_x, wind_z))

    def schedule_index(t):
        return min(int(np.floor(max(float(t), 0.0) / interval_duration)), K_pre - 1)

    def true_uncertainty(x, t):
        return true_uncertainty_fcn(x, t, schedule_values[:, schedule_index(t)])

    system.true_uncertainty_fcn = true_uncertainty
    system.set_representation(olacp.Theta)
    x = config["pretrain_initial_state"].copy()
    x_hist = np.zeros((len(t_pre), system.xdim))
    u_hist = np.zeros((len(t_pre), system.udim))
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
            u_hist[sample_index] = u
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
        raise RuntimeError("The trained representation was not installed in the planar quadrotor")

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
        fig, axs = plt.subplots(3, 2, sharex=True, figsize=(8, 6))
        for state_index, ax in enumerate(axs.flat):
            ax.plot(t_pre, x_hist[:, state_index])
            ax.set_ylabel(rf"$x_{state_index + 1}$")
            ax.grid(True)
        axs[-1, 0].set_xlabel("time (s)")
        axs[-1, 1].set_xlabel("time (s)")
        fig.suptitle("Pretraining: states")

        fig, axs = plt.subplots(system.udim, 1, sharex=True, figsize=(8, 5))
        for input_index, ax in enumerate(np.atleast_1d(axs)):
            ax.plot(t_pre, u_hist[:, input_index])
            ax.set_ylabel(rf"$u_{input_index + 1}$")
            ax.grid(True)
        np.atleast_1d(axs)[-1].set_xlabel("time (s)")
        fig.suptitle("Pretraining: control inputs")

        fig, axs = plt.subplots(system.theta_shape[0], system.theta_shape[1], sharex=True, figsize=(8, 6))
        for i in range(system.theta_shape[0]):
            for j in range(system.theta_shape[1]):
                ax = axs[i, j]
                ax.plot(np.arange(K_pre + 1) * interval_duration, theta_hist[:, i, j])
                ax.set_ylabel(rf"$\Theta_{{{i + 1},{j + 1}}}$")
                ax.grid(True)
        axs[-1, 0].set_xlabel("time (s)")
        fig.suptitle("Pretraining: representation parameters (Theta)")

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        for ax, component in zip(axs, (3, 4)):
            ax.plot(t_pre, true_uncertainty_hist[:, component], label="true uncertainty")
            ax.plot(t_pre, fitted_uncertainty_hist[:, component], "--", label=r"$Y_\Theta a_k$")
            ax.set_ylabel(rf"$w_{component + 1}$")
            ax.grid(True)
            ax.legend()
        axs[-1].set_xlabel("time (s)")
        fig.suptitle(r"Pretraining: $Y_\Theta(x)a_k$ versus true uncertainty")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(t_pre, np.maximum(prediction_error_hist, 1e-16))
        ax.set_xlabel("time (s)")
        ax.set_ylabel("squared prediction error")
        ax.grid(True)

        plt.show()

    return olacp, history


def plan_nominal_trajectory(system, config, plot=True):
    """Plan one nominal trajectory through the configured interval waypoints."""
    interval_count = int(config["K"])
    interval_steps = int(config["I_length"])
    if interval_count < 1 or interval_steps < 1:
        raise ValueError("K and I_length must be positive")

    waypoints = np.asarray(config["nominal_waypoints"], dtype=float)
    expected_shape = (interval_count + 1, system.xdim)
    if waypoints.shape != expected_shape:
        raise ValueError(f"nominal_waypoints must have shape {expected_shape}")
    if not np.all(np.isfinite(waypoints)):
        raise ValueError("nominal_waypoints must be finite")

    horizon_steps = interval_count * interval_steps
    planner = MotionPlanner(
        system=system,
        dt=config["dt"],
        Q=config["motion_planner_Q"],
        R=config["motion_planner_R"],
        Q_f=config["motion_planner_Q_f"],
        u_min=np.asarray(system.params["u_min"], dtype=float).reshape(system.udim),
        u_max=np.asarray(system.params["u_max"], dtype=float).reshape(system.udim),
    )
    
    # x_guess should be straight-line segments connecting the waypoints
    for waypoint_index in range(interval_count):
        x_guess_segment = waypoints[waypoint_index, :, None] + (
            waypoints[waypoint_index + 1, :, None] - waypoints[waypoint_index, :, None]
        ) * np.linspace(0.0, 1.0, interval_steps + 1)[:, None].T

        if waypoint_index < interval_count - 1:
            x_guess_segment = x_guess_segment[:, :-1]  # exclude the last point to avoid duplication
        else:
            x_guess_segment = x_guess_segment  # include the last point for the final segment

        if waypoint_index == 0:
            x_guess = x_guess_segment
        else:
            x_guess = np.hstack((x_guess, x_guess_segment))

    hover_input = 0.5 * system.mass * system.grav * np.ones(system.udim)
    u_guess = np.repeat(hover_input[:, None], horizon_steps, axis=1)

    x_d, u_d = planner.plan_through_waypoints(
        waypoints,
        interval_steps,
        x_guess,
        u_guess,
        hover_input,
    )
    
    t_x = config["dt"] * np.arange(horizon_steps + 1)
    t_u = t_x[:-1]
    waypoint_indices = interval_steps * np.arange(interval_count + 1)
    waypoint_error = x_d[:, waypoint_indices].T - waypoints
    maximum_waypoint_error = float(np.max(np.abs(waypoint_error)))
    regressor_norm = np.asarray(
        [
            np.linalg.norm(system.Y(x_d[:, index]), ord="fro")
            for index in range(horizon_steps + 1)
        ]
    )
    maximum_regressor_norm = float(np.max(regressor_norm))
    print(
        f"Nominal plan complete: max waypoint error={maximum_waypoint_error:.3e}, "
        f"max ||Y_Theta(x_d)||_F={maximum_regressor_norm:.3e}"
    )

    if plot:
        fig, axs = plt.subplots(3, 2, sharex=True, figsize=(8, 6))
        waypoint_times = config["interval_duration"] * np.arange(interval_count + 1)
        for state_index, ax in enumerate(axs.flat):
            ax.plot(t_x, x_d[state_index])
            ax.plot(waypoint_times, waypoints[:, state_index], "o")
            ax.set_ylabel(rf"$x_{state_index + 1}$")
            ax.grid(True)
        axs[-1, 0].set_xlabel("time (s)")
        axs[-1, 1].set_xlabel("time (s)")
        fig.suptitle("Nominal trajectory")

        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
        waypoint_times = config["interval_duration"] * np.arange(interval_count + 1)
        ax.plot(x_d[0], x_d[1])
        ax.plot(waypoints[:, 0], waypoints[:, 1], "o")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.grid(True)
        fig.suptitle("Nominal x-z trajectory")

        plt.show()

    return {
        "t_x": t_x,
        "t_u": t_u,
        "x_d": x_d,
        "u_d": u_d,
        "waypoints": waypoints.copy(),
        "waypoint_indices": waypoint_indices,
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
    """Run one planar-quadrotor CRaCCM experiment"""
    if bool(system.use_cp) != bool(use_cp) or bool(system.use_adaptive) != bool(use_adaptive):
        raise ValueError("The system must be constructed with the requested controller flags")

    K = config["K"]
    interval_duration = config["interval_duration"]

    wind_x = np.repeat(np.array([-3.00, -3.25, -3.05, 3.35, 3.05]), 4)
    wind_z = np.repeat(np.array([0.95, 0.65, 0.70, 0.85, 1.00]), 4)
    schedule_values = np.vstack((wind_x, wind_z))

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
        raise RuntimeError("The system and OLACP must own independent Theta arrays")

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

    sample_count = len(t_full)
    x_hist = np.zeros((sample_count, system.xdim))
    x_d_hist = np.zeros_like(x_hist)
    u_hist = np.full((sample_count, system.udim), np.nan)
    u_d_hist = np.full_like(u_hist, np.nan)
    energy_hist = np.full(sample_count, np.nan)
    slack_hist = np.full(sample_count, np.nan)
    a_hat_hist = np.zeros((sample_count, system.adim))
    a_k_hist = np.full((sample_count, system.adim), np.nan)
    rho_hist = np.zeros(sample_count)
    nu_hist = np.zeros(sample_count)
    quantile_hist = np.zeros(sample_count)
    theta_hist = np.zeros((sample_count,) + online_olacp.Theta.shape)
    true_uncertainty_hist = np.full((sample_count, system.xdim), np.nan)
    fitted_uncertainty_hist = np.full((sample_count, system.xdim), np.nan)
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
        terminal_x_d = np.asarray(desired_trajectory["interp_x_d"](sample_time), dtype=float)
        t_hist[index] = sample_time
        x_hist[index] = terminal_x
        x_d_hist[index] = terminal_x_d.reshape(system.xdim)
        u_hist[index] = np.asarray(held_u, dtype=float).reshape(system.udim)
        u_d_hist[index] = np.asarray(held_u_d, dtype=float).reshape(system.udim)
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
        u_hist[sample_index] = np.asarray(u).reshape(system.udim)
        u_d_hist[sample_index] = u_d
        energy_hist[sample_index] = system.Erem
        slack_hist[sample_index] = float(slack)
        xdot = system.dynamics(x, u, t)
        online_olacp.add_data_to_buffers(x, system.dynamics_nominal(x, u), xdot=xdot)

        if sample_index < sample_count - 1:
            solution = solve_ivp(
                lambda tau, state: system.dynamics_extended(state, x_d, u, geodesic_solver, tau),
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
                last_sample_index = sample_index + 1
                x_ext = event_state
                x, a_hat, rho = record_terminal_sample(
                    last_sample_index, failure_time, x_ext, u, u_d, slack
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
            failure_time = float(t_full[min(sample_index + 1, sample_count - 1)])
            failure_rho = rho
            if rho_diverged:
                failure_reason = "rho_divergence"
            elif not state_is_finite:
                failure_reason = "nonfinite_extended_state"
            else:
                failure_reason = "state_norm"
            if sample_index < sample_count - 1:
                last_sample_index = sample_index + 1
                x, a_hat, rho = record_terminal_sample(
                    last_sample_index, failure_time, x_ext, u, u_d, slack
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
        "max_control": float(np.nanmax(np.linalg.norm(u_hist, axis=1))),
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
        f"final_error={metrics['final_error']:.3e}, rms_error={metrics['rms_error']:.3e}, "
        f"max_error={metrics['max_error']:.3e}, max|u|={metrics['max_control']:.3e}, "
        f"max_slack={metrics['max_slack']:.3e}"
    )
    return result


def plot_craccm_results(results, desired_trajectory):
    """Plot and compare one or more planar-quadrotor simulations"""
    items = (
        list(results.items())
        if isinstance(results, dict)
        else [(result["label"], result) for result in results]
    )
    if not items:
        raise ValueError("At least one result is required")
    reference = max((result for _, result in items), key=lambda result: len(result["t"]))
    for label, result in items:
        if result is reference:
            continue
        if not np.allclose(result["x_hist"][0], reference["x_hist"][0]):
            raise RuntimeError(f"{label}: comparison initial state does not match")
        if not np.allclose(
            result["uncertainty_schedule_values"], reference["uncertainty_schedule_values"]
        ):
            raise RuntimeError(f"{label}: comparison uncertainty schedule does not match")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {label: colors[index % len(colors)] for index, (label, _) in enumerate(items)}
    maximum_time = max(float(result["t"][-1]) for _, result in items)
    figures = []
    state_labels = ("x (m)", "z (m)", r"$\phi$ (rad)", r"$v_x$", r"$v_z$", r"$\dot\phi$")

    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(8, 6))
    figures.append(fig)
    for state_index, ax in enumerate(axs.flat):
        ax.plot(
            desired_trajectory["t_x"],
            desired_trajectory["x_d"][state_index],
            "k--",
            label="desired" if state_index == 0 else None,
        )
        for label, result in items:
            ax.plot(result["t"], result["x_hist"][:, state_index], label=label)
            if result["status"] != "completed":
                ax.plot(result["t"][-1], result["x_hist"][-1, state_index], "x", markersize=9)
        ax.set_ylabel(state_labels[state_index])
        ax.set_xlim(0.0, maximum_time)
        ax.grid(True)
    axs[0, 0].legend()
    axs[-1, 0].set_xlabel("time (s)")
    axs[-1, 1].set_xlabel("time (s)")
    fig.suptitle("Planar-quadrotor state tracking comparison")

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
    figures.append(fig)
    ax.plot(desired_trajectory["x_d"][0], desired_trajectory["x_d"][1], "k--", label="desired")
    for label, result in items:
        ax.plot(result["x_hist"][:, 0], result["x_hist"][:, 1], label=label)
        if result["status"] != "completed":
                ax.plot(result["x_hist"][-1, 0], result["x_hist"][-1, 1], "x", markersize=9)
    ax.set_xlabel(state_labels[0])
    ax.set_ylabel(state_labels[1])
    ax.grid(True)
    ax.legend()
    fig.suptitle("Planar-quadrotor x-z position tracking comparison")

    for label, result in items:
        color = color_map[label]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
        figures.append(fig)
        for input_index, ax in enumerate(axs[:2]):
            ax.plot(result["t"], result["u_hist"][:, input_index], color=color, label="CRaCCM")
            ax.plot(result["t"], result["u_d_hist"][:, input_index], "k--", label="nominal")
            ax.set_ylabel(rf"$u_{input_index + 1}$")
            ax.grid(True)
            ax.legend()
        axs[2].semilogy(result["t"], np.maximum(result["slack_hist"], 1e-12), color=color)
        axs[2].set_ylabel("QP slack")
        axs[2].set_xlabel("time (s)")
        axs[2].grid(True)
        fig.suptitle(f"{label}: controls and slack")

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
        figures.append(fig)
        for parameter_index, ax in enumerate(axs[:2]):
            ax.plot(result["t"], result["a_hat_hist"][:, parameter_index], color=color)
            ax.plot(result["t"], result["a_k_hist"][:, parameter_index], "--", color=color)
            ax.set_ylabel(rf"$a_{parameter_index + 1}$")
            ax.grid(True)
        axs[2].plot(result["t"], result["rho_hist"], color=color)
        axs[3].plot(result["t"], result["nu_hist"], color=color)
        axs[2].set_ylabel(r"$\rho$")
        axs[3].set_ylabel(r"$\nu$")
        axs[3].set_xlabel("time (s)")
        axs[2].grid(True)
        axs[3].grid(True)
        fig.suptitle(f"{label}: CRaCCM adaptation and OLACP identification")

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
        figures.append(fig)
        axs[0].step(result["interval_times"], result["score_hist"], where="post", label=r"$s_k$")
        axs[0].step(result["t"], result["quantile_hist"], where="post", label=r"$Q_k$")
        axs[1].step(result["interval_times"], result["delta_hist"], where="post")
        axs[2].step(result["interval_times"], result["miscoverage_hist"], where="post")
        axs[0].set_ylabel("score")
        axs[1].set_ylabel(r"$\delta_k$")
        axs[2].set_ylabel("miscoverage")
        axs[2].set_xlabel("time (s)")
        for ax in axs:
            ax.grid(True)
        axs[0].legend()
        fig.suptitle(f"{label}: Algorithm 1")

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        figures.append(fig)
        for ax, component in zip(axs, (3, 4)):
            ax.plot(result["t"], result["true_uncertainty_hist"][:, component], label="true")
            ax.plot(
                result["t"],
                result["fitted_uncertainty_hist"][:, component],
                "--",
                label="fitted",
            )
            ax.set_ylabel(rf"$w_{component + 1}$")
            ax.grid(True)
            ax.legend()
        axs[-1].set_xlabel("time (s)")
        fig.suptitle(rf"{label}: $Y_\Theta(x)a_k$ versus true uncertainty")

    plt.tight_layout()
    plt.show()

    return figures


def main():
    """Pretrain Algorithm 1, plan the desired trajectory, and compare three controllers"""
    K_pre = 100
    N_cal = 80
    K = 20
    B = 4
    dt = 0.01
    interval_duration = 2.0
    I_length = int(round(interval_duration / dt))
    if K_pre < N_cal or K_pre % B != 0:
        raise ValueError("K_pre must fill N_cal and contain complete representation blocks")
    if I_length < 10 or not np.isclose(I_length * dt, interval_duration):
        raise ValueError("interval_duration must be an integer multiple of dt")

    theta_lb = np.eye(PLANAR_QUAD.theta_shape[0], PLANAR_QUAD.theta_shape[1]) - 0.2 * np.ones(PLANAR_QUAD.theta_shape)
    theta_ub = np.eye(PLANAR_QUAD.theta_shape[0], PLANAR_QUAD.theta_shape[1]) + 0.2 * np.ones(PLANAR_QUAD.theta_shape)
    theta_rng = np.random.default_rng(42)
    theta_init = theta_rng.uniform(theta_lb, theta_ub)

    a_lb = -4.0 * np.ones(PLANAR_QUAD.adim)
    a_ub = 4.0 * np.ones(PLANAR_QUAD.adim)
    projection_epsilon = 0.01
    a_hat_norm_max = 0.5 * np.linalg.norm(a_ub - a_lb) + projection_epsilon

    nominal_waypoints = np.array(
        [
            [0.00, 5.00, -0.030, 0.00, 0.00, 0.00],
            [0.57, 5.45, -0.025, 0.54, 0.40, 0.00],
            [2.07, 6.18, -0.01, 0.90, 0.25, 0.00],
            [3.93, 6.18, 0.00, 0.90, -0.25, 0.00],
            [5.43, 5.45, 0.010, 0.54, -0.40, 0.00],
            [6.00, 5.00, 0.017 , 0.00, 0.00, 0.00],
            [5.43 + 1.0, 5.45, 0.025, -0.54, 0.40, 0.00],
            [3.93 + 1.0, 6.18, 0.01, -0.90, 0.25, 0.00],
            [2.07 + 1.0, 6.18, -0.01, -0.90, -0.25, 0.00],
            [0.57 + 1.0, 5.45, -0.025, -0.54, -0.40, 0.00],
            [0.00, 5.00, -0.03, 0.00, 0.00, 0.00],
            [0.57, 5.45, -0.025, 0.54, 0.40, 0.00],
            [2.07, 6.18, -0.01, 0.90, 0.25, 0.00],
            [3.93, 6.18, 0.00, 0.90, -0.25, 0.00],
            [5.43, 5.45, 0.010, 0.54, -0.40, 0.00],
            [6.00, 5.00, 0.017, 0.00, 0.00, 0.00],
            [5.43 + 1.0, 5.45, 0.025, -0.54, 0.40, 0.00],
            [3.93 + 1.0, 6.18, 0.01, -0.90, 0.25, 0.00],
            [2.07 + 1.0, 6.18, -0.01, -0.90, -0.25, 0.00],
            [0.57 + 1.0, 5.45, -0.025, -0.54, -0.40, 0.0],
            [0.00, 5.00, -0.030, 0.00, 0.00, 0.00],
        ]
    )

    config = {
        "K_pre": K_pre,
        "N_cal": N_cal,
        "K": K,
        "B": B,
        "dt": dt,
        "interval_duration": interval_duration,
        "I_length": I_length,
        "pretrain_excitation_amplitude": 0.8,
        "pretrain_excitation_frequency": 0.12,
        "pretrain_altitude": 1.0,
        "pretrain_initial_state": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        "nominal_waypoints": nominal_waypoints,
        "motion_planner_Q": np.diag([1.0, 10.0, 1.0, 1.0, 1.0, 100.0]),
        "motion_planner_R": 10.0 * np.eye(PLANAR_QUAD.udim),
        "motion_planner_Q_f": np.diag([1500.0, 2500.0, 750.0, 250.0, 250.0, 1500.0]),
        "tracking_initial_offset": np.zeros(PLANAR_QUAD.xdim),
        "x_norm_divergence_threshold": 10.0,
        "rho_divergence_threshold": 1e6,
        "geodesic_degree": 2,
        "geodesic_nodes": 8,
        "use_qpsolvers": True,
        "verify_geodesic": False,
        "length": length,
        "mass": mass,
        "grav": grav,
        "inertia": inertia,
    }
    base_params = {
        "Theta_init": theta_init.copy(),
        "l": config["length"],
        "m": config["mass"],
        "g": config["grav"],
        "J": config["inertia"],
        "Gamma_ccm": 0.055 * np.eye(PLANAR_QUAD.adim),
        "a_ub": a_ub,
        "a_lb": a_lb,
        "a_hat_norm_max": a_hat_norm_max,
        "epsilon": projection_epsilon,
        "eta_ccm": 100.0,
        "ccm_rate": 0.2,
        "weight_slack": 1e5,
        "u_min": np.zeros(PLANAR_QUAD.udim),
        "u_max": 6.0 * np.ones(PLANAR_QUAD.udim),
        "dt": dt,
    }

    pretrain_params = dict(base_params)
    pretrain_params.update({"use_cp": False, "use_adaptive": False})
    pretrain_system = PLANAR_QUAD(pretrain_params)

    def representation_rate(update_index):
        return 1e-2 / np.sqrt(update_index)

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
        run_system = PLANAR_QUAD(run_params)
        run_olacp = copy.deepcopy(pretrained_olacp)
        if np.shares_memory(run_olacp.Theta, pretrained_olacp.Theta):
            raise RuntimeError(f"{label}: online and pretrained OLACP objects share Theta memory")

        # Run the simulation and store the results
        results[label] = run_craccm_simulation(
            run_system,
            run_olacp,
            desired_trajectory,
            config,
            use_cp,
            use_adaptive,
            label=label,
        )

    # Check if the pretrained OLACP has been mutated by any of the simulations
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
    results_filename = f"./simulations/planar_quad/sim_planar_quad_craccm_results_{timestamp}.pkl"
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

    # Plot the results
    plot_craccm_results(results, desired_trajectory)

    return pretraining_history, desired_trajectory, results


if __name__ == "__main__":
    main()
