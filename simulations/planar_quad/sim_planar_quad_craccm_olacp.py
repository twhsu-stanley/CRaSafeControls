"""Algorithm 1 and CRaCCM for the planar quadrotor"""

import copy
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, interp1d

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
    pz_reference = altitude + amplitude * np.sin(0.7 * phase_1 + 0.2)
    pz_reference_dot = 0.7 * amplitude * omega * np.cos(0.7 * phase_1 + 0.2)
    pz_reference_ddot = -0.49 * amplitude * omega**2 * np.sin(0.7 * phase_1 + 0.2)

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
    B = config["B"]
    I_length = config["I_length"]
    dt = config["dt"]
    interval_duration = config["interval_duration"]
    t_pre = np.arange(K_pre * I_length, dtype=float) * dt

    indices = np.arange(K_pre, dtype=float)
    wind_x = 3.0 * np.sin(2.0 * np.pi * indices / B)
    wind_z = 0.8 * np.cos(2.0 * np.pi * indices / B)
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
        f"Pretraining complete: quantile Q_k={quantile:.3e}; final fitted a_k={olacp.a_k}"
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

        fig, axs = plt.subplots(system.adim, 1, sharex=True, figsize=(8, 6))
        for i, ax in enumerate(np.atleast_1d(axs)):
            ax.plot(t_pre, a_k_hist[:, i])
            ax.set_ylabel(rf"$a_{{k,{i + 1}}}$")
            ax.grid(True)
        np.atleast_1d(axs)[-1].set_xlabel("time (s)")
        fig.suptitle("Pretraining: fitted parameters (a_k)")

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
    expected_shape = (interval_count + 1, 2)
    if waypoints.shape != expected_shape:
        raise ValueError(f"nominal_waypoints must have shape {expected_shape}")
    if not np.all(np.isfinite(waypoints)):
        raise ValueError("nominal_waypoints must be finite")
    x_init = np.asarray(config["reference_initial_state"], dtype=float).reshape(system.xdim)
    if not np.all(np.isfinite(x_init)) or not np.allclose(x_init[:2], waypoints[0]):
        raise ValueError("reference_initial_state must be finite and start at the first waypoint")

    horizon_steps = interval_count * interval_steps
    planner = MotionPlanner(
        system=system,
        dt=config["dt"],
        Q=config["motion_planner_Q"],
        R=config["motion_planner_R"],
        Q_terminal=config["motion_planner_Q_terminal"],
        u_min=np.asarray(system.params["u_min"], dtype=float).reshape(system.udim),
        u_max=np.asarray(system.params["u_max"], dtype=float).reshape(system.udim),
        output_matrix=np.eye(2, system.xdim),
    )
    hover_input = 0.5 * system.mass * system.grav * np.ones(system.udim)
    u_guess = np.repeat(hover_input[:, None], horizon_steps, axis=1)
    t_x = config["dt"] * np.arange(horizon_steps + 1)
    x_guess = np.repeat(x_init[:, None], horizon_steps + 1, axis=1)
    waypoint_cost_guide = build_waypoint_cost_guide(waypoints, config["interval_duration"])
    x_guess[:2] = waypoint_cost_guide(t_x)

    x_d, u_d = planner.plan_nominal_through_waypoints(
        waypoints,
        interval_steps,
        x_guess,
        u_guess,
        hover_input,
        x_init=x_init,
        output_reference=waypoint_cost_guide,
    )
    
    t_u = t_x[:-1]
    waypoint_indices = interval_steps * np.arange(interval_count + 1)
    regressor_norm = np.asarray(
        [
            np.linalg.norm(system.Y(x_d[:, index]), ord="fro")
            for index in range(horizon_steps + 1)
        ]
    )
    maximum_regressor_norm = float(np.max(regressor_norm))
    print(f"Nominal plan complete: max ||Y_Theta(x_d)||_F={maximum_regressor_norm:.3e}")

    if plot:
        fig, axs = plt.subplots(3, 2, sharex=True, figsize=(8, 6))
        waypoint_times = config["interval_duration"] * np.arange(interval_count + 1)
        for state_index, ax in enumerate(axs.flat):
            ax.plot(t_x, x_d[state_index])
            if state_index < 2:
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



def build_waypoint_cost_guide(waypoints, interval_duration):
    """Interpolate position waypoints without prescribing attitude or waypoint velocities."""
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 2 or len(waypoints) < 2:
        raise ValueError("waypoints must have shape (K + 1, 2), with K >= 1")
    if not np.all(np.isfinite(waypoints)):
        raise ValueError("waypoints must be finite")
    if not np.isfinite(interval_duration) or interval_duration <= 0.0:
        raise ValueError("interval_duration must be finite and positive")
    waypoint_times = interval_duration * np.arange(len(waypoints))
    position_curve = CubicSpline(waypoint_times, waypoints, bc_type="natural")

    def position_reference(times):
        times = np.clip(np.atleast_1d(times), waypoint_times[0], waypoint_times[-1])
        return position_curve(times).T

    return position_reference


def run_craccm_simulation(system, online_olacp, config, use_cp, use_adaptive, label=None):
    """Run one planar-quadrotor CRaCCM experiment"""
    if bool(system.use_cp) != bool(use_cp) or bool(system.use_adaptive) != bool(use_adaptive):
        raise ValueError("The system must be constructed with the requested controller flags")

    K = config["K"]
    interval_duration = config["interval_duration"]

    wind_x = np.tile(np.array([-3.00, -3.25, -3.05, -2.85, -3.05]), 4)
    wind_z = np.tile(np.array([0.95, 0.65, 0.70, 0.85, 1.00]), 4)
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
    t_hist = np.arange(len(t_full) + 1, dtype=float) * dt
    run_label = label or f"CP={bool(use_cp)}, adaptive={bool(use_adaptive)}"
    waypoints = np.asarray(config["nominal_waypoints"], dtype=float)
    if waypoints.shape != (K + 1, 2) or not np.all(np.isfinite(waypoints)):
        raise ValueError(f"nominal_waypoints must be finite with shape {(K + 1, 2)}")
    x_d = np.asarray(config["reference_initial_state"], dtype=float).reshape(system.xdim).copy()
    if not np.all(np.isfinite(x_d)) or not np.allclose(x_d[:2], waypoints[0]):
        raise ValueError("reference_initial_state must be finite and start at the first waypoint")
    if K < 1 or I_length < 1 or not np.isclose(I_length * dt, interval_duration):
        raise ValueError("K and I_length must be positive and match interval_duration / dt")
    mpc_horizon = float(config.get("mpc_horizon", 4.0))
    mpc_dt = float(config.get("mpc_dt", 0.05))
    replan_interval = float(config.get("mpc_replan_interval", 0.2))
    if not np.isfinite(replan_interval) or not 0.1 <= replan_interval <= 0.5:
        raise ValueError("mpc_replan_interval must lie between 0.1 and 0.5 seconds")
    if not np.isfinite(mpc_dt) or mpc_dt < dt:
        raise ValueError("mpc_dt must be finite and at least the simulation dt")
    if not np.isfinite(mpc_horizon) or mpc_horizon < replan_interval:
        raise ValueError("mpc_horizon must cover at least one replan interval")
    replan_steps = int(round(replan_interval / dt))
    if not np.isclose(int(round(mpc_dt / dt)) * dt, mpc_dt):
        raise ValueError("mpc_dt must be an integer multiple of simulation dt")
    if (
        not np.isclose(replan_steps * dt, replan_interval)
        or replan_steps % int(round(mpc_dt / dt))
    ):
        raise ValueError("mpc_replan_interval must be an integer multiple of mpc_dt")
    # Align representation updates with the existing replanning grid.
    if (online_olacp.representation_period * I_length) % replan_steps:
        raise ValueError(
            "The representation update period must be an integer multiple of mpc_replan_interval"
        )
    waypoint_cost_guide = build_waypoint_cost_guide(waypoints, interval_duration)
    planner = MotionPlanner(
        system=system,
        dt=dt,
        Q=config["motion_planner_Q"],
        R=config["motion_planner_R"],
        u_min=np.asarray(system.params["u_min"], dtype=float).reshape(system.udim),
        u_max=np.asarray(system.params["u_max"], dtype=float).reshape(system.udim),
        output_matrix=np.eye(2, system.xdim),
        Q_terminal=config["motion_planner_Q_terminal"],
    )
    hover_input = 0.5 * system.mass * system.grav * np.ones(system.udim)
    rho_index = system.xdim + system.adim
    adaptive_state_size = rho_index + 1
    rho_divergence_threshold = config["rho_divergence_threshold"]
    if not np.isfinite(rho_divergence_threshold) or rho_divergence_threshold <= 0.0:
        raise ValueError("rho_divergence_threshold must be finite and positive")

    def rho_divergence_event(_time, state):
        return rho_divergence_threshold - abs(float(state[rho_index]))

    rho_divergence_event.terminal = True
    rho_divergence_event.direction = -1.0

    x = x_d + config["tracking_initial_offset"]
    a_hat = system.a_center.copy()
    rho = 0.0
    x_ext = np.hstack((x, a_hat, rho, x_d))
    geodesic_solver = GeodesicSolver(
        system.xdim,
        config["geodesic_degree"],
        config["geodesic_nodes"],
        system.W_fcn,
        system.dW_dxi_fcn,
        system.dW_dai_fcn,
    )

    control_sample_count = len(t_full)
    sample_count = control_sample_count + 1  # Leave space for an early-stop terminal sample.
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
    mpc_solve_time_hist = np.zeros(sample_count)
    mpc_replan_hist = np.zeros(sample_count, dtype=bool)
    mpc_iterations_hist = np.full(sample_count, np.nan)
    mpc_horizon_steps_hist = np.full(sample_count, np.nan)
    reference_times = [0.0]
    reference_states = [x_d.copy()]
    reference_inputs = []
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
        terminal_rho = float(state[rho_index])
        terminal_x_d = state[adaptive_state_size:]
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
        return terminal_x, terminal_a_hat, terminal_rho, terminal_x_d

    def dynamics_with_reference(tau, state, held_u, held_u_d):
        desired_state = state[adaptive_state_size:]
        model_a_hat = state[system.xdim:rho_index] if use_adaptive else np.zeros(system.adim)
        adaptive_derivative = system.dynamics_extended(
            state[:adaptive_state_size], desired_state, held_u, geodesic_solver, tau
        )
        desired_derivative = system.dynamics_certainty_equivalent(
            desired_state, held_u_d, model_a_hat
        )
        return np.hstack((adaptive_derivative, desired_derivative))

    predicted_inputs = None
    u_d = hover_input.copy()
    plan_start_step = 0
    for sample_index, t in enumerate(t_full):
        last_sample_index = sample_index
        model_a_hat = a_hat if use_adaptive else np.zeros(system.adim)
        if sample_index % replan_steps == 0:
            planning_start = time.perf_counter()
            try:
                _, predicted_inputs = planner.plan_certainty_equivalent_through_waypoints(
                    x_d, waypoints, I_length, sample_index, system.Theta_hat, model_a_hat,
                    hover_input, mpc_horizon=mpc_horizon, mpc_dt=mpc_dt,
                    output_reference=waypoint_cost_guide, u_previous=u_d,
                )
            except RuntimeError as error:
                raise RuntimeError(f"{run_label}: waypoint MPC failed at t={t:.3f}") from error
            plan_start_step = sample_index
            mpc_solve_time_hist[sample_index] = time.perf_counter() - planning_start
            mpc_iterations_hist[sample_index] = planner.last_mpc_stats["iterations"]
            mpc_replan_hist[sample_index] = True
        mpc_horizon_steps_hist[sample_index] = planner.last_mpc_stats["horizon_steps"]
        input_index = (sample_index - plan_start_step) // int(round(mpc_dt / dt))
        u_d = predicted_inputs[:, input_index].copy()
        reference_inputs.append(u_d.copy())
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

        next_time = (sample_index + 1) * dt
        solution = solve_ivp(
            lambda tau, state: dynamics_with_reference(tau, state, u, u_d),
            (t, next_time),
            x_ext,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
            t_eval=[next_time],
            events=rho_divergence_event if use_adaptive else None,
        )
        if use_adaptive and solution.t_events[0].size:
            x_ext = solution.y_events[0][-1]
            status = "diverged"
            failure_reason = "rho_divergence"
            failure_time = float(solution.t_events[0][-1])
            failure_rho = float(x_ext[rho_index])
            last_sample_index = sample_index + 1
            x, a_hat, rho, x_d = record_terminal_sample(
                last_sample_index, failure_time, x_ext, u, u_d, slack
            )
            reference_times.append(failure_time)
            reference_states.append(x_d.copy())
            break
        if not solution.success:
            raise RuntimeError(
                f"{run_label}: extended dynamics failed at t={t:.3f}: {solution.message}"
            )
        x_ext = solution.y[:, -1]
        x = x_ext[:system.xdim]
        a_hat = x_ext[system.xdim:rho_index]
        rho = float(x_ext[rho_index])
        x_d = x_ext[adaptive_state_size:]
        reference_times.append(next_time)
        reference_states.append(x_d.copy())

        rho_diverged = not np.isfinite(rho) or abs(rho) >= rho_divergence_threshold
        state_is_finite = np.all(np.isfinite(x_ext))
        state_norm_diverged = np.linalg.norm(x) > config["x_norm_divergence_threshold"]
        if rho_diverged or not state_is_finite or state_norm_diverged:
            status = "diverged"
            failure_time = next_time
            failure_rho = rho
            if rho_diverged:
                failure_reason = "rho_divergence"
            elif not state_is_finite:
                failure_reason = "nonfinite_extended_state"
            else:
                failure_reason = "state_norm"
            last_sample_index = sample_index + 1
            x, a_hat, rho, x_d = record_terminal_sample(
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
    mpc_solve_time_hist = mpc_solve_time_hist[used_slice]
    mpc_replan_hist = mpc_replan_hist[used_slice]
    mpc_iterations_hist = mpc_iterations_hist[used_slice]
    mpc_horizon_steps_hist = mpc_horizon_steps_hist[used_slice]

    reference_times = np.asarray(reference_times)
    reference_states = np.asarray(reference_states).T
    reference_inputs = np.asarray(reference_inputs).T
    waypoint_times = interval_duration * np.arange(K + 1)
    desired_trajectory = {
        "mode": "certainty_equivalent_dynamics_mpc",
        "t_x": reference_times,
        "t_u": reference_times[:-1],
        "x_d": reference_states,
        "u_d": reference_inputs,
        "waypoints": waypoints.copy(),
        "waypoint_times": waypoint_times,
        "mpc_horizon": mpc_horizon,
        "mpc_dt": mpc_dt,
        "mpc_replan_interval": replan_interval,
    }

    tracking_error = np.linalg.norm(x_hist - x_d_hist, axis=1)
    metrics = {
        "rms_error": float(np.sqrt(np.mean(tracking_error ** 2))),
        "max_error": float(np.max(tracking_error)),
        "max_control": float(np.nanmax(np.linalg.norm(u_hist, axis=1))),
        "max_slack": float(np.nanmax(slack_hist)),
        "total_mpc_solve_time": float(np.nansum(mpc_solve_time_hist)),
        "mpc_replan_count": int(np.count_nonzero(mpc_replan_hist)),
        "mean_mpc_solve_time": float(np.mean(mpc_solve_time_hist[mpc_replan_hist])),
    }
    result = {
        "label": run_label,
        "use_cp": bool(use_cp),
        "use_adaptive": bool(use_adaptive),
        "t": t,
        "x_hist": x_hist,
        "x_d_hist": x_d_hist,
        "desired_trajectory": desired_trajectory,
        "mpc_solve_time_hist": mpc_solve_time_hist,
        "mpc_replan_hist": mpc_replan_hist,
        "mpc_iterations_hist": mpc_iterations_hist,
        "mpc_horizon_steps_hist": mpc_horizon_steps_hist,
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
        f"rms_error={metrics['rms_error']:.3e}, "
        f"max_error={metrics['max_error']:.3e}, max|u|={metrics['max_control']:.3e}, "
        f"max_slack={metrics['max_slack']:.3e}, "
        f"mpc_solves={metrics['mpc_replan_count']}, "
        f"mpc_time={metrics['total_mpc_solve_time']:.2f}s"
    )
    return result


def plot_craccm_results(results, desired_trajectory=None, file_format="png"):
    """Compare each run against its reference and save figures in results as PNG or PDF."""
    if file_format not in ("png", "pdf"):
        raise ValueError('file_format must be "png" or "pdf"')
    items = (
        list(results.items())
        if isinstance(results, dict)
        else [(result["label"], result) for result in results]
    )
    if not items:
        raise ValueError("At least one result is required")
    trajectories = {}
    for label, result in items:
        trajectory = result.get("desired_trajectory")
        if trajectory is None and desired_trajectory is not None:
            if "x_d" in desired_trajectory:
                trajectory = desired_trajectory
            else:
                trajectory = desired_trajectory[label]
        if trajectory is None:
            raise ValueError(f"{label}: no desired trajectory is available")
        trajectories[label] = trajectory
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

    fig, ax = plt.subplots(figsize=(8, 6))
    figures.append(fig)
    for label, result in items:
        x_d_hist = result.get("x_d_hist")
        if x_d_hist is None:
            trajectory = trajectories[label]
            x_d_hist = np.column_stack(
                [np.interp(result["t"], trajectory["t_x"], state) for state in trajectory["x_d"]]
            )
        tracking_error = np.linalg.norm(result["x_hist"] - x_d_hist, axis=1)
        ax.plot(result["t"], tracking_error, color=color_map[label], label=label)
        if result["status"] != "completed":
            ax.plot(result["t"][-1], tracking_error[-1], "x", color=color_map[label], markersize=9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$\|x(t)-x_d(t)\|_2$")
    ax.set_xlim(0.0, maximum_time)
    ax.grid(True)
    ax.legend()
    fig.suptitle("Planar-quadrotor tracking error comparison")
    fig.set_label("planar_quad_tracking_error")

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    figures.append(fig)
    for label, result in items:
        color = color_map[label]
        for input_index, ax in enumerate(axs[:2]):
            ax.plot(
                result["t"], result["u_hist"][:, input_index],
                color=color, label=f"control: {label}",
            )
            ax.plot(
                result["t"], result["u_d_hist"][:, input_index], "--",
                color=color, label=f"reference: {label}",
            )
        axs[2].semilogy(
            result["t"], np.maximum(result["slack_hist"], 1e-12), color=color, label=label
        )
    for input_index, ax in enumerate(axs[:2]):
        ax.set_ylabel(rf"$u_{input_index + 1}$")
    axs[2].set_ylabel("QP slack")
    axs[2].set_xlabel("time (s)")
    for ax in axs:
        ax.set_xlim(0.0, maximum_time)
        ax.grid(True)
        ax.legend(fontsize="small")
    fig.suptitle("Planar-quadrotor controls and QP slack comparison")
    fig.set_label("planar_quad_control_slack")

    for label, result in items:
        color = color_map[label]
        trajectory = trajectories[label]
        setting_name = "".join(c if c.isalnum() else " " for c in label.lower())
        setting_name = "_".join(setting_name.split())
        fig, axs = plt.subplots(3, 2, sharex=True, figsize=(8, 6))
        figures.append(fig)
        for state_index, ax in enumerate(axs.flat):
            ax.plot(trajectory["t_x"], trajectory["x_d"][state_index], "k--", label="desired")
            ax.plot(result["t"], result["x_hist"][:, state_index], color=color, label="actual")
            if result["status"] != "completed":
                ax.plot(
                    result["t"][-1], result["x_hist"][-1, state_index], "x",
                    color=color, markersize=9,
                )
            ax.set_ylabel(state_labels[state_index])
            ax.set_xlim(0.0, float(result["t"][-1]))
            ax.grid(True)
        axs[0, 0].legend()
        axs[-1, 0].set_xlabel("time (s)")
        axs[-1, 1].set_xlabel("time (s)")
        fig.suptitle(f"{label}: Planar-quadrotor state tracking comparison")
        fig.set_label(f"planar_quad_states_{setting_name}")

        fig, ax = plt.subplots(figsize=(8, 6))
        figures.append(fig)
        ax.plot(trajectory["x_d"][0], trajectory["x_d"][1], "k--", label="desired")
        waypoints = trajectory.get("waypoints")
        if waypoints is not None:
            ax.plot(waypoints[:, 0], waypoints[:, 1], "ko", markersize=3, label="waypoints")
        ax.plot(result["x_hist"][:, 0], result["x_hist"][:, 1], color=color, label="actual")
        if result["status"] != "completed":
            ax.plot(
                result["x_hist"][-1, 0], result["x_hist"][-1, 1], "x", color=color, markersize=9
            )
        ax.set_xlabel(state_labels[0])
        ax.set_ylabel(state_labels[1])
        ax.grid(True)
        ax.legend()
        fig.suptitle(f"{label}: Planar-quadrotor x-z position tracking comparison")
        fig.set_label(f"planar_quad_trajectory_{setting_name}")

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
        fig.set_label(f"planar_quad_adaptation_{setting_name}")

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
        fig.set_label(f"planar_quad_olacp_{setting_name}")

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    figures.append(fig)
    for ax, component in zip(axs, (3, 4)):
        for label, result in items:
            ax.plot(
                result["t"], result["true_uncertainty_hist"][:, component],
                color=color_map[label], label=f"true: {label}",
            )
            ax.plot(
                result["t"],
                result["fitted_uncertainty_hist"][:, component],
                "--",
                color=color_map[label],
                label=rf"$Y_{{\Theta_j}}(x)a_k$: {label}",
            )
        ax.set_xlim(0.0, maximum_time)
        ax.set_ylabel(rf"$w_{component + 1}$")
        ax.grid(True)
        ax.legend(fontsize="small")
    axs[-1].set_xlabel("time (s)")
    fig.suptitle(r"Planar-quadrotor: $Y_{\Theta_j}(x)a_k$ versus true uncertainty")
    fig.set_label("planar_quad_uncertainty_fit")

    plt.tight_layout()
    results_directory = Path(__file__).resolve().parent / "results"
    results_directory.mkdir(exist_ok=True)
    for figure in figures:
        filename = results_directory / f"{figure.get_label()}.{file_format}"
        figure.savefig(filename, format=file_format, dpi=300, bbox_inches="tight")
    print(f"Saved {len(figures)} {file_format.upper()} figures to {results_directory}")
    plt.show()

    return figures


def main():
    """Pretrain Algorithm 1 and compare three controllers with online waypoint references."""
    K_pre = 200
    N_cal = 80
    K = 20
    B = 5
    dt = 0.01
    interval_duration = 2.0
    I_length = int(round(interval_duration / dt))
    if K_pre < N_cal or K_pre % B != 0:
        raise ValueError("K_pre must fill N_cal and contain complete representation blocks")
    if I_length < 10 or not np.isclose(I_length * dt, interval_duration):
        raise ValueError("interval_duration must be an integer multiple of dt")

    theta_lb = np.array([
        [-1.0, -1.0],
        [-0.5, -0.5],
        [-0.5, -0.5],
        [-0.5, -0.5]
    ]).reshape(PLANAR_QUAD.theta_shape)
    theta_ub = -theta_lb
    theta_rng = np.random.default_rng(42)
    theta_init = theta_rng.uniform(theta_lb, theta_ub)

    a_lb = -1.0 * np.ones(PLANAR_QUAD.adim)
    a_ub = 1.0 * np.ones(PLANAR_QUAD.adim)
    projection_epsilon = 0.01
    a_hat_norm_max = 0.5 * np.linalg.norm(a_ub - a_lb) + projection_epsilon

    nominal_waypoints = np.array(
        [
            [0.00, 5.00],
            [0.57, 5.45],
            [2.07, 6.18],
            [3.93, 6.18],
            [5.43, 5.45],
            [6.00, 5.00],
            [6.43, 5.45],
            [4.93, 6.18],
            [3.07, 6.18],
            [1.57, 5.45],
            [0.00, 5.00],
            [0.57, 5.45],
            [2.07, 6.18],
            [3.93, 6.18],
            [5.43, 5.45],
            [6.00, 5.00],
            [6.43, 5.45],
            [4.93, 6.18],
            [3.07, 6.18],
            [1.57, 5.45],
            [0.00, 5.00],
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
        "pretrain_excitation_amplitude": 2.0,
        "pretrain_excitation_frequency": 0.12,
        "pretrain_altitude": 5.0,
        "pretrain_initial_state": np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0]),
        "nominal_waypoints": nominal_waypoints,
        "reference_initial_state": np.array([0.0, 5.0, -0.03, 0.0, 0.0, 0.0]),
        "motion_planner_Q": np.diag([5.0, 50.0]),
        "motion_planner_R": 50.0 * np.eye(PLANAR_QUAD.udim),
        "motion_planner_Q_terminal": np.diag([1.0, 10.0]),
        "mpc_horizon": 4.0,
        "mpc_dt": 0.05,
        "mpc_replan_interval": 0.2,
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
        "Gamma_ccm": 0.25 * np.eye(PLANAR_QUAD.adim),
        "a_ub": a_ub,
        "a_lb": a_lb,
        "a_hat_norm_max": a_hat_norm_max,
        "epsilon": projection_epsilon,
        "eta_ccm": 500.0,
        "ccm_rate": 0.2,
        "weight_slack": 2000.0 / 3.0,  # w_xi after normalizing the input weight to one
        "ccm_authority_threshold": 0.01,
        "u_min": np.zeros(PLANAR_QUAD.udim),
        "u_max": 6.0 * np.ones(PLANAR_QUAD.udim),
        "dt": dt,
    }

    pretrain_params = dict(base_params)
    pretrain_params.update({"use_cp": False, "use_adaptive": False})
    pretrain_system = PLANAR_QUAD(pretrain_params)

    def representation_rate(update_index):
        return 1e-3 * np.exp(-0.02 * update_index)

    olacp = OLACP(
        [],
        N_cal=N_cal,
        acp_lr=0.1,
        delta_target=0.1,
        delta_init=0.15,
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

    desired_trajectories = {label: result["desired_trajectory"] for label, result in results.items()}

    # Save results to a file
    timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S_%f")
    results_directory = Path(__file__).resolve().parent / "results"
    results_directory.mkdir(exist_ok=True)
    results_filename = results_directory / f"sim_planar_quad_craccm_results_{timestamp}.pkl"
    with results_filename.open("xb") as f:
        pickle.dump(
            {
                "config": config,
                "base_params": base_params,
                "desired_trajectories": desired_trajectories,
                "results": results,
            },
            f,
        )

    # Plot the results
    plot_craccm_results(results, desired_trajectories)

    return pretraining_history, desired_trajectories, results


if __name__ == "__main__":
    main()
