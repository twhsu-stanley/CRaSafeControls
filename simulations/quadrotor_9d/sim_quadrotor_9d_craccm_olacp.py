"""Algorithm 1 and CRaCCM for the nine-state quadrotor.

Edit the experiment setup in main(), run the simulation, then select its pickle
in plot_quadrotor_9d_craccm_olacp_results.py to generate PNG or PDF figures.
"""

import copy
from dataclasses import asdict
from datetime import datetime
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import pickle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geodesic_solver import GeodesicSolver
from motion_planner import MotionPlanner
from olacp import OLACP
from systems.quadrotor_9d.quadrotor_9d import (
    QUADROTOR_9D,
    THETA_LOWER,
    THETA_UPPER,
    initial_representation,
)
from systems.quadrotor_9d.nonlinear_uncertainty import (
    ONLINE_WIND_PATTERN,
    WindEnvironment,
    WindSchedule,
    nonlinear_uncertainty,
    pretraining_wind_pattern,
)


def validate_experiment(config, base_params):
    """Validate the experiment before collecting observations or running control."""
    numeric = [
        config["dt"],
        config["interval_duration"],
        config["physical_gamma"],
        config["payload_physical_boost"],
        config["gamma_regularization"],
        config["residual_scale"],
        config["representation_lr"],
        config["rho_divergence_threshold"],
        *config["reference_initial_state"],
    ]
    positive = [
        config["dt"],
        config["interval_duration"],
        config["physical_gamma"],
        config["gamma_regularization"],
        config["residual_scale"],
        config["representation_lr"],
        config["rho_divergence_threshold"],
    ]
    if not np.all(np.isfinite(numeric)) or min(positive) <= 0:
        raise ValueError("Times, gains, noise scale and initial reference must be finite and valid")
    if config["K"] < 1 or config["B"] < 1 or config["payload_physical_boost"] < 0:
        raise ValueError("K and B must be positive and the payload boost must be nonnegative")
    if (
        config["N_cal"] < 30
        or config["I_length"] < 10
        or config["K_pre"] < config["N_cal"]
        or config["K_pre"] % config["B"]
        or min(config["pretraining_seed"], config["theta_seed"]) < 0
    ):
        raise ValueError("Pretraining must fill the CP window and contain complete blocks")
    if not np.isclose(config["I_length"] * config["dt"], config["interval_duration"]):
        raise ValueError("interval_duration must be an integer multiple of dt")
    if base_params["ccm_rate"] <= 0 or not np.isfinite(base_params["ccm_rate"]):
        raise ValueError("ccm_rate must be finite and positive")
    validate_timing(config)


def validate_timing(config):
    """Align simulation steps, MPC input holds, replans and representation updates."""
    dt = config["dt"]
    mpc_dt, replan, horizon = (
        config["mpc_dt"],
        config["mpc_replan_interval"],
        config["mpc_horizon"],
    )
    if not np.all(np.isfinite([mpc_dt, replan, horizon])) or min(mpc_dt, replan, horizon) <= 0:
        raise ValueError("MPC times must be finite and positive")
    hold_steps, replan_steps = int(round(mpc_dt / dt)), int(round(replan / dt))
    horizon_steps = int(round(horizon / mpc_dt))
    if (
        hold_steps < 1
        or replan_steps < 1
        or horizon_steps < 2
        or not np.isclose(hold_steps * dt, mpc_dt)
        or not np.isclose(replan_steps * dt, replan)
        or not np.isclose(horizon_steps * mpc_dt, horizon)
        or replan_steps % hold_steps
        or horizon_steps * hold_steps < replan_steps
    ):
        raise ValueError("MPC times must use whole simulation steps and input holds")
    if config["B"] * config["I_length"] % replan_steps:
        raise ValueError("Representation updates must align with the MPC replanning grid")


def ascending_figure_eight(times, config):
    """Return the original analytic position guide at a fixed lap and climb rate."""
    times = np.atleast_1d(np.asarray(times, dtype=float))
    phase = 2 * np.pi * times / config["spatial_period"]
    amplitude = config["spatial_amplitude"]
    return np.column_stack(
        (
            amplitude[0] * np.sin(phase),
            amplitude[1] * np.sin(2 * phase),
            config["spatial_climb_rate"] * times,
        )
    )


class WaypointReference:
    """Keep the position cost guide separate from the integrated full reference state."""

    def __init__(self, config):
        validate_timing(config)
        self.config = config
        self.interval_duration = config["interval_duration"]
        self.times = self.interval_duration * np.arange(config["K"] + 1)
        self.initial_state = (
            np.asarray(config["reference_initial_state"], dtype=float).reshape(9).copy()
        )
        self.positions = np.asarray(config["nominal_waypoints"], dtype=float).copy()
        if (
            self.positions.shape != (config["K"] + 1, 3)
            or not np.all(np.isfinite(self.positions))
            or not np.all(np.isfinite(self.initial_state))
            or not np.allclose(self.positions[0], self.initial_state[:3])
        ):
            raise ValueError(
                "nominal_waypoints must be finite (K+1,3) positions starting at x_d(0)"
            )
        if config["position_guide"] == "ascending_figure_eight":
            expected = ascending_figure_eight(self.times, config) + self.initial_state[:3]
            if not np.allclose(self.positions, expected):
                raise ValueError("Use position_guide='natural_cubic' for custom position waypoints")
            self.spline = None
        elif config["position_guide"] == "natural_cubic":
            self.spline = CubicSpline(self.times, self.positions, axis=0, bc_type="natural")
        else:
            raise ValueError("position_guide must be 'ascending_figure_eight' or 'natural_cubic'")
        self.mpc_dt = config["mpc_dt"]
        self.horizon = config["mpc_horizon"]
        self.replan_interval = config["mpc_replan_interval"]

    def guide(self, times):
        times = np.clip(np.atleast_1d(times), self.times[0], self.times[-1])
        if self.spline is None:
            return (ascending_figure_eight(times, self.config) + self.initial_state[:3]).T
        return self.spline(times).T


def make_planner(system, config):
    """Build the unchanged position-error and hover-input certainty-equivalent MPC."""
    return MotionPlanner(
        system,
        config["dt"],
        config["motion_planner_Q"],
        config["motion_planner_R"],
        Q_terminal=config["motion_planner_Q_terminal"],
        output_matrix=np.eye(3, 9),
        S=np.zeros((4, 4)),
        u_min=system.params["u_min"],
        u_max=system.params["u_max"],
    )


def rk4_step(system, x, u, dt, t=0.0):
    """Advance the physical plant with one held control input."""

    def dynamics(state, tau):
        return system.dynamics(state, u, tau)

    k1 = dynamics(x, t)
    k2 = dynamics(x + dt * k1 / 2, t + dt / 2)
    k3 = dynamics(x + dt * k2 / 2, t + dt / 2)
    k4 = dynamics(x + dt * k3, t + dt)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def pd_input(system, x, x_d, u_ref):
    """Use nominal position, velocity and attitude feedback to excite the plant."""
    # Position/velocity errors determine the requested inertial acceleration.
    acc = system.dynamics_nominal(x_d, u_ref)[3:6]
    acc += 3 * (x_d[:3] - x[:3]) + 2.5 * (x_d[3:6] - x[3:6])
    force = np.clip(acc, -5, 5) + np.array([0, 0, system.grav])
    thrust = np.linalg.norm(force)
    b3 = force / max(thrust, 1e-9)
    # Choose roll and pitch so that thrust points along the requested force.
    yaw = x_d[8]
    bx = np.cos(yaw) * b3[0] + np.sin(yaw) * b3[1]
    by = -np.sin(yaw) * b3[0] + np.cos(yaw) * b3[1]
    eta = np.array([np.arcsin(np.clip(-by, -1, 1)), np.arctan2(bx, b3[2]), yaw])
    err = eta - x[6:9]
    err[2] = (err[2] + np.pi) % (2 * np.pi) - np.pi
    eta_dot = system.g(x_d)[6:9, 1:] @ np.asarray(u_ref)[1:] + 4 * err
    # Convert the desired Euler rates into body-rate commands.
    return np.r_[thrust, np.linalg.solve(system.g(x)[6:9, 1:], eta_dot)]


def put_interval(learner, system, block, dt):
    """Load one interval and recover w = measured xdot - nominal xdot.

    x, nominal and xdot each have shape (interval_steps, 9). Y(x) uses the
    current Theta so that a_k and its score belong to the same representation.
    """
    learner.clear_buffers()
    for x, nominal, measured in zip(block["x"], block["nominal"], block["xdot"]):
        learner.add_data_to_buffers(x, nominal, system.Y(x), xdot=measured)
    learner.estimate_uncertainty(dt)


def pretraining_intervals(system, config):
    """Yield physical observations one interval at a time under nominal PD.

    The caller learns before requesting the next interval. PD excitation does
    not depend on Theta; measured derivatives come from the physical plant.
    """
    rng = np.random.default_rng(config["pretraining_seed"])
    phases = rng.uniform(0, 2 * np.pi, 3)
    amplitude, frequency = (
        config["pretrain_excitation_amplitude"],
        config["pretrain_angular_frequency"],
    )
    winds = pretraining_wind_pattern(config["B"])
    x = config["pretrain_initial_state"].copy()
    for interval in range(config["K_pre"]):
        wind = winds[interval % len(winds)]
        env = WindEnvironment(
            *wind,
            payload=rng.uniform(*config["pretrain_payload_range"]),
            phase=rng.uniform(0, 2 * np.pi),
            residual_scale=config["residual_scale"],
        )
        system.true_uncertainty_fcn = lambda state, t, e=env: nonlinear_uncertainty(state, t, e)
        block = {key: [] for key in ("x", "u", "nominal", "xdot", "t")}
        for step in range(config["I_length"]):
            t = (interval * config["I_length"] + step) * config["dt"]
            desired = np.zeros(9)
            desired[:3] = amplitude * np.sin(frequency * t + phases)
            desired[3:6] = amplitude * frequency * np.cos(frequency * t + phases)
            u = np.clip(
                pd_input(system, x, desired, [9.81, 0, 0, 0]),
                system.params["u_min"],
                system.params["u_max"],
            )
            nominal = system.dynamics_nominal(x, u)
            measured = system.dynamics(x, u, t)
            for key, value in {
                "x": x.copy(),
                "u": u,
                "nominal": nominal,
                "xdot": measured,
                "t": t,
            }.items():
                block[key].append(value)
            x = rk4_step(system, x, u, config["dt"], t)
            if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 100:
                raise RuntimeError(f"Data collection diverged at t={t + config['dt']}")
        block["environment"] = list(asdict(env).values())
        yield {key: np.asarray(value) for key, value in block.items()}


def run_pretraining(system, olacp, config, data=None):
    """Collect, fit, score and learn in interval order, as in planar_quad.

    Scores retain the Theta used in their own interval. The most recent N_cal
    scores initialize CP; they are never rescored with the final Theta. Saved
    observations can replay the same sequence without collecting a new run.
    Return the learner and a pickle-ready history containing all observations.
    """
    if config["K_pre"] < config["N_cal"] or config["K_pre"] % config["B"]:
        raise ValueError("Pretraining must fill the CP window and complete representation blocks")
    learner = olacp
    system.set_representation(learner.Theta)
    if data is None:
        intervals = pretraining_intervals(system, config)
    else:
        if any((len(values) != config["K_pre"] for values in data.values())):
            raise ValueError("Saved pretraining arrays must contain all pretraining intervals")
        intervals = (
            {key: values[index] for key, values in data.items()} for index in range(config["K_pre"])
        )
    observations = {}
    theta_history, scores, fitted = ([learner.Theta.copy()], [], [])
    for block in intervals:
        for key in ("x", "nominal", "xdot"):
            values = block[key]
            if values.shape != (config["I_length"], 9) or not np.isfinite(values).all():
                raise ValueError(
                    f"Pretraining {key} must be finite with shape ({config['I_length']}, 9)"
                )
        put_interval(learner, system, block, config["dt"])
        score = float(learner.compute_score(system.a_ub, system.a_lb))
        scores.append(score)
        fitted.append(learner.a_k.copy())
        learner.append_score(score)
        update = learner.update_representation()
        if update is not None:
            system.set_representation(update["Theta"])
        theta_history.append(learner.Theta.copy())
        learner.clear_buffers()
        for key, values in block.items():
            observations.setdefault(key, []).append(values)
    if learner._representation_intervals:
        raise RuntimeError("Pretraining must end on a complete representation block")
    if len(scores) != config["K_pre"] or len(learner.S_cal) != config["N_cal"]:
        raise RuntimeError("Pretraining did not fill the calibration window")
    learner.compute_quantile()
    history = {
        "observations": {key: np.asarray(value) for key, value in observations.items()},
        "theta_hist": np.asarray(theta_history),
        "score_hist": np.asarray(scores),
        "a_k_hist": np.asarray(fitted),
        "quantile": learner.Q_k,
        "calibration_scores": np.asarray(learner.S_cal).copy(),
        "delta": float(learner.delta),
        "interval_index": learner.interval_index,
        "representation_update_index": learner.representation_update_index,
        "theta_lower": np.asarray(learner.Theta_lb).copy(),
        "theta_upper": np.asarray(learner.Theta_ub).copy(),
        "metric_source_sha256": system.metric.metadata["source_sha256"],
        "metric": system.metric.metadata,
        "wind_schedule": WindSchedule(
            pretraining_wind_pattern(config["B"]), config["interval_duration"]
        ).metadata(),
        "core_hashes": core_hashes(),
        "repository": repository_state(),
    }
    print(f"Pretraining complete: Q_0={learner.Q_k:.6g}")
    return (learner, history)


def normalized_gamma(theta, physical_gain, regularization):
    """Build Gamma_0 = gain * (Theta.T @ Theta + regularization * I)^(-1).

    This reduces scaling differences in the map b = Theta @ a_hat. The
    regularizer keeps the inverse well-defined for nearly dependent columns.
    """
    gram = np.asarray(theta).T @ np.asarray(theta) + regularization * np.eye(4)
    gamma = physical_gain * np.linalg.inv(gram)
    gamma = 0.5 * (gamma + gamma.T)
    np.linalg.cholesky(gamma)
    return gamma


def payload_boosted_gamma(theta, physical_gain, regularization, boost):
    """Add gain along the a_hat direction that changes payload coefficient b[2].

    With d = theta[2], the rank-one addition is boost * d d.T / ||d||^4,
    increasing d.T @ Gamma @ d by exactly boost. This is a tuning choice.
    """
    if boost < 0:
        raise ValueError("payload boost must be nonnegative")
    gamma = normalized_gamma(theta, physical_gain, regularization)
    direction = np.asarray(theta)[2]
    norm_squared = float(direction @ direction)
    if norm_squared <= np.finfo(float).eps:
        raise ValueError("Theta has no payload direction")
    gamma += boost * np.outer(direction, direction) / norm_squared**2
    gamma = 0.5 * (gamma + gamma.T)
    np.linalg.cholesky(gamma)
    return gamma


def finish_interval(learner, system, config):
    """Finish one online interval and return its score/update record.

    Order: fit a_k -> assess coverage -> append score -> possibly update Theta
    -> compute the next quantile. a_k is separate from the live adaptive a_hat.
    """
    learner.estimate_uncertainty(config["dt"])
    score = float(learner.compute_score(system.a_ub, system.a_lb))
    previous_quantile = learner.Q_k
    missed = learner.update_delta(score)
    learner.append_score(score)
    update = learner.update_representation() if config.get("online_theta", True) else None
    if update is not None:
        system.set_representation(update["Theta"])
    system.cp_quantile = learner.compute_quantile()
    event = dict(
        score=score,
        previous_quantile=previous_quantile,
        next_quantile=system.cp_quantile,
        miscoverage=missed,
        delta=float(learner.delta),
        a_fit=learner.a_k.copy(),
        representation_updated=update is not None,
    )
    learner.clear_buffers()
    return event


def clone_learner(pretrained, system):
    """Keep Theta, scores, delta and counters independent in each comparison arm.

    Rebind callbacks to the new system instead of copying the old model/metric.
    Only sample buffers are cleared; the pretraining state is preserved.
    """
    learner = copy.copy(pretrained)
    learner.__dict__ = copy.deepcopy(
        {
            key: value
            for key, value in vars(pretrained).items()
            if key not in ("Y_Theta", "representation_loss_gradient")
        }
    )
    learner.clear_buffers()
    if learner._representation_intervals:
        raise ValueError("Pretraining must end at a complete representation block")
    learner.Y_Theta = system.Y_Theta
    learner.representation_loss_gradient = system.representation_loss_gradient
    if learner.Theta is not None:
        system.set_representation(learner.Theta)
    system.cp_quantile = learner.compute_quantile()
    return learner


def compute_control(system, x, a, xd, ud, solver):
    """Return actual input u(4), scalar slack and a dictionary of diagnostics.

    The shared ctrl_craccm() solves the feedback QP. Remaining calculations
    record its energy, CP allowance, constraint residual and correction u - ud.
    """
    ud = np.asarray(ud).reshape(system.udim)
    u, slack = system.ctrl_craccm(x, a, xd, ud, solver, use_qpsolvers=True, use_slack=True)
    u = np.asarray(u).reshape(system.udim)
    # W = M^-1. Endpoint geodesic tangents give the tracking-energy variation.
    end = np.linalg.solve(system.W_fcn(x, a), system.gamma_s[:, -1])
    start = np.linalg.solve(system.W_fcn(xd, a), system.gamma_s[:, 0])
    Yx = system.Y(x) @ a if system.use_adaptive else np.zeros(system.xdim)
    Yd = system.Y(xd) @ a if system.use_adaptive else np.zeros(system.xdim)
    authority = end @ system.g(x)
    authority_norm = float(np.linalg.norm(authority))
    # Match the shared controller's CP attenuation near vanishing input authority.
    ratio = min(authority_norm / system.ccm_authority_threshold, 1.0)
    scale = ratio**3 * (10.0 - 15.0 * ratio + 6.0 * ratio**2)
    if system.use_cp:
        tightening = float(scale * np.linalg.norm(end) * system.cp_quantile)
    else:
        tightening = 0.0
    residual = (
        end @ (system.dynamics_nominal(x, u) + Yx)
        - start @ (system.dynamics_nominal(xd, ud) + Yd)
        + system.ccm_rate * system.Erem
        + tightening
        - slack
    )
    return (
        u,
        float(slack),
        dict(
            energy=system.Erem,
            tightening=tightening,
            constraint_residual=float(residual),
            correction_norm=float(np.linalg.norm(u - ud)),
            qp_correction=(u - ud).copy(),
        ),
    )


def run_craccm_simulation(system, online_olacp, config, use_cp, use_adaptive, label=None):
    """Run one independent arm using planar_quad's reference/update sequence.

    adaptive controls continuous a_hat/rho adaptation; cp controls tightening.
    Online Theta learning is independently controlled by config['online_theta'].
    Gamma is fixed from pretraining even while online Theta changes.
    Return a result dictionary containing histories, diagnostics and termination status.
    """
    if bool(system.use_cp) != bool(use_cp) or bool(system.use_adaptive) != bool(use_adaptive):
        raise ValueError("The system must have the requested CP and adaptation flags")
    adaptive = use_adaptive
    reference = WaypointReference(config)
    plant = WindSchedule(
        ONLINE_WIND_PATTERN,
        config["interval_duration"],
        WindEnvironment(residual_scale=config["residual_scale"]),
    )
    system.true_uncertainty_fcn = plant
    learner = online_olacp
    planner = make_planner(system, config)
    geodesic = GeodesicSolver(
        9,
        config["geodesic_degree"],
        config["geodesic_nodes"],
        system.W_fcn,
        system.dW_dxi_fcn,
        system.dW_dai_fcn,
    )
    times = (
        np.arange(round(config["K"] * config["interval_duration"] / config["dt"]) + 1)
        * config["dt"]
    )

    def rho_event(t, state):
        """Stop this case when the adaptive scaling state reaches its magnitude limit."""
        return config["rho_divergence_threshold"] - abs(float(state[13]))

    rho_event.terminal, rho_event.direction = (True, -1.0)
    state = np.r_[
        reference.initial_state + config["tracking_initial_offset"],
        system.a_center,
        0.0,
        reference.initial_state,
    ]
    ud = np.array([system.grav, 0, 0, 0])
    controls, plan_start_step, solve_seconds = (None, 0, [])
    rows, events, outcome = ([], [], dict(status="completed"))
    begin = time.perf_counter()
    for k, t in enumerate(times):
        x, a, rho, xd = (state[:9], state[9:13], float(state[13]), state[14:])
        reference_a = a.copy() if adaptive else np.zeros(4)
        replanned = False
        if k < len(times) - 1:
            if k % int(round(config["mpc_replan_interval"] / config["dt"])) == 0:
                start = time.perf_counter()
                _, controls = planner.plan_certainty_equivalent_through_waypoints(
                    xd,
                    reference.positions,
                    config["I_length"],
                    k,
                    system.Theta_hat,
                    reference_a,
                    np.r_[system.grav, np.zeros(3)],
                    mpc_horizon=reference.horizon,
                    mpc_dt=reference.mpc_dt,
                    output_reference=reference.guide,
                    u_previous=ud,
                )
                solve_seconds.append(time.perf_counter() - start)
                plan_start_step, replanned = (k, True)
            index = (k - plan_start_step) // int(round(config["mpc_dt"] / config["dt"]))
            ud = controls[:, index].copy()
        u, slack, diagnostics = compute_control(system, x, a, xd, ud, geodesic)
        domain = [system.metric_in_training_domain(node, a) for node in system.gamma.T]
        rows.append(
            dict(
                t=t,
                x=x.copy(),
                a=a.copy(),
                rho=rho,
                x_d=xd.copy(),
                u=u.copy(),
                u_d=ud.copy(),
                b=system.Theta_hat @ a,
                theta=system.Theta_hat.copy(),
                reference_a=reference_a,
                quantile=system.cp_quantile,
                slack=slack,
                mpc_replanned=replanned,
                in_domain=(
                    np.nan if any((value is None for value in domain)) else float(all(domain))
                ),
                **diagnostics,
            )
        )
        if k == len(times) - 1:
            break
        learner.add_data_to_buffers(
            x, system.dynamics_nominal(x, u), system.Y(x), xdot=system.dynamics(x, u, t)
        )

        def dynamics(tau, current):
            """Return derivatives of the 23 coupled states for the RK45 integrator."""
            adaptive_state, desired = (current[:14], current[14:])
            model_a = adaptive_state[9:13] if adaptive else np.zeros(4)
            return np.r_[
                system.dynamics_extended(adaptive_state, desired, u, geodesic, tau),
                system.dynamics_certainty_equivalent(desired, ud, model_a),
            ]

        solution = solve_ivp(
            dynamics,
            (t, times[k + 1]),
            state,
            method="RK45",
            rtol=1e-07,
            atol=1e-09,
            t_eval=[times[k + 1]],
            events=rho_event if adaptive else None,
        )
        if adaptive and len(solution.t_events[0]):
            outcome = dict(
                status="diverged",
                failure_reason="rho_divergence",
                failure_time=float(solution.t_events[0][-1]),
                failure_state=solution.y_events[0][-1].copy(),
            )
            break
        if not solution.success:
            outcome = dict(
                status="failed",
                failure_reason="integration_failure",
                failure_time=float(t),
                failure_message=solution.message,
            )
            break
        state = solution.y[:, -1]
        if (
            not np.all(np.isfinite(state))
            or abs(state[13]) > 1000000.0
            or np.linalg.norm(state[:9]) > 100
        ):
            outcome = dict(
                status="diverged",
                failure_reason="state_divergence",
                failure_time=float(times[k + 1]),
                failure_state=state.copy(),
            )
            break
        if (k + 1) % config["I_length"] == 0:
            event = finish_interval(learner, system, config)
            events.append(dict(event, time=float(times[k + 1])))
        if not config["quiet"] and (k + 1) % max(1, round(0.5 / config["dt"])) == 0:
            print(
                f"t={times[k + 1]:.2f}, reference error={np.linalg.norm(state[:3] - state[14:17]):.4f}",
                flush=True,
            )
    result = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    outcome.update(mpc_replans=len(solve_seconds), wall_seconds=time.perf_counter() - begin)
    result["target"] = reference.guide(result["t"]).T
    result["true_w"] = np.asarray([plant(x, t) for x, t in zip(result["x"], result["t"])])
    result["wind"] = np.asarray([plant.wind(t) for t in result["t"]])
    metrics = dict(
        summarize(result, events, system, config["I_length"]),
        **outcome,
        label=label,
        cp=bool(use_cp),
        adaptive=bool(use_adaptive),
    )
    desired_trajectory = {
        "mode": "certainty_equivalent_dynamics_mpc",
        "t_x": result["t"].copy(),
        "t_u": result["t"][:-1].copy(),
        "x_d": result["x_d"].T.copy(),
        "u_d": result["u_d"][:-1].T.copy(),
        "waypoints": reference.positions.copy(),
        "waypoint_times": reference.times.copy(),
        "mpc_horizon": reference.horizon,
        "mpc_dt": reference.mpc_dt,
        "mpc_replan_interval": reference.replan_interval,
    }
    history = {
        "label": label,
        "use_cp": bool(use_cp),
        "use_adaptive": bool(use_adaptive),
        "t": result["t"],
        "x_hist": result["x"],
        "x_d_hist": result["x_d"],
        "u_hist": result["u"],
        "u_d_hist": result["u_d"],
        "a_hat_hist": result["a"],
        "rho_hist": result["rho"],
        "theta_hist": result["theta"],
        "b_hist": result["b"],
        "reference_a_hist": result["reference_a"],
        "quantile_hist": result["quantile"],
        "slack_hist": result["slack"],
        "energy_hist": result["energy"],
        "tightening_hist": result["tightening"],
        "constraint_residual_hist": result["constraint_residual"],
        "correction_norm_hist": result["correction_norm"],
        "qp_correction_hist": result["qp_correction"],
        "mpc_replan_hist": result["mpc_replanned"],
        "in_domain_hist": result["in_domain"],
        "position_guide_hist": result["target"],
        "true_uncertainty_hist": result["true_w"],
        "wind_hist": result["wind"],
        "a_k_hist": result["a_k"],
        "fitted_uncertainty_hist": result["fitted_uncertainty"],
        "interval_events": events,
        "interval_times": np.asarray([event["time"] for event in events]),
        "score_hist": np.asarray([event["score"] for event in events]),
        "delta_hist": np.asarray([event["delta"] for event in events]),
        "miscoverage_hist": np.asarray([event["miscoverage"] for event in events]),
        "desired_trajectory": desired_trajectory,
        "uncertainty_schedule_values": plant.values.T.copy(),
        "metrics": metrics,
        "final_theta": learner.Theta.copy(),
        **outcome,
    }
    print(
        f"{label}: status={outcome['status']}, position RMS={metrics['position_rmse']:.6g}, max slack={metrics['max_slack']:.6g}, MPC solves={outcome['mpc_replans']}"
    )
    return history


def core_hashes():
    """Fingerprint shared controller, learning, planning and geodesic source files."""
    names = [
        "systems/control_affine_system.py",
        "olacp.py",
        "motion_planner.py",
        "geodesic_solver.py",
    ]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def repository_state():
    """Record the local Git branch and commit, or None when Git is unavailable."""

    def git(*args):
        """Read one property of the local repository."""
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return dict(branch=git("branch", "--show-current"), commit=git("rev-parse", "HEAD"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return dict(branch=None, commit=None)  # Source ZIPs need no Git installation.


def interval_diagnostics(result, events, system, interval_steps):
    """Reconstruct interval fits and compare both estimates with true uncertainty.

    Return fitted arrays, residual RMSEs and the fraction of interval scores
    covered by the quantile that was available before each interval ended.
    """
    fitted = np.full_like(result["true_w"], np.nan)
    a_k = np.full((len(result["t"]), system.adim), np.nan)
    fit_errors = []
    for index, event in enumerate(events):
        first = index * interval_steps
        last = min(first + interval_steps, len(result["t"]) - 1)
        parameter = np.asarray(event["a_fit"])
        for k in range(first, last):
            # Use the Theta active at this sample, not the final learned matrix.
            estimate = system.Y_Theta(result["x"][k], result["theta"][k]) @ parameter
            fitted[k], a_k[k] = estimate, parameter
            fit_errors.append(np.linalg.norm(estimate - result["true_w"][k]))
    scores = np.asarray([event["score"] for event in events])
    quantiles = np.asarray([event["previous_quantile"] for event in events])
    # a_fit minimizes an interval loss; result['a'] is the live adaptive estimate.
    controller_fit = np.asarray(
        [
            system.Y_Theta(x, theta) @ a
            for x, theta, a in zip(result["x"], result["theta"], result["a"])
        ]
    )
    return {
        "a_k": a_k,
        "fitted_uncertainty": fitted,
        "interval_coverage": float(np.mean(scores <= quantiles)) if len(scores) else None,
        "fit_residual_rmse": float(np.sqrt(np.mean(np.square(fit_errors)))) if fit_errors else None,
        "controller_residual_rmse": float(
            np.sqrt(np.mean(np.linalg.norm(controller_fit - result["true_w"], axis=1) ** 2))
        ),
    }


def summarize(result, events, system, interval_steps):
    """Add fitted arrays to result and return scalar metrics for one case."""
    # Measure both tracking of the integrated x_d and error against the common guide.
    reference_error = np.linalg.norm(result["x"][:, :3] - result["x_d"][:, :3], axis=1)
    target_error = np.linalg.norm(result["x"][:, :3] - result["target"][:, :3], axis=1)
    fits = interval_diagnostics(result, events, system, interval_steps)
    result["a_k"] = fits.pop("a_k")
    result["fitted_uncertainty"] = fits.pop("fitted_uncertainty")
    return dict(
        reference_model="estimated_mpc",
        reference_integration="coupled_RK45_live_estimate",
        controller="ControlAffineSystem.ctrl_craccm",
        qp_backend="quadprog",
        replanning=True,
        controller_input_bounds=False,
        recorded_duration=float(result["t"][-1]),
        sample_count=len(result["t"]),
        position_rmse=float(np.sqrt(np.mean(reference_error**2))),
        final_position_error=float(reference_error[-1]),
        target_rmse=float(np.sqrt(np.mean(target_error**2))),
        final_target_error=float(target_error[-1]),
        max_slack=float(result["slack"].max()),
        max_abs_rho=float(np.abs(result["rho"]).max()),
        max_constraint_residual=float(result["constraint_residual"].max()),
        training_domain_fraction=(
            float(result["in_domain"].mean()) if np.isfinite(result["in_domain"]).all() else None
        ),
        completed_intervals=len(events),
        representation_updates=sum(event["representation_updated"] for event in events),
        **fits,
    )


def common_prefix_comparison(results):
    """Compare the shared time prefix when some arms terminate early."""
    count = min((len(result["t"]) for result in results.values()))
    comparison = dict(
        sample_count=count, end_time=float(next(iter(results.values()))["t"][count - 1])
    )
    for name, reference in (
        ("position_rmse", "x_d_hist"),
        ("target_position_rmse", "position_guide_hist"),
    ):
        comparison[name] = {
            label: float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (result["x_hist"][:count, :3] - result[reference][:count, :3]) ** 2,
                            axis=1,
                        )
                    )
                )
            )
            for label, result in results.items()
        }
    return comparison


def plot_craccm_results(
    results, desired_trajectory=None, file_format="png", results_directory=None
):
    """Plot saved histories without rerunning MPC, learning or the controller."""
    if file_format not in ("png", "pdf"):
        raise ValueError('file_format must be "png" or "pdf"')
    if not results:
        raise ValueError("At least one controller result is required")
    first_result = max(results.values(), key=lambda result: len(result["t"]))
    if desired_trajectory is None:
        trajectory = first_result["desired_trajectory"]
    elif "x_d" in desired_trajectory:
        trajectory = desired_trajectory
    else:
        trajectory = desired_trajectory[first_result["label"]]
    reference = {"times": trajectory["waypoint_times"], "positions": trajectory["waypoints"]}
    results_directory = (
        Path(__file__).resolve().parent / "results"
        if results_directory is None
        else Path(results_directory)
    )
    results_directory.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    path = results_directory / f"quadrotor_9d_tracking.{file_format}"
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 2)
    axes = [fig.add_subplot(grid[j, 0]) for j in range(3)]
    path3d = fig.add_subplot(grid[0, 1], projection="3d")
    error = fig.add_subplot(grid[1, 1])
    effort = fig.add_subplot(grid[2, 1])
    first = max(results.values(), key=lambda result: len(result["t"]))
    for label, result in results.items():
        for j, ax in enumerate(axes):
            ax.plot(result["t"], result["x_hist"][:, j], label=label)
        path3d.plot(*result["x_hist"][:, :3].T, label=label)
        error.plot(
            result["t"],
            np.linalg.norm(result["x_hist"][:, :3] - result["x_d_hist"][:, :3], axis=1),
            label=label,
        )
        effort.plot(result["t"], result["correction_norm_hist"], label=label)
    for j, ax in enumerate(axes):
        ax.plot(first["t"], first["position_guide_hist"][:, j], ":", label="position cost guide")
        ax.set_ylabel(f"{'xyz'[j]} [m]")
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Time [s]")
    path3d.plot(*first["position_guide_hist"][:, :3].T, "k--", label="position cost guide")
    path3d.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    error.set(ylabel="Own reference error [m]", xlabel="Time [s]")
    effort.set(ylabel="Control correction norm", xlabel="Time [s]")
    for ax in (error, effort):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.set_label("quadrotor_9d_tracking")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(fig)
    path = results_directory / f"quadrotor_9d_control.{file_format}"
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 9), constrained_layout=True)
    names = [
        "Thrust acceleration [m/s$^2$]",
        "Body rate $\\omega_x$ [rad/s]",
        "Body rate $\\omega_y$ [rad/s]",
        "Body rate $\\omega_z$ [rad/s]",
    ]
    for label, result in results.items():
        for j, ax in enumerate(axes):
            ax.plot(result["t"], result["u_hist"][:, j], label=label, linewidth=1.0)
    for ax, name in zip(axes, names):
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.set_label("quadrotor_9d_control")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(fig)
    path = results_directory / f"quadrotor_9d_reference.{file_format}"
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7), constrained_layout=True)
    reference_data = max(results.values(), key=lambda data: len(data["t"]))
    times = reference_data["t"]
    guide = reference_data["position_guide_hist"].T
    for i, ax in enumerate(axes):
        ax.plot(times, guide[i], "k:", label="position cost guide")
        ax.plot(
            reference["times"], reference["positions"][:, i], "ko", markersize=3, label="waypoints"
        )
        for label, data in results.items():
            ax.plot(data["t"], data["x_d_hist"][:, i], label=label)
        ax.set_ylabel(f"{'xyz'[i]} [m]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Time [s]")
    fig.set_label("quadrotor_9d_reference")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(fig)
    path = results_directory / f"quadrotor_9d_wind.{file_format}"
    data = max(results.values(), key=lambda value: len(value["t"]))
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for index, axis in enumerate("xyz"):
        ax.step(data["t"], data["wind_hist"][:, index], where="post", label=f"wind {axis}")
    ax.set(xlabel="Time [s]", ylabel="Inertial wind [m/s]", title="Prescribed interval wind")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.set_label("quadrotor_9d_wind")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(fig)
    print(f"Saved {len(figures)} {file_format.upper()} figures to {results_directory}")
    return figures


def main():
    """Edit this setup, pretrain the uncertainty model and run the four controllers."""
    K_pre = 64
    N_cal = 32
    K = 20
    B = 4
    dt = 0.02
    interval_duration = 2.0
    I_length = int(round(interval_duration / dt))

    config = {
        "K_pre": K_pre,
        "N_cal": N_cal,
        "K": K,
        "B": B,
        "dt": dt,
        "interval_duration": interval_duration,
        "I_length": I_length,
        "pretraining_seed": 20260930,
        "theta_seed": 42,
        "representation_lr": 1e-4,
        "acp_lr": 0.01,
        "delta_target": 0.1,
        "delta_init": 0.1,
        "pretrain_initial_state": np.zeros(QUADROTOR_9D.xdim),
        "pretrain_excitation_amplitude": np.array([1.2, 1.0, 0.25]),
        "pretrain_angular_frequency": np.array([0.8, 0.7, 0.31]),
        "pretrain_payload_range": (0.07, 0.15),
        "reference_initial_state": np.zeros(QUADROTOR_9D.xdim),
        "tracking_initial_offset": np.zeros(QUADROTOR_9D.xdim),
        # Preserve the analytic guide; use "natural_cubic" for custom waypoints.
        "position_guide": "ascending_figure_eight",
        "spatial_period": 12.0,
        "spatial_amplitude": np.array([0.45, 0.30]),
        "spatial_climb_rate": 0.03,
        "motion_planner_Q": 8 * np.eye(3),
        "motion_planner_R": 0.03 * np.eye(QUADROTOR_9D.udim),
        "motion_planner_Q_terminal": 60 * np.eye(3),
        "mpc_horizon": 0.8,
        "mpc_dt": 0.02,
        "mpc_replan_interval": 0.4,
        "online_theta": True,
        "physical_gamma": 3e-5,
        "payload_physical_boost": 9.7e-4,
        "gamma_regularization": 1e-5,
        "residual_scale": 1.0,
        "rho_divergence_threshold": 1e4,
        "geodesic_degree": 2,
        "geodesic_nodes": 4,
        "quiet": False,
        # Optionally reuse observations from a previous results pickle.
        "pretraining_filename": None,
    }
    nominal_waypoints = (
        ascending_figure_eight(interval_duration * np.arange(K + 1), config)
        + config["reference_initial_state"][:3]
    )
    config["nominal_waypoints"] = nominal_waypoints

    saved_pretraining = None
    if config["pretraining_filename"] is not None:
        with Path(config["pretraining_filename"]).open("rb") as stream:
            saved = pickle.load(stream)
        for key in ("dt", "I_length", "residual_scale"):
            if not np.isclose(config[key], saved["config"][key]):
                raise ValueError(f"Saved pretraining uses a different {key}")
        for key in ("K_pre", "N_cal", "B", "representation_lr", "pretraining_seed", "theta_seed"):
            config[key] = saved["config"][key]
        saved_pretraining = saved["pretraining_history"]

    theta_lb, theta_ub = THETA_LOWER.copy(), THETA_UPPER.copy()
    theta_init = initial_representation(config["theta_seed"])
    a_lb, a_ub = -np.ones(QUADROTOR_9D.adim), np.ones(QUADROTOR_9D.adim)
    projection_epsilon = 0.01
    base_params = {
        "Theta_init": theta_init.copy(),
        "g": 9.81,
        "a_lb": a_lb,
        "a_ub": a_ub,
        "epsilon": projection_epsilon,
        "a_hat_norm_max": 0.5 * np.linalg.norm(a_ub - a_lb) + projection_epsilon,
        "eta_ccm": 500.0,
        "ccm_rate": 0.5,
        "weight_slack": 10000.0,
        "ccm_authority_threshold": 0.01,
        "u_min": np.array([0, -2, -2, -2]),
        "u_max": np.array([20, 2, 2, 2]),
        "dt": dt,
    }
    validate_experiment(config, base_params)
    WaypointReference(config)
    pretrain_params = dict(base_params, use_cp=False, use_adaptive=False)
    pretrain_system = QUADROTOR_9D(pretrain_params)
    olacp = OLACP(
        [],
        N_cal=config["N_cal"],
        acp_lr=config["acp_lr"],
        delta_target=config["delta_target"],
        delta_init=config["delta_init"],
        buffer_maxlen=config["I_length"],
        Theta_init=theta_init,
        Theta_lb=theta_lb,
        Theta_ub=theta_ub,
        representation_period=config["B"],
        representation_lr=config["representation_lr"],
        Y_Theta=pretrain_system.Y_Theta,
        representation_loss_gradient=pretrain_system.representation_loss_gradient,
    )
    observations = None
    if saved_pretraining is not None:
        if (
            saved_pretraining["metric_source_sha256"]
            != pretrain_system.metric.metadata["source_sha256"]
        ):
            raise ValueError("Saved pretraining used a different neural metric")
        if not np.array_equal(saved_pretraining["theta_lower"], theta_lb) or not np.array_equal(
            saved_pretraining["theta_upper"], theta_ub
        ):
            raise ValueError("Saved pretraining used a different Theta box")
        expected_wind = WindSchedule(
            pretraining_wind_pattern(config["B"]), config["interval_duration"]
        ).metadata()
        if saved_pretraining["wind_schedule"] != expected_wind:
            raise ValueError("Saved pretraining uses a different wind schedule")
        observations = saved_pretraining["observations"]
    pretrained_olacp, pretraining_history = run_pretraining(
        pretrain_system, olacp, config, data=observations
    )
    pretrained_theta = pretrained_olacp.Theta.copy()
    base_params["Gamma_ccm"] = payload_boosted_gamma(
        pretrained_theta,
        config["physical_gamma"],
        config["gamma_regularization"],
        config["payload_physical_boost"],
    )
    settings = (
        ("Proposed", True, True),
        ("No CP", False, True),
        ("Yes CP + no adaptation", True, False),
        ("No CP + no adaptation", False, False),
    )
    results = {}
    for label, use_cp, use_adaptive in settings:
        run_params = dict(
            base_params,
            Theta_init=pretrained_theta.copy(),
            use_cp=use_cp,
            use_adaptive=use_adaptive,
        )
        run_system = QUADROTOR_9D(run_params)
        run_olacp = clone_learner(pretrained_olacp, run_system)
        results[label] = run_craccm_simulation(
            run_system, run_olacp, config, use_cp, use_adaptive, label=label
        )
    if not np.array_equal(pretrained_olacp.Theta, pretrained_theta):
        raise RuntimeError("A controller run mutated the pretrained representation")
    if not np.array_equal(pretrained_olacp.S_cal, pretraining_history["calibration_scores"]):
        raise RuntimeError("A controller run mutated the pretrained calibration scores")

    desired_trajectories = {
        label: result["desired_trajectory"] for label, result in results.items()
    }
    timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S_%f")
    results_directory = Path(__file__).resolve().parent / "results"
    results_directory.mkdir(exist_ok=True)
    results_filename = results_directory / f"sim_quadrotor_9d_craccm_results_{timestamp}.pkl"
    with results_filename.open("xb") as stream:
        pickle.dump(
            {
                "config": config,
                "base_params": base_params,
                "desired_trajectories": desired_trajectories,
                "results": results,
                "pretraining_history": pretraining_history,
                "comparison": common_prefix_comparison(results),
            },
            stream,
        )
    print(f"Saved results to {results_filename}")
    return pretraining_history, desired_trajectories, results


if __name__ == "__main__":
    main()
