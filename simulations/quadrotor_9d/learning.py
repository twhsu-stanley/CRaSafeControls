"""Collect observations, learn the uncertainty model and update conformal scores.

run_example.py calls prepare_from_scratch() before simulation; simulation.py
calls clone_learner() once per case and finish_interval() at interval boundaries.

Names: Theta is the shared 5x4 representation, a_k is a fitted interval parameter,
and a_hat is the continuously adapted estimate maintained by the simulation.
CP (conformal prediction) uses residual scores to set an uncertainty allowance.
Default pretraining spans 64 x 2 s = 128 s of separate PD-driven simulation.
The online clock then restarts at zero with learned Theta and CP scores;
a_hat and rho restart at zero, and the saved CCM weights remain fixed.
"""


import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from olacp import OLACP
from systems.quadrotor_9d.quadrotor_9d import THETA_LOWER, THETA_UPPER, initial_representation
from systems.quadrotor_9d.nonlinear_uncertainty import (
    WindEnvironment, WindSchedule, nonlinear_uncertainty, pretraining_wind_pattern,
)
from .results import dump_json, core_hashes, repository_state


# Bounds for the four coefficients fitted independently in each observation interval.
A_LOWER, A_UPPER = -np.ones(4), np.ones(4)


def rk4_step(system, x, u, dt, t=0.):
    """Advance the physical plant with one held control input."""
    def dynamics(state, tau):
        return system.dynamics(state, u, tau)
    k1 = dynamics(x, t)
    k2 = dynamics(x+dt*k1/2, t+dt/2)
    k3 = dynamics(x+dt*k2/2, t+dt/2)
    k4 = dynamics(x+dt*k3, t+dt)
    return x+dt*(k1+2*k2+2*k3+k4)/6


def pd_input(system, x, x_d, u_ref):
    """Use nominal position, velocity and attitude feedback to excite the plant."""
    # Position/velocity errors determine the requested inertial acceleration.
    acc = system.dynamics_nominal(x_d, u_ref)[3:6]
    acc += 3*(x_d[:3]-x[:3])+2.5*(x_d[3:6]-x[3:6])
    force = np.clip(acc, -5, 5)+np.array([0, 0, system.grav])
    thrust = np.linalg.norm(force)
    b3 = force/max(thrust, 1e-9)
    # Choose roll and pitch so that thrust points along the requested force.
    yaw = x_d[8]
    bx = np.cos(yaw)*b3[0]+np.sin(yaw)*b3[1]
    by = -np.sin(yaw)*b3[0]+np.cos(yaw)*b3[1]
    eta = np.array([np.arcsin(np.clip(-by, -1, 1)), np.arctan2(bx, b3[2]), yaw])
    err = eta-x[6:9]
    err[2] = (err[2]+np.pi) % (2*np.pi)-np.pi
    eta_dot = system.g(x_d)[6:9, 1:] @ np.asarray(u_ref)[1:]+4*err
    # Convert the desired Euler rates into body-rate commands.
    return np.r_[thrust, np.linalg.solve(system.g(x)[6:9, 1:], eta_dot)]


def make_learner(system, args, learning_rate):
    """Fit a_k per interval and learn Theta per block within a fixed box.

    The learning rate updates Theta; Gamma separately controls continuous a_hat
    adaptation. N_cal is the number of recent interval scores kept for CP.
    """
    return OLACP(
        [], N_cal=args.calibration_intervals, acp_lr=args.acp_lr,
        delta_target=args.delta, delta_init=args.delta,
        buffer_maxlen=args.interval_steps,
        Theta_init=system.Theta_hat.copy(), Theta_lb=THETA_LOWER,
        Theta_ub=THETA_UPPER, representation_period=args.representation_period,
        representation_lr=learning_rate, Y_Theta=system.Y_Theta,
        representation_loss_gradient=system.representation_loss_gradient,
    )


def put_interval(learner, system, block, dt):
    """Load one interval and recover w = measured xdot - nominal xdot.

    x, nominal and xdot each have shape (interval_steps, 9). Y(x) uses the
    current Theta so that a_k and its score belong to the same representation.
    """
    learner.clear_buffers()
    for x, nominal, measured in zip(block["x"], block["nominal"], block["xdot"]):
        learner.add_data_to_buffers(x, nominal, system.Y(x), xdot=measured)
    learner.estimate_uncertainty(dt)


def pretraining_intervals(system, args):
    """Yield physical observations one interval at a time under nominal PD.

    The caller learns before requesting the next interval. PD excitation does
    not depend on Theta; measured derivatives come from the physical plant.
    """
    rng = np.random.default_rng(args.seed)
    # Smooth position motion supplies excitation for fitting the uncertainty model.
    phases = rng.uniform(0, 2*np.pi, 3)
    amplitude, frequency = np.array([1.2, 1., .25]), np.array([.8, .7, .31])
    winds = pretraining_wind_pattern(args.representation_period)
    x = np.zeros(9)
    for interval in range(args.pretrain_intervals):
        # One wind cycle spans one Theta-learning block, as in planar pretraining.
        wind = winds[interval % len(winds)]
        env = WindEnvironment(*wind, payload=rng.uniform(.07, .15),
                              phase=rng.uniform(0, 2*np.pi),
                              residual_scale=args.residual_scale)
        system.true_uncertainty_fcn = lambda state, t, e=env: nonlinear_uncertainty(state, t, e)
        # Keep both nominal and actual derivatives so the residual can be reconstructed.
        block = {key: [] for key in ("x", "u", "nominal", "xdot", "t")}
        for step in range(args.interval_steps):
            t = (interval*args.interval_steps+step)*args.dt
            desired = np.zeros(9)
            desired[:3] = amplitude*np.sin(frequency*t+phases)
            desired[3:6] = amplitude*frequency*np.cos(frequency*t+phases)
            u = np.clip(pd_input(system, x, desired, [9.81, 0, 0, 0]),
                        args.u_min, args.u_max)
            nominal = system.dynamics_nominal(x, u)
            measured = system.dynamics(x, u, t)
            for key, value in {"x": x.copy(), "u": u, "nominal": nominal,
                               "xdot": measured, "t": t}.items():
                block[key].append(value)
            x = rk4_step(system, x, u, args.dt, t)
            if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 100:
                raise RuntimeError(f"Data collection diverged at t={t+args.dt}")
        block["environment"] = list(asdict(env).values())
        yield {key: np.asarray(value) for key, value in block.items()}


def run_pretraining(system, args, data=None):
    """Collect, fit, score and learn in interval order, as in planar_quad.

    Scores retain the Theta used in their own interval. The most recent N_cal
    scores initialize CP; they are never rescored with the final Theta. Saved
    observations can replay the same sequence without collecting a new run.
    Return the learner, observations and history; theta[k] precedes score[k].
    """
    if (args.pretrain_intervals < args.calibration_intervals
            or args.pretrain_intervals % args.representation_period):
        raise ValueError("Pretraining must fill the CP window and complete representation blocks")
    # Initialize within the same fixed box later used to clip Theta updates.
    theta_initial = initial_representation(getattr(args, "theta_seed", 42))
    system.set_representation(theta_initial)
    learner = make_learner(system, args, args.learning_rate)
    if data is None:
        intervals = pretraining_intervals(system, args)
    else:
        if any(len(values) != args.pretrain_intervals for values in data.values()):
            raise ValueError("Saved pretraining arrays must contain all pretraining intervals")
        intervals = ({key: values[index] for key, values in data.items()}
                     for index in range(args.pretrain_intervals))
    # Histories retain every interval; the learner's CP window keeps only recent scores.
    observations = {}
    theta_history, scores, fitted = [learner.Theta.copy()], [], []
    for block in intervals:
        for key in ("x", "nominal", "xdot"):
            values = block[key]
            if values.shape != (args.interval_steps, 9) or not np.isfinite(values).all():
                raise ValueError(f"Pretraining {key} must be finite with shape ({args.interval_steps}, 9)")
        put_interval(learner, system, block, args.dt)
        # Bounded least squares fits a_k; the score is the maximum residual norm.
        score = float(learner.compute_score(A_UPPER, A_LOWER))
        scores.append(score)
        fitted.append(learner.a_k.copy())
        learner.append_score(score)
        # Pretraining fills the score window without adapting delta. A Theta
        # update takes effect in the next interval, after this score is stored.
        update = learner.update_representation()
        if update is not None:
            system.set_representation(update["Theta"])
        theta_history.append(learner.Theta.copy())
        learner.clear_buffers()
        for key, values in block.items():
            observations.setdefault(key, []).append(values)
    if learner._representation_intervals:
        raise RuntimeError("Pretraining must end on a complete representation block")
    if len(scores) != args.pretrain_intervals or len(learner.S_cal) != args.calibration_intervals:
        raise RuntimeError("Pretraining did not fill the calibration window")
    # Initial CP calibration uses the retained historical scores and fixed delta.
    learner.compute_quantile()
    return (learner, {key: np.asarray(value) for key, value in observations.items()},
            dict(theta=np.asarray(theta_history), score=np.asarray(scores), a_fit=np.asarray(fitted)))


def prepare_from_scratch(system, args, output):
    """Save streaming observations, learning history and initialization for replay."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    learner, training, history = run_pretraining(system, args)
    theta = learner.Theta.copy()
    np.savez_compressed(output/"data_pretraining.npz", **training)
    np.savez_compressed(output/"pretraining_history.npz", **history)
    metadata = dict(
        schema_version=4, generation="fresh_physical_plant_observations",
        calibration_method="rolling_pretraining_scores",
        dt=args.dt, interval_steps=args.interval_steps,
        intervals={"pretraining":args.pretrain_intervals,
                   "calibration_window":args.calibration_intervals},
        representation_period=args.representation_period,
        learning_rate=args.learning_rate, theta_initial=history["theta"][0],
        theta_initialization="uniform_fixed_box", theta_lower=THETA_LOWER,
        theta_upper=THETA_UPPER,
        theta_learned=theta, controller_rate=system.metric.rate,
        seeds={"pretraining":args.seed, "theta":getattr(args, "theta_seed", 42)},
        pretraining_wind_schedule=WindSchedule(
            pretraining_wind_pattern(args.representation_period),
            args.interval_steps*args.dt,
        ).metadata(),
        initial_quantile=learner.Q_k,
        residual_scale=args.residual_scale, core_hashes=core_hashes(),
        repository=repository_state(),
        metric_source_sha256=system.metric.metadata["source_sha256"],
        representation_updates=learner.representation_update_index,
        theta_change_norm=float(np.linalg.norm(theta-history["theta"][0])),
    )
    dump_json(output/"summary.json", metadata)
    return metadata, learner


def normalized_gamma(theta, physical_gain, regularization):
    """Build Gamma_0 = gain * (Theta.T @ Theta + regularization * I)^(-1).

    This reduces scaling differences in the map b = Theta @ a_hat. The
    regularizer keeps the inverse well-defined for nearly dependent columns.
    """
    gram = np.asarray(theta).T @ np.asarray(theta) + regularization*np.eye(4)
    gamma = physical_gain*np.linalg.inv(gram)
    gamma = 0.5*(gamma+gamma.T)
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
    gamma += boost*np.outer(direction, direction)/norm_squared**2
    gamma = 0.5*(gamma+gamma.T)
    np.linalg.cholesky(gamma)
    return gamma


def finish_interval(learner, system, args):
    """Finish one online interval and return its score/update record.

    Order: fit a_k -> assess coverage -> append score -> possibly update Theta
    -> compute the next quantile. a_k is separate from the live adaptive a_hat.
    """
    learner.estimate_uncertainty(args.dt)
    score = float(learner.compute_score(system.a_ub, system.a_lb))
    previous_quantile = learner.Q_k
    # Assess this score against the quantile available before observing it.
    missed = learner.update_delta(score)
    learner.append_score(score)
    update = learner.update_representation() if getattr(args, "online_theta", True) else None
    if update is not None:
        system.set_representation(update["Theta"])
    system.cp_quantile = learner.compute_quantile()
    event = dict(score=score, previous_quantile=previous_quantile,
                 next_quantile=system.cp_quantile, miscoverage=missed,
                 delta=float(learner.delta), a_fit=learner.a_k.copy(),
                 representation_updated=update is not None)
    # Discard this interval's samples; keep scores, Theta and block-update state.
    learner.clear_buffers()
    return event


def clone_learner(pretrained, system):
    """Keep Theta, scores, delta and counters independent in each comparison arm.

    Rebind callbacks to the new system instead of copying the old model/metric.
    Only sample buffers are cleared; the pretraining state is preserved.
    """
    learner = copy.copy(pretrained)
    learner.__dict__ = copy.deepcopy({
        key: value for key, value in vars(pretrained).items()
        if key not in ("Y_Theta", "representation_loss_gradient")
    })
    learner.clear_buffers()
    if learner._representation_intervals:
        raise ValueError("Pretraining must end at a complete representation block")
    learner.Y_Theta = system.Y_Theta
    learner.representation_loss_gradient = system.representation_loss_gradient
    if learner.Theta is not None:
        system.set_representation(learner.Theta)
    system.cp_quantile = learner.compute_quantile()
    return learner
