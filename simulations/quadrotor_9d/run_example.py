"""Entry point: prepare learning, compare four controllers and save results.

File connections:
    learning.py   -> initial Theta, conformal scores and adaptation gain.
    reference.py  -> position targets and MPC settings.
    simulation.py -> one controller case and its time history.
    results.py    -> error metrics, JSON reports and plots.

Start reading at main(). --source replays saved pretraining observations.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Allow this entry point to run directly from the repository without installation.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systems.quadrotor_9d.ccm import DEFAULT_METRIC, load_metric
from systems.quadrotor_9d.nonlinear_uncertainty import (
    ONLINE_WIND_PATTERN, WindEnvironment, WindSchedule, pretraining_wind_pattern,
)
from simulations.quadrotor_9d import learning, results
from simulations.quadrotor_9d.simulation import run_simulation
from simulations.quadrotor_9d.reference import WaypointReference, validate_timing


# Tuple order: file key, plot label, continuous adaptation, CP tightening.
# Theta/OLACP updates follow the common configuration in all four arms.
ARMS = (("proposed", "Proposed", True, True),
        ("no_cp", "No CP", True, False),
        ("cp_no_adaptation", "Yes CP + no adaptation", False, True),
        ("no_adaptation", "No CP + no adaptation", False, False))


def parse_args(argv=None):
    """Return experiment settings; step counts are converted to seconds using dt."""
    parser = argparse.ArgumentParser(description=__doc__)
    # Output location, optional observation replay and the saved CCM to evaluate.
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        help="Saved summary.json/data_pretraining.npz using the current wind schedule")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_METRIC,
                        help="Trained neural CCM NPZ (default: bundled metric)")
    # Theta pretraining, CP score-window size and independent random seeds.
    parser.add_argument("--pretrain-intervals", type=int, default=64)
    parser.add_argument("--calibration-intervals", type=int, default=32,
                        help="CP window size: recent pretraining/online scores, not extra data")
    parser.add_argument("--interval-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--theta-seed", type=int, default=42,
                        help="Independent seed for uniform Theta initialization in the fixed box")
    # Mission duration, position targets and MPC prediction/replanning grids.
    parser.add_argument("--sim-time", type=float, default=40.)
    parser.add_argument("--dt", type=float, default=.02)
    parser.add_argument("--trajectory", choices=("spatial", "point"), default="spatial",
                        help="Spatial: ascending figure eight, 12 s per lap and 0.03 m/s climb")
    parser.add_argument("--waypoints", type=Path, help="JSON array of K+1 position waypoints [x,y,z]")
    parser.add_argument("--initial-reference", type=float, nargs=9, default=np.zeros(9),
                        help="Complete initial reference state; position must match the first waypoint")
    parser.add_argument("--mpc-horizon-steps",type=int,default=40)
    parser.add_argument("--mpc-input-steps", type=int, default=1,
                        help="Simulation steps per held MPC reference input")
    parser.add_argument("--mpc-replan-steps",type=int,default=20)
    parser.add_argument("--goal", type=float, nargs=3, default=[.5,.3,.3])
    # Online learning, continuous adaptation gains and physical disturbance size.
    parser.add_argument("--online-theta", action=argparse.BooleanOptionalAction, default=True,
                        help="Continue Theta updates after pretraining (pretraining always learns)")
    parser.add_argument("--physical-gamma", type=float, default=3e-5)
    parser.add_argument("--payload-physical-boost", type=float, default=9.7e-4)
    parser.add_argument("--gamma-regularization", type=float, default=1e-5)
    parser.add_argument("--representation-period", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=1.,
                        help="Scale gusts and varying rate biases; drag/payload coefficients stay fixed")
    parser.add_argument("--rho-divergence-threshold",type=float,default=1e4)
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(
        acp_lr=.01, delta=.1, eta=500., slack_weight=10000., geo_degree=2, geo_nodes=4,
        initial_offset=np.zeros(9), u_min=np.array([0, -2, -2, -2]),
        u_max=np.array([20, 2, 2, 2]),
    )
    args = parser.parse_args(argv)
    # Reject invalid files and timing before creating experiment outputs.
    numeric = [args.sim_time, args.dt, args.physical_gamma, args.payload_physical_boost,
               args.gamma_regularization, args.residual_scale, args.learning_rate,
               args.rho_divergence_threshold, *args.goal, *args.initial_reference]
    if (not np.all(np.isfinite(numeric)) or min(args.sim_time, args.dt,
            args.physical_gamma, args.gamma_regularization, args.residual_scale,
            args.learning_rate, args.rho_divergence_threshold) <= 0
            or args.payload_physical_boost < 0 or args.representation_period < 1
            or args.mpc_horizon_steps < 2 or args.mpc_replan_steps < 1
            or not np.isclose(args.sim_time/args.dt, round(args.sim_time/args.dt))):
        parser.error("Times, gains, periods, scale and goal must be finite and valid")
    paths = [args.checkpoint]
    if args.source is not None:
        paths += [args.source/"summary.json",args.source/"data_pretraining.npz"]
    if args.waypoints is not None:
        paths.append(args.waypoints)
    for path in paths:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    if args.source is None and (args.calibration_intervals < 30 or args.interval_steps < 10
            or args.pretrain_intervals < args.calibration_intervals or min(args.seed, args.theta_seed) < 0
            or args.pretrain_intervals % args.representation_period
            or round(args.sim_time/args.dt) % args.interval_steps):
        parser.error("Pretraining must fill the CP window (at least 30 scores), use complete blocks and at least 10 steps")
    if args.output.is_file() or (args.output.exists() and any(args.output.iterdir())):
        parser.error("Choose an empty output directory")
    try:
        validate_timing(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def prepare_learning(config, metric):
    """Prepare the common initial learner used by all four controller cases.

    Return (source folder, metadata, Theta, learner, Gamma). Theta maps adaptive
    parameters to physical coefficients; Gamma controls continuous adaptation.
    """
    source = config.source or config.output/"pretraining"
    if config.source is None:
        # Collect and learn together; the retained interval scores initialize CP.
        print("Collecting observations and learning interval by interval...", flush=True)
        metadata, learner = learning.prepare_from_scratch(metric, config, source)
    else:
        # Replay observations in their original order to reconstruct learner state.
        metadata = json.loads((source/"summary.json").read_text(encoding="utf-8"))
        if config.interval_steps != int(metadata["interval_steps"]):
            raise ValueError("Saved observations use a different --interval-steps value")
        if "dt" in metadata and not np.isclose(metadata["dt"], config.dt):
            raise ValueError("Saved observations use a different dt")
        if metadata.get("metric_source_sha256", metric.metadata["source_sha256"]) != metric.metadata["source_sha256"]:
            raise ValueError("Saved pretraining used a different metric checkpoint")
        if not np.isclose(config.residual_scale, metadata.get("residual_scale", 1.)):
            raise ValueError("Saved pretraining uses a different --residual-scale; omit --source to collect new observations")
        for key, bound in (("theta_lower", learning.THETA_LOWER), ("theta_upper", learning.THETA_UPPER)):
            if key in metadata and not np.array_equal(metadata[key], bound):
                raise ValueError("Saved pretraining used a different fixed Theta box")
        config.pretrain_intervals = int(metadata["intervals"]["pretraining"])
        # Restore the saved learning settings before checking the wind protocol.
        counts = metadata["intervals"]
        config.calibration_intervals = int(counts.get("calibration_window", counts.get("calibration")))
        config.representation_period = int(metadata["representation_period"])
        config.learning_rate = float(metadata["learning_rate"])
        config.seed = int(metadata["seeds"]["pretraining"])
        config.theta_seed = int(metadata["seeds"].get("theta", 42))
        expected_wind = WindSchedule(pretraining_wind_pattern(config.representation_period),
                                     config.interval_steps*config.dt).metadata()
        if metadata.get("pretraining_wind_schedule") != expected_wind:
            raise ValueError("Saved data use a different pretraining wind schedule; omit --source to collect new data")
        validate_timing(config)
        with np.load(source/"data_pretraining.npz", allow_pickle=False) as data:
            training = {key: data[key].copy() for key in data.files}
        print("Replaying interval-wise pretraining from saved observations...", flush=True)
        learner, _, history = learning.run_pretraining(metric, config, training)
        metadata = dict(metadata, schema_version=4,
                        calibration_method="rolling_pretraining_scores",
                        intervals={"pretraining":config.pretrain_intervals,
                                   "calibration_window":config.calibration_intervals},
                        seeds={"pretraining":config.seed, "theta":config.theta_seed},
                        theta_initial=history["theta"][0], theta_learned=learner.Theta.copy(),
                        theta_initialization="uniform_fixed_box", theta_lower=learning.THETA_LOWER,
                        theta_upper=learning.THETA_UPPER,
                        initial_quantile=learner.Q_k,
                        representation_updates=learner.representation_update_index,
                        theta_change_norm=float(np.linalg.norm(learner.Theta-history["theta"][0])))
    config.rate = float(metadata["controller_rate"])
    config.learning_rate = float(metadata["learning_rate"])
    if not np.isfinite(config.rate) or config.rate <= 0:
        raise ValueError("Controller rate must be finite and positive")
    if (config.interval_steps < 10 or config.calibration_intervals < 30
            or config.pretrain_intervals < config.calibration_intervals
            or config.pretrain_intervals % config.representation_period
            or round(config.sim_time/config.dt) % config.interval_steps):
        raise ValueError("Pretraining must fill the CP window and simulation must use complete intervals/blocks")
    theta = learner.Theta.copy()
    # Compute Gamma once from pretrained Theta; all arms use this fixed gain.
    gamma = learning.payload_boosted_gamma(theta, config.physical_gamma,
                                          config.gamma_regularization, config.payload_physical_boost)
    results.dump_json(config.output/"calibration.json", dict(
        scores=list(learner.S_cal), quantile=learner.Q_k, delta=learner.delta, theta=theta,
        method="rolling_pretraining_scores", interval_index=learner.interval_index,
        representation_update_index=learner.representation_update_index,
        source=str(source.resolve()), source_mode="reused" if config.source else "generated",
    ))
    print(f"Pretraining complete: CP window={len(learner.S_cal)}, Q_0={learner.Q_k:.6g}", flush=True)
    return source, metadata, theta, learner, gamma


def main(argv=None):
    """Load a CCM, prepare OLACP, run independent arms and save their comparisons."""
    # 1. Read settings and construct the common position cost guide.
    config = parse_args(argv)
    reference = WaypointReference(config)
    config.output.mkdir(parents=True, exist_ok=True)
    # 2. Load fixed CCM weights, then learn Theta and initialize the CP allowance.
    metric = load_metric(config.checkpoint)
    source, metadata, theta, learner, gamma = prepare_learning(config, metric)
    # The true plant has physical disturbances beyond the learned Psi(x) Theta a.
    environment = WindEnvironment(residual_scale=config.residual_scale)
    # All cases receive the same prescribed wind, changing once per observation interval.
    plant = WindSchedule(ONLINE_WIND_PATTERN, config.interval_steps*config.dt, environment)
    # 3. Run each CP/adaptation combination with its own learner copy.
    histories, intervals, summaries = {}, {}, {}
    for key, label, adaptive, cp in ARMS:
        print(f"Running {label}...", flush=True)
        data, events, outcome = run_simulation(metric, config, reference, learner, theta, gamma, adaptive, cp, plant)
        data["target"] = reference.guide(data["t"]).T  # Cost guide, not x_d.
        data["true_w"] = np.asarray([plant(x, t) for x, t in zip(data["x"], data["t"])])
        data["wind"] = np.asarray([plant.wind(t) for t in data["t"]])
        summary = results.summarize(data, events, learning.make_system(metric, theta), config.interval_steps)
        summaries[key] = dict(summary, **outcome, label=label, cp=cp, adaptive=adaptive)
        histories[label], intervals[key] = data, events
        np.savez_compressed(config.output/f"{key}.npz", **data)
    # 4. Save numerical histories, provenance, scalar metrics and comparison plots.
    hashes = results.core_hashes()
    report = dict(
        schema_version=3,
        method="Position cost guide; native CE-MPC; coupled live reference",
        status="completed" if all(arm["status"] == "completed" for arm in summaries.values()) else "controller_failures",
        source=str(source.resolve()), source_mode="reused" if config.source else "generated",
        pretraining_summary=metadata, metric=metric.metadata, trajectory=config.trajectory,
        environment=asdict(environment), wind_schedule=plant.metadata(),
        configuration=vars(config), arms=summaries,
        reference=reference.metadata(),
        common_prefix_comparison=results.common_prefix_comparison(histories),
        repository=results.repository_state(), core_hashes=hashes,
        changed_core_files_from_local_source=[name for name in hashes if metadata.get("core_hashes", {}).get(name) != hashes[name]],
    )
    results.dump_json(config.output/"metric.json", metric.metadata)
    results.dump_json(config.output/"intervals.json", intervals)
    results.dump_json(config.output/"summary.json", report)
    results.save_plot(histories, config.output/"comparison.png")
    results.save_input_plot(histories, config.output/"inputs.png")
    results.save_reference_plot(histories, reference, config.output/"reference.png")
    results.save_wind_plot(histories, config.output/"wind.png")
    print(json.dumps(summaries, default=lambda value: value.tolist(), indent=2))
    return report


if __name__ == "__main__":
    raise SystemExit(0 if main()["status"] == "completed" else 1)

