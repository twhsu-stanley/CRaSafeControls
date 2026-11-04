"""Fit W(x) = J(x) Q J(x).T using LMIs at fixed (x, b) samples.

Model: x_dot = F(x, b) + B(x)u, with F = f_nominal + Psi(x)b.
State derivatives hold b fixed at each sample; this does not certify the true plant.

J is prescribed and only Q is optimized. Positive Q ensures W > 0 on regular
Euler coordinates, and the frame structure gives full C2 = 0. Finite C1 checks
do not certify the entire state and parameter domain.
"""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
# Support direct script execution and python -m from the repository root.
REPOSITORY_ROOT = ROOT.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from systems.quadrotor_9d.ccm_conditions import model_derivatives, projection
from systems.quadrotor_9d.LMICCM.structured_metric import invariant_frame, normalize_domain


def coefficients(x, b):
    """Return constant coefficients U and V (each 5x9) for one sample's C1(Q).

    With U = P.T J and V = P.T (A J - D_F J),
    C1(Q) = V Q U.T + U Q V.T + 2 * rate * U Q U.T.
    For fixed x, b, and rate, this expression is linear in the unknown Q.
    """
    F, A, B, *_ = model_derivatives(x, b)
    J, dJ_dx = invariant_frame(x)
    P = projection(B)  # P: 9×5, B.T @ P = 0.

    # dJ_dx[k, i, j] = partial J[i, j] / partial x[k]; sum over k with F.
    # Flatten F from (9, 1) to (9,) to compute D_F J = sum_k F_k * dJ/dx_k.
    directional_J = np.einsum("k,kij->ij", F.ravel(), dJ_dx)
    U = P.T @ J
    V = P.T @ (A @ J - directional_J)
    return U, V


def domain_pool(domain, count, seed):
    """Return (states, parameters) combining the origin, random points, and corners.

    main sets symmetric velocity and attitude bounds. Corner positions are zero
    because this model's CCM conditions do not depend on position.
    Default sample count: 1 + 8192 + 2**6 * 2**5 = 10241.
    """
    rng = np.random.default_rng(seed)
    random_states = rng.uniform(domain["x_min"], domain["x_max"], (count, 9))
    random_parameters = rng.uniform(domain["b_min"], domain["b_max"], (count, 5))

    # Positive/negative bounds for three velocities and three Euler angles: 64 corners.
    state_signs = np.array(list(product((-1.0, 1.0), repeat=6)))
    state_corners = np.zeros((len(state_signs), 9))
    state_corners[:, 3:] = state_signs * np.asarray(domain["x_max"])[3:]

    # All lower/upper-bound combinations of the five b components: 32 corners.
    parameter_bounds = zip(domain["b_min"], domain["b_max"])
    parameter_corners = np.array(list(product(*parameter_bounds)))
    boundary_states = np.repeat(state_corners, len(parameter_corners), axis=0)
    boundary_parameters = np.tile(parameter_corners, (len(state_corners), 1))

    states = np.vstack((np.zeros((1, 9)), random_states, boundary_states))
    parameters = np.vstack((np.zeros((1, 5)), random_parameters, boundary_parameters))
    return states, parameters


def parse_args(argv=None):
    """Parse CLI options and validate the output path and training settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--pool-size", type=int, default=8192)
    parser.add_argument("--velocity-limit", type=float, default=0.75)
    parser.add_argument("--tilt-limit", type=float, default=0.25)
    parser.add_argument("--yaw-limit", type=float, default=0.2)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--q-floor", type=float, default=0.001)
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args(argv)

    positive_settings = [
        args.velocity_limit, args.tilt_limit, args.yaw_limit,
        args.rate, args.margin, args.q_floor,
    ]
    if (
        args.output.exists()
        or args.pool_size < 256
        or args.seed < 0
        or not np.isfinite(positive_settings).all()
        or min(positive_settings + [args.rounds]) <= 0
        or args.tilt_limit >= 1.5
    ):
        parser.error("Choose a new output folder and finite positive settings")
    return args


def load_training_reference(args):
    """Load the target and domain, then apply this run's velocity and angle limits."""
    reference_path = ROOT / "metrics/training_reference.npz"
    with np.load(reference_path, allow_pickle=False) as data:
        reference = json.loads(str(data["metadata"].item()))
        target = np.array(data["target"], dtype=float)

    # target anchors the objective; the optimization variable Q is newly defined.
    domain = normalize_domain(reference["domain"])
    domain["x_min"][3:6] = [-args.velocity_limit] * 3
    domain["x_max"][3:6] = [args.velocity_limit] * 3
    angle_limits = [args.tilt_limit, args.tilt_limit, args.yaw_limit]
    domain["x_min"][6:9] = [-limit for limit in angle_limits]
    domain["x_max"][6:9] = angle_limits
    reference["domain"] = domain
    return target, reference, reference_path


def solve_sampled_lmi(U, V, target, active, args):
    """Solve the active samples' LMIs and return numeric Q and the solver status."""
    import cvxpy as cp  # Runtime metric evaluation does not require CVXPY.

    Q = cp.Variable((9, 9), symmetric=True)
    constraints = [
        Q >> args.q_floor * np.eye(9),  # λ_min(Q) >= q_floor.
        cp.trace(Q) == np.trace(target),  # Normalize the overall metric scale.
    ]
    for index in active:
        u, v = U[index], V[index]
        C1 = v @ Q @ u.T + u @ Q @ v.T + 2 * args.rate * u @ Q @ u.T
        # << imposes negative semidefiniteness, not an elementwise comparison.
        constraints.append(C1 << -args.margin * np.eye(5))

    # Solve the convex problem minimizing ||Q - target||_F^2 with CLARABEL.
    problem = cp.Problem(cp.Minimize(cp.sum_squares(Q - target)), constraints)
    problem.solve(solver="CLARABEL", max_iter=150, tol_feas=1e-8, tol_gap_abs=1e-8)
    if Q.value is None:
        raise RuntimeError(
            f"No feasible matrix: {problem.status}; output contains no deployable metric"
        )
    # Q.value is the numerical solution; remove floating-point asymmetry.
    value = (Q.value + Q.value.T) / 2
    return value, problem.status


def maximum_c1_eigenvalues(Q, U, V, rate):
    """Evaluate C1 across the pool and return each sample's largest eigenvalue (N,)."""
    # U, V: (N, 5, 9). transpose(0, 2, 1) transposes each sample's matrix.
    U_T = U.transpose(0, 2, 1)
    V_T = V.transpose(0, 2, 1)
    C1 = V @ Q @ U_T + U @ Q @ V_T
    C1 += 2 * rate * U @ Q @ U_T
    symmetric_C1 = (C1 + C1.transpose(0, 2, 1)) / 2
    return np.linalg.eigvalsh(symmetric_C1)[:, -1]


def fit_metric(U, V, target, args):
    """Solve a subset, check the full pool, and add the most difficult samples.

    Selection uses only the initial pool. This is not a continuous-state search
    or a proof of C1 for every untested x and b.
    """
    active = list(range(257))  # Origin plus the first 256 random samples.
    history = []
    for iteration in range(args.rounds):
        Q, solver_status = solve_sampled_lmi(U, V, target, active, args)
        eigenvalues = maximum_c1_eigenvalues(Q, U, V, args.rate)
        worst_c1 = float(eigenvalues.max())
        minimum_q = float(np.linalg.eigvalsh(Q)[0])
        record = dict(
            round=iteration,
            constraints=len(active),
            solver_status=solver_status,
            worst_pool_eigenvalue=worst_c1,
            minimum_Q_eigenvalue=minimum_q,
        )
        history.append(record)
        (args.output / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)

        # Retain the numerical stopping tolerances: 90% of margin, q_floor - 1e-7.
        c1_passes = worst_c1 < -0.9 * args.margin
        q_passes = minimum_q >= args.q_floor - 1e-7
        if c1_passes and q_passes:
            return Q, active, history

        # Add up to 128 inactive samples, ordered by decreasing largest eigenvalue.
        active_set = set(active)
        worst_first = np.argsort(eigenvalues)[::-1]
        new_indices = [int(index) for index in worst_first if index not in active_set][:128]
        active.extend(new_indices)

    raise RuntimeError("C1 search did not converge; output contains no deployable metric")


def save_result(Q, states, parameters, active, history, reference, reference_path, args, started):
    """Save Q, the domain, and guarantee scope; save only active training samples."""
    fingerprint = hashlib.sha256(Q.tobytes()).hexdigest()
    max_velocity_norm_squared = 3 * args.velocity_limit**2
    w_lower_bound = np.linalg.eigvalsh(Q)[0] / (9 + 3 * max_velocity_norm_squared)
    metadata = dict(
        architecture="right_invariant_constant",
        parameterization="physical",
        source_checkpoint="sampled_LMI/right_invariant_Q",
        source_sha256=fingerprint,
        task="QUADROTOR_9D_EULER_PAYLOAD_PLANAR_STYLE",
        rate=args.rate,
        domain=reference["domain"],
        w_lb=float(w_lower_bound),
        baseline_metric_sha256=reference["baseline_metric_sha256"],
        training_reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        training=dict(
            seed=args.seed,
            pool_size=len(states),
            margin=args.margin,
            q_floor=args.q_floor,
            history=history,
            wall_seconds=time.perf_counter() - started,
        ),
        guarantee=dict(
            positivity="Q positive definite and J invertible on regular Euler coordinates",
            C2="Full matrix identity from right invariance; floating point evaluation has rounding error",
            C1="Finite sampled LMI constraints and search only; no whole-domain certificate",
        ),
    )
    np.savez_compressed(args.output / "metric.npz", Q=Q, metadata=json.dumps(metadata))
    np.savez_compressed(
        args.output / "training_samples.npz", x=states[active], b=parameters[active]
    )
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv=None):
    """Load settings, sample the domain, compute coefficients, fit Q, and save."""
    args = parse_args(argv)
    target, reference, reference_path = load_training_reference(args)
    states, parameters = domain_pool(reference["domain"], args.pool_size, args.seed)

    # Collect per-sample (U_i, V_i) into separate arrays of shape (N, 5, 9).
    coefficient_pairs = [coefficients(x, b) for x, b in zip(states, parameters)]
    U = np.asarray([pair[0] for pair in coefficient_pairs])
    V = np.asarray([pair[1] for pair in coefficient_pairs])

    started = time.perf_counter()
    args.output.mkdir(parents=True)
    Q, active, history = fit_metric(U, V, target, args)
    return save_result(
        Q, states, parameters, active, history, reference, reference_path, args, started
    )


if __name__ == "__main__":
    main()
