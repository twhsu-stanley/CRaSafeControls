"""Train the 9D dual CCM W = L.T L + w_lb I from random neural weights.

Data flow: settings -> fixed (x, b) samples -> W and state derivatives -> CCM
matrices -> weighted losses -> Adam updates -> validation -> selected NPZ.
State order is [px, py, pz, vx, vy, vz, roll, pitch, yaw]; b = Theta @ a_hat
contains five physical coefficients. State derivatives hold b fixed.

The metric architecture and sampled losses follow NeuralCCM.zip's adaptive
payload task. C1, projected C2, and the metric upper bound are soft penalties.
Only positivity is structural. Deployment uses the existing CRaCCM controller.
"""

import argparse
import copy
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
# Support direct script execution and python -m from the repository root.
REPOSITORY_ROOT = ROOT.parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from systems.quadrotor_9d.ccm_conditions import model_derivatives, projection
from systems.quadrotor_9d.NeuralCCM.neural_metric import WEIGHT_NAMES


class MetricNetwork(nn.Module):
    """Random 11 -> hidden -> hidden -> 81 network, compatible with the runtime NPZ."""

    def __init__(self, hidden=128, w_lb=0.03, rate=0.5):
        """Initialize three Linear layers; no saved metric is loaded."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(11, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 81, bias=False),
        ).double()
        self.w_lb = float(w_lb)
        self.rate = float(rate)

    def forward(self, x, b):
        """Return W (N, 9, 9) and dW/dx (N, 9, 9, 9).

        N is the batch size. Inputs are x (N, 9) and physical b (N, 5).
        The last derivative axis follows the nine-state order. The analytic
        chain rule remains differentiable with respect to the trainable weights.
        """
        weight_in, bias_in, weight_hidden, bias_hidden, weight_out = tuple(self.layers.parameters())
        features = torch.cat((x[:, 3:9], b), dim=1)
        h1 = torch.tanh(torch.nn.functional.linear(features, weight_in, bias_in))
        h2 = torch.tanh(torch.nn.functional.linear(h1, weight_hidden, bias_hidden))
        L = torch.nn.functional.linear(h2, weight_out).reshape(-1, 9, 9)
        identity = torch.eye(9, dtype=x.dtype, device=x.device)
        W = L.transpose(1, 2) @ L + self.w_lb * identity

        # The network uses six state features; b is fixed in these state derivatives.
        # The layer biases above are separate from the physical coefficient vector b.
        dh1_dx = (1 - h1.square())[:, :, None] * weight_in[None, :, :6]
        dh2_dx = (1 - h2.square())[:, :, None] * torch.einsum("ij,njk->nik", weight_hidden, dh1_dx)
        dL_dx = torch.einsum("ij,njk->nik", weight_out, dh2_dx).reshape(-1, 9, 9, 6)

        # dW/dx_k = (dL/dx_k).T L + L.T (dL/dx_k).
        dW_dvelocity_attitude = torch.einsum("njik,njl->nilk", dL_dx, L)
        dW_dvelocity_attitude = (
            dW_dvelocity_attitude + dW_dvelocity_attitude.transpose(1, 2)
        )
        position_derivatives = torch.zeros_like(dW_dvelocity_attitude[..., :3])
        dW_dx = torch.cat((position_derivatives, dW_dvelocity_attitude), dim=-1)
        return W, dW_dx


def training_domain(args):
    """Build x (9 entries) and b (5 entries) bounds from independent CLI settings."""
    limits = [args.position_limit] * 3 + [args.velocity_limit] * 3
    limits += [args.tilt_limit, args.tilt_limit, args.yaw_limit]
    return dict(
        x_min=[-limit for limit in limits], x_max=limits,
        b_min=list(args.b_min), b_max=list(args.b_max),
    )


def _counterexample_samples(domain, path, rng):
    """Return 128 training samples: a reported failing point and nearby points."""
    point = np.asarray(
        json.loads(Path(path).read_text())["adversarial_search"]["point"], dtype=float
    )
    # Report order: six velocity/attitude states, then five physical coefficients.
    lower = np.r_[domain["x_min"][3:], domain["b_min"]]
    upper = np.r_[domain["x_max"][3:], domain["b_max"]]
    if (
        point.shape != (11,)
        or not np.isfinite(point).all()
        or np.any(point < lower)
        or np.any(point > upper)
    ):
        raise ValueError("Counterexample must be a finite 11-vector inside the training domain")
    nearby = rng.uniform(-0.04, 0.04, (128, 11)) * (upper - lower)
    hard_points = np.clip(point + nearby, lower, upper)
    hard_points[0] = point
    states = np.c_[np.zeros((128, 3)), hard_points[:, :6]]
    return states, hard_points[:, 6:]


def observations(domain, count, seed, boundary_samples=0, counterexamples=None):
    """Sample states/parameters and precompute model data for all training epochs.

    Return (x, b, F, A, B, DB, P) as CPU float64 tensors with shapes:
    (N,9), (N,5), (N,9), (N,9,9), (N,9,4), (N,4,9,9), (N,9,5).
    F = f_nominal + Psi b, A = dF/dx, DB[:,j] = dB_j/dx, and B.T P = 0.
    These model values do not depend on neural weights, so they are cached once.
    """
    rng = np.random.default_rng(seed)
    states = rng.uniform(domain["x_min"], domain["x_max"], (count, 9))
    parameters = rng.uniform(domain["b_min"], domain["b_max"], (count, 5))

    if boundary_samples:
        # Six state features plus five coefficients give 2**11 corner combinations.
        corners = np.array(list(product((False, True), repeat=11)))
        corners = corners[rng.permutation(len(corners))[:boundary_samples]]
        corner_states = np.zeros((len(corners), 9))
        corner_states[:, 3:] = np.where(
            corners[:, :6], domain["x_max"][3:], domain["x_min"][3:]
        )
        corner_parameters = np.where(corners[:, 6:], domain["b_max"], domain["b_min"])
        states = np.vstack((states, corner_states))
        parameters = np.vstack((parameters, corner_parameters))

    if counterexamples is not None:
        hard_states, hard_parameters = _counterexample_samples(domain, counterexamples, rng)
        states = np.vstack((states, hard_states))
        parameters = np.vstack((parameters, hard_parameters))

    rows = [model_derivatives(x, b) for x, b in zip(states, parameters)]
    F, A, B = (np.stack([row[index] for row in rows]) for index in range(3))
    DB = np.stack([np.stack(row[3:]) for row in rows])
    P = np.stack([projection(matrix) for matrix in B])
    arrays = (states, parameters, F[..., 0], A, B, DB, P)
    return tuple(torch.from_numpy(array).double() for array in arrays)


def matrices(model, batch, rate=None):
    """Return W (N,9,9), C1 (N,5,5), and projected C2 (N,4,5,5).

    P spans the input-orthogonal dual directions. These definitions agree with
    the shared evaluator in CRaSafeControls-review/systems/quadrotor_9d.
    """
    x, b, F, A, B, DB, P = batch
    rate = model.rate if rate is None else rate
    W, dW_dx = model(x, b)
    P_T = P.transpose(1, 2)

    # dW_dx[n,i,j,k] differentiates state k; einsum sums that axis with F.
    directional_W = torch.einsum("nijk,nk->nij", dW_dx, F)
    H = -directional_W + A @ W + W @ A.transpose(1, 2) + 2 * rate * W
    C1 = P_T @ H @ P

    # Evaluate all four input vector fields together: D_Bj W - DB_j W - W DB_j.T.
    C2_full = torch.einsum("nijk,nkl->nlij", dW_dx, B)
    C2_full = C2_full - DB @ W[:, None] - W[:, None] @ DB.transpose(2, 3)
    projected_C2 = P_T[:, None] @ C2_full @ P[:, None]
    # Projection is weaker than full C2 = 0; only projected C2 enters this loss.
    return W, (C1 + C1.transpose(1, 2)) / 2, projected_C2


def losses(model, batch, margin, w_ub):
    """Return three scalar losses and per-sample largest C1/W eigenvalues.

    C1 training targets eigenvalues <= -margin; the reported pass rate uses < 0.
    W >= w_lb I is structural. C2 and W <= w_ub I remain sampled soft penalties.
    """
    W, C1, projected_C2 = matrices(model, batch)
    c1_eigenvalues = torch.linalg.eigvalsh(C1)
    w_eigenvalues = torch.linalg.eigvalsh(W)
    c1_loss = torch.relu(c1_eigenvalues + margin).mean()
    c2_loss = projected_C2.square().sum((1, 2, 3)).mean()
    upper_loss = torch.relu(w_eigenvalues - w_ub).mean()
    terms = (c1_loss, c2_loss, upper_loss)
    return terms, c1_eigenvalues[:, -1], w_eigenvalues[:, -1]


def batches(data, size, shuffle=False, device="cpu"):
    """Yield shuffled or ordered batches on the target device, retaining the tail."""
    indices = torch.randperm(len(data[0])) if shuffle else torch.arange(len(data[0]))
    for index in indices.split(size):
        if len(index):
            yield tuple(value[index].to(device) for value in data)


@torch.no_grad()
def validate(model, data, hard, args):
    """Return (record, score) without building a parameter-gradient graph.

    data contains independent random validation samples. hard contains corners
    and optional counterexamples already used for training; report them separately.
    A smaller score wins, using tuple order: coverage, worst margin violation, loss.
    """
    weighted_losses, maximum_c1_values, maximum_w_values, hard_c1_values = [], [], [], []
    for batch in batches(data, args.batch_size, device=args.device):
        terms, maximum_c1, maximum_w = losses(model, batch, args.margin, args.w_ub)
        # Weight by batch size so a smaller final batch has the correct contribution.
        weighted_losses.append(np.array([term.item() for term in terms]) * len(batch[0]))
        maximum_c1_values.extend(maximum_c1.cpu().tolist())
        maximum_w_values.extend(maximum_w.cpu().tolist())
    for batch in batches(hard, args.batch_size, device=args.device):
        C1 = matrices(model, batch)[1]
        hard_c1_values.extend(torch.linalg.eigvalsh(C1)[:, -1].cpu().tolist())

    c1, c2, upper = np.sum(weighted_losses, axis=0) / len(data[0])
    fraction = float(np.mean(np.asarray(maximum_c1_values) < 0))
    worst = float(max(maximum_c1_values))
    boundary_fraction = float(np.mean(np.asarray(hard_c1_values) < 0)) if hard_c1_values else None
    boundary_worst = float(max(hard_c1_values)) if hard_c1_values else None
    record = dict(
        C1_pass_fraction=fraction, worst_C1_eigenvalue=worst,
        C1_hinge=float(c1), C2_mean_squared=float(c2), W_upper_hinge=float(upper),
        maximum_W_eigenvalue=float(max(maximum_w_values)),
        worst_boundary_eigenvalue=boundary_worst, boundary_pass_fraction=boundary_fraction,
    )
    score = (
        1 - min(fraction, boundary_fraction if hard_c1_values else 1.0),
        max(0.0, max(worst, boundary_worst if hard_c1_values else -np.inf) + args.margin),
        float(c1 * args.c1_weight + c2 * args.c2_weight + upper * args.upper_weight),
    )
    return record, score


def train_epoch(model, data, optimizer, loss_weights, args):
    """Update trainable weights once over all samples, including the final batch."""
    for batch in batches(data, args.batch_size, shuffle=True, device=args.device):
        terms, _, _ = losses(model, batch, args.margin, args.w_ub)
        objective = (torch.stack(terms) * loss_weights).sum()
        if not torch.isfinite(objective):
            raise RuntimeError("Nonfinite training loss; no metric exported")
        optimizer.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()


def fit_model(model, train, validation, hard, args):
    """Train, record epoch diagnostics, and restore the best trained model.

    Epoch zero evaluates random weights only. Selection starts after an Adam
    update, and the returned history keeps both validation and training-boundary
    statistics. This loop does not perform a whole-domain verification search.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    loss_weights = torch.tensor(
        [args.c1_weight, args.c2_weight, args.upper_weight], dtype=torch.float64, device=args.device
    )
    history, best_score, best_state, selected_epoch = [], None, None, None
    for epoch in range(args.epochs + 1):
        if epoch:
            train_epoch(model, train, optimizer, loss_weights, args)

        record, score = validate(model, validation, hard, args)
        record.update(epoch=epoch, learning_rate=optimizer.param_groups[0]["lr"])
        if not np.isfinite(score).all():
            raise RuntimeError("Nonfinite validation score; no metric exported")
        if epoch and (best_score is None or score < best_score):
            best_score = score
            # Copy tensors so later optimizer steps cannot change the saved candidate.
            best_state = copy.deepcopy(model.state_dict())
            selected_epoch = epoch
            record["selected"] = True
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if epoch:
            scheduler.step()

    model.load_state_dict(best_state)
    return history, selected_epoch


def weight_arrays(model):
    """Detach weights, move them to CPU, and copy them in runtime NPZ key order."""
    return {
        name: parameter.detach().cpu().numpy().copy()
        for name, parameter in zip(WEIGHT_NAMES, model.layers.parameters())
    }


def weight_hash(arrays):
    """Identify weight contents for provenance; this value is not a training loss."""
    payload = b"".join(arrays[name].tobytes() for name in WEIGHT_NAMES)
    return hashlib.sha256(payload).hexdigest()


def export(model, domain, output, training):
    """Save fixed weights and metadata for NeuralCCMMetric's NumPy-only runtime."""
    arrays = weight_arrays(model)
    metadata = dict(
        architecture="neural", parameterization="physical", rate=model.rate, w_lb=model.w_lb,
        domain=copy.deepcopy(domain), task="QUADROTOR_9D_EULER_PAYLOAD_PLANAR_STYLE",
        source_checkpoint="neural_from_scratch/weights", source_sha256=weight_hash(arrays),
        training=training,
        guarantee=dict(
            positivity="W = L.T L + w_lb I by construction",
            C1="Finite sampled training and validation only; no whole-domain certificate",
            C2="Projected residual penalty, following the reference ZIP; no exact full-C2 guarantee",
        ),
    )
    np.savez_compressed(output / "metric.npz", metadata=json.dumps(metadata), **arrays)
    return metadata


# CLI defaults are independent of the currently bundled model's metadata.
DEFAULTS = dict(
    epochs=60, train_samples=16384, validation_samples=4096, boundary_samples=2048,
    batch_size=256, seed=20261031, learning_rate=0.001, lr_step=20,
    hidden=128, rate=0.5, w_lb=0.03, w_ub=10.0, margin=0.05,
    c1_weight=100.0, c2_weight=1.0, upper_weight=1.0,
    position_limit=5.0, velocity_limit=2.0, tilt_limit=0.35, yaw_limit=0.4,
)


def parse_args(argv=None):
    """Read CLI settings and validate bounds, numeric values, paths, and device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counterexamples", type=Path, default=None,
                        help="Optional past training points; disabled by default.")
    parser.add_argument("--b-min", type=float, nargs=5, default=[-0.72, -0.72, -0.66, -0.2, -0.2])
    parser.add_argument("--b-max", type=float, nargs=5, default=[0.72, 0.72, 0.66, 0.2, 0.2])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for name, default in DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args(argv)

    positive = [getattr(args, key) for key in DEFAULTS if key not in ("seed", "boundary_samples")]
    if (
        args.output.exists()
        or args.seed < 0
        or not 0 <= args.boundary_samples <= 2048
        or not np.isfinite(positive).all()
        or min(positive) <= 0
        or args.tilt_limit >= np.pi / 2
        or args.w_ub <= args.w_lb
        or not np.isfinite([args.b_min, args.b_max]).all()
        or np.any(np.asarray(args.b_min) > np.asarray(args.b_max))
        or (args.counterexamples is not None and not args.counterexamples.is_file())
    ):
        parser.error("Choose a new output folder and finite, ordered, positive training settings")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu")
    return args


def main(argv=None):
    """Prepare fixed samples, fit a randomly initialized metric, and export it."""
    args = parse_args(argv)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    domain = training_domain(args)
    model = MetricNetwork(args.hidden, args.w_lb, args.rate).to(args.device)
    # Keep initial weights only to record provenance and detect an unchanged model.
    initial_arrays = weight_arrays(model)

    train = observations(
        domain, args.train_samples, args.seed, args.boundary_samples, args.counterexamples
    )
    validation = observations(domain, args.validation_samples, args.seed + 1)
    hard = tuple(value[args.train_samples:] for value in train)
    args.output.mkdir(parents=True)
    source_path = args.output / "train_source.py"
    source_path.write_bytes(Path(__file__).read_bytes())
    if args.counterexamples is not None:
        (args.output / "counterexample_source.json").write_bytes(args.counterexamples.read_bytes())

    history, selected_epoch = fit_model(model, train, validation, hard, args)
    arrays = weight_arrays(model)
    delta = float(np.sqrt(sum(np.sum((arrays[key] - initial_arrays[key])**2) for key in WEIGHT_NAMES)))
    if delta <= 1e-12:
        raise RuntimeError("Training did not change the random weights; no metric exported")
    training = dict(
        method="Neural CCM: Adam training from random initialization",
        initialization="random", initial_weights_sha256=weight_hash(initial_arrays),
        configuration={key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        selected_epoch=selected_epoch, parameter_change_l2=delta,
        training_script_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        actual_train_samples=len(train[0]), validation_samples=len(validation[0]),
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
        reference="NeuralCCM.zip/main_adaptive.py and payload-planar-style model/config",
        C2_definition="Projected C2 penalty, as in the reference ZIP",
        selection="Held-out random C1 and training-boundary C1, then weighted validation loss",
        selected_validation=history[selected_epoch],
    )
    metadata = export(model, domain, args.output, training)
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    main()
