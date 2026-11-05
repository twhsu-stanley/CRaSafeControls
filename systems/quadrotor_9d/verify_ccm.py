"""Independent-seed neural CCM sampling and optional adversarial C1 search.

No auxiliary neural feedback is evaluated. Reported C1 rates are empirical;
positivity follows from the neural metric's structure, while C2 is approximate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[2]
# Support direct script execution as well as python -m systems.quadrotor_9d.verify_ccm.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from systems.quadrotor_9d.ccm import DEFAULT_METRIC, load_metric
from systems.quadrotor_9d.ccm_conditions import ccm_matrices, evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_METRIC,
                        help="Trained neural CCM NPZ (default: bundled metric)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--search-iterations", type=int, default=80)
    parser.add_argument("--simulation", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.samples < 1 or args.seed < 0 or args.search_iterations < 0:
        parser.error("Choose a new output folder and valid sample/search settings")
    metric = load_metric(args.checkpoint)
    rng = np.random.default_rng(args.seed)
    d = metric.domain
    x = rng.uniform(d["x_min"], d["x_max"], (args.samples, 9))
    b = rng.uniform(d["b_min"], d["b_max"], (args.samples, 5))
    summary, samples = evaluate(metric, x, b)
    report = dict(checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  source=metric.metadata, seed=args.seed, random_domain=summary,
                  scope="Frozen b model f+Psi b; empirical C1 check, not a global certificate")
    print(json.dumps(summary), flush=True)
    if args.search_iterations:
        bounds = list(zip(d["x_min"][3:], d["x_max"][3:])) + list(zip(d["b_min"], d["b_max"]))
        def objective(point):
            C1 = ccm_matrices(metric, np.r_[np.zeros(3), point[:6]], point[6:])[1]
            return -np.linalg.eigvalsh((C1+C1.T)/2)[-1]
        search = differential_evolution(objective, bounds, rng=np.random.default_rng(args.seed+1),
                                        maxiter=args.search_iterations, popsize=12,
                                        tol=1e-7, polish=True)
        report["adversarial_search"] = dict(
            worst_C1_eigenvalue=float(-search.fun), point=search.x.tolist(),
            evaluations=search.nfev, converged=bool(search.success), message=str(search.message))
        print(json.dumps(report["adversarial_search"]), flush=True)
    if args.simulation:
        report["simulation"] = str(args.simulation.resolve())
        report["trajectories"] = {}
        for key in ("proposed", "no_cp", "cp_no_adaptation", "no_adaptation"):
            with np.load(args.simulation/f"{key}.npz", allow_pickle=False) as data:
                for location, state in (("plant", "x"), ("reference", "x_d")):
                    report["trajectories"][key+"_"+location] = evaluate(metric, data[state], data["b"])[0]
    args.output.mkdir(parents=True)
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(args.output/"samples.npz", x=x, b=b, checks=samples)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(np.sort(samples[:, 1]))
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set(xlabel="Ordered held-out sample", ylabel="max eig(C1)", title=f"C1, lambda={metric.rate}")
    axes[1].hist(samples[:, 2], bins=40)
    axes[1].set(xlabel="Full C2 Frobenius norm (max input)", ylabel="Count", title="C2 equality residual")
    for axis in axes:
        axis.grid(alpha=.25)
    fig.savefig(args.output/"conditions.png", dpi=170)
    plt.close(fig)
    return report


if __name__ == "__main__":
    main()
