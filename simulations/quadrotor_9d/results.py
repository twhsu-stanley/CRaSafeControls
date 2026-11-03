"""Turn simulation histories into metrics, reproducibility records and plots.

Called by run_example.py after each controller case and after the comparison.
Inputs are the histories and interval events returned by simulation.py.
Outputs are scalar summaries, JSON reports and PNG figures.
"""


import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def dump_json(path, value):
    """Write NumPy values and paths as portable JSON."""
    def convert(item):
        """Convert array values and filesystem paths into JSON-compatible objects."""
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, (np.integer, np.floating, np.bool_)):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(
        json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def core_hashes():
    """Fingerprint shared controller, learning, planning and geodesic source files."""
    names = [
        "systems/control_affine_system.py", "olacp.py",
        "motion_planner.py", "geodesic_solver.py",
    ]
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}


def repository_state():
    """Record the local Git branch and commit, or None when Git is unavailable."""
    def git(*args):
        """Read one property of the local repository."""
        return subprocess.check_output(["git","-C",str(ROOT),*args],text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        return dict(branch=git("branch","--show-current"),commit=git("rev-parse","HEAD"))
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
        first = index*interval_steps
        last = min(first+interval_steps, len(result["t"])-1)
        parameter = np.asarray(event["a_fit"])
        for k in range(first, last):
            # Use the Theta active at this sample, not the final learned matrix.
            estimate = system.Y_Theta(result["x"][k], result["theta"][k]) @ parameter
            fitted[k], a_k[k] = estimate, parameter
            fit_errors.append(np.linalg.norm(estimate-result["true_w"][k]))
    scores = np.asarray([event["score"] for event in events])
    quantiles = np.asarray([event["previous_quantile"] for event in events])
    # a_fit minimizes an interval loss; result['a'] is the live adaptive estimate.
    controller_fit = np.asarray([
        system.Y_Theta(x, theta) @ a
        for x, theta, a in zip(result["x"], result["theta"], result["a"])
    ])
    return {
        "a_k": a_k, "fitted_uncertainty": fitted,
        "interval_coverage": float(np.mean(scores <= quantiles)) if len(scores) else None,
        "fit_residual_rmse": float(np.sqrt(np.mean(np.square(fit_errors)))) if fit_errors else None,
        "controller_residual_rmse": float(np.sqrt(np.mean(
            np.linalg.norm(controller_fit-result["true_w"], axis=1)**2
        ))),
    }


def summarize(result, events, system, interval_steps):
    """Add fitted arrays to result and return scalar metrics for one case."""
    # Measure both tracking of the integrated x_d and error against the common guide.
    reference_error = np.linalg.norm(result["x"][:, :3]-result["x_d"][:, :3], axis=1)
    target_error = np.linalg.norm(result["x"][:, :3]-result["target"][:, :3], axis=1)
    fits = interval_diagnostics(result, events, system, interval_steps)
    result["a_k"] = fits.pop("a_k")
    result["fitted_uncertainty"] = fits.pop("fitted_uncertainty")
    return dict(
        reference_model="estimated_mpc", reference_integration="coupled_RK45_live_estimate",
        controller="ControlAffineSystem.ctrl_craccm", qp_backend="quadprog",
        replanning=True, controller_input_bounds=False,
        recorded_duration=float(result["t"][-1]), sample_count=len(result["t"]),
        position_rmse=float(np.sqrt(np.mean(reference_error**2))),
        final_position_error=float(reference_error[-1]),
        target_rmse=float(np.sqrt(np.mean(target_error**2))), final_target_error=float(target_error[-1]),
        max_slack=float(result["slack"].max()), max_abs_rho=float(np.abs(result["rho"]).max()),
        max_constraint_residual=float(result["constraint_residual"].max()),
        training_domain_fraction=(float(result["in_domain"].mean())
                                  if np.isfinite(result["in_domain"]).all() else None),
        completed_intervals=len(events),
        representation_updates=sum(event["representation_updated"] for event in events), **fits,
    )


def common_prefix_comparison(results):
    """Compare the shared time prefix when some arms terminate early."""
    count = min(len(result["t"]) for result in results.values())
    comparison = dict(sample_count=count, end_time=float(next(iter(results.values()))["t"][count-1]))
    for name, reference in (("position_rmse", "x_d"), ("target_position_rmse", "target")):
        comparison[name] = {
            label: float(np.sqrt(np.mean(np.sum(
                (result["x"][:count, :3]-result[reference][:count, :3])**2, axis=1))))
            for label, result in results.items()
        }
    return comparison


def save_plot(results, path):
    """Plot position traces, 3D paths, tracking errors and control corrections."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,9),constrained_layout=True)
    grid = fig.add_gridspec(3,2)
    axes = [fig.add_subplot(grid[j,0]) for j in range(3)]
    path3d = fig.add_subplot(grid[0,1],projection="3d")
    error = fig.add_subplot(grid[1,1])
    effort = fig.add_subplot(grid[2,1])
    first = max(results.values(),key=lambda result:len(result["t"]))
    for label,result in results.items():
        for j,ax in enumerate(axes):
            ax.plot(result["t"],result["x"][:,j],label=label)
        path3d.plot(*result["x"][:,:3].T,label=label)
        error.plot(result["t"],np.linalg.norm(result["x"][:,:3]-result["x_d"][:,:3],axis=1),label=label)
        effort.plot(result["t"],result["correction_norm"],label=label)
    for j,ax in enumerate(axes):
        ax.plot(first["t"],first["target"][:,j],":",label="position cost guide")
        ax.set_ylabel(f"{'xyz'[j]} [m]")
        ax.grid(alpha=.3)
    axes[0].legend()
    axes[-1].set_xlabel("Time [s]")
    path3d.plot(*first["target"][:,:3].T,"k--",label="position cost guide")
    path3d.set(xlabel="x [m]",ylabel="y [m]",zlabel="z [m]")
    error.set(ylabel="Own reference error [m]",xlabel="Time [s]")
    effort.set(ylabel="Control correction norm",xlabel="Time [s]")
    for ax in (error,effort):
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def save_input_plot(results,path):
    """Plot actual thrust acceleration and the three body-rate commands over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(4,1,sharex=True,figsize=(12,9),constrained_layout=True)
    names = [r"Thrust acceleration [m/s$^2$]",r"Body rate $\omega_x$ [rad/s]",
             r"Body rate $\omega_y$ [rad/s]",r"Body rate $\omega_z$ [rad/s]"]
    for label,result in results.items():
        for j,ax in enumerate(axes):
            ax.plot(result["t"],result["u"][:,j],label=label,linewidth=1.)
    for ax,name in zip(axes,names):
        ax.set_ylabel(name)
        ax.grid(alpha=.3)
    axes[0].legend(ncol=2,fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.savefig(path,dpi=160)
    plt.close(fig)


def save_reference_plot(histories, reference, path):
    """Compare common position targets with each case's integrated reference x_d."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7), constrained_layout=True)
    times = max(histories.values(), key=lambda data: len(data["t"]))["t"]
    guide = reference.guide(times)
    for i, ax in enumerate(axes):
        ax.plot(times, guide[i], "k:", label="position cost guide")
        ax.plot(reference.times, reference.positions[:, i], "ko", markersize=3, label="waypoints")
        for label, data in histories.items():
            ax.plot(data["t"], data["x_d"][:, i], label=label)
        ax.set_ylabel(f"{'xyz'[i]} [m]")
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Time [s]")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_wind_plot(histories, path):
    """Show the common prescribed wind with its interval-boundary jumps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = max(histories.values(), key=lambda value: len(value["t"]))
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for index, axis in enumerate("xyz"):
        ax.step(data["t"], data["wind"][:, index], where="post", label=f"wind {axis}")
    ax.set(xlabel="Time [s]", ylabel="Inertial wind [m/s]", title="Prescribed interval wind")
    ax.grid(alpha=.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1., 1.))
    fig.savefig(path, dpi=160)
    plt.close(fig)
