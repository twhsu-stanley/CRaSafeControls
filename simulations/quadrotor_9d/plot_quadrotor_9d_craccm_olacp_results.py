"""Load a saved quadrotor comparison and regenerate its figures."""

from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulations.quadrotor_9d.sim_quadrotor_9d_craccm_olacp import plot_craccm_results

if __name__ == "__main__":
    results_filename = "enter by user"
    with open(results_filename, "rb") as stream:
        data = pickle.load(stream)
    plot_craccm_results(
        data["results"],
        data["desired_trajectories"],
        file_format="png",
        results_directory=Path(results_filename).resolve().parent,
    )
