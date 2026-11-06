"""Load and plot one saved nine-state quadrotor experiment."""

import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from simulations.quadrotor_9d.sim_quadrotor_9d_craccm_olacp import plot_craccm_results

if __name__ == "__main__":
    results_filename = "enter by user"
    with open(results_filename, "rb") as f:
        data = pickle.load(f)
        results = data["results"]
        desired_trajectory = data.get("desired_trajectories", data.get("desired_trajectory"))

    for setting, result in results.items():
        print(f"Setting: {setting}")
        print(f"  RMS tracking error: {result['metrics']['rms_error']:.4f}")
        print(f"  Max tracking error: {result['metrics']['max_error']:.4f}")
        print(f"  Max control input: {result['metrics']['max_control']:.4f}")
        print(f"  Max slack: {result['metrics']['max_slack']:.2e}")
        print(f"  Status: {result['status']}")

    plot_craccm_results(results, desired_trajectory, file_format="png")
