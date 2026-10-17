import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from simulations.pendubot.sim_pendubot_craclf_olacp import plot_craclf_results

if __name__ == "__main__":
    results_filename = (
        "./simulations/pendubot/results/sim_pendubot_craclf_results_2026_1016_020859_844929.pkl"
    )
    with open(results_filename, "rb") as stream:
        data = pickle.load(stream)
        results = data["results"]

    plot_craclf_results(
        results, interval_duration=data["config"]["interval_duration"], file_format="png"
    )
