import os
import pickle
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from simulations.acc.sim_acc_cracbf_olacp import plot_cracbf_results

if __name__ == "__main__":
    results_filename = "./simulations/acc/sim_acc_cracbf_results_2026_1016_022118_899549.pkl"
    with open(results_filename, "rb") as stream:
        data = pickle.load(stream)
        results = data["results"]

    plot_cracbf_results(results)
    plt.show()
