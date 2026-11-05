"""NumPy runtime and saved neural CCM for the 9D quadrotor."""

from pathlib import Path

from .neural_metric import NeuralCCMMetric

DEFAULT_METRIC = Path(__file__).resolve().parent / "metrics/metric.npz"

__all__ = ["NeuralCCMMetric", "DEFAULT_METRIC"]
