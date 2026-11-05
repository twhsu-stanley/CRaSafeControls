"""Load the trained neural CCM used by the 9D quadrotor controller.

Metric evaluation and its analytic derivatives need only NumPy.
"""

from pathlib import Path

from .NeuralCCM.neural_metric import NeuralCCMMetric


DEFAULT_METRIC = Path(__file__).resolve().parent / "NeuralCCM" / "metrics" / "metric.npz"


def load_metric(checkpoint=None):
    """Load the bundled neural CCM or a custom trained neural checkpoint."""
    return NeuralCCMMetric(DEFAULT_METRIC if checkpoint is None else checkpoint)
