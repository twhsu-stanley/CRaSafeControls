"""Load the trained neural CCM used by the 9D quadrotor controller.

Metric evaluation and its analytic derivatives need only NumPy.
"""

from .NeuralCCM import DEFAULT_METRIC, NeuralCCMMetric


def load_metric(checkpoint=None):
    """Load the bundled neural CCM or a custom trained neural checkpoint."""
    return NeuralCCMMetric(DEFAULT_METRIC if checkpoint is None else checkpoint)
