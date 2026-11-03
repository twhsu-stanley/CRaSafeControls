"""Right-invariant LMI CCM inference; CVXPY is needed only for training."""

from pathlib import Path

from .structured_metric import InvariantCCMMetric

METRIC_DIR = Path(__file__).resolve().parent / "metrics"
DEFAULT_METRIC = METRIC_DIR / "invariant_metric.npz"

__all__ = ["InvariantCCMMetric", "DEFAULT_METRIC"]
