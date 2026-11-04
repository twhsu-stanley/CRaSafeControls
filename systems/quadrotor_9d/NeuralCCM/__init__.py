"""Neural CCM runtime; PyTorch is needed only for offline training."""

from pathlib import Path

from .neural_metric import NeuralCCMMetric

DEFAULT_METRIC = Path(__file__).resolve().parent / "metrics/metric.npz"

__all__ = ["NeuralCCMMetric", "DEFAULT_METRIC"]
