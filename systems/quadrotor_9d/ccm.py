"""Load bundled CCMs from this package's NeuralCCM and LMICCM subpackages.

Runtime evaluation needs only NumPy. Training frameworks are imported by the
offline trainers, never by this adapter or the controller.
"""

import importlib
import json
from pathlib import Path

import numpy as np


# Resolve saved models relative to this package, independent of the working directory.
PACKAGE_ROOT = Path(__file__).resolve().parent
METRICS = {
    "neural": ("NeuralCCM", "metric.npz"),
    "lmi": ("LMICCM", "invariant_metric.npz"),
}


def metric_checkpoint(name="neural"):
    """Return a bundled model path, independently of the working directory."""
    try:
        project, filename = METRICS[name]
    except KeyError:
        raise ValueError(f"Unknown CCM {name!r}; choose from {tuple(METRICS)}") from None
    return PACKAGE_ROOT / project / "metrics" / filename


def load_metric(checkpoint=None):
    """Dispatch a saved metric by its metadata; custom NPZs remain supported."""
    checkpoint = metric_checkpoint() if checkpoint is None else Path(checkpoint)
    with np.load(checkpoint, allow_pickle=False) as data:
        kind = json.loads(str(data["metadata"].item())).get("architecture", "neural")
    implementations = {
        "neural": ("NeuralCCM", "NeuralCCMMetric"),
        "right_invariant_constant": ("LMICCM", "InvariantCCMMetric"),
    }
    if kind not in implementations:
        raise ValueError(f"Unknown metric architecture: {kind}")
    module, name = implementations[kind]
    return getattr(importlib.import_module(f".{module}", package=__package__), name)(checkpoint)
