"""Load a trained 9D dual CCM and evaluate it using NumPy only.

State order: x = [px, py, pz, vx, vy, vz, roll, pitch, yaw]. The eleven network
features are x[3:9] and the five physical coefficients b = Theta @ a_hat.
The fixed weights define W = M^{-1} = L.T @ L + w_lb I. The controller requests
W, dW/dx, and dW/db through this class; training is implemented separately.
"""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

# NPZ keys follow the three Linear layers at Sequential indices 0, 2, and 4.
WEIGHT_NAMES = ("weight0", "bias0", "weight2", "bias2", "weight4")


class NeuralCCMMetric:
    """Evaluate the saved metric and its analytic derivatives with fixed weights."""

    def __init__(self, checkpoint):
        """Read weights and metadata, then validate their shapes and domain bounds."""
        self.checkpoint_path = Path(checkpoint).resolve()
        with np.load(self.checkpoint_path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"].item()))
            self.weights = tuple(np.array(data[key], dtype=float) for key in WEIGHT_NAMES)

        self.parameterization = self.metadata["parameterization"]
        if self.parameterization != "physical":
            raise ValueError("The 9D metrics use five physical coefficients Theta@a")
        # Historical API name: adim counts b's five entries, not a_hat's four.
        self.adim = 5
        hidden = len(self.weights[1])
        expected_shapes = (
            (hidden, 6 + self.adim), (hidden,),
            (hidden, hidden), (hidden,),
            (81, hidden),
        )
        if tuple(weight.shape for weight in self.weights) != expected_shapes:
            raise ValueError("Unexpected NeuralCCM metric architecture")
        if not all(np.all(np.isfinite(weight)) for weight in self.weights):
            raise ValueError("Metric weights must be finite")
        for weight in self.weights:
            weight.setflags(write=False)

        self.w_lb = float(self.metadata["w_lb"])
        self.rate = float(self.metadata["rate"])
        if not np.isfinite(self.w_lb) or self.w_lb <= 0:
            raise ValueError("w_lb must be finite and positive")
        if not np.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("The contraction rate must be finite and positive")

        domain = self.metadata.get("domain")
        if domain is not None:
            # Normalize legacy names so new exports and runtime access use b_*.
            for bound in ("min", "max"):
                current_key = f"b_{bound}"
                legacy_key = f"a_{bound}"
                if current_key not in domain and legacy_key in domain:
                    domain[current_key] = domain[legacy_key]
                domain.pop(legacy_key, None)
        self.domain = None if domain is None else {
            name: np.asarray(domain[name], dtype=float).reshape(size)
            for name, size in (
                ("x_min", 9), ("x_max", 9),
                ("b_min", self.adim), ("b_max", self.adim),
            )
        }
        if self.domain is not None:
            finite_bounds = all(np.all(np.isfinite(value)) for value in self.domain.values())
            ordered_states = np.all(self.domain["x_min"] <= self.domain["x_max"])
            ordered_parameters = np.all(self.domain["b_min"] <= self.domain["b_max"])
            if not (finite_bounds and ordered_states and ordered_parameters):
                raise ValueError("Training bounds must be finite and ordered")

    def _features(self, x, parameter):
        """Validate inputs and return the eleven features as a hashable cache key."""
        x = np.asarray(x, dtype=float).reshape(9)
        parameter = np.asarray(parameter, dtype=float).reshape(self.adim)
        features = np.r_[x[3:9], parameter]
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(features)):
            raise ValueError("Metric state and parameters must be finite")
        return tuple(features)

    @lru_cache(maxsize=128)
    def _evaluate(self, features):
        """Return W (9, 9) and dW/dfeatures (9, 9, 11), both read-only.

        Compute all feature derivatives together by the chain rule. Repeated
        controller calls at the same features reuse this bounded cache.
        """
        weight_in, bias_in, weight_hidden, bias_hidden, weight_out = self.weights
        h1 = np.tanh(weight_in @ features + bias_in)
        h2 = np.tanh(weight_hidden @ h1 + bias_hidden)
        L = (weight_out @ h2).reshape(9, 9)
        W = L.T @ L + self.w_lb * np.eye(9)

        # d(tanh(z))/dz = 1 - tanh(z)^2. The final axis indexes features.
        dh1_dfeatures = (1 - h1 * h1)[:, None] * weight_in
        dh2_dfeatures = (1 - h2 * h2)[:, None] * (weight_hidden @ dh1_dfeatures)
        dL_dfeatures = (weight_out @ dh2_dfeatures).reshape(9, 9, -1)

        # Product rule for W = L.T L: (dL).T L + L.T (dL).
        dW_dfeatures = np.einsum("jip,jk->ikp", dL_dfeatures, L)
        dW_dfeatures += np.einsum("ji,jkp->ikp", L, dL_dfeatures)
        W.setflags(write=False)
        dW_dfeatures.setflags(write=False)
        return W, dW_dfeatures

    def W(self, x, parameter):
        """Return the dual metric W(x, b), with W >= w_lb I by construction."""
        return self._evaluate(self._features(x, parameter))[0]

    def derivatives(self, x, parameter):
        """Return (dW/dx, dW/db), shaped (9, 9, 9) and (9, 9, 5).

        Derivative axes are last: dW_dx[i, j, k] = partial W[i, j]/partial x[k].
        Position derivatives vanish because position is not a network input.
        The quadrotor adapter applies dW/db @ Theta to obtain dW/da_hat.
        """
        dW_dfeatures = self._evaluate(self._features(x, parameter))[1]
        dW_dx = np.zeros((9, 9, 9))
        dW_dx[:, :, 3:9] = dW_dfeatures[:, :, :6]
        dW_db = dW_dfeatures[:, :, 6:]
        return dW_dx, dW_db

    def in_training_domain(self, x, parameter, atol=1e-10):
        """Check stored bounds; return None when domain metadata is unavailable.

        Membership records the sampling scope, not a whole-domain C1/C2 proof.
        """
        if self.domain is None:
            return None
        x = np.asarray(x, dtype=float).reshape(9)
        parameter = np.asarray(parameter, dtype=float).reshape(self.adim)
        finite_inputs = np.all(np.isfinite(x)) and np.all(np.isfinite(parameter))
        state_in_domain = (
            np.all(x >= self.domain["x_min"] - atol)
            and np.all(x <= self.domain["x_max"] + atol)
        )
        parameter_in_domain = (
            np.all(parameter >= self.domain["b_min"] - atol)
            and np.all(parameter <= self.domain["b_max"] + atol)
        )
        return bool(finite_inputs and state_in_domain and parameter_in_domain)
