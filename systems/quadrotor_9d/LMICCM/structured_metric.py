"""Right-invariant dual CCM for the 9D quadrotor: W(x) = J(x) Q J(x).T.

State order: x = [p_x, p_y, p_z, v_x, v_y, v_z, roll, pitch, yaw].
Only the constant symmetric matrix Q is learned; J(x) follows the input geometry.
Positive Q and invertible J ensure W > 0, while this frame gives full C2 = 0.
The separate trainer checks negative definiteness of C1 at finite (x, b) samples.

Here (v, R) are treated as SE(3) coordinates; physical position p stays separate.
Runtime metric evaluation in this module requires only NumPy.
"""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np


def normalize_domain(domain):
    """Return a copy with legacy a_min/a_max renamed to b_min/b_max."""
    domain = dict(domain)
    for bound in ("min", "max"):
        current_key = f"b_{bound}"
        legacy_key = f"a_{bound}"
        if current_key not in domain and legacy_key in domain:
            domain[current_key] = domain[legacy_key]
        domain.pop(legacy_key, None)
    return domain


def skew(v):
    """Return the 3x3 skew-symmetric matrix satisfying skew(v) @ z = v cross z."""
    vx, vy, vz = v
    return np.array([
        [0.0, -vz, vy],
        [vz, 0.0, -vx],
        [-vy, vx, 0.0],
    ])


def invariant_frame(x):
    """Map invariant tangent coordinates to [delta p, delta v, delta Euler].

    J is a prescribed 9x9 frame, distinct from the dynamics Jacobian A.
    dJ_dx[k, i, j] = partial J[i, j] / partial x[k]: derivative axis first.
    Invariance under the input flows makes full C2 vanish for W = J Q J.T.
    """
    velocity = x[3:6]
    pitch, yaw = x[7:9]
    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) < 1e-5:
        raise ValueError("Euler pitch singularity")

    sin_yaw = np.sin(yaw)
    cos_yaw = np.cos(yaw)
    tan_pitch = np.tan(pitch)
    sec_pitch = 1.0 / cos_pitch

    J = np.eye(9)
    # An infinitesimal rotation changes velocity by delta alpha cross v.
    J[3:6, 6:9] = -skew(velocity)
    # Convert a world-frame infinitesimal rotation to Euler-angle increments.
    J[6:9, 6:9] = [
        [cos_yaw * sec_pitch, sin_yaw * sec_pitch, 0.0],
        [-sin_yaw, cos_yaw, 0.0],
        [cos_yaw * tan_pitch, sin_yaw * tan_pitch, 1.0],
    ]

    # Position and roll derivatives are zero because J does not depend on them.
    dJ_dx = np.zeros((9, 9, 9))
    for axis in range(3):
        dJ_dx[3 + axis, 3:6, 6:9] = -skew(np.eye(3)[axis])

    dJ_dx[7, 6:9, 6:9] = [
        [cos_yaw * sec_pitch * tan_pitch, sin_yaw * sec_pitch * tan_pitch, 0.0],
        [0.0, 0.0, 0.0],
        [cos_yaw * sec_pitch**2, sin_yaw * sec_pitch**2, 0.0],
    ]
    dJ_dx[8, 6:9, 6:9] = [
        [-sin_yaw * sec_pitch, cos_yaw * sec_pitch, 0.0],
        [-cos_yaw, -sin_yaw, 0.0],
        [-sin_yaw * tan_pitch, cos_yaw * tan_pitch, 0.0],
    ]
    return J, dJ_dx


class InvariantCCMMetric:
    """Runtime interface for W = M^{-1} and its derivatives using a saved Q.

    W does not depend on the five physical coefficients b in this architecture.
    Their bounds are still stored because the C1 dynamics F(x, b) depend on b.
    """

    parameterization = "physical"
    adim = 5

    def __init__(self, checkpoint):
        """Load an NPZ and validate Q, the rate, the domain, and the W lower bound."""
        self.checkpoint_path = Path(checkpoint).resolve()
        with np.load(self.checkpoint_path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"].item()))
            self.Q = np.asarray(data["Q"], dtype=float)

        if (
            self.Q.shape != (9, 9)
            or not np.isfinite(self.Q).all()
            or not np.allclose(self.Q, self.Q.T, atol=1e-12)
        ):
            raise ValueError("Q must be a finite symmetric 9x9 matrix")
        # Cholesky rejects Q if it is not positive definite.
        np.linalg.cholesky(self.Q)
        self.Q.setflags(write=False)

        self.rate = float(self.metadata["rate"])
        if not np.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("Contraction rate must be finite and positive")

        domain = normalize_domain(self.metadata["domain"])
        self.metadata["domain"] = domain
        self.domain = {key: np.asarray(value, dtype=float) for key, value in domain.items()}
        for low_key, high_key, size in (("x_min", "x_max", 9), ("b_min", "b_max", 5)):
            lower = self.domain[low_key]
            upper = self.domain[high_key]
            if (
                lower.shape != (size,)
                or upper.shape != (size,)
                or not np.isfinite([lower, upper]).all()
                or np.any(lower > upper)
            ):
                raise ValueError("Metric domain must contain finite ordered bounds")

        max_abs_pitch = max(abs(self.domain["x_min"][7]), abs(self.domain["x_max"][7]))
        if max_abs_pitch >= np.pi / 2:
            raise ValueError("Metric domain crosses an Euler singularity")

        # Conservative eigenvalue bound from ||J^{-1}||_F^2 <= 9 + 3||v||^2.
        self.w_lb = float(self.metadata["w_lb"])
        max_abs_velocity = np.maximum(
            abs(self.domain["x_min"][3:6]), abs(self.domain["x_max"][3:6])
        )
        certified_lower_bound = np.linalg.eigvalsh(self.Q)[0] / (
            9 + 3 * (max_abs_velocity @ max_abs_velocity)
        )
        if not np.isfinite(self.w_lb) or not 0 < self.w_lb <= certified_lower_bound * (1 + 1e-6):
            raise ValueError("Invalid certified lower bound for W on the declared domain")

    @lru_cache(maxsize=128)
    def _evaluate(self, state):
        """Cache W and dW/dx at repeated states; return read-only arrays."""
        J, dJ_dx = invariant_frame(np.asarray(state))
        W = J @ self.Q @ J.T

        # Product rule: dW/dx_k = (dJ/dx_k) Q J.T + J Q (dJ/dx_k).T.
        dW_dx_first = dJ_dx @ self.Q @ J.T
        dW_dx_first += dW_dx_first.transpose(0, 2, 1)
        W.setflags(write=False)
        dW_dx_first.setflags(write=False)

        # The shared CCM interface expects the derivative axis last: [i, j, k].
        dW_dx = np.moveaxis(dW_dx_first, 0, -1)
        return W, dW_dx

    def _state(self, x, parameter):
        """Check input sizes and finiteness; return a hashable state tuple."""
        x = np.asarray(x, dtype=float).reshape(9)
        parameter = np.asarray(parameter, dtype=float).reshape(self.adim)
        if not np.isfinite(x).all() or not np.isfinite(parameter).all():
            raise ValueError("Metric state and parameters must be finite")
        return tuple(x)

    def W(self, x, parameter):
        """Return the 9x9 dual metric at state x."""
        return self._evaluate(self._state(x, parameter))[0]

    def derivatives(self, x, parameter):
        """Return (dW/dx, dW/db) with shapes (9, 9, 9) and (9, 9, 5)."""
        dW_dx = self._evaluate(self._state(x, parameter))[1]
        dW_db = np.zeros((9, 9, self.adim))
        return dW_dx, dW_db

    def in_training_domain(self, x, parameter, atol=1e-10):
        """Check stored x and b bounds; membership is not a global C1 certificate."""
        x = np.asarray(x)
        parameter = np.asarray(parameter)
        finite_inputs = np.isfinite(x).all() and np.isfinite(parameter).all()
        state_in_domain = (
            np.all(x >= self.domain["x_min"] - atol)
            and np.all(x <= self.domain["x_max"] + atol)
        )
        parameter_in_domain = (
            np.all(parameter >= self.domain["b_min"] - atol)
            and np.all(parameter <= self.domain["b_max"] + atol)
        )
        return bool(finite_inputs and state_in_domain and parameter_in_domain)
