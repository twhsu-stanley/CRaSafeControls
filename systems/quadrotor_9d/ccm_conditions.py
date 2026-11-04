"""CCM conditions for the 9D estimated model F(x, b) = f_nominal(x) + Psi(x)b.

Neural and LMI CCMs share these dynamics and definitions of C1 and C2.
The five physical coefficients b are held fixed when taking state derivatives.
This estimated model is distinct from the simulated true plant with extra disturbances.
"""

from functools import lru_cache
from types import SimpleNamespace

import casadi as ca
import numpy as np

from systems.quadrotor_9d.quadrotor_9d import QUADROTOR_9D


@lru_cache(maxsize=1)
def dynamics_function():
    """Build a CasADi function mapping (x, b) to F, A, B, and dB_j/dx.

    SX represents symbolic variables before numerical values are supplied.
    Build derivative expressions once and reuse them through lru_cache.
    """
    system = QUADROTOR_9D(SimpleNamespace(rate=.5))
    x = ca.SX.sym("x", 9)
    b = ca.SX.sym("b", 5)
    state = np.array([x[i] for i in range(9)], dtype=object)

    def matrix(rows):
        # horzcat joins entries into rows; vertcat stacks those rows vertically.
        return ca.vertcat(*[ca.horzcat(*row) for row in rows])

    F = ca.vertcat(*system.f(state)) + matrix(system.psi(state)) @ b
    A = ca.jacobian(F, x)
    B = matrix(system.g(state))
    input_jacobians = [ca.jacobian(B[:, j], x) for j in range(4)]
    # Declare two symbolic inputs and seven outputs; numerical evaluation comes later.
    outputs = [F, A, B, *input_jacobians]
    return ca.Function("metric_model_derivatives", [x, b], outputs)


def model_derivatives(x, b):
    """Evaluate numeric (x, b) and return a tuple of seven NumPy arrays.

    Order and shapes: F (9, 1), A (9, 9), B (9, 4), and four dB_j/dx (9, 9).
    np.asarray converts each CasADi result; it does not select particular outputs.
    """
    return tuple(np.asarray(value) for value in dynamics_function()(x, b))


def projection(B):
    """Return an orthonormal basis P (9x5) satisfying B.T @ P = 0.

    B has column rank four on regular Euler coordinates. Its remaining left
    singular vectors span the orthogonal complement. P is not a square projector.
    """
    left_vectors, _, _ = np.linalg.svd(B, full_matrices=True)
    return left_vectors[:, B.shape[1]:]


def ccm_matrices(metric, x, b):
    """Return (W, C1, full C2, projected C2).

    Shapes: (9, 9), (5, 5), (4, 9, 9), and (4, 5, 5), respectively.
    C1 < 0 gives contraction; C2_j = 0 gives metric invariance under input flows.
    """
    F, A, B, *input_jacobians = model_derivatives(x, b)
    W = metric.W(x, b)
    dW_dx, _ = metric.derivatives(x, b)

    # dW_dx[i, j, k] = partial W[i, j] / partial x[k]: derivative axis last.
    # "ijk,k->ij" sums over k: D_F W = sum_k F_k * dW/dx_k.
    directional_W = np.einsum("ijk,k->ij", dW_dx, F.ravel())
    H = -directional_W + A @ W + W @ A.T + 2 * metric.rate * W

    # Restrict to dual directions satisfying B.T @ P = 0 to eliminate input terms.
    P = projection(B)
    C1 = P.T @ H @ P

    C2 = []
    for j, input_jacobian in enumerate(input_jacobians):
        directional_W = np.einsum("ijk,k->ij", dW_dx, B[:, j])
        residual = directional_W - input_jacobian @ W - W @ input_jacobian.T
        C2.append(residual)
    C2 = np.array(C2)

    # Full C2 = 0 implies projected C2 = 0; the converse generally does not hold.
    # Return both to distinguish full-matrix invariance from projected residuals.
    projected_C2 = np.array([P.T @ residual @ P for residual in C2])
    return W, C1, C2, projected_C2


def evaluate(metric, states, parameters):
    """Return a summary dictionary and per-sample checks (Nx5) as (summary, data).

    Columns: minimum eig(W), maximum eig(C1), maximum full C2 Frobenius norm,
    maximum projected C2 Frobenius norm, and training-domain membership.
    C1 pass fractions describe these samples, not a whole-domain certificate.
    """
    rows = []
    for x, b in zip(states, parameters):
        W, C1, C2, projected_C2 = ccm_matrices(metric, x, b)
        minimum_w = np.linalg.eigvalsh(W)[0]
        maximum_c1 = np.linalg.eigvalsh((C1 + C1.T) / 2)[-1]
        # axis=(1, 2) computes the Frobenius norm of each input's residual matrix.
        full_c2_norm = np.max(np.linalg.norm(C2, axis=(1, 2)))
        projected_c2_norm = np.max(np.linalg.norm(projected_C2, axis=(1, 2)))
        in_domain = metric.in_training_domain(x, b)
        rows.append((minimum_w, maximum_c1, full_c2_norm, projected_c2_norm, in_domain))

    data = np.asarray(rows)
    summary = dict(
        samples=len(data),
        minimum_W_eigenvalue=float(data[:, 0].min()),
        C1_pass_fraction=float(np.mean(data[:, 1] < 0)),
        worst_C1_eigenvalue=float(data[:, 1].max()),
        max_full_C2_frobenius=float(data[:, 2].max()),
        max_projected_C2_frobenius=float(data[:, 3].max()),
        in_domain_fraction=float(data[:, 4].mean()),
    )
    return summary, data
