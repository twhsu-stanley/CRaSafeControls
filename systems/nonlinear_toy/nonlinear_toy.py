import numpy as np

from systems.control_affine_system import ControlAffineSystem


class NONLINEAR_TOY(ControlAffineSystem):
    """Three-state nonlinear system used by the CRaCCM example.

    The learned uncertainty model is

        Y_Theta(x) a = Psi(x) Theta a,

    where Theta has shape (3, 2), a has length two, and

        Psi(x) = [[-x1, 0, 0],
                  [0, 0, 0],
                  [0, -x3, -x1**2]].

    The physical uncertainty is supplied independently through
    ``true_uncertainty`` and is zero by default.
    """

    theta_shape = (3, 2)
    xdim = 3
    udim = 1
    adim = 2

    def __init__(self, params=None):
        if params is None:
            params = {}
        elif not isinstance(params, dict):
            raise TypeError("Parameters must be a dictionary.")

        self.Theta_hat = np.asarray(
            params.get("Theta_init", np.zeros(self.theta_shape)), dtype=float
        ).reshape(self.theta_shape)
        if not np.all(np.isfinite(self.Theta_hat)):
            raise ValueError("Theta_init must be finite")

        self.true_uncertainty_fcn = params.get("true_uncertainty")
        if self.true_uncertainty_fcn is not None and not callable(self.true_uncertainty_fcn):
            raise TypeError("true_uncertainty must be callable as w(x, t)")

        super().__init__(params)

    def f(self, x):
        """Return the nominal drift.

        This deliberately avoids coercing ``x`` to floats so the same function
        can also be used by the CasADi-based nominal motion planner.
        """
        x1, x2, x3 = x
        return np.array([x3, x1**2 - x2, np.tanh(x2)])

    def g(self, x):
        """Return the constant control matrix."""
        return np.array([[0.0], [0.0], [1.0]])

    def psi(self, x):
        """Return the fixed feature matrix Psi(x)."""
        x1, _, x3 = np.asarray(x, dtype=float).reshape(self.xdim)
        return np.array(
            [
                [-x1, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, -x3, -(x1**2)],
            ]
        )

    def Y(self, x):
        """Evaluate the currently installed uncertainty representation."""
        return self.Y_Theta(x, self.Theta_hat)

    def Y_Theta(self, x, theta):
        """Evaluate Y_Theta(x) = Psi(x) @ Theta."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.theta_shape:
            raise ValueError(f"theta must have shape {self.theta_shape}")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta must be finite")
        return self._validate_Y_shape(self.psi(x) @ theta)

    def representation_loss_gradient(self, x, theta, a, w):
        """Return grad_Theta ||Y_Theta(x) @ a - w||_2**2."""
        a = np.asarray(a, dtype=float).reshape(-1)
        w = np.asarray(w, dtype=float).reshape(-1)
        if a.size != self.adim:
            raise ValueError(f"a must have length {self.adim}")
        if w.size != self.xdim:
            raise ValueError(f"w must have length {self.xdim}")

        Psi_x = self.psi(x)
        residual = self.Y_Theta(x, theta) @ a - w
        return 2.0 * np.outer(Psi_x.T @ residual, a)

    def set_representation(self, Theta_hat):
        """Install the representation used by the controller and CCM."""
        Theta_hat = np.asarray(Theta_hat, dtype=float)
        if Theta_hat.shape != self.theta_shape:
            raise ValueError(f"Theta_hat must have shape {self.theta_shape}")
        if not np.all(np.isfinite(Theta_hat)):
            raise ValueError("Theta_hat must be finite")
        self.Theta_hat = Theta_hat.copy()

    def true_uncertainty(self, x, t=0.0):
        """Return the physical uncertainty w(x, t), which is zero by default."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        if self.true_uncertainty_fcn is None:
            uncertainty = np.zeros(self.xdim)
        else:
            uncertainty = np.asarray(self.true_uncertainty_fcn(x, t), dtype=float)
        if uncertainty.shape != (self.xdim,):
            raise ValueError(f"true_uncertainty(x, t) must return shape ({self.xdim},)")
        if not np.all(np.isfinite(uncertainty)):
            raise ValueError("true_uncertainty(x, t) must be finite")
        return uncertainty

    def dynamics(self, x, u, t=0.0):
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        u = np.asarray(u, dtype=float).reshape(self.udim)
        return self.f(x) + self.g(x) @ u + self.true_uncertainty(x, t)

    def dynamics_extended(self, x_ext, x_d, u, geodesic_solver, t=0.0):
        x_ext = np.asarray(x_ext, dtype=float).reshape(self.xdim + self.adim + 1)
        x = x_ext[: self.xdim]
        a_hat = x_ext[self.xdim : self.xdim + self.adim]
        rho = x_ext[self.xdim + self.adim]

        dxdt_ext = np.zeros(self.xdim + self.adim + 1)
        dxdt_ext[: self.xdim] = self.dynamics(x, u, t)
        if self.use_adaptive:
            a_hat_dot, rho_dot = self.adaptation_craccm(
                x, x_d, a_hat, rho, geodesic_solver
            )
        else:
            a_hat_dot = np.zeros(self.adim)
            rho_dot = 0.0
        dxdt_ext[self.xdim : self.xdim + self.adim] = a_hat_dot
        dxdt_ext[self.xdim + self.adim] = rho_dot
        return dxdt_ext

    def _metric_parameter(self, a):
        """Return a1 = Theta[0, :] @ a used by the explicit dual CCM."""
        a = np.asarray(a, dtype=float).reshape(self.adim)
        return float(self.Theta_hat[0] @ a)

    def W_fcn(self, x, a):
        """Return the explicit dual CCM W(x, a) = M(x, a)^-1."""
        x1 = float(np.asarray(x, dtype=float).reshape(self.xdim)[0])
        a1 = self._metric_parameter(a)
        return np.array(
            [
                [1.42, 0.0, 1.42 * (a1 - 1.0)],
                [0.0, 6.21, -2.85 * x1],
                [
                    1.42 * (a1 - 1.0),
                    -2.85 * x1,
                    1.42 * a1**2 - 2.84 * a1 + 1.30 * x1**2 + 5.79,
                ],
            ]
        )

    def dW_dxi_fcn(self, i, x, a):
        """Return the partial derivative of W with respect to x[i]."""
        if i < 0 or i >= self.xdim:
            raise IndexError(f"state index must lie in [0, {self.xdim})")
        if i != 0:
            return np.zeros((self.xdim, self.xdim))
        x1 = float(np.asarray(x, dtype=float).reshape(self.xdim)[0])
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -2.85],
                [0.0, -2.85, 2.60 * x1],
            ]
        )

    def dW_dai_fcn(self, i, x, a):
        """Return the partial derivative of W with respect to a[i]."""
        if i < 0 or i >= self.adim:
            raise IndexError(f"parameter index must lie in [0, {self.adim})")
        a1 = self._metric_parameter(a)
        da1_dai = float(self.Theta_hat[0, i])
        return np.array(
            [
                [0.0, 0.0, 1.42 * da1_dai],
                [0.0, 0.0, 0.0],
                [1.42 * da1_dai, 0.0, (2.84 * a1 - 2.84) * da1_dai],
            ]
        )
