import numpy as np

from systems.control_affine_system import ControlAffineSystem


class ACC(ControlAffineSystem):
    """Adaptive-cruise-control system from the CRaCBF example.

    The controller represents the unknown plant term as ``Y_theta(x) @ a``.
    The scalar representation parameter ``theta`` controls the wake-effect
    decay, while the seven-dimensional interval parameter ``a`` contains the
    drag coefficients and lead-vehicle velocity.  The physical uncertainty is
    supplied independently so online representation updates never alter the
    simulated plant.
    """

    theta_shape = (1,)
    xdim = 3
    udim = 1
    adim = 7

    def __init__(self, params=None):
        if params is None:
            params = {}
        elif not isinstance(params, dict):
            raise TypeError("Parameters must be a dictionary.")

        theta_init = np.asarray(
            params.get("Theta_init", np.array([0.05])), dtype=float
        ).reshape(-1)
        if theta_init.size != 1:
            raise ValueError("Theta_init must contain exactly one value")
        if not np.isfinite(theta_init[0]) or theta_init[0] <= 0.0:
            raise ValueError("Theta_init must be finite and strictly positive")
        self.Theta_hat = theta_init.reshape(self.theta_shape).copy()

        self.mass = float(params.get("m", 2000.0))
        self.desired_velocity = float(params.get("vd", 20.0))
        self.nominal_gain = float(params.get("Kp", 200.0))
        self.z_min = float(params.get("z_min", 5.0))
        self.lookahead_time = float(params.get("T_h", params.get("T", 1.0)))
        self.cbf_smoothing_epsilon = float(
            params.get("cbf_smoothing_epsilon", 0.1)
        )
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("m must be finite and strictly positive")
        if not np.isfinite(self.lookahead_time) or self.lookahead_time <= 0.0:
            raise ValueError("T_h must be finite and strictly positive")
        if (
            not np.isfinite(self.cbf_smoothing_epsilon)
            or self.cbf_smoothing_epsilon <= 0.0
        ):
            raise ValueError(
                "cbf_smoothing_epsilon must be finite and strictly positive"
            )

        self.true_uncertainty_fcn = params.get("true_uncertainty")
        if self.true_uncertainty_fcn is None:
            self.true_uncertainty_fcn = lambda x, t: np.zeros(self.xdim)
        if not callable(self.true_uncertainty_fcn):
            raise TypeError("true_uncertainty must be callable as w(x, t)")

        super().__init__(params)

    def f(self, x):
        """Return the nominal drift ``[v, 0, -v]``."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        v = x[1]
        return np.array([v, 0.0, -v])

    def g(self, x):
        """Return the wheel-force input matrix."""
        return np.array([[0.0], [1.0 / self.mass], [0.0]])

    @staticmethod
    def _theta_scalar(theta):
        theta_array = np.asarray(theta, dtype=float).reshape(-1)
        if theta_array.size != 1:
            raise ValueError("theta must contain exactly one value")
        theta_value = float(theta_array[0])
        if not np.isfinite(theta_value) or theta_value <= 0.0:
            raise ValueError("theta must be finite and strictly positive")
        return theta_value

    @classmethod
    def psi_theta(cls, x, theta):
        """Evaluate the six wake-effect features from Section V-B."""
        x = np.asarray(x, dtype=float).reshape(cls.xdim)
        v, z = x[1], x[2]
        theta_value = cls._theta_scalar(theta)
        wake_decay = np.exp(-theta_value * z)
        return np.array(
            [
                1.0,
                v,
                v**2,
                wake_decay,
                v * wake_decay,
                v**2 * wake_decay,
            ]
        )

    def Y(self, x):
        """Evaluate the currently installed uncertainty representation."""
        return self.Y_theta(x, self.Theta_hat)

    def Y_theta(self, x, theta):
        """Evaluate the paper's ``3 x 7`` representation matrix."""
        Yx = np.zeros((self.xdim, self.adim))
        Yx[1, :6] = self.psi_theta(x, theta)
        Yx[2, 6] = 1.0
        return self._validate_Y_shape(Yx)

    def representation_loss_gradient(self, x, theta, a, w):
        """Return ``grad_theta ||Y_theta(x) @ a - w||_2**2``."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        a = np.asarray(a, dtype=float).reshape(-1)
        w = np.asarray(w, dtype=float).reshape(-1)
        if a.size != self.adim:
            raise ValueError(f"a must have length {self.adim}")
        if w.size != self.xdim:
            raise ValueError(f"w must have length {self.xdim}")

        theta_value = self._theta_scalar(theta)
        v, z = x[1], x[2]
        wake_decay = np.exp(-theta_value * z)
        residual = self.Y_theta(x, theta) @ a - w
        model_derivative = -z * wake_decay * (
            a[3] + v * a[4] + v**2 * a[5]
        )
        return np.array([2.0 * residual[1] * model_derivative])

    def set_representation(self, theta):
        """Install a positive scalar representation parameter."""
        theta_value = self._theta_scalar(theta)
        self.Theta_hat = np.array([theta_value])

    @staticmethod
    def smooth_max(r, epsilon):
        """Smooth approximation ``0.5 * (r + sqrt(r**2 + epsilon**2))``."""
        r = np.asarray(r, dtype=float)
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and strictly positive")
        return 0.5 * (r + np.sqrt(r**2 + epsilon**2))

    @staticmethod
    def smooth_max_derivative(r, epsilon):
        """Derivative of :meth:`smooth_max` with respect to ``r``."""
        r = np.asarray(r, dtype=float)
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and strictly positive")
        return 0.5 * (1.0 + r / np.sqrt(r**2 + epsilon**2))

    def cbf(self, x, a_hat):
        """Return the smoothed, parameter-dependent collision-avoidance CBF."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        relative_velocity = x[1] - a_hat[6]
        h = (
            x[2]
            - self.z_min
            - self.lookahead_time
            * self.smooth_max(
                relative_velocity, self.cbf_smoothing_epsilon
            )
        )
        return np.asarray(h, dtype=float)

    def dcbfdx(self, x, a_hat):
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        relative_velocity = x[1] - a_hat[6]
        phi_prime = float(
            self.smooth_max_derivative(
                relative_velocity, self.cbf_smoothing_epsilon
            )
        )
        return np.array(
            [[0.0], [-self.lookahead_time * phi_prime], [1.0]]
        )

    def dcbfda(self, x, a_hat):
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        relative_velocity = x[1] - a_hat[6]
        phi_prime = float(
            self.smooth_max_derivative(
                relative_velocity, self.cbf_smoothing_epsilon
            )
        )
        gradient = np.zeros((self.adim, 1))
        gradient[6, 0] = self.lookahead_time * phi_prime
        return gradient

    def true_uncertainty(self, x, t):
        uncertainty = np.asarray(self.true_uncertainty_fcn(x, t), dtype=float)
        if uncertainty.shape != (self.xdim,):
            raise ValueError(
                f"true_uncertainty(x, t) must return shape ({self.xdim},)"
            )
        if not np.all(np.isfinite(uncertainty)):
            raise ValueError("true_uncertainty(x, t) must be finite")
        return uncertainty

    def dynamics(self, x, u, t=0.0):
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        u = np.asarray(u, dtype=float).reshape(self.udim)
        return self.f(x) + self.g(x) @ u + self.true_uncertainty(x, t)

    def dynamics_extended(self, x_ext, u, t=0.0):
        x_ext = np.asarray(x_ext, dtype=float).reshape(
            self.xdim + self.adim + 1
        )
        x = x_ext[: self.xdim]
        a_hat = x_ext[self.xdim : self.xdim + self.adim]
        rho = x_ext[self.xdim + self.adim]

        dxdt_ext = np.zeros(self.xdim + self.adim + 1)
        dxdt_ext[: self.xdim] = self.dynamics(x, u, t)
        if self.use_adaptive:
            a_hat_dot, rho_dot = self.adaptation_cracbf(x, a_hat, rho)
        else:
            a_hat_dot = np.zeros(self.adim)
            rho_dot = 0.0
        dxdt_ext[self.xdim : self.xdim + self.adim] = a_hat_dot
        dxdt_ext[self.xdim + self.adim] = rho_dot
        return dxdt_ext

    def ctrl_nominal(self, x):
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        return np.array(
            [self.nominal_gain * (self.desired_velocity - x[1])]
        )
