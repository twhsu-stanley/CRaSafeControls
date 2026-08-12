import numpy as np

from systems.control_affine_system import ControlAffineSystem


class ACC(ControlAffineSystem):
    """Adaptive-cruise-control system from the CRaCBF example.

    The learned uncertainty model is

        Y_Theta(x) a =
            [0, Psi(x) Theta a[:3], lead_velocity_scale * a[3]]^T,

    where Psi(x) = [1, v, v^2] kron [1, z, z^2] and Theta has
    shape (9, 3). The fourth interval parameter is the lead-vehicle
    velocity deviation Delta v_l divided by lead_velocity_scale and is
    used by the parameter-dependent CRaCBF. The physical uncertainty is
    supplied independently so representation updates never alter the
    simulated plant.
    """

    theta_shape = (9, 3)
    xdim = 3
    udim = 1
    adim = 4
    nominal_lead_velocity = 25.0
    lead_velocity_scale = 10.0
    cbf_smoothing_epsilon = 0.1

    def __init__(self, params=None):
        if params is None:
            params = {}
        elif not isinstance(params, dict):
            raise TypeError("Parameters must be a dictionary.")

        Theta_init = np.asarray(
            params.get("Theta_init", np.zeros(self.theta_shape)),
            dtype=float,
        )
        if Theta_init.shape != self.theta_shape:
            raise ValueError(
                f"Theta_init must have shape {self.theta_shape}"
            )
        if np.any(~np.isfinite(Theta_init)):
            raise ValueError("Theta_init must be finite")
        self.Theta_hat = Theta_init.copy()

        self.mass = float(params.get("m", 2000.0))
        self.desired_velocity = float(params.get("vd", 20.0))
        self.nominal_gain = float(params.get("Kp", 200.0))
        self.nominal_lead_velocity = float(
            params.get(
                "nominal_lead_velocity",
                self.nominal_lead_velocity,
            )
        )
        self.lead_velocity_scale = float(
            params.get(
                "lead_velocity_scale",
                self.lead_velocity_scale,
            )
        )
        self.cbf_smoothing_epsilon = float(
            params.get(
                "cbf_smoothing_epsilon",
                self.cbf_smoothing_epsilon,
            )
        )
        self.z_min = float(params.get("z_min", 5.0))
        self.lookahead_time = float(params.get("T_h", params.get("T", 1.0)))
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("m must be finite and strictly positive")
        if not np.isfinite(self.desired_velocity):
            raise ValueError("vd must be finite")
        if not np.isfinite(self.nominal_gain) or self.nominal_gain < 0.0:
            raise ValueError("Kp must be finite and nonnegative")
        if not np.isfinite(self.nominal_lead_velocity):
            raise ValueError(
                "nominal_lead_velocity must be finite"
            )
        if (
            not np.isfinite(self.lead_velocity_scale)
            or self.lead_velocity_scale <= 0.0
        ):
            raise ValueError(
                "lead_velocity_scale must be finite and strictly positive"
            )
        if (
            not np.isfinite(self.cbf_smoothing_epsilon)
            or self.cbf_smoothing_epsilon <= 0.0
        ):
            raise ValueError(
                "cbf_smoothing_epsilon must be finite and strictly positive"
            )
        if not np.isfinite(self.lookahead_time) or self.lookahead_time <= 0.0:
            raise ValueError("T_h must be finite and strictly positive")
        if not np.isfinite(self.z_min):
            raise ValueError("z_min must be finite")
        
        self.true_uncertainty_fcn = params.get("true_uncertainty")
        if self.true_uncertainty_fcn is None:
            self.true_uncertainty_fcn = lambda x, t: np.zeros(self.xdim)
        if not callable(self.true_uncertainty_fcn):
            raise TypeError("true_uncertainty must be callable as w(x, t)")

        super().__init__(params)

    def f(self, x):
        """Return the nominal drift [v, 0, nominal_lead_velocity - v]"""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        v = x[1]
        return np.array([v, 0.0, self.nominal_lead_velocity - v])

    def g(self, x):
        """Return the wheel-force input matrix"""
        return np.array([[0.0], [1.0 / self.mass], [0.0]])

    @staticmethod
    def _theta_matrix(theta):
        theta_array = np.asarray(theta, dtype=float)
        if theta_array.shape != ACC.theta_shape:
            raise ValueError(f"theta must have shape {ACC.theta_shape}")
        if np.any(~np.isfinite(theta_array)):
            raise ValueError("theta must be finite")
        return theta_array

    @classmethod
    def psi(cls, x):
        """Evaluate [1, v, v^2] kron [1, z, z^2]"""
        x = np.asarray(x, dtype=float).reshape(cls.xdim)
        v, z = x[1], x[2]
        return np.kron(
            np.array([1.0, v, v**2]),
            np.array([1.0, z, z**2]),
        )

    def Y(self, x):
        """Evaluate the currently installed uncertainty representation"""
        return self.Y_Theta(x, self.Theta_hat)

    def Y_Theta(self, x, theta):
        """Evaluate the 3 x 4 representation matrix from Section V-B"""
        theta_matrix = self._theta_matrix(theta)
        Yx = np.zeros((self.xdim, self.adim))
        Yx[1, :3] = self.psi(x) @ theta_matrix
        Yx[2, 3] = self.lead_velocity_scale
        return self._validate_Y_shape(Yx)

    def representation_loss_gradient(self, x, theta, a, w):
        """Return grad_theta ||Y_Theta(x) @ a - w||_2**2"""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        a = np.asarray(a, dtype=float).reshape(-1)
        w = np.asarray(w, dtype=float).reshape(-1)
        if a.size != self.adim:
            raise ValueError(f"a must have length {self.adim}")
        if w.size != self.xdim:
            raise ValueError(f"w must have length {self.xdim}")

        self._theta_matrix(theta)
        residual = self.Y_Theta(x, theta) @ a - w
        return 2.0 * residual[1] * np.outer(self.psi(x), a[:3])

    def set_representation(self, theta):
        """Install a finite 9 x 3 representation matrix"""
        self.Theta_hat = self._theta_matrix(theta).copy()

    @staticmethod
    def smooth_max(r, epsilon):
        """Smooth approximation 0.5 * (r + sqrt(r**2 + epsilon**2))"""
        r = np.asarray(r, dtype=float)
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and strictly positive")
        return 0.5 * (r + np.sqrt(r**2 + epsilon**2))

    @staticmethod
    def smooth_max_derivative(r, epsilon):
        """Derivative of :meth:`smooth_max` with respect to r"""
        r = np.asarray(r, dtype=float)
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and strictly positive")
        return 0.5 * (1.0 + r / np.sqrt(r**2 + epsilon**2))

    def cbf(self, x, a_hat):
        """Return the smoothed, parameter-dependent collision-avoidance CBF"""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        relative_velocity = (
            x[1] - self.nominal_lead_velocity - self.lead_velocity_scale * a_hat[3]
        )
        h = (
            x[2]
            - self.z_min
            - self.lookahead_time
            * self.smooth_max(relative_velocity, self.cbf_smoothing_epsilon)
        )
        return np.asarray(h, dtype=float)

    def dcbfdx(self, x, a_hat):
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        relative_velocity = (
            x[1] - self.nominal_lead_velocity - self.lead_velocity_scale * a_hat[3]
        )
        phi_prime = self.smooth_max_derivative(
            relative_velocity,
            self.cbf_smoothing_epsilon,
        )
        return np.array(
            [[0.0], [-self.lookahead_time * phi_prime], [1.0]]
        )

    def dcbfda(self, x, a_hat):
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        relative_velocity = (
            x[1] - self.nominal_lead_velocity - self.lead_velocity_scale * a_hat[3]
        )
        phi_prime = self.smooth_max_derivative(
            relative_velocity,
            self.cbf_smoothing_epsilon,
        )
        gradient = np.zeros((self.adim, 1))
        gradient[3, 0] = (
            self.lead_velocity_scale * self.lookahead_time * phi_prime
        )
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
