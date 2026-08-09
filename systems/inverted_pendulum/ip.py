import numpy as np
from scipy.linalg import eigvalsh, solve_continuous_are, solve_continuous_lyapunov

from systems.control_affine_system import ControlAffineSystem


class IP(ControlAffineSystem):
    """Direct-torque inverted pendulum with matched parametric uncertainty.

    A four-feature model approximates the damping, gravity, and wind uncertainty by

        Y_Theta(x) a = Psi(x) Theta a.

    The CRaCLF candidate is a parameter-dependent quadratic form obtained from
    the LQR Riccati equation for the upright certainty-equivalent linearization.
    """

    theta_shape = (4, 5)
    xdim = 2
    udim = 1
    adim = 5

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

        self.length = float(params.get("length", 1.0))
        self.mass = float(params.get("mass", 1.0))
        self.inertia = float(params.get("inertia", self.mass * self.length**2 / 3.0))
        self.grav = float(params.get("grav", params.get("g", 9.81)))
        self.true_grav = float(params.get("true_grav", self.grav))
        self.damping = float(params.get("damping", 0.01))
        self.true_damping = float(params.get("true_damping", self.damping))
        self.drag_coefficient = float(params.get("c_w", 0.0))

        self.wind_velocity_fcn = params.get("wind_velocity", lambda t: np.zeros(2))
        if not callable(self.wind_velocity_fcn):
            wind_velocity = np.asarray(self.wind_velocity_fcn, dtype=float).reshape(2)
            self.wind_velocity_fcn = lambda t: wind_velocity

        self.true_uncertainty_fcn = params.get("true_uncertainty")
        if self.true_uncertainty_fcn is not None and not callable(self.true_uncertainty_fcn):
            raise TypeError("true_uncertainty must be callable as w(x, t)")

        positive_parameters = {
            "length": self.length,
            "mass": self.mass,
            "inertia": self.inertia,
            "grav": self.grav,
            "true_grav": self.true_grav,
        }
        for name, value in positive_parameters.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if not np.isfinite(self.damping) or self.damping < 0.0:
            raise ValueError("damping must be finite and nonnegative")
        if not np.isfinite(self.true_damping) or self.true_damping < 0.0:
            raise ValueError("true_damping must be finite and nonnegative")
        if not np.isfinite(self.drag_coefficient) or self.drag_coefficient < 0.0:
            raise ValueError("c_w must be finite and nonnegative")

        self.lqr_Q = np.asarray(
            params.get("lqr_Q", np.diag([25.0, 4.0])), dtype=float
        ).reshape(self.xdim, self.xdim)
        self.lqr_R = float(params.get("lqr_R", 1.0))
        if not np.all(np.isfinite(self.lqr_Q)):
            raise ValueError("lqr_Q must be finite")
        if not np.allclose(self.lqr_Q, self.lqr_Q.T):
            raise ValueError("lqr_Q must be symmetric")
        if np.min(np.linalg.eigvalsh(self.lqr_Q)) <= 0.0:
            raise ValueError("lqr_Q must be positive definite")
        if not np.isfinite(self.lqr_R) or self.lqr_R <= 0.0:
            raise ValueError("lqr_R must be finite and strictly positive")

        super().__init__(params)
        self._invalidate_riccati_cache()

    def f(self, x):
        """Return the nominal drift for x = [phi, phi_dot]."""
        phi, phi_dot = np.asarray(x, dtype=float).reshape(self.xdim)
        angular_acceleration = (
            -self.damping * phi_dot
            + 0.5 * self.mass * self.grav * self.length * np.sin(phi)
        ) / self.inertia
        return np.array([phi_dot, angular_acceleration])

    def g(self, x):
        """Return the direct-torque input matrix."""
        np.asarray(x, dtype=float).reshape(self.xdim)
        return np.array([[0.0], [1.0 / self.inertia]])

    @staticmethod
    def psi(x):
        """Return the coarse four-feature acceleration-level matrix Psi(x)."""
        phi, phi_dot = np.asarray(x, dtype=float).reshape(2)
        return np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, np.sin(phi), np.cos(phi) - 1.0, phi_dot],
            ],
            dtype=float,
        )

    def Y(self, x):
        """Evaluate the currently installed uncertainty representation."""
        return self.Y_Theta(x, self.Theta_hat)

    def Y_Theta(self, x, theta):
        """Evaluate Y_Theta(x) = Psi(x) Theta."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.theta_shape:
            raise ValueError(f"theta must have shape {self.theta_shape}")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta must be finite")
        return self._validate_Y_shape(self.psi(x) @ theta)

    def representation_loss_gradient(self, x, theta, a, w):
        """Return grad_Theta ||Y_Theta(x) a - w||_2**2."""
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
        """Install a representation and invalidate its Riccati certificate."""
        Theta_hat = np.asarray(Theta_hat, dtype=float)
        if Theta_hat.shape != self.theta_shape:
            raise ValueError(f"Theta_hat must have shape {self.theta_shape}")
        if not np.all(np.isfinite(Theta_hat)):
            raise ValueError("Theta_hat must be finite")
        self.Theta_hat = Theta_hat.copy()
        self._invalidate_riccati_cache()

    def wind_torque(self, x, t=0.0):
        """Return the wind torque about the pivot."""
        phi, phi_dot = np.asarray(x, dtype=float).reshape(self.xdim)
        wind_velocity = np.asarray(self.wind_velocity_fcn(t), dtype=float).reshape(2)
        if not np.all(np.isfinite(wind_velocity)):
            raise ValueError("wind_velocity(t) must be finite")
        relative_velocity = np.array(
            [
                wind_velocity[0] - self.length * phi_dot * np.cos(phi),
                wind_velocity[1] + self.length * phi_dot * np.sin(phi),
            ]
        )
        wind_force = (
            self.drag_coefficient
            * np.linalg.norm(relative_velocity, ord=2)
            * relative_velocity
        )
        return self.length * (np.cos(phi) * wind_force[0] - np.sin(phi) * wind_force[1])

    def true_trim_input(self, t=0.0):
        """Return the physical input that trims the upright state."""
        return -float(self.wind_torque(np.zeros(self.xdim), t))

    def true_uncertainty(self, x, t=0.0):
        """Return the damping, gravity, and wind acceleration mismatch."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        if self.true_uncertainty_fcn is not None:
            uncertainty = np.asarray(self.true_uncertainty_fcn(x, t), dtype=float)
        else:
            phi, phi_dot = x
            angular_uncertainty = (
                (self.damping - self.true_damping) * phi_dot
                + 0.5 * self.mass * (self.true_grav - self.grav) * self.length * np.sin(phi)
                + self.wind_torque(x, t)
            ) / self.inertia
            uncertainty = np.array([0.0, angular_uncertainty])

        if uncertainty.shape != (self.xdim,):
            raise ValueError(f"true_uncertainty(x, t) must return shape ({self.xdim},)")
        if not np.all(np.isfinite(uncertainty)):
            raise ValueError("true_uncertainty(x, t) must be finite")
        return uncertainty

    def dynamics(self, x, u, t=0.0):
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        u = np.asarray(u, dtype=float).reshape(self.udim)
        return self.f(x) + self.g(x) @ u + self.true_uncertainty(x, t)

    def dynamics_extended(self, x_ext, u, t=0.0):
        x_ext = np.asarray(x_ext, dtype=float).reshape(self.xdim + self.adim + 1)
        x = x_ext[: self.xdim]
        a_hat = x_ext[self.xdim : self.xdim + self.adim]
        rho = x_ext[-1]

        dxdt_ext = np.zeros(self.xdim + self.adim + 1)
        dxdt_ext[: self.xdim] = self.dynamics(x, u, t)
        if self.use_adaptive:
            a_hat_dot, rho_dot = self.adaptation_craclf(x, a_hat, rho)
        else:
            a_hat_dot = np.zeros(self.adim)
            rho_dot = 0.0
        dxdt_ext[self.xdim : self.xdim + self.adim] = a_hat_dot
        dxdt_ext[-1] = rho_dot
        return dxdt_ext

    @staticmethod
    def _representation_linear_terms(beta):
        """Return position and velocity terms in the upright linearization."""
        beta = np.asarray(beta, dtype=float).reshape(4)
        return float(beta[1]), float(beta[3])

    def estimated_trim_input(self, a_hat):
        """Return the torque that centers the estimated upright drift."""
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        return -self.inertia * float((self.Theta_hat @ a_hat)[0])

    def certainty_equivalent_F(self, x, a_hat):
        """Return the equilibrium-centered certainty-equivalent drift."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        return (
            self.f(x)
            + self.Y(x) @ a_hat
            + self.g(x).reshape(self.xdim) * self.estimated_trim_input(a_hat)
        )

    def linearized_certainty_equivalent_F(self, a_hat):
        """Return the upright Jacobian A(a_hat) of the centered drift."""
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        position_term, velocity_term = self._representation_linear_terms(self.Theta_hat @ a_hat)
        nominal_position = 0.5 * self.mass * self.grav * self.length / self.inertia
        nominal_velocity = -self.damping / self.inertia
        return np.array(
            [[0.0, 1.0], [nominal_position + position_term, nominal_velocity + velocity_term]]
        )

    def _linearization_parameter_derivatives(self):
        """Return the exact derivatives dA/da_i for fixed Theta."""
        derivatives = np.zeros((self.adim, self.xdim, self.xdim))
        for index in range(self.adim):
            position_term, velocity_term = self._representation_linear_terms(
                self.Theta_hat[:, index]
            )
            derivatives[index, 1, 0] = position_term
            derivatives[index, 1, 1] = velocity_term
        return derivatives

    def _invalidate_riccati_cache(self):
        self._riccati_cache_a = None
        self._riccati_cache_P = None
        self._riccati_cache_A_cl = None
        self._riccati_cache_dP = None

    def _riccati_solution(self, a_hat):
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        if self._riccati_cache_a is not None and np.array_equal(a_hat, self._riccati_cache_a):
            return self._riccati_cache_P

        A = self.linearized_certainty_equivalent_F(a_hat)
        B = self.g(np.zeros(self.xdim))
        try:
            P = solve_continuous_are(A, B, self.lqr_Q, np.array([[self.lqr_R]]))
        except (ValueError, np.linalg.LinAlgError) as error:
            raise ValueError("failed to solve the CARE") from error
        P = 0.5 * (P + P.T)
        A_cl = A - B @ (B.T @ P) / self.lqr_R
        if np.min(np.linalg.eigvalsh(P)) <= 0.0:
            raise ValueError("the CARE solution P is not positive definite")
        if np.max(np.real(np.linalg.eigvals(A_cl))) >= 0.0:
            raise ValueError("the LQR closed loop is unstable (A-BK is not Hurwitz)")

        self._riccati_cache_a = a_hat.copy()
        self._riccati_cache_P = P
        self._riccati_cache_A_cl = A_cl
        self._riccati_cache_dP = None
        return P

    def _riccati_sensitivities(self, a_hat):
        self._riccati_solution(a_hat)
        if self._riccati_cache_dP is not None:
            return self._riccati_cache_dP

        P = self._riccati_cache_P
        A_cl = self._riccati_cache_A_cl
        sensitivities = np.zeros((self.adim, self.xdim, self.xdim))
        for index, A_i in enumerate(self._linearization_parameter_derivatives()):
            source = A_i.T @ P + P @ A_i
            P_i = solve_continuous_lyapunov(A_cl.T, -source)
            sensitivities[index] = 0.5 * (P_i + P_i.T)
        self._riccati_cache_dP = sensitivities
        return sensitivities

    def clf_matrix(self, a_hat):
        """Return the stabilizing CARE matrix P(a_hat)."""
        return self._riccati_solution(a_hat).copy()

    def clf(self, x, a_hat):
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        P = self._riccati_solution(a_hat)
        return np.asarray(0.5 * x @ P @ x)

    def dclfdx(self, x, a_hat):
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        P = self._riccati_solution(a_hat)
        return (P @ x).reshape(self.xdim, 1)

    def dclfda(self, x, a_hat):
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        sensitivities = self._riccati_sensitivities(a_hat)
        gradient = np.array([0.5 * x @ P_i @ x for P_i in sensitivities])
        return gradient.reshape(self.adim, 1)

    def lqr_gain(self, a_hat):
        """Return the local certainty-equivalent LQR gain."""
        P = self._riccati_solution(a_hat)
        B = self.g(np.zeros(self.xdim))
        return B.T @ P / self.lqr_R

    def local_lqr_control(self, x, a_hat):
        """Return estimated trim torque plus local LQR feedback."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        feedback = float((self.lqr_gain(a_hat) @ x).item())
        return np.array([self.estimated_trim_input(a_hat) - feedback])

    def local_decay_rate(self, a_hat):
        """Return the linear LQR decay rate certified by P(a_hat)."""
        P = self._riccati_solution(a_hat)
        gain = self.lqr_gain(a_hat)
        dissipation = self.lqr_Q + self.lqr_R * gain.T @ gain
        return float(np.min(eigvalsh(dissipation, P)))

    def ctrl_nominal(self, x):
        return np.zeros(self.udim)
