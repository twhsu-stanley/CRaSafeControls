import numpy as np
from scipy.linalg import (
    eigvalsh,
    solve_continuous_are,
    solve_continuous_lyapunov,
)

from systems.control_affine_system import ControlAffineSystem


class Pendubot(ControlAffineSystem):
    """Pendubot with learned damping, gravity, and wind uncertainty.

    The learned model is ``Y_Theta(x) @ a = Psi(x) @ Theta @ a``.
    Its only constant generalized-torque feature acts at the actuated joint,
    so every certainty-equivalent model retains the upright equilibrium after
    applying its estimated trim input. A parameter-dependent CARE supplies the
    local CRaCLF and its exact parameter sensitivities.
    """

    theta_shape = (13, 5)
    xdim = 4
    udim = 1
    adim = 5

    def __init__(self, params=None):
        if params is None:
            params = {}
        elif not isinstance(params, dict):
            raise TypeError("Parameters must be a dictionary.")

        self.Theta_hat = np.asarray(
            params.get("Theta_init", np.zeros(self.theta_shape)),
            dtype=float,
        ).reshape(self.theta_shape)
        if not np.all(np.isfinite(self.Theta_hat)):
            raise ValueError("Theta_init must be finite")

        self.mass_1 = float(params.get("m1", 1.0))
        self.mass_2 = float(params.get("m2", 1.0))
        self.length_1 = float(params.get("L1", 1.0))
        self.length_2 = float(params.get("L2", 1.0))
        self.com_1 = float(params.get("r1", 0.5 * self.length_1))
        self.com_2 = float(params.get("r2", 0.5 * self.length_2))
        self.inertia_1 = float(
            params.get("I1", self.mass_1 * self.length_1**2 / 12.0)
        )
        self.inertia_2 = float(
            params.get("I2", self.mass_2 * self.length_2**2 / 12.0)
        )
        self.grav = float(params.get("grav", params.get("g", 9.81)))
        self.true_grav = float(params.get("true_grav", self.grav))

        damping_default = np.array(
            [params.get("b1", 0.05), params.get("b2", 0.05)],
            dtype=float,
        )
        self.damping = np.asarray(
            params.get("damping", damping_default), dtype=float
        ).reshape(2)
        true_damping_default = np.array(
            [
                params.get("true_b1", self.damping[0]),
                params.get("true_b2", self.damping[1]),
            ],
            dtype=float,
        )
        self.true_damping = np.asarray(
            params.get("true_damping", true_damping_default), dtype=float
        ).reshape(2)

        self.aerodynamic_center = float(
            params.get("L_w", 0.5 * self.length_1)
        )
        self.drag_coefficient = float(params.get("c_w", 0.0))
        self.wind_velocity_fcn = params.get(
            "wind_velocity", lambda t: np.zeros(2)
        )
        if not callable(self.wind_velocity_fcn):
            wind_velocity = np.asarray(
                self.wind_velocity_fcn, dtype=float
            ).reshape(2)
            self.wind_velocity_fcn = lambda t: wind_velocity

        self.true_uncertainty_fcn = params.get("true_uncertainty")
        if (
            self.true_uncertainty_fcn is not None
            and not callable(self.true_uncertainty_fcn)
        ):
            raise TypeError("true_uncertainty must be callable as w(x, t)")

        positive_parameters = {
            "m1": self.mass_1,
            "m2": self.mass_2,
            "L1": self.length_1,
            "L2": self.length_2,
            "r1": self.com_1,
            "r2": self.com_2,
            "I1": self.inertia_1,
            "I2": self.inertia_2,
            "grav": self.grav,
            "true_grav": self.true_grav,
            "L_w": self.aerodynamic_center,
        }
        for name, value in positive_parameters.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if self.com_1 > self.length_1 or self.com_2 > self.length_2:
            raise ValueError("each center-of-mass distance must not exceed its link length")
        if self.aerodynamic_center > self.length_1:
            raise ValueError("L_w must not exceed the first-link length")
        if np.any(~np.isfinite(self.damping)) or np.any(self.damping < 0.0):
            raise ValueError("damping must be finite and nonnegative")
        if np.any(~np.isfinite(self.true_damping)) or np.any(
            self.true_damping < 0.0
        ):
            raise ValueError("true_damping must be finite and nonnegative")
        if not np.isfinite(self.drag_coefficient) or self.drag_coefficient < 0.0:
            raise ValueError("c_w must be finite and nonnegative")

        self.mu_1 = (
            self.inertia_1
            + self.inertia_2
            + self.mass_1 * self.com_1**2
            + self.mass_2 * (self.length_1**2 + self.com_2**2)
        )
        self.mu_2 = self.mass_2 * self.length_1 * self.com_2
        self.mu_3 = self.inertia_2 + self.mass_2 * self.com_2**2
        self.input_direction = np.array([1.0, 0.0])

        self.lqr_Q = np.asarray(
            params.get("lqr_Q", np.diag([25.0, 25.0, 4.0, 4.0])),
            dtype=float,
        ).reshape(self.xdim, self.xdim)
        self.lqr_R = float(params.get("lqr_R", 1.0))
        if not np.allclose(self.lqr_Q, self.lqr_Q.T):
            raise ValueError("lqr_Q must be symmetric")
        if np.min(np.linalg.eigvalsh(self.lqr_Q)) <= 0.0:
            raise ValueError("lqr_Q must be positive definite")
        if not np.isfinite(self.lqr_R) or self.lqr_R <= 0.0:
            raise ValueError("lqr_R must be finite and strictly positive")

        for q2 in (0.0, np.pi):
            if np.min(np.linalg.eigvalsh(self.mass_matrix([0.0, q2]))) <= 0.0:
                raise ValueError("the physical parameters must define a positive-definite mass matrix")

        super().__init__(params)
        self._invalidate_riccati_cache()

    def mass_matrix(self, q):
        """Return the two-link inertia matrix ``M(q)``."""
        q = np.asarray(q, dtype=float).reshape(2)
        cosine = np.cos(q[1])
        return np.array(
            [
                [
                    self.mu_1 + 2.0 * self.mu_2 * cosine,
                    self.mu_3 + self.mu_2 * cosine,
                ],
                [self.mu_3 + self.mu_2 * cosine, self.mu_3],
            ]
        )

    def coriolis_vector(self, q, q_dot):
        """Return the Coriolis/centrifugal vector ``h(q, q_dot)``."""
        q = np.asarray(q, dtype=float).reshape(2)
        q_dot = np.asarray(q_dot, dtype=float).reshape(2)
        sine = np.sin(q[1])
        return np.array(
            [
                -self.mu_2
                * sine
                * (2.0 * q_dot[0] * q_dot[1] + q_dot[1] ** 2),
                self.mu_2 * sine * q_dot[0] ** 2,
            ]
        )

    def gravity_shape(self, q):
        """Return the vector ``s(q)`` multiplying gravitational acceleration."""
        q = np.asarray(q, dtype=float).reshape(2)
        total_angle = q[0] + q[1]
        distal_term = self.mass_2 * self.com_2 * np.sin(total_angle)
        return np.array(
            [
                (self.mass_1 * self.com_1 + self.mass_2 * self.length_1)
                * np.sin(q[0])
                + distal_term,
                distal_term,
            ]
        )

    def f(self, x):
        """Return the nominal Pendubot drift."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        q, q_dot = x[:2], x[2:]
        generalized_force = (
            -self.coriolis_vector(q, q_dot)
            - self.damping * q_dot
            + self.grav * self.gravity_shape(q)
        )
        q_ddot = np.linalg.solve(self.mass_matrix(q), generalized_force)
        return np.hstack((q_dot, q_ddot))

    def g(self, x):
        """Return the first-joint torque input matrix."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        acceleration_direction = np.linalg.solve(
            self.mass_matrix(x[:2]), self.input_direction
        )
        return np.vstack((np.zeros((2, 1)), acceleration_direction[:, None]))

    @staticmethod
    def feature_vector(x):
        """Return the six nonconstant generalized-torque features."""
        q1, q2, q1_dot, q2_dot = np.asarray(x, dtype=float).reshape(4)
        return np.array(
            [
                np.sin(q1),
                np.sin(q1 + q2),
                np.cos(q1) - 1.0,
                np.cos(q1 + q2) - 1.0,
                q1_dot,
                q2_dot,
            ]
        )

    @classmethod
    def generalized_torque_features(cls, x):
        """Return the draft's ``2 x 13`` matrix ``Psi_tau(x)``."""
        features = cls.feature_vector(x)
        Psi_tau = np.zeros((2, 13))
        Psi_tau[0, 0] = 1.0
        Psi_tau[0, 1:7] = features
        Psi_tau[1, 7:13] = features
        return Psi_tau

    def psi(self, x):
        """Return the full state-space feature matrix ``Psi(x)``."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        acceleration_features = np.linalg.solve(
            self.mass_matrix(x[:2]), self.generalized_torque_features(x)
        )
        return np.vstack((np.zeros((2, 13)), acceleration_features))

    def Y(self, x):
        """Evaluate the currently installed Pendubot representation."""
        return self.Y_theta(x, self.Theta_hat)

    def Y_theta(self, x, theta):
        """Evaluate ``Y_Theta(x) = Psi(x) @ Theta``."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.theta_shape:
            raise ValueError(f"theta must have shape {self.theta_shape}")
        return self._validate_Y_shape(self.psi(x) @ theta)

    def representation_loss_gradient(self, x, theta, a, w):
        """Return ``grad_Theta ||Y_Theta(x) @ a - w||_2**2``."""
        a = np.asarray(a, dtype=float).reshape(-1)
        w = np.asarray(w, dtype=float).reshape(-1)
        if a.size != self.adim:
            raise ValueError(f"a must have length {self.adim}")
        if w.size != self.xdim:
            raise ValueError(f"w must have length {self.xdim}")

        Psi_x = self.psi(x)
        residual = self.Y_theta(x, theta) @ a - w
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
        """Return the generalized torque from wind acting on link one."""
        q1, _, q1_dot, _ = np.asarray(x, dtype=float).reshape(self.xdim)
        wind_velocity = np.asarray(
            self.wind_velocity_fcn(t), dtype=float
        ).reshape(2)
        relative_velocity = np.array(
            [
                wind_velocity[0]
                - self.aerodynamic_center * q1_dot * np.cos(q1),
                wind_velocity[1]
                + self.aerodynamic_center * q1_dot * np.sin(q1),
            ]
        )
        wind_force = (
            self.drag_coefficient
            * np.linalg.norm(relative_velocity, ord=2)
            * relative_velocity
        )
        actuated_torque = self.aerodynamic_center * (
            np.cos(q1) * wind_force[0] - np.sin(q1) * wind_force[1]
        )
        return np.array([actuated_torque, 0.0])

    def true_trim_input(self, t=0.0):
        """Return the unknown physical input that trims the upright state."""
        return -float(self.wind_torque(np.zeros(self.xdim), t)[0])

    def true_uncertainty(self, x, t=0.0):
        """Return the damping, gravity, and wind acceleration mismatch."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        if self.true_uncertainty_fcn is not None:
            uncertainty = np.asarray(
                self.true_uncertainty_fcn(x, t), dtype=float
            )
        else:
            q, q_dot = x[:2], x[2:]
            generalized_mismatch = (
                (self.damping - self.true_damping) * q_dot
                + (self.true_grav - self.grav) * self.gravity_shape(q)
                + self.wind_torque(x, t)
            )
            acceleration_mismatch = np.linalg.solve(
                self.mass_matrix(q), generalized_mismatch
            )
            uncertainty = np.hstack((np.zeros(2), acceleration_mismatch))

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
            a_hat_dot, rho_dot = self.adaptation_craclf(x, a_hat, rho)
        else:
            a_hat_dot = np.zeros(self.adim)
            rho_dot = 0.0
        dxdt_ext[self.xdim : self.xdim + self.adim] = a_hat_dot
        dxdt_ext[self.xdim + self.adim] = rho_dot
        return dxdt_ext

    @staticmethod
    def _representation_linear_terms(beta):
        """Return the position/velocity linearization of ``Psi_tau beta``."""
        beta = np.asarray(beta, dtype=float).reshape(13)
        torque_position = np.array(
            [
                [beta[1] + beta[2], beta[2]],
                [beta[7] + beta[8], beta[8]],
            ]
        )
        torque_velocity = np.array(
            [[beta[5], beta[6]], [beta[11], beta[12]]]
        )
        return torque_position, torque_velocity

    def estimated_trim_input(self, a_hat):
        """Return ``-e1.T @ Psi_tau(0) @ Theta @ a_hat``."""
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        return -float((self.Theta_hat @ a_hat)[0])

    def certainty_equivalent_drift(self, x, a_hat):
        """Return the equilibrium-centered estimated drift ``F_a(x)``."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        return (
            self.f(x)
            + self.Y(x) @ a_hat
            + self.g(x).reshape(self.xdim) * self.estimated_trim_input(a_hat)
        )

    def _gravity_shape_jacobian(self):
        distal = self.mass_2 * self.com_2
        return np.array(
            [
                [
                    self.mass_1 * self.com_1
                    + self.mass_2 * self.length_1
                    + distal,
                    distal,
                ],
                [distal, distal],
            ]
        )

    def linearized_certainty_equivalent_drift(self, a_hat):
        """Return ``A(a_hat) = dF_a/dx`` at the upright state."""
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        beta = self.Theta_hat @ a_hat
        representation_q, representation_v = self._representation_linear_terms(
            beta
        )
        torque_q = self.grav * self._gravity_shape_jacobian() + representation_q
        torque_v = -np.diag(self.damping) + representation_v
        mass_upright = self.mass_matrix(np.zeros(2))

        A = np.zeros((self.xdim, self.xdim))
        A[:2, 2:] = np.eye(2)
        A[2:, :2] = np.linalg.solve(mass_upright, torque_q)
        A[2:, 2:] = np.linalg.solve(mass_upright, torque_v)
        return A

    def _linearization_parameter_derivatives(self):
        """Return the exact derivatives ``dA/da_i`` for fixed ``Theta``."""
        mass_upright = self.mass_matrix(np.zeros(2))
        derivatives = np.zeros((self.adim, self.xdim, self.xdim))
        for index in range(self.adim):
            torque_q, torque_v = self._representation_linear_terms(
                self.Theta_hat[:, index]
            )
            derivatives[index, 2:, :2] = np.linalg.solve(
                mass_upright, torque_q
            )
            derivatives[index, 2:, 2:] = np.linalg.solve(
                mass_upright, torque_v
            )
        return derivatives

    def _invalidate_riccati_cache(self):
        self._riccati_cache_a = None
        self._riccati_cache_P = None
        self._riccati_cache_A_cl = None
        self._riccati_cache_dP = None

    def _riccati_solution(self, a_hat):
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        if (
            self._riccati_cache_a is not None
            and np.array_equal(a_hat, self._riccati_cache_a)
        ):
            return self._riccati_cache_P

        A = self.linearized_certainty_equivalent_drift(a_hat)
        B = self.g(np.zeros(self.xdim))
        try:
            P = solve_continuous_are(
                A, B, self.lqr_Q, np.array([[self.lqr_R]])
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            raise ValueError(
                "failed to solve the CARE"
            ) from error
        P = 0.5 * (P + P.T)
        A_cl = A - B @ (B.T @ P) / self.lqr_R
        if np.min(np.linalg.eigvalsh(P)) <= 0.0:
            raise ValueError("the CARE solution P is not positive definite")
        if np.max(np.real(np.linalg.eigvals(A_cl))) >= 0.0:
            raise ValueError("the LQR closed loop is unstable (i.e., A-BK is not Hurwitz)")

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
        for index, A_i in enumerate(
            self._linearization_parameter_derivatives()
        ):
            source = A_i.T @ P + P @ A_i
            P_i = solve_continuous_lyapunov(A_cl.T, -source)
            sensitivities[index] = 0.5 * (P_i + P_i.T)
        self._riccati_cache_dP = sensitivities
        return sensitivities

    def clf_matrix(self, a_hat):
        """Return the stabilizing CARE matrix ``P(a_hat)``."""
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
        gradient = np.array(
            [0.5 * x @ P_i @ x for P_i in sensitivities]
        )
        return gradient.reshape(self.adim, 1)

    def lqr_gain(self, a_hat):
        """Return the local certainty-equivalent LQR gain."""
        P = self._riccati_solution(a_hat)
        B = self.g(np.zeros(self.xdim))
        return B.T @ P / self.lqr_R

    def local_lqr_control(self, x, a_hat):
        """Return the estimated trim input plus local LQR feedback."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        return np.array(
            [
                self.estimated_trim_input(a_hat)
                - float((self.lqr_gain(a_hat) @ x).item())
            ]
        )

    def local_decay_rate(self, a_hat):
        """Return the linear LQR decay rate certified by ``P(a_hat)``."""
        P = self._riccati_solution(a_hat)
        gain = self.lqr_gain(a_hat)
        dissipation = self.lqr_Q + self.lqr_R * gain.T @ gain
        return float(np.min(eigvalsh(dissipation, P)))

    def ctrl_nominal(self, x):
        return np.zeros(self.udim)
        #return self.local_lqr_control(x, self.a_center)


PENDUBOT = Pendubot
