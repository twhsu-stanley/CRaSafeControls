import numpy as np
import sympy as sp

from systems.control_affine_system import ControlAffineSystem


class IP(ControlAffineSystem):
    """Inverted pendulum with actuator dynamics and unmatched uncertainty.

    The controller represents the unknown plant and wind terms as

        Y_Theta(x) a = Psi(x) Theta a,

    where the uncertainty enters the angular-acceleration equation and the
    commanded torque enters the actuator equation.  The CRaCLF is the
    parameter-dependent backstepping construction from the example.
    """

    theta_shape = (5, 4)
    xdim = 3
    udim = 1
    adim = 4

    def __init__(self, params=None):
        if params is None:
            params = {}

        self.Theta_hat = np.asarray(
            params.get(
                "Theta_init",
                np.vstack((np.eye(self.adim), np.zeros((1, self.adim)))),
            ),
            dtype=float,
        ).reshape(self.theta_shape)

        self.length = float(params.get("l", params.get("L", 1.0)))
        self.mass = float(params.get("m", 1.0))
        self.grav = float(params.get("grav", params.get("g", 9.81)))
        self.damping = float(params.get("b", 0.01))
        self.inertia = float(
            params.get("I", self.mass * self.length**2 / 3.0)
        )
        self.actuator_time_constant = float(params.get("T_a", 0.1))
        self.drag_coefficient = float(params.get("c_w", 0.0))
        self.true_damping = float(params.get("b_star", self.damping))
        self.true_mass = float(params.get("m_star", self.mass))

        if self.length <= 0.0:
            raise ValueError("pendulum length must be positive")
        if self.mass <= 0.0:
            raise ValueError("pendulum mass must be positive")
        if self.inertia <= 0.0:
            raise ValueError("pendulum inertia must be positive")
        if self.actuator_time_constant <= 0.0:
            raise ValueError("T_a must be positive")
        if self.drag_coefficient < 0.0:
            raise ValueError("c_w must be nonnegative")

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

        super().__init__(params)
        self._lambdify_symbolic_clf()

    def f(self, x):
        """Return the nominal drift for [phi, phi_dot, tau_a]."""
        phi, phi_dot, tau_a = np.asarray(x, dtype=float).reshape(self.xdim)
        angular_acceleration = (
            -self.damping * phi_dot
            - 0.5 * self.mass * self.grav * self.length * np.sin(phi)
            - tau_a
        ) / self.inertia
        return np.array(
            [
                phi_dot,
                angular_acceleration,
                -tau_a / self.actuator_time_constant,
            ]
        )

    def g(self, x):
        """Return the commanded-torque input matrix."""
        return np.array(
            [[0.0], [0.0], [1.0 / self.actuator_time_constant]]
        )

    @staticmethod
    def psi(x):
        """Return the centered feature matrix Psi(x)."""
        phi, phi_dot, _ = np.asarray(x, dtype=float).reshape(3)
        return np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    phi_dot,
                    np.sin(phi),
                    np.cos(phi) - 1.0,
                    phi_dot * np.sin(phi),
                    phi_dot * np.cos(phi),
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    def Y(self, x):
        """Evaluate the currently installed pendulum representation."""
        return self.Y_theta(x, self.Theta_hat)

    def Y_theta(self, x, theta):
        """Evaluate Y_theta(x) = Psi(x) @ theta."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.theta_shape:
            raise ValueError(f"theta must have shape {self.theta_shape}")
        return self._validate_Y_shape(self.psi(x) @ theta)

    def representation_loss_gradient(self, x, theta, a, w):
        """Return grad_theta ||Y_theta(x) @ a - w||_2**2."""
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
        """Install a new representation without recompiling certificates."""
        Theta_hat = np.asarray(Theta_hat, dtype=float)
        if Theta_hat.shape != self.theta_shape:
            raise ValueError(f"Theta_hat must have shape {self.theta_shape}")
        self.Theta_hat = Theta_hat.copy()

    def wind_torque(self, x, t=0.0):
        """Return the torque generated by the two-dimensional wind field."""
        phi, phi_dot, _ = np.asarray(x, dtype=float).reshape(self.xdim)
        wind_velocity = np.asarray(
            self.wind_velocity_fcn(t), dtype=float
        ).reshape(2)
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
        return self.length * (
            np.sin(phi) * wind_force[1]
            - np.cos(phi) * wind_force[0]
        )

    def true_uncertainty(self, x, t=0.0):
        """Return the unknown damping, mass, and wind mismatch."""
        if self.true_uncertainty_fcn is not None:
            uncertainty = np.asarray(
                self.true_uncertainty_fcn(x, t), dtype=float
            )
        else:
            phi, phi_dot, _ = np.asarray(x, dtype=float).reshape(self.xdim)
            angular_uncertainty = (
                (self.damping - self.true_damping) * phi_dot
                + 0.5
                * (self.mass - self.true_mass)
                * self.grav
                * self.length
                * np.sin(phi)
                + self.wind_torque(x, t)
            ) / self.inertia
            uncertainty = np.array([0.0, angular_uncertainty, 0.0])

        if uncertainty.shape != (self.xdim,):
            raise ValueError(
                f"true_uncertainty(x, t) must return shape ({self.xdim},)"
            )
        return uncertainty

    def dynamics(self, x, u, t=0.0):
        return (
            np.asarray(self.f(x), dtype=float).reshape(self.xdim)
            + (
                np.asarray(self.g(x), dtype=float)
                @ np.asarray(u, dtype=float).reshape(self.udim)
            ).reshape(self.xdim)
            + self.true_uncertainty(x, t)
        )

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

    def _lambdify_symbolic_clf(self):
        """Generate the exact backstepping CRaCLF and its gradients."""
        phi, phi_dot, tau_a = sp.symbols(
            "phi phi_dot tau_a", real=True
        )
        x = sp.Matrix([phi, phi_dot, tau_a])
        a_symbols = sp.symbols(f"a0:{self.adim}", real=True)
        a = sp.Matrix(a_symbols)
        theta_symbols = sp.symbols(
            f"theta0:{np.prod(self.theta_shape)}", real=True
        )
        Theta = sp.Matrix(*self.theta_shape, theta_symbols)

        feature = sp.Matrix(
            [[
                phi_dot,
                sp.sin(phi),
                sp.cos(phi) - 1,
                phi_dot * sp.sin(phi),
                phi_dot * sp.cos(phi),
            ]]
        )
        d = (feature @ Theta @ a)[0]
        q = (
            -self.damping * phi_dot
            - 0.5 * self.mass * self.grav * self.length * sp.sin(phi)
        ) / self.inertia + d

        z1 = phi
        z2 = phi_dot + 2 * phi
        tau_desired = self.inertia * (q + 2 * phi_dot + z1 + 2 * z2)
        z3 = (tau_desired - tau_a) / self.inertia
        z = sp.Matrix([z1, z2, z3])
        clf = sp.Rational(1, 2) * (z1**2 + z2**2 + z3**2)

        tau_desired_dot_ce = (
            sp.diff(tau_desired, phi) * phi_dot
            + sp.diff(tau_desired, phi_dot)
            * (q - tau_a / self.inertia)
        )
        u_backstepping = (
            tau_a
            + self.actuator_time_constant * tau_desired_dot_ce
            + self.inertia
            * self.actuator_time_constant
            * (z2 + 2 * z3)
        )

        dclfdx = sp.Matrix([sp.diff(clf, state) for state in x])
        dclfda = sp.Matrix([sp.diff(clf, parameter) for parameter in a])

        arguments = [x, a, theta_symbols]
        self._clf_function = sp.lambdify(arguments, clf, modules="numpy")
        self._dclfdx_function = sp.lambdify(
            arguments, dclfdx, modules="numpy"
        )
        self._dclfda_function = sp.lambdify(
            arguments, dclfda, modules="numpy"
        )
        self._z_function = sp.lambdify(arguments, z, modules="numpy")
        self._tau_desired_function = sp.lambdify(
            arguments, tau_desired, modules="numpy"
        )
        self._u_backstepping_function = sp.lambdify(
            arguments, u_backstepping, modules="numpy"
        )
        self._clf_rate_backstepping = 4.0

    def _certificate_arguments(self, x, a):
        return (
            np.asarray(x, dtype=float).reshape(self.xdim),
            np.asarray(a, dtype=float).reshape(self.adim),
            self.Theta_hat.reshape(-1),
        )

    def clf(self, x, a):
        return np.asarray(
            self._clf_function(*self._certificate_arguments(x, a)),
            dtype=float,
        )

    def dclfdx(self, x, a):
        return np.asarray(
            self._dclfdx_function(*self._certificate_arguments(x, a)),
            dtype=float,
        ).reshape(self.xdim, 1)

    def dclfda(self, x, a):
        return np.asarray(
            self._dclfda_function(*self._certificate_arguments(x, a)),
            dtype=float,
        ).reshape(self.adim, 1)

    def backstepping_coordinates(self, x, a_hat):
        return np.asarray(
            self._z_function(*self._certificate_arguments(x, a_hat)),
            dtype=float,
        ).reshape(self.xdim)

    def desired_actuator_torque(self, x, a_hat):
        return float(
            np.asarray(
                self._tau_desired_function(
                    *self._certificate_arguments(x, a_hat)
                )
            ).item()
        )

    def backstepping_control(self, x, a_hat):
        return np.array(
            [
                float(
                    np.asarray(
                        self._u_backstepping_function(
                            *self._certificate_arguments(x, a_hat)
                        )
                    ).item()
                )
            ]
        )

    def ctrl_nominal(self, x):
        # The backstepping controller establishes the CLF property; the
        # implemented control is the min-norm solution of the CRaCLF-QP.
        return np.zeros(self.udim)
