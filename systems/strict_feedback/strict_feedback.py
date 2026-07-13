import numpy as np
import sympy as sp

from systems.control_affine_system import ControlAffineSystem


class StrictFeedbackSystem(ControlAffineSystem):
    """Strict-feedback system and backstepping CRaCLF from Example 1.

    The learned parametric model is

        Y_Theta(x) a = Psi(x) Theta a,

    where Psi has the polynomial features [x1, x1**2, x1**3] in its
    first row. The physical uncertainty remains the sinusoidal model from
    Section V-A and is intentionally unknown to the controller.
    """

    theta_shape = (3, 2)
    xdim = 3
    udim = 1
    adim = 2

    def __init__(self, params=None):
        if params is None:
            params = {}
        self.Theta_hat = np.asarray(
            params.get(
                "Theta_init",
                np.array([[-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]),
            ),
            dtype=float,
        ).reshape(self.theta_shape)
        self.true_uncertainty_fcn = params.get("true_uncertainty")
        if self.true_uncertainty_fcn is None:
            self.true_uncertainty_fcn = lambda x, t: np.array(
                [-np.sin(x[0]) - 0.5 * x[0] ** 2, 0.0, 0.0]
            )
        if not callable(self.true_uncertainty_fcn):
            raise TypeError("true_uncertainty must be callable as w(x, t)")

        super().__init__(params)
        self._lambdify_symbolic_clf()

    def f(self, x):
        """Return the numerical nominal drift."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        return np.array([x[1], x[2], 0.0])

    def g(self, x):
        """Return the numerical control matrix."""
        return np.array([[0.0], [0.0], [1.0]])

    @staticmethod
    def psi(x):
        """Return the Example 1 feature matrix Psi(x)."""
        x1 = float(np.asarray(x).reshape(-1)[0])
        return np.array(
            [
                [x1, x1**2, x1**3],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

    def Y(self, x):
        """Evaluate the currently installed strict-feedback representation."""
        return self.Y_theta(x, self.Theta_hat)

    def Y_theta(self, x, theta):
        """Evaluate Y_theta(x) = Psi(x) @ theta."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.theta_shape:
            raise ValueError(f"theta must have shape {self.theta_shape}")
        return self._validate_Y_shape(self.psi(x) @ theta)

    def representation_loss_gradient(self, x, theta, a, w):
        """Return the analytic strict-feedback representation gradient."""
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

    def _lambdify_symbolic_clf(self):
        """Generate exact CLF/backstepping derivatives once with SymPy."""
        x1, x2, x3 = sp.symbols("x1 x2 x3", real=True)
        x = sp.Matrix([x1, x2, x3])
        a1, a2 = sp.symbols("a1 a2", real=True)
        a = sp.Matrix([a1, a2])
        theta_symbols = sp.symbols(
            f"theta0:{np.prod(self.theta_shape)}", real=True
        )
        Theta = sp.Matrix(*self.theta_shape, theta_symbols)

        feature = sp.Matrix([[x1, x1**2, x1**3]])
        d = (feature @ Theta @ a)[0]
        d_prime = sp.diff(d, x1)

        z1 = x1
        z2 = x2 + 2 * x1 + d
        z3 = x1 + x3 + 2 * z2 + (2 + d_prime) * (x2 + d)
        clf = sp.Rational(1, 2) * (z1**2 + z2**2 + z3**2)

        dz3_dx1 = sp.diff(z3, x1)
        dz3_dx2 = sp.diff(z3, x2)
        u_backstepping = (
            -dz3_dx1 * (x2 + d)
            - dz3_dx2 * x3
            - z2
            - 2 * z3
        )

        dclfdx = sp.Matrix([sp.diff(clf, state) for state in x])
        dclfda = sp.Matrix([sp.diff(clf, parameter) for parameter in a])
        z = sp.Matrix([z1, z2, z3])

        arguments = [x, a, theta_symbols]
        self._clf_function = sp.lambdify(arguments, clf, modules="numpy")
        self._dclfdx_function = sp.lambdify(
            arguments, dclfdx, modules="numpy"
        )
        self._dclfda_function = sp.lambdify(
            arguments, dclfda, modules="numpy"
        )
        self._z_function = sp.lambdify(arguments, z, modules="numpy")
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

    def true_uncertainty(self, x, t):
        uncertainty = np.asarray(self.true_uncertainty_fcn(x, t), dtype=float)
        if uncertainty.shape != (self.xdim,):
            raise ValueError(
                f"true_uncertainty(x, t) must return shape ({self.xdim},)"
            )
        return uncertainty

    def dynamics(self, x, u, t=0.0):
        return (
            np.asarray(self.f(x), dtype=float).reshape(self.xdim)
            + (
                np.asarray(self.g(x), dtype=float) @ np.asarray(u).reshape(-1)
            ).reshape(self.xdim)
            + self.true_uncertainty(x, t)
        )

    def dynamics_extended(self, x_ext, u, t=0.0):
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

    def ctrl_nominal(self, x):
        # The backstepping controller establishes the CLF property; the
        # implemented control is the min-norm solution of the CRaCLF-QP.
        return np.zeros(self.udim)


# Short alias consistent with the existing IP class naming style.
StrictFeedback = StrictFeedbackSystem
