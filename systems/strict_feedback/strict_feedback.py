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
        self._lambdify_backstepping_coordinates()

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
        """Install a new representation and regenerate symbolic derivatives."""
        Theta_hat = np.asarray(Theta_hat, dtype=float)
        if Theta_hat.shape != self.theta_shape:
            raise ValueError(f"Theta_hat must have shape {self.theta_shape}")
        self.Theta_hat = Theta_hat.copy()

        x_sym, f_sym, g_sym, a_sym = self.define_system_symbolic()
        clf_sym = self.define_clf_symbolic(x_sym, a_sym)
        self.lambdify_certificate_funcs(
            x_sym,
            f_sym,
            g_sym,
            a_sym,
            clf_sym,
            None,
            None,
        )
        self._lambdify_backstepping_coordinates(x_sym, a_sym)

    def define_system_symbolic(self):
        x1, x2, x3 = sp.symbols("x1 x2 x3", real=True)
        x = sp.Matrix([x1, x2, x3])

        f = sp.Matrix([x2, x3, 0])
        g = sp.Matrix([0, 0, 1])

        a1, a2 = sp.symbols("a1 a2", real=True)
        a = sp.Matrix([a1, a2])

        return x, f, g, a

    def define_clf_symbolic(self, x, a):
        x1, x2, x3 = x
        feature = sp.Matrix([[x1, x1**2, x1**3]])
        feature_prime = sp.Matrix([[1, 2 * x1, 3 * x1**2]])
        Theta = sp.Matrix(self.Theta_hat.tolist())

        d = (feature @ Theta @ a)[0]
        d_prime = (feature_prime @ Theta @ a)[0]

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

        self._z_sym = sp.Matrix([z1, z2, z3])
        self._u_backstepping_sym = sp.simplify(u_backstepping)
        self._clf_rate_backstepping = 4.0

        return sp.simplify(clf)

    def _lambdify_backstepping_coordinates(self, x_sym=None, a_sym=None):
        if x_sym is None or a_sym is None:
            x_sym, _, _, a_sym = self.define_system_symbolic()
            # Build expressions using the same symbolic state objects.
            self.define_clf_symbolic(x_sym, a_sym)
        self._z_function = sp.lambdify(
            [x_sym, a_sym], self._z_sym, modules="numpy"
        )
        self._u_backstepping_function = sp.lambdify(
            [x_sym, a_sym], self._u_backstepping_sym, modules="numpy"
        )

    def backstepping_coordinates(self, x, a_hat):
        return np.asarray(self._z_function(x, a_hat), dtype=float).reshape(3)

    def backstepping_control(self, x, a_hat):
        return np.array(
            [float(np.asarray(self._u_backstepping_function(x, a_hat)).item())]
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
