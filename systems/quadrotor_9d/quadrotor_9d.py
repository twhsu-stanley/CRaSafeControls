"""9D Euler quadrotor adapter for the repository's existing CRaCCM controller.

Dynamics and features follow the supplied NeuralCCM payload-adaptive system:
x = [p_x, p_y, p_z, v_x, v_y, v_z, roll, pitch, yaw]
u = [thrust_acceleration, omega_x, omega_y, omega_z]
This is an attitude-rate model, not
a 12-state rigid-body model. Physical plant uncertainty is supplied separately.
"""

from pathlib import Path

import numpy as np

from systems.control_affine_system import ControlAffineSystem
from .NeuralCCM.neural_metric import NeuralCCMMetric


class QUADROTOR_9D(ControlAffineSystem):
    """Nine-state quadrotor with a learned uncertainty model and neural CCM."""

    theta_shape = (5, 4)
    xdim = 9
    udim = 4
    adim = 4

    def __init__(self, params=None):
        if params is None:
            params = {}
        elif not isinstance(params, dict):
            raise TypeError("Parameters must be a dictionary.")

        self.Theta_hat = np.asarray(params.get("Theta_init"), dtype=float).reshape(self.theta_shape)
        if not np.all(np.isfinite(self.Theta_hat)):
            raise ValueError("Theta_init must be finite")
        self.Theta_hat = self.Theta_hat.copy()

        self.grav = float(params.get("g", 9.81))
        if self.grav != 9.81:
            raise ValueError("The neural CCM uses g=9.81")
        self.true_uncertainty_fcn = params.get("true_uncertainty")
        if self.true_uncertainty_fcn is not None and not callable(self.true_uncertainty_fcn):
            raise TypeError("true_uncertainty must be callable as w(x, t)")

        checkpoint = Path(__file__).resolve().parent / "NeuralCCM/metrics/metric.npz"
        self.metric = NeuralCCMMetric(checkpoint)
        super().__init__(params)

    def f(self, x):
        """Return position kinematics and free-fall acceleration without input."""
        return np.array([x[3], x[4], x[5], 0., 0., -self.grav, 0., 0., 0.])

    @staticmethod
    def _thrust_direction(x):
        """Return R(roll, pitch, yaw) e_z in inertial coordinates."""
        phi, pitch, yaw = x[6:9]
        if type(x[0]).__module__.startswith("casadi"):
            import casadi as math
        else:
            math = np
        sp, cp = math.sin(phi), math.cos(phi)
        st, ct = math.sin(pitch), math.cos(pitch)
        sy, cy = math.sin(yaw), math.cos(yaw)
        return cy*st*cp+sy*sp, sy*st*cp-cy*sp, ct*cp

    def g(self, x):
        """Map thrust acceleration and body angular rates into state derivatives.

        The last three rows convert body rates to Euler rates. This model
        assumes rate-command tracking and omits rotor/torque dynamics.
        """
        phi, pitch = x[6:8]
        # Use the same trigonometric model for numerical simulation and symbolic MPC.
        symbolic = type(x[0]).__module__.startswith("casadi")
        if not symbolic and abs(np.cos(pitch)) < 1e-5:
            raise ValueError("Euler pitch singularity")
        if symbolic:
            import casadi as math
        else:
            math = np
        sp, cp = math.sin(phi), math.cos(phi)
        tangent, secant = math.tan(pitch), 1/math.cos(pitch)
        bx, by, bz = self._thrust_direction(x)
        rows = (
            [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
            [bx, 0., 0., 0.], [by, 0., 0., 0.], [bz, 0., 0., 0.],
            [0., 1., sp*tangent, cp*tangent],
            [0., 0., cp, -sp], [0., 0., sp*secant, cp*secant],
        )
        # Scalar matrix entries match the branch's symbolic MotionPlanner interface.
        if symbolic:
            output = np.empty((self.xdim,self.udim), dtype=object)
            for i, row in enumerate(rows):
                for j,value in enumerate(row):
                    output[i,j] = value
            return output
        return np.vstack(rows)

    def psi(self, x):
        """Return the prescribed 9x5 feature matrix used in Y_Theta = Psi @ Theta.

        Columns represent x/y linear drag, a payload-like thrust-direction term,
        and additive roll/pitch Euler-rate biases. These features approximate
        the physical disturbance; they do not generate the simulated true w.
        """
        symbolic = type(x[0]).__module__.startswith("casadi")
        x = np.asarray(x, dtype=object if symbolic else float).reshape(self.xdim)
        bx, by, bz = self._thrust_direction(x)
        result = np.zeros((self.xdim, 5),dtype=object if symbolic else float)
        result[3, 0], result[4, 1] = -x[3], -x[4]
        for i,value in enumerate((bx,by,bz)):
            result[3+i,2] = -self.grav*value
        result[6, 3], result[7, 4] = 1., 1.
        return result

    def Y(self, x):
        return self.Y_Theta(x, self.Theta_hat)

    def Y_Theta(self, x, theta):
        """Map a four-component parameter to uncertainty using a 5x4 Theta."""
        theta = np.asarray(theta)
        if theta.shape != self.theta_shape:
            raise ValueError(f"theta must have shape {self.theta_shape}")
        if theta.dtype != object or all(np.isscalar(entry) for entry in theta.flat):
            theta = np.asarray(theta, dtype=float)
            if not np.all(np.isfinite(theta)):
                raise ValueError("theta must be finite")
        Y_theta = self.psi(x) @ theta
        if Y_theta.dtype == object:
            return Y_theta
        return self._validate_Y_shape(Y_theta)

    def dynamics_certainty_equivalent(self, x, u, a_hat):
        """Return the certainty-equivalent dynamics f(x) + g(x)u + Y_Theta(x)a_hat."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        u = np.asarray(u, dtype=float).reshape(self.udim)
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.adim)
        return self.f(x) + self.g(x) @ u + self.Y(x) @ a_hat

    def representation_loss_gradient(self, x, theta, a, w):
        """Return grad_Theta ||Y_Theta(x) @ a - w||_2**2"""
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
        """Install the representation used by the controller and CCM"""
        Theta_hat = np.asarray(Theta_hat, dtype=float)
        if Theta_hat.shape != self.theta_shape:
            raise ValueError(f"Theta_hat must have shape {self.theta_shape}")
        if not np.all(np.isfinite(Theta_hat)):
            raise ValueError("Theta_hat must be finite")
        self.Theta_hat = Theta_hat.copy()

    def true_uncertainty(self, x, t=0.0):
        """Return the true uncertainty. Set to zero by default"""
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
        """Evaluate the true simulated plant: nominal dynamics plus physical w."""
        x = np.asarray(x, dtype=float).reshape(self.xdim)
        u = np.asarray(u, dtype=float).reshape(self.udim)
        return self.f(x) + self.g(x) @ u + self.true_uncertainty(x, t)

    def dynamics_extended(self, x_ext, x_d, u, geodesic_solver, t=0.0):
        """Integrate plant x and continuous a_hat/rho; Theta updates are external."""
        x_ext = np.asarray(x_ext, dtype=float).reshape(self.xdim + self.adim + 1)
        x = x_ext[: self.xdim]
        a_hat = x_ext[self.xdim : self.xdim + self.adim]
        rho = x_ext[-1]

        dxdt_ext = np.zeros(self.xdim + self.adim + 1)
        dxdt_ext[: self.xdim] = self.dynamics(x, u, t)
        if self.use_adaptive:
            a_hat_dot, rho_dot = self.adaptation_craccm(x, x_d, a_hat, rho, geodesic_solver)
        else:
            a_hat_dot = np.zeros(self.adim)
            rho_dot = 0.0
        dxdt_ext[self.xdim : self.xdim + self.adim] = a_hat_dot
        dxdt_ext[-1] = rho_dot
        return dxdt_ext

    def _metric_parameter(self, a):
        """Convert adaptive coordinates a(4) to the CCM's physical coordinates b(5)."""
        a = np.asarray(a, dtype=float).reshape(self.adim)
        return self.Theta_hat @ a

    def W_fcn(self, x, a):
        """Evaluate the saved dual metric W(x, Theta @ a) = M^-1."""
        return self.metric.W(x, self._metric_parameter(a))

    def dW_dxi_fcn(self, i, x, a):
        if i < 0 or i >= self.xdim:
            raise IndexError(f"state index must lie in [0, {self.xdim})")
        return self.metric.derivatives(x, self._metric_parameter(a))[0][:, :, i]

    def dW_dai_fcn(self, i, x, a):
        if i < 0 or i >= self.adim:
            raise IndexError(f"parameter index must lie in [0, {self.adim})")
        derivative = self.metric.derivatives(x, self._metric_parameter(a))[1]
        # Chain rule: dW/da_i = sum_r (dW/db_r) * Theta[r, i].
        return derivative @ self.Theta_hat[:, i]
