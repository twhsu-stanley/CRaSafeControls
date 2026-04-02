import sympy as sp
import numpy as np
from systems.control_affine_system import ControlAffineSystem
from utils import sindy_prediction_symbolic

class ACC_SINDY(ControlAffineSystem):
    def __init__(self, params=None):
        super().__init__(params)

    def define_system_symbolic(self):
        # Symbolic states
        p, v, z = sp.symbols('p v z')
        x = sp.Matrix([p, v, z])

        feature_names = self.params["feature_names"]
        coefficients = self.params["coefficients"]
        idx_x = self.params["idx_x"]
        idx_u = self.params["idx_u"]

        f = sindy_prediction_symbolic(x, np.array([0.0]), feature_names, coefficients, idx_x)
        g = sindy_prediction_symbolic(x, np.array([1.0]), feature_names, coefficients, idx_u)

        # Define the symbolic uncertainty term Y(x)
        m = self.params['m']
        Y = sp.Matrix([[0, 0, 0, 0], [-1/m, -v/m, -v**2/m, 0], [0, 0, 0, 1]])

        a0, a1, a2, a3 = sp.symbols('a0 a1 a2 a3')
        a = sp.Matrix([a0, a1, a2, a3])

        return x, f, g, Y, a

    def dynamics_extended(self, x_ext, u):
        """Extended true dynamics for state propagation in sim"""
        x = x_ext[0:self.xdim]
        a_hat = x_ext[self.xdim:(self.xdim+self.adim)]
        rho = x_ext[(self.xdim+self.adim)]

        dxdt_ext = np.zeros((self.xdim+self.adim+1,))

        # True dynamics
        m = self.params['m']
        v = x[1]
        fx = np.array([v, -1/m *(0.5 + 5.0 * v + 1.0 * v**2), self.params["v0"] - v])
        gx = np.array([[0], [1/self.params["m"]], [0]])
        dxdt_ext[0:self.xdim] = fx + (gx @ u).ravel() + self.Y(x) @ self.a_true

        if self.use_adaptive:
            a_hat_dot, rho_dot = self.adaptation_cracbf(x, a_hat, rho)
        else:
            a_hat_dot= np.zeros(self.adim)
            rho_dot = 0.0    
        dxdt_ext[self.xdim:(self.xdim+self.adim)] = a_hat_dot
        dxdt_ext[(self.xdim+self.adim)] = rho_dot

        return dxdt_ext
    
    def define_cbf_symbolic(self, x, a):
        v = x[1]
        z = x[2]
        a3 = a[3]
        T = self.params['T']
        return z - T * (v - a3)

    def ctrl_nominal(self, x):
        v = x[1]
        vd = self.params['vd']
        Kp = self.params['Kp']
        return np.array([Kp * (vd - v)])