import sympy as sp
import numpy as np
from systems.control_affine_system import ControlAffineSystem

class ACC(ControlAffineSystem):
    def __init__(self, params=None):
        super().__init__(params)

    def define_system_symbolic(self):
        # Symbolic states
        p, v, z = sp.symbols('p v z')
        x = sp.Matrix([p, v, z])

        v0 = self.params['v0']
        m = self.params['m']

        f = sp.Matrix([[v], [0], [v0 - v]])
        g = sp.Matrix([[0], [1/m], [0]])

        # Define the symbolic uncertainty term Y(x)
        Y = sp.Matrix([[0, 0, 0, 0], [-1/m, -v/m, -v**2/m, 0], [0, 0, 0, 1]])

        a0, a1, a2, a3 = sp.symbols('a0 a1 a2 a3')
        a = sp.Matrix([a0, a1, a2, a3])

        return x, f, g, Y, a

    def dynamics_extended(self, x_ext, u):
        x = x_ext[0:self.xdim]
        a_hat = x_ext[self.xdim:(self.xdim+self.adim)].reshape(-1,1)
        rho = x_ext[(self.xdim+self.adim)]

        dxdt_ext = np.zeros((self.xdim+self.adim+1,))

        dxdt_ext[0:self.xdim] = self.dynamics(x, u)

        if self.use_adaptive:
            a_hat_dot, rho_dot = self.adaptation_cracbf(x, a_hat, rho)
        else:
            a_hat_dot= np.zeros((self.adim,1))
            rho_dot = 0.0    
        dxdt_ext[self.xdim:(self.xdim+self.adim)] = a_hat_dot.ravel()
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