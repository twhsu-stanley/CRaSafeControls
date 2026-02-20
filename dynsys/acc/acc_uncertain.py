import sympy as sp
import numpy as np
from dynsys.ctrl_affine_sys import CtrlAffineSys

class ACC_UNCERTAIN(CtrlAffineSys):
    def __init__(self, params=None):
        super().__init__(params)

    def define_system_symbolic(self):
        # Symbolic states
        p, v, z = sp.symbols('p v z')
        x = sp.Matrix([p, v, z])

        f0 = self.params['f0']
        f1 = self.params['f1']
        f2 = self.params['f2']
        v0 = self.params['v0']
        m = self.params['m']

        f = sp.Matrix([[v], [0], [v0 - v]])
        g = sp.Matrix([[0], [1/m], [0]])

        # Define the symbolic uncertainty term Y(x)
        Y = sp.Matrix([[0, 0, 0], [1, x[1], x[1]**2], [0, 0, 0]])

        a0, a1, a2 = sp.symbols('a0 a1 a2')
        a = sp.Matrix([a0, a1, a2])

        return x, f, g, Y, a

    def define_clf_symbolic(self, x):
        v = x[1]
        vd = self.params['vd']
        return (v - vd)**2

    def define_cbf_symbolic(self, x):
        v = x[1]
        z = x[2]
        T = self.params['T']
        return z - T * v

    def ctrl_nominal(self, x):
        v = x[1]
        vd = self.params['vd']
        Kp = self.params['Kp']
        return np.array([Kp * (vd - v)])