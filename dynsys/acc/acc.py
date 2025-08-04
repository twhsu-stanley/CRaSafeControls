import sympy as sp
import numpy as np
from dynsys.ctrl_affine_sys import CtrlAffineSys

class ACC(CtrlAffineSys):
    def __init__(self, params=None):
        super().__init__(params)

    def define_system_symbolic(self, params):
        # Symbolic states
        p, v, z = sp.symbols('p v z')
        x = sp.Matrix([p, v, z])

        f0 = params['f0']
        f1 = params['f1']
        f2 = params['f2']
        v0 = params['v0']
        m = params['m']

        Fr = f0 + f1 * v + f2 * v**2
        f = sp.Matrix([[v], [-Fr/m], [v0 - v]])
        g = sp.Matrix([[0], [1/m], [0]])

        return x, f, g

    def define_clf_symbolic(self, params, x):
        v = x[1]
        vd = params['vd']
        return (v - vd)**2

    def define_cbf_symbolic(self, params, x):
        v = x[1]
        z = x[2]
        T = params['T']
        return z - T * v

    def ctrl_nominal(self, x):
        v = x[1]
        vd = self.params['vd']
        Kp = self.params['Kp']
        return np.array([Kp * (vd - v)])