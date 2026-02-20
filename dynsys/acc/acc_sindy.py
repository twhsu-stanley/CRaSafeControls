import sympy as sp
import numpy as np
from dynsys.ctrl_affine_sys import CtrlAffineSys
from dynsys.utils import sindy_prediction_symbolic

class ACC_SINDY(CtrlAffineSys):
    def __init__(self, params=None):
        super().__init__(params)

    def define_system_symbolic(self):
        # Symbolic states
        x0, x1, x2 = sp.symbols('x0 x1 x2')
        x = sp.Matrix([x0, x1, x2])

        feature_names = self.params["feature_names"]
        coefficients = self.params["coefficients"]
        idx_x = self.params["idx_x"]
        idx_u = self.params["idx_u"]

        f = sindy_prediction_symbolic(x, np.array([0.0]), feature_names, coefficients, idx_x)
        g = sindy_prediction_symbolic(x, np.array([1.0]), feature_names, coefficients, idx_u)
        
        # Define the symbolic uncertainty term Y(x)
        Y = sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

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