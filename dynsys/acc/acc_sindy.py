import sympy as sp
import numpy as np
from dynsys.ctrl_affine_sys import CtrlAffineSys
from dynsys.utils import sindy_prediction_symbolic

class ACC_SINDY(CtrlAffineSys):
    def __init__(self, params=None):
        super().__init__(params)

    def define_system_symbolic(self, params):
        # Symbolic states
        x0, x1, x2 = sp.symbols('x0 x1 x2')
        x = sp.Matrix([x0, x1, x2])

        feature_names = params["feature_names"]
        coefficients = params["coefficients"]
        idx_x = params["idx_x"]
        idx_u = params["idx_u"]

        f = sindy_prediction_symbolic(x, np.array([0.0]), feature_names, coefficients, idx_x)
        g = sindy_prediction_symbolic(x, np.array([1.0]), feature_names, coefficients, idx_u)
        
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