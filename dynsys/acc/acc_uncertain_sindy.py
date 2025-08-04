import sympy as sp
import numpy as np
from dynsys.ctrl_affine_sys import CtrlAffineSys
from dynsys.utils import sindy_prediction_symbolic

class ACC_UNCERTAIN_SINDY(CtrlAffineSys):
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
        
        #p, v, z = sp.symbols('p v z')
        #x = sp.Matrix([p, v, z])
        #v0 = params['v0']
        #m = params['m']

        #f = sp.Matrix([[v], [0], [v0 - v]])
        #g = sp.Matrix([[0], [1/m], [0]])

        return x, f, g
    
    def define_Y_symbolic(self, x):
        # Define the symbolic uncertainty term Y(x)
        # TODO: this should be given by some neural network
        v = x[1]
        return sp.Matrix([[0, 0, 0], [1, v, v**2], [0, 0, 0]])
    
    def define_a_symbolic(self):
        # Symbolic states
        a0, a1, a2 = sp.symbols('a0 a1 a2')
        return sp.Matrix([a0, a1, a2])

    def define_aclf_symbolic(self, params, x, a_L_hat=None):
        v = x[1]
        vd = params['vd']
        return (v - vd)**2

    def define_acbf_symbolic(self, params, x, a_b_hat=None):
        v = x[1]
        z = x[2]
        T = params['T']
        return z - T * v
    
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