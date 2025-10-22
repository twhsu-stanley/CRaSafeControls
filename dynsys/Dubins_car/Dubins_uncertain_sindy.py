import sympy as sp
import numpy as np
from dynsys.ctrl_affine_sys import CtrlAffineSys
from dynsys.utils import sindy_prediction_symbolic, wrapToPi

class DUBINS_UNCERTAIN_SINDY(CtrlAffineSys):
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
        
        """
        # Symbolic states
        p_x, p_y, theta = sp.symbols('p_x p_y theta')
        x = sp.Matrix([p_x, p_y, theta])

        v = self.params['v']

        # System dynamics based on MATLAB code
        f = sp.Matrix([
            [v * sp.cos(theta)],
            [v * sp.sin(theta)],
            [0]
        ])
        g = sp.Matrix([
            [0],
            [0],
            [1]
        ])
        """

        return x, f, g
    
    def define_Y_symbolic(self, x):
        # Define the symbolic uncertainty term Y(x)
        # TODO: this should be given by some neural network
        theta = x[2]
        return sp.Matrix([[sp.cos(theta), 0, 0], [0, sp.sin(theta), 0], [0, 0, theta]])
        #return sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    
    def define_a_symbolic(self):
        # Symbolic states
        a0, a1, a2 = sp.symbols('a0 a1 a2')
        return sp.Matrix([a0, a1, a2])

    def define_clf_symbolic(self, x):
        p_x = x[0]
        p_y = x[1]
        theta = x[2]

        x_target = self.params["target"]["x"]
        y_target = self.params["target"]["y"]

        clf = (sp.cos(theta) * (p_y - y_target) - sp.sin(theta) * (p_x - x_target))**2
        return clf

    def define_cbf_symbolic(self, x):
        p_x = x[0]
        p_y = x[1]
        theta = x[2]

        v = self.params["v"]
        cbf_gamma = self.params["cbf"]["gamma"]

        xo = self.params["obstacle"]["x"]
        yo = self.params["obstacle"]["y"]
        ro = self.params["obstacle"]["r"]

        distance = (p_x - xo)**2 + (p_y - yo)**2 - ro**2
        deriv_distance = 2 * (p_x - xo) * v * sp.cos(theta) + 2 * (p_y - yo) * v * sp.sin(theta)
        cbf = deriv_distance + cbf_gamma * distance
        return cbf
    
    def define_aclf_symbolic(self, x, a_L_hat=None):
        p_x = x[0]
        p_y = x[1]
        theta = x[2]

        x_target = self.params["target"]["x"]
        y_target = self.params["target"]["y"]

        clf = (sp.cos(theta) * (p_y - y_target) - sp.sin(theta) * (p_x - x_target))**2
        return clf
    
    def define_acbf_symbolic(self, x, a_b_hat=None):
        p_x = x[0]
        p_y = x[1]
        theta = x[2]

        v = self.params["v"]
        cbf_gamma = self.params["cbf"]["gamma"]

        xo = self.params["obstacle"]["x"]
        yo = self.params["obstacle"]["y"]
        ro = self.params["obstacle"]["r"]

        distance = (p_x - xo)**2 + (p_y - yo)**2 - ro**2
        deriv_distance = 2 * (p_x - xo) * v * sp.cos(theta) + 2 * (p_y - yo) * v * sp.sin(theta)
        cbf = deriv_distance + cbf_gamma * distance
        return cbf

    def ctrl_nominal(self, x):
        """Proportional navigation towards the target point"""
        x_target = self.params["target"]["x"]
        y_target = self.params["target"]["y"]

        theta_d = np.arctan2(y_target - x[1], x_target - x[0])
        theta_err = theta_d - x[2]
        theta_err = wrapToPi(theta_err)

        return np.array([self.params["Kp"] * theta_err])