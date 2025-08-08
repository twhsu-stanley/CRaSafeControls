import sympy as sp
import numpy as np
from dynsys.ctrl_affine_sys import CtrlAffineSys

class DUBINS_UNCERTAIN(CtrlAffineSys):
    def __init__(self, params=None):
        super().__init__(params)

    def define_system_symbolic(self, params):
        # Symbolic states
        p_x, p_y, theta = sp.symbols('p_x p_y theta')
        x = sp.Matrix([p_x, p_y, theta])

        v = params['v']

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

        # True uncertainty term: Y(x)a(theta)
        Y = sp.Matrix([[sp.cos(theta), 0, 0], [0, sp.sin(theta), 0], [0, 0, theta]]) # true Y(x)
        a = np.copy(params["a_true"]) # true a(Theta)
        f += Y @ a  # Adding the true uncertainty to the system dynamics

        return x, f, g

    def define_clf_symbolic(self, params, x):
        p_x = x[0]
        p_y = x[1]
        theta = x[2]
        clf = (sp.cos(theta) * (p_y - params["yd"]) - sp.sin(theta) * (p_x - params["xd"]))**2
        return clf

    def define_cbf_symbolic(self, params, x):
        p_x = x[0]
        p_y = x[1]
        theta = x[2]

        v = params["v"]
        cbf_gamma = params["cbf_gamma"]

        xo = params["obstacle"]["xo"]
        yo = params["obstacle"]["yo"]
        ro = params["obstacle"]["ro"]

        distance = (p_x - xo)**2 + (p_y - yo)**2 - ro**2
        deriv_distance = 2 * (p_x - xo) * v * sp.cos(theta) + 2 * (p_y - yo) * v * sp.sin(theta)
        cbf = deriv_distance + cbf_gamma * distance
        return cbf

    def ctrl_nominal(self, x):
        """Proportional navigation towards the target point"""
        x_target = self.params[""]
        theta_d = np.arctan2(x_d[1] - x[1], x_d[0] - x[0])
        theta_err = theta_d - x[2]
        theta_err = (theta_err + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

        u = Kp * theta_err