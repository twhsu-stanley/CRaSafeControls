import sympy as sp
import numpy as np
from dynsys.ctrl_affine_sys import CtrlAffineSys


class NONLINEAR_TOY(CtrlAffineSys):
    """
    Simple 3-state control-affine toy system

    x = [x1, x2, x3]^T

    x1_dot = x3
    x2_dot = x1^2 - x2
    x3_dot = tanh(x2) + u - a2*x3 - a3*x1^2

    plus an unmatched param term -a1 * [x1; 0; 0]

    We represent the parameter dependence as Y(x) * a where
    a = [a1, a2, a3]^T and
    Y = [[-x1,    0,        0   ],
         [  0,    0,        0   ],
         [  0,   -x3,    -x1**2 ]]

    and g(x) = [0; 0; 1].
    """

    def __init__(self, params=None):
        super().__init__(params)

    def define_system_symbolic(self):
        # Symbolic states
        x1, x2, x3 = sp.symbols('x1 x2 x3')
        x = sp.Matrix([x1, x2, x3])

        # Parameter symbols (a1, a2, a3)
        a1, a2, a3 = sp.symbols('a1 a2 a3')
        a = sp.Matrix([a1, a2, a3])

        # Drift dynamics (nominal)
        f = sp.Matrix([
            x3,
            x1**2 - x2,
            sp.tanh(x2)
        ])

        # Control influence (enters only in x3)
        g = sp.Matrix([[0], [0], [1]])

        # Uncertainty / param-dependent terms Y(x) * a
        Y = sp.Matrix([
            [-x1,     0,        0 ],
            [  0,     0,        0 ],
            [  0,   -x3,    -x1**2]
        ])

        return x, f, g, Y, a
    
    def W_fcn(self, x, a):
        """Dual CCM metric W(x,a)"""
        x1 = x[0]
        a1 = a[0].item()
        return np.array([
            [             1.42,        0.0,                              1.42 * (a1 - 1.0)],
            [              0.0,       6.21,                                     -2.85 * x1],
            [1.42 * (a1 - 1.0), -2.85 * x1, 1.42 * a1**2 - 2.84 * a1 + 1.30 * x1**2 + 5.79]
        ])

    def dW_dx1_fcn(self, x, a):
        """Partial derivative of W with respect to x1"""
        x1 = x[0]
        return np.array([
            [0.0,   0.0,       0.0],
            [0.0,   0.0,     -2.85],
            [0.0, -2.85, 2.60 * x1]
        ])

    def dW_dxi_fcn(self, i, x, a):
        if i == 0:
            return self.dW_dx1_fcn(x, a)
        else:
            # no dependence on other states
            return np.zeros((3, 3))
    
    def dW_da1_fcn(self, x, a):
        """Partial derivative of W with respect to a1"""
        a1 = a[0].item()
        return np.array([
            [0.0,  0.0,             1.42],
            [0.0,  0.0,              0.0],
            [1.42, 0.0, 2.84 * a1 - 2.84]
        ])

    def dW_dai_fcn(self, i, x, a):
        if i == 0:
            return self.dW_da1_fcn(x, a)
        else:
            return np.zeros((3, 3))

