import sympy as sp
import numpy as np
from scipy.linalg import solve_continuous_lyapunov as lyap
from dynsys.ctrl_affine_sys import CtrlAffineSys
from dynsys.utils import sindy_prediction_symbolic

class IP_SINDY(CtrlAffineSys):
    def __init__(self, params=None):
        super().__init__(params)

        self.A = None # Placeholder for the linearized system matrix
        self.B = None # Placeholder for the linearized input matrix

    def define_system_symbolic(self):
        # Symbolic states
        x0, x1 = sp.symbols('x0 x1')
        x = sp.Matrix([x0, x1])

        feature_names = self.params["feature_names"]
        coefficients = self.params["coefficients"]
        idx_x = self.params["idx_x"]
        idx_u = self.params["idx_u"]

        f = sindy_prediction_symbolic(x, np.array([0.0]), feature_names, coefficients, idx_x)
        g = sindy_prediction_symbolic(x, np.array([1.0]), feature_names, coefficients, idx_u)

        # Linearization of the system dynamics
        A = f.jacobian(x)
        A = sp.lambdify([x], A, modules='numpy')
        self.A = A(np.array([0,0])) # evaluate at equilibrium point (0,0)
        B = sp.lambdify([x], g, modules='numpy')
        self.B = B(np.array([0,0])) # evaluate at equilibrium point (0,0)
        
        # Define the symbolic uncertainty term Y(x)
        Y = sp.Matrix([[0, 0], [0, 0]])

        a0, a1 = sp.symbols('a0 a1')
        a = sp.Matrix([a0, a1])
        
        return x, f, g, Y, a

    def define_clf_symbolic(self, x):
        # x: symbolic states

        # Linearized Dynamics with state feedback : u0 = params.Kp * x0 + params.Kd * x1
        A_cl = self.A + self.B @ np.array([[self.params["Kp"], self.params["Kd"]]])
        Q = self.params['clf']['rate'] * np.eye(self.A.shape[0])
        P = lyap(A_cl.T, -Q) # Cost Matrix for quadratic CLF. (V = e'*P*e)
        clf = (x.T @ P @ x)[0,0]
        
        # Find c1: c1*||x||^2 <= V(x) = x'Px <= c2*||x||^2
        # TODO: iniitialize c1 and c2 in the superclass constructor
        self.c1 = np.min(np.linalg.eigvals(P))
        self.c2 = np.max(np.linalg.eigvals(P))

        return clf

    def ctrl_nominal(self, x):
        return np.zeros((self.udim, 1))