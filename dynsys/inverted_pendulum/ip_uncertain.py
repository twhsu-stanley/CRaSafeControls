import sympy as sp
import numpy as np
from scipy.linalg import solve_continuous_lyapunov as lyap
from scipy.linalg import solve_continuous_are
from dynsys.ctrl_affine_sys import CtrlAffineSys

class IP_UNCERTAIN(CtrlAffineSys):
    def __init__(self, params=None):
        super().__init__(params)

        self.A = None # Placeholder for the linearized system matrix
        self.B = None # Placeholder for the linearized input matrix

    def define_system_symbolic(self):
        # Symbolic states
        theta, theta_dot = sp.symbols('theta theta_dot')
        x = sp.Matrix([theta, theta_dot])
        
        l = self.params['l']    # length of pendulum (m)
        m = self.params['m']    # mass of pendulum (kg)
        grav = self.params['g']    # gravity (m/s^2)
        b = self.params['b']    # friction coefficient (s*Nm/rad)
        I = self.params['I']    # moment of inertia (kg*m^2)
        assert I == m * l**2 / 3, "I = m*l^2/3"

        f = sp.Matrix([
            [x[1]],
            [(-b * x[1] + m * grav * l * sp.sin(x[0]) / 2) / I]
        ])
        g = sp.Matrix([
            [0],
            [-1 / I]
        ])

        return x, f, g

    def define_Y_symbolic(self, x):
        # Define the symbolic uncertainty term Y(x)
        theta = x[0]
        theta_dot = x[1]
        Y = sp.Matrix([[0, 0], [sp.sin(theta), theta_dot]])
        return Y

    def define_a_symbolic(self):
        # Symbolic states
        a0, a1 = sp.symbols('a0 a1')
        return sp.Matrix([a0, a1])
    
    def define_clf_symbolic(self, x):
        I = self.params['I'] # moment of inertia (kg*m^2)
        c_bar = self.params['m'] * self.params['g'] * self.params['l'] / (2 * I)
        b_bar = self.params['b'] / I

        # Linearized Dynamics with state feedback : u0 = params.Kp * x0 + params.Kd * x1
        A = np.array([
            [0, 1],
            [c_bar - self.params['Kp'] / I, -b_bar - self.params['Kd'] / I]
        ])
        Q = self.params['clf']['rate'] * np.eye(A.shape[0])
        P = lyap(A.T, -Q)
        clf = (x.T @ P @ x)[0,0]

        # Find c1: c1*||x||^2 <= V(x) = x'Px <= c2*||x||^2
        # TODO: iniitialize c1 and c2 in the superclass constructor
        self.c1 = np.min(np.linalg.eigvals(P))
        self.c2 = np.max(np.linalg.eigvals(P))

        # LQR
        """
        A = np.array([
            [0, 1],
            [c_bar, -b_bar]
        ])
        B = np.array([[0], [-1 / I]])
        Q = params['clf']['Q']
        R = params['clf']['R']
        P_lqr = solve_continuous_are(A, B, Q, R)
        K_lqr = np.linalg.inv(R) @ B.T @ P_lqr
        self.K_lqr = K_lqr
        self.P_lqr = P_lqr
        clf = x.T * sp.Matrix(P_lqr) * x
        """

        return clf

    def ctrl_nominal(self, x):
        return np.array([0])