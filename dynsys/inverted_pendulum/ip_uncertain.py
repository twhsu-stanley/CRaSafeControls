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

    def define_system_symbolic(self, params):
        # Symbolic states
        theta, theta_dot = sp.symbols('theta theta_dot')
        x = sp.Matrix([theta, theta_dot])
        
        l = params['l']    # length of pendulum (m)
        m = params['m']    # mass of pendulum (kg)
        grav = params['g']    # gravity (m/s^2)
        b = params['b']    # friction coefficient (s*Nm/rad)
        I = params['I']    # moment of inertia (kg*m^2)
        assert I == m * l**2 / 3, "I = m*l^2/3"

        f = sp.Matrix([
            [x[1]],
            [(-b * x[1] + m * grav * l * sp.sin(x[0]) / 2) / I]
        ])
        g = sp.Matrix([
            [0],
            [-1 / I]
        ])

        # True uncertainty term: Y(x)a(Theta)
        Y = sp.Matrix([[0, 0], [sp.sin(theta), theta_dot]]) # true Y(x)
        a = np.copy(params["a_true"]) # true a(Theta)
        f += Y @ a  # Adding the true uncertainty to the system dynamics

        return x, f, g

    def define_clf_symbolic(self, params, x):
        I = params['I'] # moment of inertia (kg*m^2)
        c_bar = params['m'] * params['g'] * params['l'] / (2 * I)
        b_bar = params['b'] / I

        # Linearized Dynamics with state feedback : u0 = params.Kp * x0 + params.Kd * x1
        A = np.array([
            [0, 1],
            [c_bar - params['Kp'] / I, -b_bar - params['Kd'] / I]
        ])
        Q = params['clf']['rate'] * np.eye(A.shape[0])
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