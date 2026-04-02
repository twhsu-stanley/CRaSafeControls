import sympy as sp
import numpy as np
from scipy.linalg import solve_continuous_lyapunov as lyap
from scipy.linalg import solve_continuous_are
from systems.control_affine_system import ControlAffineSystem

class IP(ControlAffineSystem):
    def __init__(self, params=None):
        super().__init__(params)

    def define_system_symbolic(self):
        # Symbolic states
        theta, theta_dot = sp.symbols('theta theta_dot')
        x = sp.Matrix([theta, theta_dot])
        
        l = self.params['l']    # length of pendulum (m)
        m = self.params['m']    # mass of pendulum (kg)
        grav = self.params['grav'] # gravity (m/s^2)
        b = self.params['b']    # friction coefficient (s*Nm/rad) 
        I = m * l**2 / 3        # moment of inertia (kg*m^2)

        f = sp.Matrix([
            [x[1]],
            [(-b * x[1] + m * grav * l * sp.sin(x[0]) / 2) / I]
        ])
        g = sp.Matrix([
            [0],
            [-1 / I]
        ])

        # Define the symbolic uncertainty matrix Y(x)
        Y = sp.Matrix([[x[0], 0, 0], 
                       [0, sp.sin(x[0]), x[1]]])
        a0, a1, a2 = sp.symbols('a0 a1 a2')
        a = sp.Matrix([a0, a1, a2])

        return x, f, g, Y, a
    
    def dynamics_extended(self, x_ext, u):
        x = x_ext[0:self.xdim]
        a_hat = x_ext[self.xdim:(self.xdim+self.adim)]
        rho = x_ext[(self.xdim+self.adim)]

        dxdt_ext = np.zeros((self.xdim+self.adim+1,))

        dxdt_ext[0:self.xdim] = self.dynamics(x, u)

        if self.use_adaptive:
            a_hat_dot, rho_dot = self.adaptation_craclf(x, a_hat, rho)
        else:
            a_hat_dot= np.zeros(self.adim)
            rho_dot = 0.0    
        dxdt_ext[self.xdim:(self.xdim+self.adim)] = a_hat_dot
        dxdt_ext[(self.xdim+self.adim)] = rho_dot

        return dxdt_ext
    
    def define_clf_symbolic(self, x, a):
        a0 = a[0]
        a1 = a[1]
        a2 = a[2]
        
        l = self.params['l']    # length of pendulum (m)
        m = self.params['m']    # mass of pendulum (kg)
        grav = self.params['grav'] # gravity (m/s^2)
        b = self.params['b']    # friction coefficient (s*Nm/rad) 
        I = m * l**2 / 3        # moment of inertia (kg*m^2)
        c_bar = m * grav * l / (2 * I)
        b_bar = b / I

        # Linearized Dynamics with state feedback : u0 = params.Kp * x0 + params.Kd * x1
        A_cl = sp.Matrix([[a0, 1],
                          [c_bar - self.params['Kp']/I + a1, -b_bar - self.params['Kd']/I + a2]
        ])
        Q = self.params['clf']['rate'] * sp.eye(2)

        # Get P(a) by solving the Lyapunov equation: A_cl^T P(a) + P(a) A_cl + Q = 0
        p11, p12, p22 = sp.symbols('p11 p12 p22', real=True)
        P = sp.Matrix([
            [p11, p12],
            [p12, p22]
        ])

        lyap_mat = sp.expand(A_cl.T @ P + P @ A_cl + Q)

        eqs = [
            sp.Eq(lyap_mat[0, 0], 0),
            sp.Eq(lyap_mat[0, 1], 0),
            sp.Eq(lyap_mat[1, 1], 0),
        ]

        sol = sp.solve(eqs, (p11, p12, p22), dict=True)
        if not sol:
            raise ValueError("Could not solve the parameter-dependent Lyapunov equation.")

        P = sp.simplify(P.subs(sol[0]))
        clf = sp.simplify((x.T @ P @ x)[0, 0])

        #A_cl = np.array([[0, 1],
        #              [c_bar - self.params['Kp']/I, -b_bar - self.params['Kd']/I]
        #])
        #Q = self.params['clf']['rate'] * np.eye(2)
        #P = lyap(A_cl.T, -Q)
        #clf = (x.T @ P @ x)[0,0]
        
        return clf

    def ctrl_nominal(self, x):
        return np.array([0.0])