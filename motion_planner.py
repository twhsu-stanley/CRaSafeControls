import numpy as np
import casadi as ca
from systems.control_affine_system import ControlAffineSystem

class MotionPlanner:

    def __init__(
        self,
        system: ControlAffineSystem,
        dt: float,
        Q: np.ndarray,
        R: np.ndarray,
        Q_f: np.ndarray,
        u_min: np.ndarray = None,
        u_max: np.ndarray = None,
    ):
        
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        self.system = system
        self.dt = dt
        self.xdim = system.xdim
        self.udim = system.udim
        self.Q = Q
        self.R = R
        self.Q_f = Q_f
        self.u_min = u_min
        self.u_max = u_max
        # TODO: inclde obstacles

        self.ipopt_options = {
            "print_time": False,
            "ipopt": {"print_level": 0, "sb": "yes", "max_iter": 300},
        }

        self._nominal_dynamics = self._build_nominal_dynamics_function(system)

    def _build_nominal_dynamics_function(self, system):

        x_ca = ca.MX.sym("x", self.xdim)
        u_ca = ca.MX.sym("u", self.udim)
        f = ca.vertcat(*(system.f(ca.vertsplit(x_ca)).flatten().tolist()))
        g = ca.vertcat(*(system.g(ca.vertsplit(x_ca)).flatten().tolist()))
        xdot_ca = f + g @ u_ca
        return ca.Function("nominal_dynamics", [x_ca, u_ca], [xdot_ca])

    def _rk4_step_symbolic(self, xk, uk):
        dt = self.dt
        k1 = self._nominal_dynamics(xk, uk)
        k2 = self._nominal_dynamics(xk + 0.5 * dt * k1, uk)
        k3 = self._nominal_dynamics(xk + 0.5 * dt * k2, uk)
        k4 = self._nominal_dynamics(xk + dt * k3, uk)
        return xk + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _rk4_step_numpy(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        x_vec = np.asarray(xk, dtype=float).reshape(self.xdim)
        u_vec = np.asarray(uk, dtype=float).reshape(self.udim)

        def f_eval(x_eval: np.ndarray) -> np.ndarray:
            return np.array(self._nominal_dynamics(x_eval, u_vec), dtype=float).reshape(self.xdim)

        dt = self.dt
        k1 = f_eval(x_vec)
        k2 = f_eval(x_vec + 0.5 * dt * k1)
        k3 = f_eval(x_vec + 0.5 * dt * k2)
        k4 = f_eval(x_vec + dt * k3)
        return x_vec + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def plan(self, x_init, x_goal, horizon_steps, x_guess, u_guess):
        """ Plan a trajectory from x_init towards x_goal by solving the following optimization problem
        minimize_{x_0,...,x_N, u_0,...,u_{N-1}} sum_{k=0}^{N-1} (x_k - x_goal)^T Q (x_k - x_goal) + u_k^T R u_k + (x_N - x_goal)^T Q_f (x_N - x_goal)
        subject to: 1) x_0 = x_init
                    2) x_{k+1} = f_d(x_k, u_k), for k=0,...,N-1, where f_d is the discrete-time nominal dynamics obtained by applying RK4
                    3) u_min <= u_k <= u_max, for k=0,...,N-1
                    4) TODO: include inequality constraints for obstacle avoidance
        """

        horizon_steps = int(horizon_steps)
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be at least 1")

        x_init = np.asarray(x_init, dtype=float).reshape(self.xdim)
        x_goal = np.asarray(x_goal, dtype=float).reshape(self.xdim)

        opti = ca.Opti()
        X = opti.variable(self.xdim, horizon_steps + 1)
        U = opti.variable(self.udim, horizon_steps)

        opti.subject_to(X[:, 0] == x_init)

        Q = ca.DM(self.Q)
        R = ca.DM(self.R)
        Q_f = ca.DM(self.Q_f)

        objective = 0
        for k in range(horizon_steps):
            xk = X[:, k]
            uk = U[:, k]
            x_next = self._rk4_step_symbolic(xk, uk)
            opti.subject_to(X[:, k + 1] == x_next)

            dx = xk - x_goal
            objective += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([uk.T, R, uk])

            if self.u_min is not None and self.u_max is not None:
                opti.subject_to(opti.bounded(ca.DM(self.u_min), uk, ca.DM(self.u_max)))
            else:
                if self.u_min is not None:
                    opti.subject_to(ca.DM(self.u_min) <= uk)
                if self.u_max is not None:
                    opti.subject_to(uk <= ca.DM(self.u_max))

        dx_terminal = X[:, horizon_steps] - x_goal
        objective += ca.mtimes([dx_terminal.T, Q_f, dx_terminal])
        opti.minimize(objective)

        opti.set_initial(X, x_guess)
        opti.set_initial(U, u_guess)

        solver_options = {"expand": False, **self.ipopt_options}
        opti.solver("ipopt", solver_options)

        solution = opti.solve()

        x_d = np.asarray(solution.value(X), dtype=float).reshape(self.xdim, horizon_steps + 1)
        u_d = np.asarray(solution.value(U), dtype=float).reshape(self.udim, horizon_steps)
        
        return x_d, u_d
