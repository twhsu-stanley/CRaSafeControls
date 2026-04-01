import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import casadi as ca

class MotionPlanner:

    def __init__(
        self,
        system,
        dt,
        horizon_steps,
        x_goal,
        Qx,
        Ru,
        Qf,
        u_min,
        u_max,
    ):
        
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be at least 1.")

        self.system = system
        self.dt = float(dt)
        self.default_horizon_steps = int(horizon_steps) # TODO: duplicated
        self.xdim = system.xdim
        self.udim = system.udim
        self.x_goal = x_goal
        self.Qx = Qx
        self.Ru = Ru
        self.Qf = Qf
        self.u_min = u_min
        self.u_max = u_max

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

    def plan(self, x0, t0, horizon_steps, x_guess, u_guess):
        horizon_steps = self.default_horizon_steps if horizon_steps is None else int(horizon_steps)
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be at least 1.")

        x0 = np.asarray(x0, dtype=float).reshape(self.xdim)
        t0 = float(t0)

        opti = ca.Opti()
        X = opti.variable(self.xdim, horizon_steps + 1)
        U = opti.variable(self.udim, horizon_steps)

        opti.subject_to(X[:, 0] == x0)

        Qx = ca.DM(self.Qx)
        Ru = ca.DM(self.Ru)
        Qf = ca.DM(self.Qf)
        x_goal = ca.DM(self.x_goal)

        objective = 0
        for k in range(horizon_steps):
            xk = X[:, k]
            uk = U[:, k]
            x_next = self._rk4_step_symbolic(xk, uk)
            opti.subject_to(X[:, k + 1] == x_next)

            dx = xk - x_goal
            objective += ca.mtimes([dx.T, Qx, dx]) + ca.mtimes([uk.T, Ru, uk])

            if self.u_min is not None and self.u_max is not None:
                opti.subject_to(opti.bounded(ca.DM(self.u_min), uk, ca.DM(self.u_max)))
            else:
                if self.u_min is not None:
                    opti.subject_to(ca.DM(self.u_min) <= uk)
                if self.u_max is not None:
                    opti.subject_to(uk <= ca.DM(self.u_max))

        dx_terminal = X[:, horizon_steps] - x_goal
        objective += ca.mtimes([dx_terminal.T, Qf, dx_terminal])
        opti.minimize(objective)

        opti.set_initial(X, x_guess)
        opti.set_initial(U, u_guess)

        solver_options = {"expand": False, **self.ipopt_options}
        opti.solver("ipopt", solver_options)

        solution = opti.solve()

        x_nom = np.asarray(solution.value(X), dtype=float).reshape(self.xdim, horizon_steps + 1)
        u_nom = np.asarray(solution.value(U), dtype=float).reshape(self.udim, horizon_steps)
        t_grid = t0 + self.dt * np.arange(horizon_steps + 1)
        return t_grid, x_nom, u_nom
