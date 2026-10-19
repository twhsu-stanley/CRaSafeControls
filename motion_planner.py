import numpy as np
import casadi as ca
from systems.control_affine_system import ControlAffineSystem

class MotionPlanner:
    """Plan full-state trajectories with discrete-time output and input costs."""

    def __init__(
        self,
        system: ControlAffineSystem,
        dt: float,
        Q: np.ndarray,
        R: np.ndarray,
        Q_f: np.ndarray,
        u_min: np.ndarray = None,
        u_max: np.ndarray = None,
        *,
        output_matrix: np.ndarray = None,
        Q_terminal: np.ndarray = None,
        S: np.ndarray = None,
    ):
        """Track y = output_matrix @ x, defaulting to full-state tracking.

        Q_f weights scheduled waypoints; Q_terminal weights the MPC horizon end.
        S optionally penalizes input increments in the certainty-equivalent MPC.
        All weights are discrete-time weights, without a time-step multiplier.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        self.system = system
        self.dt = dt
        self.xdim = system.xdim
        self.udim = system.udim
        self.output_matrix = np.eye(self.xdim) if output_matrix is None else np.asarray(
            output_matrix, dtype=float
        )
        if self.output_matrix.ndim != 2 or self.output_matrix.shape[1] != self.xdim:
            raise ValueError("output_matrix must have shape (ydim, xdim)")
        self.ydim = self.output_matrix.shape[0]
        if self.ydim < 1 or not np.all(np.isfinite(self.output_matrix)):
            raise ValueError("output_matrix must be finite and have at least one row")
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.Q_f = np.asarray(Q_f, dtype=float)
        self.Q_terminal = (
            self.Q.copy() if Q_terminal is None else np.asarray(Q_terminal, dtype=float)
        )
        self.S = np.zeros((self.udim, self.udim)) if S is None else np.asarray(S, dtype=float)
        for name, weight, dim in (
            ("Q", self.Q, self.ydim), ("Q_f", self.Q_f, self.ydim),
            ("Q_terminal", self.Q_terminal, self.ydim), ("R", self.R, self.udim),
            ("S", self.S, self.udim),
        ):
            if weight.shape != (dim, dim) or not np.all(np.isfinite(weight)):
                raise ValueError(f"{name} must be a finite ({dim}, {dim}) matrix")
            if not np.allclose(weight, weight.T) or np.min(np.linalg.eigvalsh(weight)) < -1e-10:
                raise ValueError(f"{name} must be symmetric positive semidefinite")
        self.u_min = u_min
        self.u_max = u_max
        # TODO: inclde obstacles

        self.ipopt_options = {
            "print_time": False,
            "ipopt": {"print_level": 0, "sb": "yes", "max_iter": 300},
        }

        self._nominal_dynamics = self._build_nominal_dynamics_function()
        self._nominal_rk4 = self._build_rk4_function("nominal_rk4", self._nominal_dynamics)
        self._certainty_equivalent_dynamics = None
        self._certainty_equivalent_rk4 = None
        self._mpc_problem = None
        self._mpc_warm_start = None
        self.last_mpc_stats = None

    @staticmethod
    def _to_casadi_matrix(values, shape):
        """Convert vector or matrix entries to CasADi without flattening their layout."""
        if isinstance(values, (ca.MX, ca.SX, ca.DM)):
            if values.shape != shape:
                raise ValueError(f"Expected shape {shape}, got {values.shape}")
            return values
        entries = np.asarray(values, dtype=object)
        if entries.ndim == 1:
            entries = entries[:, None]
        if entries.shape != shape:
            raise ValueError(f"Expected shape {shape}, got {entries.shape}")
        return ca.vertcat(*(ca.horzcat(*row) for row in entries))

    def _build_nominal_dynamics_function(self):
        """Build the nominal continuous-time dynamics from the system's f and g."""
        x = ca.MX.sym("x", self.xdim)
        u = ca.MX.sym("u", self.udim)
        x_entries = ca.vertsplit(x)
        f = self._to_casadi_matrix(self.system.f(x_entries), (self.xdim, 1))
        g = self._to_casadi_matrix(self.system.g(x_entries), (self.xdim, self.udim))
        return ca.Function("nominal_dynamics", [x, u], [f + g @ u])

    def _build_certainty_equivalent_dynamics_function(self):
        """Build the certainty-equivalent dynamics using the system's whole Y_Theta."""
        x = ca.MX.sym("x", self.xdim)
        u = ca.MX.sym("u", self.udim)
        theta = ca.MX.sym("theta", *self.system.theta_shape)
        a_hat = ca.MX.sym("a_hat", self.system.adim)
        theta_entries = np.array(ca.vertsplit(ca.vec(theta)), dtype=object)
        theta_entries = theta_entries.reshape(self.system.theta_shape, order="F")
        Y_theta = self._to_casadi_matrix(
            self.system.Y_Theta(ca.vertsplit(x), theta_entries), (self.xdim, self.system.adim)
        )
        xdot = self._nominal_dynamics(x, u) + Y_theta @ a_hat
        return ca.Function("certainty_equivalent_dynamics", [x, u, theta, a_hat], [xdot])

    def _build_rk4_function(self, name, dynamics, dt=None):
        """Build an RK4 state-transition map with inputs and parameters held constant."""
        x, u = dynamics.mx_in(0), dynamics.mx_in(1)
        parameters = [dynamics.mx_in(i) for i in range(2, dynamics.n_in())]
        dt = self.dt if dt is None else dt
        k1 = dynamics(x, u, *parameters)
        k2 = dynamics(x + 0.5 * dt * k1, u, *parameters)
        k3 = dynamics(x + 0.5 * dt * k2, u, *parameters)
        k4 = dynamics(x + dt * k3, u, *parameters)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return ca.Function(name, [x, u, *parameters], [x_next]).expand()

    def _build_certainty_equivalent_mpc_problem(self, horizon_steps, prediction_dt):
        """Build a reusable fixed-size problem with model and reference parameters."""
        if self._certainty_equivalent_dynamics is None:
            self._certainty_equivalent_dynamics = (
                self._build_certainty_equivalent_dynamics_function()
            )
        self._certainty_equivalent_rk4 = self._build_rk4_function(
            "certainty_equivalent_rk4", self._certainty_equivalent_dynamics, prediction_dt
        )
        opti = ca.Opti()
        X = opti.variable(self.xdim, horizon_steps + 1)
        U = opti.variable(self.udim, horizon_steps)
        parameters = {
            "x_init": opti.parameter(self.xdim),
            "theta": opti.parameter(*self.system.theta_shape),
            "a_hat": opti.parameter(self.system.adim),
            "u_reference": opti.parameter(self.udim),
            "u_previous": opti.parameter(self.udim),
            "output_reference": opti.parameter(self.ydim, horizon_steps + 1),
            "waypoint_reference": opti.parameter(self.ydim, horizon_steps),
            "waypoint_mask": opti.parameter(1, horizon_steps),
            "waypoint_fraction": opti.parameter(1, horizon_steps),
        }
        opti.subject_to(X[:, 0] == parameters["x_init"])
        predicted_next = self._certainty_equivalent_rk4.map(horizon_steps)(
            X[:, :-1], U, ca.repmat(parameters["theta"], 1, horizon_steps),
            ca.repmat(parameters["a_hat"], 1, horizon_steps),
        )
        opti.subject_to(X[:, 1:] == predicted_next)
        outputs = ca.DM(self.output_matrix) @ X
        output_error = outputs[:, :-1] - parameters["output_reference"][:, :-1]
        input_error = U - ca.repmat(parameters["u_reference"], 1, horizon_steps)
        input_increment = ca.horzcat(U[:, 0] - parameters["u_previous"], U[:, 1:] - U[:, :-1])
        objective = ca.dot(output_error, ca.DM(self.Q) @ output_error)
        objective += ca.dot(input_error, ca.DM(self.R) @ input_error)
        objective += ca.dot(input_increment, ca.DM(self.S) @ input_increment)
        fraction = ca.repmat(parameters["waypoint_fraction"], self.ydim, 1)
        waypoint_outputs = outputs[:, :-1] + fraction * (outputs[:, 1:] - outputs[:, :-1])
        waypoint_error = waypoint_outputs - parameters["waypoint_reference"]
        waypoint_cost = ca.sum1(waypoint_error * (ca.DM(self.Q_f) @ waypoint_error))
        objective += ca.dot(parameters["waypoint_mask"], waypoint_cost)
        terminal_error = outputs[:, -1] - parameters["output_reference"][:, -1]
        objective += ca.mtimes([terminal_error.T, ca.DM(self.Q_terminal), terminal_error])
        opti.minimize(objective)
        if self.u_min is not None:
            lower = ca.repmat(ca.DM(self.u_min), 1, horizon_steps)
            opti.subject_to(ca.vec(lower) <= ca.vec(U))
        if self.u_max is not None:
            upper = ca.repmat(ca.DM(self.u_max), 1, horizon_steps)
            opti.subject_to(ca.vec(U) <= ca.vec(upper))
        opti.solver("ipopt", {"expand": False, **self.ipopt_options})
        return {
            "opti": opti, "X": X, "U": U, "parameters": parameters,
            "horizon_steps": horizon_steps, "prediction_dt": prediction_dt,
        }

    def plan_certainty_equivalent_through_waypoints(
        self, x_init, waypoints, interval_steps, start_step, theta, a_hat, u_reference=None,
        *, horizon_duration=4.0, prediction_dt=None, output_reference=None, u_previous=None,
    ):
        """Replan a fixed-length certainty-equivalent horizon from the desired state.

        Waypoint times stay on the mission clock. Every future waypoint within
        the horizon receives Q_f at its scheduled time, interpolating adjacent
        predicted outputs for off-grid times. The running output reference is only a
        cost guide, not the dynamically generated desired state. Beyond the
        mission end the guide is padded with the final waypoint. Theta and
        a_hat are frozen during prediction and refreshed at each replan. All
        cost weights are discrete-time weights; no time-step multiplier is applied.
        """
        interval_steps_int, start_step_int = int(interval_steps), int(start_step)
        if interval_steps_int != interval_steps or interval_steps_int < 1:
            raise ValueError("interval_steps must be a positive integer")
        if start_step_int != start_step or start_step_int < 0:
            raise ValueError("start_step must be a nonnegative integer")
        interval_steps, start_step = interval_steps_int, start_step_int
        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != self.ydim or len(waypoints) < 2:
            raise ValueError("waypoints must have shape (K + 1, ydim), with K >= 1")
        if start_step >= (len(waypoints) - 1) * interval_steps:
            raise ValueError("start_step must precede the last waypoint deadline")
        prediction_dt = self.dt if prediction_dt is None else float(prediction_dt)
        horizon_duration = float(horizon_duration)
        interval_duration = interval_steps * self.dt
        if not np.isfinite(prediction_dt) or not 0.0 < prediction_dt <= interval_duration:
            raise ValueError("prediction_dt must be positive and no longer than an interval")
        if not np.isfinite(horizon_duration) or horizon_duration < prediction_dt:
            raise ValueError("horizon_duration must be finite and at least prediction_dt")
        horizon_steps = int(round(horizon_duration / prediction_dt))
        if not np.isclose(horizon_steps * prediction_dt, horizon_duration):
            raise ValueError("horizon_duration must be an integer multiple of prediction_dt")
        x_init = np.asarray(x_init, dtype=float).reshape(self.xdim)
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.system.theta_shape:
            raise ValueError(f"theta must have shape {self.system.theta_shape}")
        a_hat = np.asarray(a_hat, dtype=float).reshape(self.system.adim)
        if u_reference is None:
            u_reference = np.zeros(self.udim)
        u_reference = np.asarray(u_reference, dtype=float).reshape(self.udim)
        u_previous = u_reference if u_previous is None else u_previous
        u_previous = np.asarray(u_previous, dtype=float).reshape(self.udim)
        arrays = (waypoints, x_init, theta, a_hat, u_reference, u_previous)
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("MPC states, parameters, and references must be finite")

        start_time = start_step * self.dt
        prediction_times = start_time + prediction_dt * np.arange(horizon_steps + 1)
        waypoint_times = interval_duration * np.arange(len(waypoints))
        if output_reference is None:
            guide = np.array(
                [np.interp(prediction_times, waypoint_times, row) for row in waypoints.T]
            )
        else:
            guide = np.asarray(output_reference(prediction_times), dtype=float)
        if guide.shape != (self.ydim, horizon_steps + 1) or not np.all(np.isfinite(guide)):
            raise ValueError(
                "output_reference must return a finite (ydim, horizon_steps + 1) array"
            )
        waypoint_reference = np.zeros((self.ydim, horizon_steps))
        waypoint_mask = np.zeros((1, horizon_steps))
        waypoint_fraction = np.zeros((1, horizon_steps))
        included_waypoints = []
        for index, waypoint_time in enumerate(waypoint_times[1:], start=1):
            if not start_time + 1e-9 < waypoint_time <= prediction_times[-1] + 1e-9:
                continue
            offset = (waypoint_time - start_time) / prediction_dt
            stage = min(horizon_steps - 1, max(0, int(np.ceil(offset - 1e-9)) - 1))
            waypoint_reference[:, stage] = waypoints[index]
            waypoint_mask[0, stage] = 1.0
            waypoint_fraction[0, stage] = np.clip(offset - stage, 0.0, 1.0)
            included_waypoints.append(index)

        problem = self._mpc_problem
        if problem is None or (problem["horizon_steps"], problem["prediction_dt"]) != (
            horizon_steps, prediction_dt
        ):
            problem = self._build_certainty_equivalent_mpc_problem(horizon_steps, prediction_dt)
            self._mpc_problem = problem
            self._mpc_warm_start = None
        opti, X, U = problem["opti"], problem["X"], problem["U"]
        parameter_values = {
            "x_init": x_init, "theta": theta, "a_hat": a_hat, "u_reference": u_reference,
            "u_previous": u_previous, "output_reference": guide,
            "waypoint_reference": waypoint_reference,
            "waypoint_mask": waypoint_mask, "waypoint_fraction": waypoint_fraction,
        }
        for name, value in parameter_values.items():
            opti.set_value(problem["parameters"][name], value)

        warm_start = self._mpc_warm_start
        if warm_start is not None:
            previous_time, previous_waypoints, previous_interval, previous_u = warm_start
            shift_float = (start_time - previous_time) / prediction_dt
            shift = int(round(shift_float))
            compatible = previous_interval == interval_duration
            compatible &= np.array_equal(previous_waypoints, waypoints)
            compatible &= np.isclose(shift, shift_float)
            compatible &= 0 <= shift < previous_u.shape[1]
        else:
            compatible = False
        if compatible:
            tail = np.repeat(previous_u[:, -1:], shift, axis=1)
            u_guess = np.hstack((previous_u[:, shift:], tail))
        else:
            u_guess = np.repeat(u_reference[:, None], horizon_steps, axis=1)
        x_guess = np.empty((self.xdim, horizon_steps + 1))
        x_guess[:, 0] = x_init
        for k in range(horizon_steps):
            next_state = self._certainty_equivalent_rk4(x_guess[:, k], u_guess[:, k], theta, a_hat)
            x_guess[:, k + 1] = np.asarray(next_state).reshape(self.xdim)
        opti.set_initial(X, x_guess)
        opti.set_initial(U, u_guess)
        solution = opti.solve()
        x_d = np.asarray(solution.value(X), dtype=float).reshape(self.xdim, horizon_steps + 1)
        u_d = np.asarray(solution.value(U), dtype=float).reshape(self.udim, horizon_steps)
        self._mpc_warm_start = (start_time, waypoints.copy(), interval_duration, u_d)
        stats = solution.stats()
        self.last_mpc_stats = {
            "horizon_steps": horizon_steps,
            "prediction_dt": prediction_dt,
            "waypoint_indices": included_waypoints,
            "iterations": int(stats["iter_count"]),
            "return_status": stats["return_status"],
        }
        return x_d, u_d

    def plan_nominal_towards_goal(
        self,
        x_init,
        x_goal,
        horizon_steps,
        x_guess,
        u_guess,
        u_reference=None,
    ):
        """Plan a nominal trajectory towards x_goal"""

        horizon_steps = int(horizon_steps)
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be at least 1")

        x_init = np.asarray(x_init, dtype=float).reshape(self.xdim)
        x_goal = np.asarray(x_goal, dtype=float).reshape(self.xdim)

        if u_reference is None:
            u_reference = np.zeros(self.udim)
        u_reference = np.asarray(u_reference, dtype=float).reshape(self.udim)
        if not np.all(np.isfinite(u_reference)):
            raise ValueError("u_reference must be finite")

        opti = ca.Opti()
        X = opti.variable(self.xdim, horizon_steps + 1)
        U = opti.variable(self.udim, horizon_steps)

        opti.subject_to(X[:, 0] == x_init)

        Q = ca.DM(self.Q)
        R = ca.DM(self.R)
        Q_f = ca.DM(self.Q_f)
        u_reference = ca.DM(u_reference)

        objective = 0
        for k in range(horizon_steps):
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = self._nominal_rk4(x_k, u_k)
            opti.subject_to(X[:, k + 1] == x_next)

            dx = ca.DM(self.output_matrix) @ (x_k - x_goal)
            du = u_k - u_reference
            objective += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([du.T, R, du])

            if self.u_min is not None and self.u_max is not None:
                opti.subject_to(opti.bounded(ca.DM(self.u_min), u_k, ca.DM(self.u_max)))
            else:
                if self.u_min is not None:
                    opti.subject_to(ca.DM(self.u_min) <= u_k)
                if self.u_max is not None:
                    opti.subject_to(u_k <= ca.DM(self.u_max))

        dx_terminal = ca.DM(self.output_matrix) @ (X[:, horizon_steps] - x_goal)
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

    def plan_nominal_through_waypoints(
        self,
        waypoints,
        interval_steps,
        x_guess,
        u_guess,
        u_reference=None,
        *,
        x_init=None,
    ):
        """Plan a nominal trajectory through output waypoints from a full initial state."""

        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != self.ydim or len(waypoints) < 2:
            raise ValueError("waypoints must have shape (K + 1, ydim), with K >= 1")
        if int(interval_steps) != interval_steps or interval_steps < 1:
            raise ValueError("interval_steps must be a positive integer")
        interval_steps = int(interval_steps)
        if x_init is None:
            if self.ydim != self.xdim or not np.array_equal(self.output_matrix, np.eye(self.xdim)):
                raise ValueError("x_init is required when tracking outputs instead of full states")
            x_init = waypoints[0]
        x_init = np.asarray(x_init, dtype=float).reshape(self.xdim)
        if not np.all(np.isfinite(waypoints)) or not np.all(np.isfinite(x_init)):
            raise ValueError("waypoints and x_init must be finite")

        if u_reference is None:
            u_reference = np.zeros(self.udim)
        u_reference = np.asarray(u_reference, dtype=float).reshape(self.udim)
        if not np.all(np.isfinite(u_reference)):
            raise ValueError("u_reference must be finite")

        interval_count = waypoints.shape[0] - 1
        horizon_steps = interval_count * interval_steps
        opti = ca.Opti()
        X = opti.variable(self.xdim, horizon_steps + 1)
        U = opti.variable(self.udim, horizon_steps)

        opti.subject_to(X[:, 0] == ca.DM(x_init))
        
        Q = ca.DM(self.Q)
        R = ca.DM(self.R)
        Q_f = ca.DM(self.Q_f)
        u_reference_ca = ca.DM(u_reference)
        objective = 0
        for k in range(horizon_steps):
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = self._nominal_rk4(x_k, u_k)
            opti.subject_to(X[:, k + 1] == x_next)

            output_error = ca.DM(self.output_matrix) @ x_k - waypoints[k // interval_steps + 1]
            input_error = u_k - u_reference_ca
            objective += ca.mtimes([output_error.T, Q, output_error])
            objective += ca.mtimes([input_error.T, R, input_error])

            # Apply Q_f cost at the end of each interval to encourage reaching the waypoints
            if (k + 1) % interval_steps == 0:
                waypoint_index = (k + 1) // interval_steps
                waypoint_error = (
                    ca.DM(self.output_matrix) @ X[:, k + 1] - waypoints[waypoint_index]
                )
                objective += ca.mtimes([waypoint_error.T, Q_f, waypoint_error])

            if self.u_min is not None and self.u_max is not None:
                opti.subject_to(opti.bounded(ca.DM(self.u_min), u_k, ca.DM(self.u_max)))
            else:
                if self.u_min is not None:
                    opti.subject_to(ca.DM(self.u_min) <= u_k)
                if self.u_max is not None:
                    opti.subject_to(u_k <= ca.DM(self.u_max))

        opti.minimize(objective)
        opti.set_initial(X, x_guess)
        opti.set_initial(U, u_guess)
        opti.solver("ipopt", {"expand": False, **self.ipopt_options})

        solution = opti.solve()
        x_d = np.asarray(solution.value(X), dtype=float).reshape(self.xdim, horizon_steps + 1)
        u_d = np.asarray(solution.value(U), dtype=float).reshape(self.udim, horizon_steps)

        return x_d, u_d
