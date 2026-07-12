import numpy as np
from scipy.optimize import lsq_linear
from collections import deque

class ACP:
    """
    Time-interval-wise Adaptive Conformal Prediction (ACP), parameter fitting,
    and optional block-wise representation learning following Algorithm 1.

    Notes
    1) The paper allows delta_k to temporarily leave [0, 1]. To keep the order
       statistic well-defined, this implementation clamps the quantile rank to
       [1, |S_cal| + 1]. The ``quantile_edge_policy`` option chooses either
       the supplied finite score bounds or +inf for the appended-infinity
       order statistic.
    2) The controller can read ``acp.Q_k`` and write it into
       ``system.cp_quantile`` at the beginning of every interval.
    3) If `N_cal` is provided, S_cal is maintained as a
       moving FIFO window. Once the window is full, appending a new score
       automatically drops the oldest score.
    4) Representation learning is enabled by supplying ``theta_init`` and
       samples of Psi(x) to ``add_data_to_buffers``. The model is
       ``Y_Theta(x) = Psi(x) @ Theta``. Call ``update_representation`` once
       after fitting a_k and before clearing the interval buffers.
    """

    def __init__(
        self,
        S_cal_init, # numpy array or list
        N_cal: int = 1000,
        lr: float = 0.5, # learning rate
        delta_target: float = 0.05,
        delta_init: float = 0.05,
        score_max: float = 1.0, # max possible score
        score_min: float = 0.0, # min possible score
        buffer_maxlen: int = 1000,
        theta_init=None,
        representation_period: int = 1,
        representation_learning_rate=1e-3,
        theta_lb=None,
        theta_ub=None,
        quantile_edge_policy: str = "bounds",
    ):
        if N_cal < 100:
            raise ValueError("N_cal must be at least 100")
        if len(S_cal_init) == 0:
            raise ValueError("S_cal_init must be nonempty")
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if delta_target >= 1.0 or delta_target <= 0.0:
            raise ValueError("delta_target must be in (0,1)")
        if delta_init >= 1.0 or delta_init <= 0.0:
            raise ValueError("delta_init must be in (0,1)")
        if buffer_maxlen < 10:
            raise ValueError("buffer_maxlen must be at least 10")
        if quantile_edge_policy not in {"bounds", "infinity"}:
            raise ValueError("quantile_edge_policy must be 'bounds' or 'infinity'")
        
        if len(S_cal_init) > N_cal:
            S_cal_init = S_cal_init[-N_cal:]
        self.S_cal = deque(S_cal_init, maxlen=N_cal)

        self.N_cal = N_cal
        self.lr = lr
        self.delta_target = float(delta_target)
        self.delta = delta_init
        self.score_max = score_max
        self.score_min = score_min
        self.quantile_edge_policy = quantile_edge_policy
        self.compute_quantile() # update self.Q_k
        self.buffer_maxlen = buffer_maxlen

        # Moving window of data used to solve a_k
        self._x_buffer = deque(maxlen=self.buffer_maxlen) # to store x_t
        self._xdot_buffer = deque(maxlen=self.buffer_maxlen) # optional measured xdot_t
        self._xdot_nom_buffer = deque(maxlen=self.buffer_maxlen) # to store f(x_t) + g(x_t) u_t
        self._Y_buffer = deque(maxlen=self.buffer_maxlen) # to store Y(x_t)
        self._w_buffer = deque(maxlen=self.buffer_maxlen) # to store w_t
        self._Psi_buffer = deque(maxlen=self.buffer_maxlen) # to store Psi(x_t)

        # Optional block-wise representation learning (lines 19--23).
        self.Theta = None if theta_init is None else np.asarray(theta_init, dtype=float).copy()
        self.representation_period = int(representation_period)
        self.representation_learning_rate = representation_learning_rate
        self.theta_lb = theta_lb
        self.theta_ub = theta_ub
        self.interval_index = 0
        self.representation_update_index = 0
        self._representation_intervals = []
        self.last_representation_gradient = None

        if self.Theta is not None:
            if self.Theta.ndim != 2:
                raise ValueError("theta_init must be a two-dimensional matrix")
            if self.representation_period < 1:
                raise ValueError("representation_period must be at least 1")
            if (theta_lb is None) != (theta_ub is None):
                raise ValueError("theta_lb and theta_ub must either both be set or both be None")
            if theta_lb is not None:
                self.theta_lb = np.broadcast_to(
                    np.asarray(theta_lb, dtype=float), self.Theta.shape
                ).copy()
                self.theta_ub = np.broadcast_to(
                    np.asarray(theta_ub, dtype=float), self.Theta.shape
                ).copy()
                if np.any(self.theta_lb > self.theta_ub):
                    raise ValueError("theta_lb must be less than or equal to theta_ub")
                self.Theta = np.clip(self.Theta, self.theta_lb, self.theta_ub)

    def add_data_to_buffers(self, x, xdot_nom, Yx=None, Psi_x=None, xdot=None):
        """Append one sample from the current interval.

        ``Yx`` remains optional only when representation learning is enabled;
        in that case it is evaluated as ``Psi_x @ self.Theta``.
        """
        if Yx is None:
            if self.Theta is None or Psi_x is None:
                raise ValueError("Yx is required unless both theta_init and Psi_x are provided")
            Yx = np.asarray(Psi_x, dtype=float) @ self.Theta
        if self.Theta is not None and Psi_x is None:
            raise ValueError("Psi_x is required when representation learning is enabled")

        self._x_buffer.append(np.asarray(x, dtype=float).copy())
        self._xdot_nom_buffer.append(np.asarray(xdot_nom, dtype=float).copy())
        if xdot is not None:
            self._xdot_buffer.append(np.asarray(xdot, dtype=float).copy())
        self._Y_buffer.append(np.asarray(Yx, dtype=float).copy())
        if Psi_x is not None:
            self._Psi_buffer.append(np.asarray(Psi_x, dtype=float).copy())

    def clear_buffers(self):
        self._x_buffer = deque(maxlen=self.buffer_maxlen)
        self._xdot_buffer = deque(maxlen=self.buffer_maxlen)
        self._xdot_nom_buffer = deque(maxlen=self.buffer_maxlen)
        self._Y_buffer = deque(maxlen=self.buffer_maxlen)
        self._w_buffer = deque(maxlen=self.buffer_maxlen)
        self._Psi_buffer = deque(maxlen=self.buffer_maxlen)

    def estimate_uncertainty(self, dt):
        """
        Compute uncertainty data for t in I_k:
            w_t = xdot_t - (f_bar(x_t) + g_bar(x_t) u_t)
        """
        if len(self._x_buffer) < 2:
            raise ValueError("At least two state samples are required to estimate uncertainty")
        if len(self._x_buffer) != len(self._xdot_nom_buffer):
            raise ValueError("State and nominal-derivative buffers have inconsistent lengths")

        # Prefer sampled state derivatives, as written on line 6 of Algorithm
        # 1. Fall back to finite differences for simulations without an xdot
        # sensor or observer.
        if len(self._xdot_buffer) == len(self._x_buffer):
            x_dot_buffer = np.asarray(self._xdot_buffer)
        elif len(self._xdot_buffer) == 0:
            x_dot_buffer = np.gradient(np.array(self._x_buffer), dt, axis=0)
        else:
            raise ValueError("xdot must be supplied for either every sample or no samples")
        self._w_buffer = deque(
            (x_dot - x_dot_nom for (x_dot, x_dot_nom) in zip(x_dot_buffer, self._xdot_nom_buffer)),
            maxlen=self.buffer_maxlen,
        )

    def compute_score(self, a_ub, a_lb):
        """
        1. Fit the true (fictitious) parameter by solving the constrained least squares: 
                a_k = argmin_a sum_{t in I_k} ||Y(x_t) a - w_t||_2^2
                    s.t. a_lb <= a <= a_ub
           
        2. Compute the score: 
                s_k = sup_{t in I_k} ||Y(x_t) a_k - w_t||_2
        """

        if len(self._w_buffer) == 0:
            raise ValueError("Call estimate_uncertainty() before compute_score()")
        if len(self._Y_buffer) != len(self._w_buffer):
            raise ValueError("Y and uncertainty buffers have inconsistent lengths")

        # Fit the fictitious true parameter a_k.
        Y_stack = np.vstack(self._Y_buffer) # shape: (#sample * xdim, adim)
        w_stack = np.hstack(self._w_buffer) # shape: (#sample * xdim, )
        
        result = lsq_linear(Y_stack, w_stack, bounds=(a_lb, a_ub))
        self.a_k = result.x
        #a_k, *_ = np.linalg.lstsq(Y_stack, w_stack, rcond=None)
        #self.a_k = a_k

        # Compute the score s_k
        residual_norms = []
        for Y_t, w_t in zip(self._Y_buffer, self._w_buffer):
            r_t = Y_t @ self.a_k - w_t
            residual_norms.append(float(np.linalg.norm(r_t, ord=2)))
        s_k = np.max(residual_norms)

        return s_k

    def append_score(self, s_k):
        """Append the completed interval's score to the calibration window."""
        self.S_cal.append(float(s_k))

    def update_representation(self, a_k=None):
        """Apply lines 19--23 of Algorithm 1 when a block is complete.

        The gradient treats every fitted a_k as fixed, exactly as line 21:

            grad_Theta sum ||Psi(x_t) Theta a_k - w_t||^2.

        Returns ``None`` between representation updates. At an update, returns
        a dictionary containing the new Theta, gradient, and learning rate.
        """
        if self.Theta is None:
            return None
        if len(self._Psi_buffer) != len(self._w_buffer):
            raise ValueError("Psi and uncertainty buffers have inconsistent lengths")
        if len(self._Psi_buffer) == 0:
            raise ValueError("No interval data are available for representation learning")

        if a_k is None:
            if not hasattr(self, "a_k"):
                raise ValueError("Call compute_score() before update_representation()")
            a_k = self.a_k
        a_k = np.asarray(a_k, dtype=float).reshape(-1)

        self._representation_intervals.append({
            "Psi": np.asarray(self._Psi_buffer, dtype=float).copy(),
            "w": np.asarray(self._w_buffer, dtype=float).copy(),
            "a": a_k.copy(),
        })
        self.interval_index += 1

        if self.interval_index % self.representation_period != 0:
            return None

        gradient = np.zeros_like(self.Theta)
        for interval in self._representation_intervals:
            a_interval = interval["a"]
            for Psi_t, w_t in zip(interval["Psi"], interval["w"]):
                residual = Psi_t @ self.Theta @ a_interval - w_t
                gradient += 2.0 * np.outer(Psi_t.T @ residual, a_interval)

        self.representation_update_index += 1
        learning_rate = self._representation_step_size(self.representation_update_index)
        theta_next = self.Theta - learning_rate * gradient
        if self.theta_lb is not None:
            theta_next = np.clip(theta_next, self.theta_lb, self.theta_ub)

        self.Theta = theta_next
        self.last_representation_gradient = gradient
        self._representation_intervals.clear()

        return {
            "Theta": self.Theta.copy(),
            "gradient": gradient.copy(),
            "learning_rate": learning_rate,
            "update_index": self.representation_update_index,
        }

    def _representation_step_size(self, update_index):
        schedule = self.representation_learning_rate
        if callable(schedule):
            value = schedule(update_index)
        elif np.isscalar(schedule):
            value = schedule
        else:
            values = np.asarray(schedule, dtype=float).reshape(-1)
            if update_index > len(values):
                raise ValueError("representation learning-rate schedule is exhausted")
            value = values[update_index - 1]
        value = float(value)
        if value <= 0.0:
            raise ValueError("representation learning rates must be positive")
        return value

    def compute_quantile(self):
        """
        Return Q_k, the adaptive conformal quantile computed from the current
        calibration set and the current ACP failure estimate delta_k.
        """
        S_cal_sort = np.sort(np.asarray(self.S_cal, dtype=float))
        
        S_cal_size = len(self.S_cal) 
        assert S_cal_size <= self.N_cal

        rank = int(np.ceil((1.0 - self.delta) * (S_cal_size + 1)))
        rank = min(max(rank, 1), S_cal_size + 1)
        if rank == S_cal_size + 1:
            self.Q_k = (
                np.inf if self.quantile_edge_policy == "infinity" else self.score_max
            )
        else:
            self.Q_k = S_cal_sort[rank - 1]
        return self.Q_k

    def update_delta(self, s_k):
        """ 
        Update delta: delta_{k+1} = delta_k + lr * (delta_target - e_k) 
        """
        if self.Q_k is None:
            raise ValueError("Call compute_quantile() before update_delta().")
        
        e_k = int(s_k > self.Q_k) # assuming self.Q_k has already been updated

        self.delta = self.delta + self.lr * (self.delta_target - e_k)

        return e_k
