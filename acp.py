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
       [1, |S_cal| + 1]. For the appended-infinity order statistic, it uses the
       largest score in the current calibration set.
    2) The controller can read ``acp.Q_k`` and write it into
       ``system.cp_quantile`` at the beginning of every interval.
    3) If `N_cal` is provided, S_cal is maintained as a
       moving FIFO window. Once the window is full, appending a new score
       automatically drops the oldest score.
    4) Representation learning is enabled by supplying ``theta_init``, a
       callable ``Y_theta(x, theta)``, and a callable
       ``representation_loss_gradient(x, theta, a, w)``. The first callable
       may have any structure (for example, a feature model or neural network)
       as long as it returns the matrix Y_Theta(x). The second returns the
       gradient of ``||Y_Theta(x) @ a - w||_2**2`` with respect to theta.
       Call ``update_representation`` once after fitting a_k and before
       clearing the interval buffers.
    """

    def __init__(
        self,
        S_cal_init, # numpy array or list
        N_cal: int = 1000,
        acp_lr: float = 0.5, # learning rate
        delta_target: float = 0.05,
        delta_init: float = 0.05,
        buffer_maxlen: int = 1000,
        theta_init=None,
        representation_period: int = 1,
        representation_lr=1e-3,
        theta_lb=None,
        theta_ub=None,
        Y_theta=None,
        representation_loss_gradient=None,
    ):
        if N_cal < 100:
            raise ValueError("N_cal must be at least 100")
        if len(S_cal_init) == 0:
            raise ValueError("S_cal_init must be nonempty")
        if acp_lr <= 0.0:
            raise ValueError("acp_lr must be positive")
        if delta_target >= 1.0 or delta_target <= 0.0:
            raise ValueError("delta_target must be in (0,1)")
        if delta_init >= 1.0 or delta_init <= 0.0:
            raise ValueError("delta_init must be in (0,1)")
        if buffer_maxlen < 10:
            raise ValueError("buffer_maxlen must be at least 10")
        
        if len(S_cal_init) > N_cal:
            S_cal_init = S_cal_init[-N_cal:]
        self.S_cal = deque(S_cal_init, maxlen=N_cal)

        self.N_cal = N_cal
        self.acp_lr = acp_lr
        self.delta_target = float(delta_target)
        self.delta = delta_init
        self.compute_quantile() # update self.Q_k
        self.buffer_maxlen = buffer_maxlen

        # Moving window of data used to solve a_k
        self._x_buffer = deque(maxlen=self.buffer_maxlen) # to store x_t
        self._xdot_buffer = deque(maxlen=self.buffer_maxlen) # optional measured xdot_t
        self._xdot_nom_buffer = deque(maxlen=self.buffer_maxlen) # to store f(x_t) + g(x_t) u_t
        self._Y_buffer = deque(maxlen=self.buffer_maxlen) # to store Y(x_t)
        self._w_buffer = deque(maxlen=self.buffer_maxlen) # to store w_t

        # Optional block-wise representation learning (lines 19--23).
        self.Theta = None if theta_init is None else np.asarray(theta_init, dtype=float).copy()
        self.representation_period = int(representation_period)
        self.representation_lr = representation_lr
        self.Y_theta = Y_theta
        self.representation_loss_gradient = representation_loss_gradient
        self.theta_lb = theta_lb
        self.theta_ub = theta_ub
        self.interval_index = 0
        self.representation_update_index = 0
        self._representation_intervals = []
        self.last_representation_gradient = None

        if self.Theta is not None:
            if self.Theta.size == 0:
                raise ValueError("theta_init must be nonempty")
            if not callable(self.Y_theta):
                raise TypeError("Y_theta must be callable when theta_init is provided")
            if not callable(self.representation_loss_gradient):
                raise TypeError(
                    "representation_loss_gradient must be callable when "
                    "theta_init is provided"
                )
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

    def add_data_to_buffers(self, x, xdot_nom, Yx=None, xdot=None):
        """Append one sample from the current interval.

        ``Yx`` remains optional only when representation learning is enabled;
        in that case it is evaluated as ``Y_theta(x, self.Theta)``.
        """
        if Yx is None:
            if self.Theta is None:
                raise ValueError("Yx is required when representation learning is disabled")
            Yx = self.Y_theta(x, self.Theta)

        Yx = np.asarray(Yx, dtype=float)
        if Yx.ndim != 2:
            raise ValueError("Yx must be a two-dimensional matrix")

        xdot_nom = np.asarray(xdot_nom, dtype=float).reshape(-1)
        if Yx.shape[0] != xdot_nom.size:
            raise ValueError(
                "Yx must have one row per state derivative: expected "
                f"{xdot_nom.size}, got {Yx.shape[0]}"
            )
        if self._Y_buffer and Yx.shape != self._Y_buffer[0].shape:
            raise ValueError(
                "Yx shape must remain constant within an interval: expected "
                f"{self._Y_buffer[0].shape}, got {Yx.shape}"
            )

        if xdot is not None:
            xdot = np.asarray(xdot, dtype=float).reshape(-1)
            if xdot.size != xdot_nom.size:
                raise ValueError(
                    "xdot and xdot_nom must contain the same number of elements"
                )

        self._x_buffer.append(np.asarray(x, dtype=float).reshape(-1).copy())
        self._xdot_nom_buffer.append(xdot_nom.copy())
        if xdot is not None:
            self._xdot_buffer.append(xdot.copy())
        self._Y_buffer.append(Yx.copy())

    def clear_buffers(self):
        self._x_buffer = deque(maxlen=self.buffer_maxlen)
        self._xdot_buffer = deque(maxlen=self.buffer_maxlen)
        self._xdot_nom_buffer = deque(maxlen=self.buffer_maxlen)
        self._Y_buffer = deque(maxlen=self.buffer_maxlen)
        self._w_buffer = deque(maxlen=self.buffer_maxlen)

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

        parameter_dimension = self._Y_buffer[0].shape[1]
        a_lb = np.asarray(a_lb, dtype=float)
        a_ub = np.asarray(a_ub, dtype=float)
        if a_lb.size not in {1, parameter_dimension}:
            raise ValueError(
                f"a_lb must be scalar or have length {parameter_dimension}"
            )
        if a_ub.size not in {1, parameter_dimension}:
            raise ValueError(
                f"a_ub must be scalar or have length {parameter_dimension}"
            )
        a_lb = float(a_lb.item()) if a_lb.size == 1 else a_lb.reshape(-1)
        a_ub = float(a_ub.item()) if a_ub.size == 1 else a_ub.reshape(-1)

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

            grad_Theta sum ||Y_Theta(x_t) a_k - w_t||^2.

        Returns ``None`` between representation updates. At an update, returns
        a dictionary containing the new Theta, gradient, and learning rate.
        """
        if self.Theta is None:
            return None
        if len(self._x_buffer) != len(self._w_buffer):
            raise ValueError("State and uncertainty buffers have inconsistent lengths")
        if len(self._x_buffer) == 0:
            raise ValueError("No interval data are available for representation learning")

        if a_k is None:
            if not hasattr(self, "a_k"):
                raise ValueError("Call compute_score() before update_representation()")
            a_k = self.a_k
        a_k = np.asarray(a_k, dtype=float).reshape(-1)

        self._representation_intervals.append({
            "x": np.asarray(self._x_buffer, dtype=float).copy(),
            "w": np.asarray(self._w_buffer, dtype=float).copy(),
            "a": a_k.copy(),
        })
        self.interval_index += 1

        if self.interval_index % self.representation_period != 0:
            return None

        gradient = np.zeros_like(self.Theta)
        for interval in self._representation_intervals:
            a_interval = interval["a"]
            for x_t, w_t in zip(interval["x"], interval["w"]):
                sample_gradient = np.asarray(
                    self.representation_loss_gradient(
                        x_t, self.Theta, a_interval, w_t
                    ),
                    dtype=float,
                )
                if sample_gradient.shape != self.Theta.shape:
                    raise ValueError(
                        "representation_loss_gradient must return an array "
                        f"with shape {self.Theta.shape}, got "
                        f"{sample_gradient.shape}"
                    )
                gradient += sample_gradient

        self.representation_update_index += 1
        representation_lr = self._representation_learning_rate(self.representation_update_index)
        theta_next = self.Theta - representation_lr * gradient
        if self.theta_lb is not None:
            theta_next = np.clip(theta_next, self.theta_lb, self.theta_ub)

        self.Theta = theta_next
        self.last_representation_gradient = gradient
        self._representation_intervals.clear()

        return {
            "Theta": self.Theta.copy(),
            "gradient": gradient.copy(),
            "representation_lr": representation_lr,
            "update_index": self.representation_update_index,
        }

    def _representation_learning_rate(self, update_index):
        schedule = self.representation_lr
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
            self.Q_k = S_cal_sort[-1]
        else:
            self.Q_k = S_cal_sort[rank - 1]
        return self.Q_k

    def update_delta(self, s_k):
        """ 
        Update delta: delta_{k+1} = delta_k + acp_lr * (delta_target - e_k) 
        """
        if self.Q_k is None:
            raise ValueError("Call compute_quantile() before update_delta().")
        
        e_k = int(s_k > self.Q_k) # assuming self.Q_k has already been updated

        self.delta = self.delta + self.acp_lr * (self.delta_target - e_k)

        return e_k
