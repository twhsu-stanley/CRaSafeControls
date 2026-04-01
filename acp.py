import numpy as np
from scipy.optimize import lsq_linear
from collections import deque

class ACP:
    """
    Time-interval-wise Adaptive Conformal Prediction (ACP) with online parameter fitting,
    following Algorithm 1 in the paper.

    Notes
    1) The paper allows delta_k to temporarily leave [0, 1]. To keep the order
       statistic well-defined, this implementation clamps the quantile rank to
       [1, |S_cal| + 1]. If the requested rank is |S_cal| + 1, the returned
       quantile is +inf, matching the "union infinity" construction.
    2) This class only manages ACP and the interval-wise online learning step.
       Your controller can simply read `acp.Q_k` (or the value returned
       by `acc.compute_quantile()`) and write it into `system.cp_quantile`.
    3) If `N_cal` is provided, S_cal is maintained as a
       moving FIFO window. Once the window is full, appending a new score
       automatically drops the oldest score.
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
        buffer_maxlen: int = 1000
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
        
        if len(S_cal_init) > N_cal:
            S_cal_init = S_cal_init[-N_cal:]
        self.S_cal = deque(S_cal_init, maxlen=N_cal)

        self.N_cal = N_cal
        self.lr = lr
        self.delta_target = float(delta_target)
        self.delta = delta_init
        self.score_max = score_max
        self.score_min = score_min
        self.compute_quantile() # update self.Q_k
        self.buffer_maxlen = buffer_maxlen

        # Moving window of data used to solve a_k
        self._x_buffer = deque(maxlen=self.buffer_maxlen) # to store x_t
        self._xdot_nom_buffer = deque(maxlen=self.buffer_maxlen) # to store f(x_t) + g(x_t) u_t
        self._Y_buffer = deque(maxlen=self.buffer_maxlen) # to store Y(x_t)
        self._w_buffer = deque(maxlen=self.buffer_maxlen) # to store w_t

    def add_data_to_buffers(self, x, xdot_nom, Yx):
        self._x_buffer.append(x)
        self._xdot_nom_buffer.append(xdot_nom)
        self._Y_buffer.append(Yx)

    def clear_buffers(self):
        self._x_buffer = deque(maxlen=self.buffer_maxlen)
        self._xdot_nom_buffer = deque(maxlen=self.buffer_maxlen)
        self._Y_buffer = deque(maxlen=self.buffer_maxlen)
        self._w_buffer = deque(maxlen=self.buffer_maxlen)

    def estimate_uncertainty(self, dt):
        """
        Compute uncertainty data for t in I_k:
            w_t = xdot_t - (f_bar(x_t) + g_bar(x_t) u_t)
        """
        #Given a state trajectory x_t, compute a vector of w_t
        x_dot_buffer = np.gradient(np.array(self._x_buffer), dt, axis=0)
        self._w_buffer =  [x_dot - x_dot_nom for (x_dot, x_dot_nom) in zip(x_dot_buffer, self._xdot_nom_buffer)] 

    def compute_score(self, a_ub, a_lb):
        """
        1. Fit the true (fictitious) parameter by solving the constrained least squares: 
                a_k = argmin_a sum_{t in I_k} ||Y(x_t) a - w_t||_2
                    s.t. a_lb <= a <= a_ub
           
        2. Compute the score: 
                s_k = sup_{t in I_k} ||Y(x_t) a_k - w_t||_2
        """

        # Fit true parameter a_k
        Y_stack = np.vstack(self._Y_buffer) # shape: (#sample * xdim, adim)
        w_stack = np.hstack(self._w_buffer) # shape: (#sample * xdim, )
        
        print("Fitting a_k")
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

    def compute_quantile(self):
        """
        Return Q_k, the adaptive conformal quantile computed from the current
        calibration set and the current ACP failure estimate delta_k.
        """
        S_cal_sort = np.sort(self.S_cal)
        
        S_cal_size = len(self.S_cal) 
        assert S_cal_size <= self.N_cal

        if 1.0 - self.delta <= 0.0:
            self.Q_k = self.score_min #-np.inf
            return self.Q_k
        if 1.0 - self.delta >= 1.0:
            self.Q_k = self.score_max #np.inf
            return self.Q_k

        rank = int(np.ceil((1.0 - self.delta) * (S_cal_size + 1)))
        if rank >= S_cal_size + 1:
            self.Q_k = self.score_max #np.inf
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

        #if self.delta < -self.lr or self.delta > 1 + self.lr:
        #    raise ValueError(f"self.delta = {self.delta} out of [-lr, 1+lr]")
