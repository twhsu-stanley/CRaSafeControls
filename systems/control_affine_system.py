import numpy as np
from qpsolvers import solve_qp

class ControlAffineSystem:
    def __init__(self, params=None):
        # System parameters
        if params is None:
            params = {}
        elif not isinstance(params, dict):
            raise TypeError("Parameters must be a dictionary.")
        self.params = params

        self.use_cp = self.params.get("use_cp", False)
        self.cp_quantile = self.params.get("cp_quantile", 0.0)
        self.use_adaptive = self.params.get("use_adaptive", False)
        self.weight_slack = self.params.get("weight_slack", 100)
        self.dt = self.params.get("dt")

        # Subclasses provide numerical dynamics and explicit dimensions.
        for name in ("xdim", "udim", "adim"):
            value = self.params.get(name, getattr(self, name, None))
            if value is None or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))

        # True uncertainty parameters
        self.a_true = np.copy(self.params["a_true"]) if "a_true" in self.params else np.zeros((self.adim,1))

        # Rates
        self.clf_rate = self.params.get("clf_rate", None)
        self.cbf_rate = self.params.get("cbf_rate", None)
        self.ccm_rate = self.params.get("ccm_rate", None)

        # Constant term for the adaptation laws
        self.eta_clf = float(self.params.get("eta_clf", 0.1))
        self.eta_cbf = float(self.params.get("eta_cbf", 0.1))
        self.eta_ccm = float(self.params.get("eta_ccm", 0.1))

        # Adaptation gain matrices
        self.Gamma_cbf = self.params.get("Gamma_cbf", None)
        self.Gamma_clf = self.params.get("Gamma_clf", None)
        self.Gamma_ccm = self.params.get("Gamma_ccm", None)
        
        self.safe_set_tightening = 0.0

        self.a_ub = np.asarray(self.params["a_ub"], dtype=float).reshape(-1)
        self.a_lb = np.asarray(self.params["a_lb"], dtype=float).reshape(-1)
        if self.a_ub.shape != self.a_lb.shape:
            raise ValueError("a_ub and a_lb must have the same shape")
        if np.any(self.a_lb > self.a_ub):
            raise ValueError("a_lb must be less than or equal to a_ub")
        if self.a_lb.shape[0] != self.adim or self.a_ub.shape[0] != self.adim:
            raise ValueError(
                f"Dimension mismatch: a has length {self.adim}, but a_lb has "
                f"length {self.a_lb.shape[0]} and a_ub has length "
                f"{self.a_ub.shape[0]}"
            )
        
        self.a_center = 0.5 * (self.a_ub + self.a_lb) # center of the convex set where a_hat belongs to
        self.a_hat_norm_max = float(self.params["a_hat_norm_max"])
        if not np.isfinite(self.a_hat_norm_max) or self.a_hat_norm_max <= 0.0:
            raise ValueError("a_hat_norm_max must be finite and positive")

        if self.use_adaptive:
            # For projection-based adaptive controls
            a_err_norm_max = self.a_hat_norm_max + 0.5 * np.linalg.norm(self.a_ub - self.a_lb, ord=2)
            self.epsilon = float(self.params.get("epsilon", 1e-3))
            if not 0.0 < self.epsilon < self.a_hat_norm_max:
                raise ValueError(
                    "epsilon must satisfy 0 < epsilon < a_hat_norm_max"
                )

            if self.Gamma_cbf is not None:
                # NOTE: self.a_err_max is only used by the CRaCBF
                # NOTE: assuming Gamma_cbf is positive definite and symmetric
                # Find min_a a^T @ inv(Gamma_cbf) @ a subject to ||a|| == a_err_norm_max
                eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(self.Gamma_cbf))
                #self.a_err_max = a_err_norm_max * eigvecs[:,np.argmin(eigvals)]
                #self.a_err_max = a_err_norm_max * (self.a_ub - self.a_lb)/np.linalg.norm(self.a_ub - self.a_lb, ord=2)
                self.safe_set_tightening = (a_err_norm_max ** 2) * np.max(eigvals)

    def dynamics(self, x, u):
        raise NotImplementedError("Dynamics are not implemented for this system")

    def f(self, x):
        """Evaluate the nominal drift as an ``xdim``-vector."""
        raise NotImplementedError("f is not implemented for this system")

    def g(self, x):
        """Evaluate the control matrix with shape ``(xdim, udim)``."""
        raise NotImplementedError("g is not implemented for this system")

    def Y(self, x):
        """Evaluate the currently installed uncertainty regressor."""
        raise NotImplementedError("Y is not implemented for this system")

    def _validate_Y_shape(self, Yx):
        """Validate and return a numerical uncertainty-regressor matrix."""
        Yx = np.asarray(Yx, dtype=float)
        expected_shape = (self.xdim, self.adim)
        if Yx.shape != expected_shape:
            raise ValueError(
                f"Y(x) must have shape {expected_shape}, got {Yx.shape}"
            )
        return Yx

    def Y_theta(self, x, theta):
        """Evaluate a candidate representation Y_theta(x).

        Representation-learning subclasses must override this method. The
        dependence on ``theta`` may be arbitrary, including a neural network,
        but the result must have shape ``(xdim, adim)`` so the uncertainty
        remains linear in the interval parameter ``a``. A numerical neural
        subclass must also override ``Y`` and ``set_representation`` so the
        controller uses the installed weights.
        """
        raise NotImplementedError("Y_theta is not implemented for this system")

    def representation_loss_gradient(self, x, theta, a, w):
        """Return grad_theta ||Y_theta(x) @ a - w||_2**2.

        A neural representation can implement this hook with backpropagation
        and return its weights as one packed NumPy array.
        """
        raise NotImplementedError(
            "Representation-loss gradient is not implemented for this system"
        )

    def set_representation(self, theta):
        """Install representation parameters used by the controller.

        Certificate functions should accept the installed parameters at
        runtime. Neural subclasses can instead update their model state here.
        """
        raise NotImplementedError("Representation updates are not implemented")

    def clf(self, x, a):
        raise NotImplementedError("CLF is not implemented for this system")

    def dclfdx(self, x, a):
        raise NotImplementedError("CLF state gradient is not implemented")

    def dclfda(self, x, a):
        raise NotImplementedError("CLF parameter gradient is not implemented")

    def cbf(self, x, a):
        raise NotImplementedError("CBF is not implemented for this system")

    def dcbfdx(self, x, a):
        raise NotImplementedError("CBF state gradient is not implemented")

    def dcbfda(self, x, a):
        raise NotImplementedError("CBF parameter gradient is not implemented")

    def lf_clf(self, x, a):
        gradient = np.asarray(self.dclfdx(x, a), dtype=float).reshape(1, self.xdim)
        return gradient @ self.f(x)

    def lg_clf(self, x, a):
        gradient = np.asarray(self.dclfdx(x, a), dtype=float).reshape(1, self.xdim)
        return gradient @ self.g(x)

    def lY_clf(self, x, a):
        gradient = np.asarray(self.dclfdx(x, a), dtype=float).reshape(1, self.xdim)
        return gradient @ self._validate_Y_shape(self.Y(x))

    def lf_cbf(self, x, a):
        gradient = np.asarray(self.dcbfdx(x, a), dtype=float).reshape(1, self.xdim)
        return gradient @ self.f(x)

    def lg_cbf(self, x, a):
        gradient = np.asarray(self.dcbfdx(x, a), dtype=float).reshape(1, self.xdim)
        return gradient @ self.g(x)

    def lY_cbf(self, x, a):
        gradient = np.asarray(self.dcbfdx(x, a), dtype=float).reshape(1, self.xdim)
        return gradient @ self._validate_Y_shape(self.Y(x))
    
    def dynamics_nominal(self, x, u):
        u = np.asarray(u, dtype=float).reshape(self.udim)
        return self.f(x) + self.g(x) @ u

    def ctrl_nominal(self, x):
        raise NotImplementedError("Nominal control not implemented.")

    # Control laws
    def ctrl_craclf(self, x, a_hat_clf, u_ref, use_slack=True):
        """CRaCLF-QP Controller"""  

        #NOTE: using reshape to enforce correct shape
        V = self.clf(x, a_hat_clf)
        LfV = self.lf_clf(x, a_hat_clf)
        LgV = self.lg_clf(x, a_hat_clf).reshape(1,self.udim)
        LYV = self.lY_clf(x, a_hat_clf).reshape(1,self.adim)
        dclfdx = self.dclfdx(x, a_hat_clf).reshape(self.xdim,1)

        if self.use_cp:
            tightening = self.cp_quantile * np.linalg.norm(dclfdx, 2)
        else:
            tightening = 0.0

        if use_slack:
            # Constraints: A[u; slack] <= b
            A = np.hstack([LgV, np.array([[-1]])])
        else:
            A = LgV

        b = (-LfV
            -LYV @ a_hat_clf
            -tightening
            -self.clf_rate * V
        )

        if "u_max" in self.params:
            A = np.vstack([A, np.hstack([np.eye(self.udim), np.zeros((self.udim, 1))])]) if use_slack else np.vstack([A, np.eye(self.udim)])
            umax = self.params["u_max"]
            if np.isscalar(umax):
                b = np.vstack([b, umax * np.ones((self.udim, 1))])
            elif umax.shape == (self.udim, 1) or umax.shape == (self.udim,):
                b = np.vstack([b, umax.reshape(-1, 1)])
            else:
                raise ValueError("params['u_max'] should be either a scalar or an (udim, 1) array")

        if "u_min" in self.params:
            A = np.vstack([A, np.hstack([-np.eye(self.udim), np.zeros((self.udim, 1))])]) if use_slack else np.vstack([A, -np.eye(self.udim)])
            umin = self.params["u_min"]
            if np.isscalar(umin):
                b = np.vstack([b, -umin * np.ones((self.udim, 1))])
            elif umin.shape == (self.udim, 1) or umin.shape == (self.udim,):
                b = np.vstack([b, -umin.reshape(-1, 1)])
            else:
                raise ValueError("params['u_min'] should be either a scalar or an (udim, 1) array")

        # Solve QP: min_u 0.5 u^T P u + f^T u subject to A u <= b
        if use_slack:
            P = np.block([
                [np.eye(self.udim), np.zeros((self.udim, 1))],
                [np.zeros((1, self.udim)), self.weight_slack],
            ])
            f = np.concatenate([-u_ref, [0]])

            # Enforce slack >= 0
            A = np.vstack([A, np.hstack([np.zeros((1, self.udim)), np.array([[-1.0]])])])
            b = np.vstack([b, np.array([[0.0]])])
        else:
            P = np.eye(self.udim)
            f = -u_ref

        qp_sol = solve_qp(P=P, q=f, G=A, h=b, solver='quadprog')
        if qp_sol is None:
            raise ValueError("solve_qp returns None")
        
        if use_slack:
            u_qp = qp_sol[: self.udim].reshape(self.udim, 1)
            slack = qp_sol[-1]
        else:
            u_qp = qp_sol.reshape(self.udim, 1)
            slack = 0.0

        return u_qp, slack
    
    def ctrl_cracbf(self, x, a_hat_cbf, u_ref, rho_cbf):
        """CRaCBF QP Controller"""

        x = np.asarray(x, dtype=float).reshape(self.xdim)
        a_hat_cbf = np.asarray(a_hat_cbf, dtype=float).reshape(self.adim)
        u_ref = np.asarray(u_ref, dtype=float).reshape(self.udim)
        rho_cbf = float(np.asarray(rho_cbf, dtype=float).item())

        h = self.cbf(x, a_hat_cbf)
        Lfh = self.lf_cbf(x, a_hat_cbf)
        Lgh = self.lg_cbf(x, a_hat_cbf).reshape(1,self.udim)
        LYh = self.lY_cbf(x, a_hat_cbf).reshape(1,self.adim)
        dcbfdx = self.dcbfdx(x, a_hat_cbf).reshape(self.xdim,1)
        
        dcbfda = self.dcbfda(x, a_hat_cbf).reshape(self.adim,1)

        if self.use_cp:
            tightening =  self.cp_quantile * np.linalg.norm(dcbfdx, 2)
        else:
            tightening = 0.0

        if self.use_adaptive:
            a_hat_cbf_dot = ControlAffineSystem.projection_operator(a_hat_cbf, 
                                              -self.nu_cbf(rho_cbf) * self.Gamma_cbf @ self.Y(x).T @ dcbfdx,
                                              self.a_center,
                                              self.a_hat_norm_max,
                                              self.epsilon,
                                              self.Gamma_cbf)
        
            correction_term = -self.eta_cbf/(h + self.eta_cbf).item() * (dcbfda.T @ a_hat_cbf_dot).item()
        else:
            correction_term = 0.0

        # A u <= b
        A = -Lgh
        b = (
            Lfh
            + LYh @ a_hat_cbf
            - tightening
            + float(self.cbf_rate) * (h - 0.5 / self.nu_cbf(rho_cbf) * self.safe_set_tightening)
            - correction_term
        )
        if "u_max" in self.params:
            A = np.vstack([A, np.eye(self.udim)])
            umax = self.params["u_max"]
            if np.isscalar(umax):
                b = np.hstack([b, umax * np.ones(self.udim)])
            elif np.asarray(umax).shape in {
                (self.udim, 1),
                (self.udim,),
            }:
                b = np.hstack([b, np.asarray(umax).reshape(-1)])
            else:
                raise ValueError("params['u_max'] should be either a scalar or an (udim, 1) array")
        if "u_min" in self.params:
            A = np.vstack([A, -np.eye(self.udim)])
            umin = self.params["u_min"]
            if np.isscalar(umin):
                b = np.hstack([b, -umin * np.ones(self.udim)])
            elif np.asarray(umin).shape in {
                (self.udim, 1),
                (self.udim,),
            }:
                b = np.hstack([b, -np.asarray(umin).reshape(-1)])
            else:
                raise ValueError("params['u_min'] should be either a scalar or an (udim, 1) array")

        # Solve QP: min_u 0.5 * u^T P u + f^T u  subject to A u <= b
        P = np.eye(self.udim)
        f = -u_ref
        A = np.asarray(A, dtype=float).reshape(-1, self.udim)
        b = np.asarray(b, dtype=float).reshape(-1)
        qp_sol = solve_qp(P=P, q=f, G=A, h=b, solver='quadprog')
        if qp_sol is None:
            raise ValueError("solve_qp returns None")
        u_qp = qp_sol.reshape(-1,1)

        return u_qp
    
    def ctrl_craccm(self, x, a_hat_ccm, x_d, u_d, geodesic_solver, use_qpsolvers=False, use_slack=True, verify_geodesic=False):
        """CRaCCM control law"""
        # x: current state
        # x_d: desired state
        # u_d: nominal control input; u_d.shape = (self.udim,) or (self.udim, 1)

        u_d = u_d.reshape(self.udim, 1) # ensure correct shape

        # Compute geodesic
        self.calc_geodesic(geodesic_solver, x, x_d, a_hat_ccm, verify_geodesic) # update gamma, gamma_s, and E_rem

        gamma_s1_M_x = self.gamma_s[:, -1].reshape(1,-1) @ np.linalg.inv(self.W_fcn(x, a_hat_ccm))
        gamma_s0_M_d = self.gamma_s[:, 0].reshape(1,-1) @ np.linalg.inv(self.W_fcn(x_d, a_hat_ccm))
        
        if self.use_adaptive: 
            Y_x_a = (self.Y(x) @ a_hat_ccm).reshape(-1,1)
            Y_d_a = (self.Y(x_d) @ a_hat_ccm).reshape(-1,1)
        else:
            Y_x_a = 0.0
            Y_d_a = 0.0

        if self.use_cp:
            #Theta = np.linalg.cholesky(M_x)
            #sigma_max = np.max(np.linalg.svd(Theta, compute_uv=False))  # maximum singular value
            #tightening = sigma_max * self.cp_quantile * np.sqrt(self.Erem)
            tightening = np.linalg.norm(gamma_s1_M_x, 2) * self.cp_quantile
        else:
            tightening = 0.0
        
        A = gamma_s1_M_x @ self.g(x)
        B = (gamma_s1_M_x @ (self.f(x) + self.g(x) @ u_d + Y_x_a)
            - gamma_s0_M_d @ (self.f(x_d) + self.g(x_d) @ u_d + Y_d_a)
            + self.ccm_rate * self.Erem).item()

        if use_qpsolvers is True: 
            if use_slack:
                P = np.block([[np.eye(self.udim),        np.zeros((self.udim, 1))],
                              [np.zeros((1, self.udim)), np.array([[self.weight_slack]])],
                ])
                q = np.zeros(self.udim + 1)
                G = np.vstack([np.hstack([A, np.array([[-1.0]])]),
                               np.hstack([np.zeros((1, self.udim)), np.array([[-1.0]])]),
                ])
                h = np.array([-(B + tightening), 0.0])
                qp_sol = solve_qp(P, q, G, h, solver = 'quadprog')
                u_qp = qp_sol[0:self.udim].reshape(-1,1)
                slack = qp_sol[-1]
            else:
                # no slack
                P = np.eye(self.udim)
                q = np.zeros(self.udim)
                G = A
                h = np.array([-(B + tightening)])
                qp_sol = solve_qp(P, q, G, h, solver = 'quadprog')
                u_qp = qp_sol.reshape(-1,1)
                slack = 0.0
            
        else:
            # Analytic solution
            if use_slack:
                denom = (1 + self.weight_slack * A @ A.T).item()
                #tightening = (tightening * denom + B)/(denom-1)
                A_norm = np.linalg.norm(A, 2)
                if A_norm > 1e-5:
                    if B + tightening <= 0:
                        u_qp = np.zeros((self.udim,1))
                        slack = 0.0
                    else:
                        u_qp = (-self.weight_slack * (B + tightening) * A.T) / denom
                        slack = (B + tightening) / denom
                else:
                    print(f"Loss of control authority: norm(A)={A_norm:2E}")
                    if B + tightening <= 0:
                        u_qp = np.zeros((self.udim,1))
                        slack = 0.0
                    else:
                        u_qp = np.zeros((self.udim,1))
                        slack = B + tightening
            else:
                #TODO: complete this
                raise ValueError(f"Analytic QP solution for CRaCCM with no slack is not supported")

        uc = u_d + u_qp

        # Pint uncertainty terms for debugging
        #U1 = -((a_hat_ccm - self.a_true).T @ self.Y(x).T @ gamma_s1_M_x.T).item() # term to be cancelled by adaptive a_dot
        #U2 = (gamma_s0_M_d @ self.Y(x_d) @ a_hat_ccm).item() # term to be cancelled by adaptive rho_dot
        #print("U1=", U1, "; U2=", U2)

        return uc, slack

    # Solve for geodesics for CCM-based controllers
    def calc_geodesic(self, solver, x, x_d, a_hat_ccm=None, verify_geodesic=False):
        
        # Initialize optimization variables and constraints internally
        c0, beq = solver.initialize_conditions(x_d, x)
        
        # Solve the geodesic optimization problem
        gamma, gamma_s, Erem = solver.solve_geodesic(c0, beq, a_hat_ccm) # TODO: check if a_hat_ccm is None?
        self.gamma = gamma
        self.gamma_s = gamma_s
        self.Erem = Erem.item()
        
        # Verify whether the curve found is really a geodesic
        if verify_geodesic and self.Erem > 1e-3:
            error = 0
            for k in range(solver.N + 1):
                gk = gamma[:, k]
                gsk = gamma_s[:, k]
                M = np.linalg.inv(solver.W_fcn(gk,a_hat_ccm))
                error += ((gsk.T @ M @ gsk - self.Erem)**2) * solver.w_cheby[k]
            error = np.sqrt(error)/self.Erem
            if error > 1e-5:
                print(f"geodesic error={error:2E} exceeds threshold = 1e-5")
                #if error > 1e-2:
                #    raise ValueError(f"geodesic error={error:2E} exceeds threshold = 1e-2")

    # Adaptation laws
    def adaptation_craclf(self, x, a_hat_clf, rho_clf):
        """CRaCLF adaptation law"""
        V = self.clf(x, a_hat_clf)
        #NOTE: using reshape to enforce correct shape
        dclfda = self.dclfda(x, a_hat_clf).reshape(self.adim,1)
        dclfdx = self.dclfdx(x, a_hat_clf).reshape(self.xdim,1)
        
        a_hat_clf_dot =  ControlAffineSystem.projection_operator(a_hat_clf, 
                                              self.nu_clf(rho_clf) * self.Gamma_clf @ self.Y(x).T @ dclfdx,
                                              self.a_center,
                                              self.a_hat_norm_max,
                                              self.epsilon,
                                              self.Gamma_clf)
        
        rho_clf_dot = -self.nu_clf(rho_clf)/(self.dnu_drho_clf(rho_clf) * (V + self.eta_clf)).item() * (dclfda.T @ a_hat_clf_dot).item()

        return a_hat_clf_dot.ravel(), rho_clf_dot

    def adaptation_cracbf(self, x, a_hat_cbf, rho_cbf):
        """CRaCBF adaptation law"""
        h = self.cbf(x, a_hat_cbf)
        #NOTE: using reshape to enforce correct shape
        dcbfda = self.dcbfda(x, a_hat_cbf).reshape(self.adim,1)
        dcbfdx = self.dcbfdx(x, a_hat_cbf).reshape(self.xdim,1)

        # Projection operator to enforce bounds on a_hat_cbf
        a_hat_cbf_dot = ControlAffineSystem.projection_operator(a_hat_cbf, 
                                              -self.nu_cbf(rho_cbf) * self.Gamma_cbf @ self.Y(x).T @ dcbfdx,
                                              self.a_center,
                                              self.a_hat_norm_max,
                                              self.epsilon,
                                              self.Gamma_cbf)
        
        rho_cbf_dot = -self.nu_cbf(rho_cbf)/(self.dnu_drho_cbf(rho_cbf) * (h + self.eta_cbf)).item() * (dcbfda.T @ a_hat_cbf_dot).item()

        return a_hat_cbf_dot.ravel(), rho_cbf_dot

    def adaptation_craccm(self, x, x_d, a_hat_ccm, rho_ccm, geodesic_solver):
        """CRaCCM adaptation law"""

        # Make sure self.gamma_s and self.gamma are already updated by calc_geodesic
        gamma_s1_M_x = self.gamma_s[:, -1].reshape(1,-1) @ np.linalg.inv(self.W_fcn(x, a_hat_ccm))
        gamma_s0_M_d = self.gamma_s[:, 0].reshape(1,-1) @ np.linalg.inv(self.W_fcn(x_d, a_hat_ccm))

        dErem_dai = np.zeros(self.adim)
        for i in range(self.adim):
            for k in range(geodesic_solver.N + 1):
                gk = self.gamma[:, k]
                gsk = self.gamma_s[:, k]
                dW_dai = self.dW_dai_fcn(i,gk,a_hat_ccm)
                M = np.linalg.inv(self.W_fcn(gk,a_hat_ccm)) # TODO: check correctness
                dM_dai = -M @ dW_dai @ M # TODO: check correctness
                dErem_dai[i] += (gsk.T @ dM_dai @ gsk) * geodesic_solver.w_cheby[k]

        a_hat_ccm_dot = ControlAffineSystem.projection_operator(a_hat_ccm, 
                                              self.nu_ccm(rho_ccm) * self.Gamma_ccm @ self.Y(x).T @ gamma_s1_M_x.T,
                                              self.a_center,
                                              self.a_hat_norm_max,
                                              self.epsilon,
                                              self.Gamma_ccm)
        #a_hat_dot = self.nu_ccm(rho_ccm) * self.Gamma_ccm @ self.Y(x).T @ gamma_s1_M_x.T
        
        c1 = (2 * gamma_s0_M_d @ self.Y(x_d) @ a_hat_ccm).item()
        c2 = (dErem_dai @ a_hat_ccm_dot).item()
        # Printing for debugging
        #print("dErem_dai = ", dErem_dai, "; a_hat_dot = ", a_hat_ccm_dot)
        #print("c1 = ", c1, "; c2 = ", c2)
        rho_ccm_dot = -(self.nu_ccm(rho_ccm) * (c1 + c2)) / (self.dnu_drho_ccm(rho_ccm) * (self.Erem + self.eta_ccm))

        return a_hat_ccm_dot.ravel(), rho_ccm_dot

    # Scaling functions for unmatched adaptive controls
    @staticmethod
    def nu_clf(rho_clf):
        # Smooth, strictly increasing, and bounded in (1, 2), as required by
        # Definition 1 of the paper.
        nu = np.arctan(rho_clf)/np.pi + 1.5
        return nu
    
    @staticmethod
    def dnu_drho_clf(rho_clf):
        dnu_drho = 1/(1+(rho_clf)**2)/np.pi
        return max(dnu_drho, 1e-20)

    @staticmethod
    def nu_cbf(rho_cbf):
        nu = np.arctan(rho_cbf)/np.pi + 1.5
        return nu
    
    @staticmethod
    def dnu_drho_cbf(rho_cbf):
        dnu_drho = 1/(1+(rho_cbf)**2)/np.pi
        return max(dnu_drho, 1e-20)
    
    @staticmethod
    def nu_ccm(rho_ccm):
        nu = np.arctan(rho_ccm)/np.pi + 1.5
        return nu
    
    @staticmethod
    def dnu_drho_ccm(rho_ccm):
        dnu_drho = 1/(1+(rho_ccm)**2)/np.pi
        return max(dnu_drho, 1e-20)
    
    # Functions for projection-based adaptive controls
    @staticmethod
    def phi(a_hat, a_center, a_hat_norm_max, epsilon):
        """Compute the barrier function φ(â)"""
        denominator = 2 * epsilon * a_hat_norm_max - epsilon**2
        return (
            np.linalg.norm(a_hat - a_center, ord=2)**2
            - (a_hat_norm_max - epsilon)**2
        ) / denominator

    @staticmethod
    def grad_phi(a_hat, a_center, a_hat_norm_max, epsilon):
        """Compute the gradient ∇φ(â)"""
        denominator = 2 * epsilon * a_hat_norm_max - epsilon**2
        return (2 * (a_hat - a_center)).reshape(-1,1) / denominator

    @staticmethod
    def projection_operator(a_hat, y, a_center, a_hat_norm_max, epsilon, Gamma=None):
        """
        Implements the adaptive control projection operator:
        Proj(a_hat, y, φ)

        Parameters:
        - a_hat: current parameter estimate (np.ndarray)
        - y: nominal adaptation signal (np.ndarray)x
        - a_center: center of the box-linit set where the true parameter belongs to
        - a_hat_norm_max: upper bound on ||a_hat - a_center||
        - epsilon: small positive scalar (for soft boundary enforcement)
        - Gamma: symmetric positive-definite adaptation gain matrix

        Returns:
        - projected update (np.ndarray)
        """
        if not 0.0 < epsilon < a_hat_norm_max:
            raise ValueError("epsilon must satisfy 0 < epsilon < a_hat_norm_max")

        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if Gamma is None:
            Gamma = np.eye(y.shape[0])
        Gamma = np.asarray(Gamma, dtype=float)

        phi_val = ControlAffineSystem.phi(a_hat, a_center, a_hat_norm_max, epsilon)
        grad_phi_val = ControlAffineSystem.grad_phi(a_hat, a_center, a_hat_norm_max, epsilon)

        if phi_val > 0 and (y.T @ grad_phi_val).item() > 0:
            denominator = (grad_phi_val.T @ Gamma @ grad_phi_val).item()
            correction = (
                Gamma @ grad_phi_val
                * ((grad_phi_val.T @ y).item() / denominator)
                * phi_val
            )
            return y - correction
        else:
            return y
