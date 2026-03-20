import numpy as np
import sympy as sp
from qpsolvers import solve_qp
from dynsys.utils import *

class CtrlAffineSys:
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

        # Let subclass define symbolic system
        x_sym, f_sym, g_sym, Y_sym, a_sym = self.define_system_symbolic()
        self.xdim = x_sym.shape[0]
        self.udim = g_sym.shape[1]
        self.adim = Y_sym.shape[1]
        assert(f_sym.shape[0] == x_sym.shape[0])
        assert(g_sym.shape[0] == x_sym.shape[0])
        assert(Y_sym.shape[1] == a_sym.shape[0])

        # Treu uncertainty parameters
        self.a_true = np.copy(self.params["a_true"]) if "a_true" in self.params else np.zeros((self.adim,1))

        # Constant term for the adaptation laws
        self.eta_clf = self.params.get("eta_clf", 0.1)
        self.eta_cbf = self.params.get("eta_cbf", 0.1)
        self.eta_ccm = self.params.get("eta_ccm", 0.1)

        # Adaptation gain matrices
        self.Gamma_cbf = self.params.get("Gamma_cbf", np.eye(self.adim))
        self.Gamma_clf = self.params.get("Gamma_clf", np.eye(self.adim))
        self.Gamma_ccm = self.params.get("Gamma_ccm", np.eye(self.adim))

        # Define symbolic CLF and CBF 
        # NOTE: To be general and to handle both regular and adaptive CLF/CBF, 
        #       these functions depend on the uncertainty parameters. 
        clf_sym = self.define_clf_symbolic(x_sym, a_sym)
        cbf_sym = self.define_cbf_symbolic(x_sym, a_sym)

        if self.use_adaptive:
            # For projection-based adaptive controls
            self.a_hat_norm_max = self.params["a_hat_norm_max"]
            self.a_err_max = np.ones((self.adim, 1))/np.sqrt(self.adim) * self.a_hat_norm_max * 2 #TODO: check correctness
            self.epsilon = self.params.get("epsilon", 1e-3) # a small value for numerical stability of projection operator
        else:
            self.a_err_max = np.zeros((self.adim,1))

        # Convert symbolic functions into Python functions
        self.lambdify_symbolic_funcs(x_sym, f_sym, g_sym, Y_sym, a_sym, clf_sym, cbf_sym)

    def dynamics(self, x, u):
        return (self.f(x) + self.g(x) @ u + self.Y(x) @ self.a_true).ravel()
    
    def dynamics_nominal(self, x, u):
        return (self.f(x) + self.g(x) @ u).ravel()

    def ctrl_nominal(self, x):
        raise NotImplementedError("Nominal control not implemented.")

    def define_system_symbolic(self):
        raise NotImplementedError("System definition not implemented.")

    def define_clf_symbolic(self, x_sym, a_hat_clf=None):
        pass

    def define_cbf_symbolic(self, x_sym, a_hat_cbf=None):
        pass

    def lambdify_symbolic_funcs(self, x_sym, f_sym, g_sym, Y_sym, a_hat_sym, clf_sym=None, cbf_sym=None):
        if x_sym is None or f_sym is None or g_sym is None or Y_sym is None or a_hat_sym is None:
            raise ValueError("Symbolic x, f, and g must be provided.")

        self.xdim = len(x_sym)
        self.udim = g_sym.shape[1] if isinstance(g_sym, sp.Matrix) else np.shape(g_sym)[1]

        self.f = sp.lambdify([x_sym], f_sym, modules='numpy')
        self.g = sp.lambdify([x_sym], g_sym, modules='numpy')
        self.Y = sp.lambdify([x_sym], Y_sym, modules='numpy')

        # CBF
        if cbf_sym is not None:
            self.cbf = sp.lambdify([x_sym, a_hat_sym], cbf_sym, modules='numpy')

            dcbfdx = sp.simplify(sp.derive_by_array(cbf_sym, x_sym))
            dcbfdx = sp.Matrix(dcbfdx)  # Convert to Matrix for compatibility
            self.dcbfdx = sp.lambdify([x_sym, a_hat_sym], dcbfdx, modules='numpy')
            self.lf_cbf = sp.lambdify([x_sym, a_hat_sym], dcbfdx.T @ f_sym, modules='numpy')
            self.lg_cbf = sp.lambdify([x_sym, a_hat_sym], dcbfdx.T @ g_sym, modules='numpy')
            self.lY_cbf = sp.lambdify([x_sym, a_hat_sym], dcbfdx.T @ Y_sym, modules='numpy') if Y_sym is not None else None

            dcbfda = sp.simplify(sp.derive_by_array(cbf_sym, a_hat_sym))
            dcbfda = sp.Matrix(dcbfda)  # Convert to Matrix for compatibility
            self.dcbfda = sp.lambdify([x_sym, a_hat_sym], dcbfda, modules='numpy')

        # CLF
        if clf_sym is not None:
            self.clf = sp.lambdify([x_sym, a_hat_sym], clf_sym, modules='numpy')
            dclfdx = sp.simplify(sp.derive_by_array(clf_sym, x_sym))
            dclfdx = sp.Matrix(dclfdx)  # Convert to Matrix for compatibility
            self.dclfdx = sp.lambdify([x_sym, a_hat_sym], dclfdx, modules='numpy')
            self.lf_clf = sp.lambdify([x_sym, a_hat_sym], dclfdx.T @ f_sym, modules='numpy')
            self.lg_clf = sp.lambdify([x_sym, a_hat_sym], dclfdx.T @ g_sym, modules='numpy')
            self.lY_clf = sp.lambdify([x_sym, a_hat_sym], dclfdx.T @ Y_sym, modules='numpy') if Y_sym is not None else None
    
            dclfda = sp.simplify(sp.derive_by_array(clf_sym, a_hat_sym))
            dclfda = sp.Matrix(dclfda)
            self.dclfda = sp.lambdify([x_sym, a_hat_sym], dclfda, modules='numpy')            

    # Control laws
    def ctrl_craclf(self, x, a_hat_clf, u_ref, use_slack=True):
        """CRaCLF-QP Controller"""
        if self.clf is None:
            raise ValueError("aCLF not defined.")
        if u_ref is None:
            u_ref = np.zeros((self.udim, 1))
        if u_ref.shape[0] != self.udim:
            raise ValueError("u_ref shape mismatch.")

        V = self.clf(x, a_hat_clf)
        LfV = self.lf_clf(x, a_hat_clf)
        LgV = self.lg_clf(x, a_hat_clf)
        LYV = self.lY_clf(x, a_hat_clf)
        dclfdx = self.dclfdx(x, a_hat_clf)
        dclfda = self.dclfda(x, a_hat_clf)

        if self.use_cp:
            tightening =  self.cp_quantile * np.linalg.norm(dclfdx, 2)
        else:
            tightening = 0.0

        if use_slack:
            # Constraints : A[u; slack] <= b
            A = np.hstack([LgV, np.array([[-1]])])
            b = (
                -LfV
                -tightening
                -LYV @ (a_hat_clf + self.Gamma_clf @ dclfda) #TODO: check sign
                - self.params["clf"]["rate"] * V
            ).item()
        else:
            A = LgV
            b = (
                -LfV
                -LYV @ (a_hat_clf + self.Gamma_clf @ dclfda)
                - tightening
                - self.params["clf"]["rate"] * V
            ).item()

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
                [np.zeros((1, self.udim)), self.params["weight"]["slack"]],
            ])
            f = np.concatenate([-u_ref, [0]])

            # Enforce slack >= 0
            A = np.vstack([A, np.hstack([np.zeros((1, self.udim)), np.array([[-1.0]])])])
            b = np.vstack([b, np.array([[0.0]])])
        else:
            P = np.eye(self.udim)
            f = -u_ref

        qp_sol = solve_qp(P, f, A, b, solver='quadprog')
        if qp_sol is None:
            raise ValueError("solve_qp returns None")
        
        if use_slack:
            u_qp = qp_sol[: self.udim].reshape(self.udim, 1)
            slack = qp_sol[-1]
        else:
            u_qp = qp_sol.reshape(self.udim, 1)
            slack = 0.0

        return u_qp, slack
    
    def ctrl_cracbf(self, x, a_hat_cbf, u_ref):
        """CRaCBF QP Controller"""
        if self.cbf is None:
            raise ValueError("CBF not defined")
        if u_ref is None:
            u_ref = np.zeros((self.udim, 1))
        if u_ref.shape[0] != self.udim:
            raise ValueError("u_ref shape mismatch")

        h = self.cbf(x, a_hat_cbf)
        Lfh = self.lf_cbf(x, a_hat_cbf)
        Lgh = self.lg_cbf(x, a_hat_cbf)
        LYh = self.lY_cbf(x, a_hat_cbf)
        dcbfdx = self.dcbfdx(x, a_hat_cbf)
        
        if self.use_cp:
            tightening =  self.cp_quantile * np.linalg.norm(dcbfdx, 2)
        else:
            tightening = 0.0
        
        # A u <= b
        A = -Lgh
        b = (
            Lfh 
            + LYh @ a_hat_cbf #TODO: check sign
            - tightening
            + self.params["cbf"]["rate"] * (h - 0.5 * self.a_err_max.T @ np.linalg.inv(self.Gamma_cbf) @ self.a_err_max)
        )
        if "u_max" in self.params:
            A = np.vstack([A, np.eye(self.udim)])
            umax = self.params["u_max"]
            if np.isscalar(umax):
                b = np.vstack([b, umax * np.ones((self.udim, 1))])
            elif umax.shape == (self.udim, 1) or umax.shape == (self.udim,):
                b = np.vstack([b, umax.reshape(-1, 1)])
            else:
                raise ValueError("params['u_max'] should be either a scalar or an (udim, 1) array")
        if "u_min" in self.params:
            A = np.vstack([A, -np.eye(self.udim)])
            umin = self.params["u_min"]
            if np.isscalar(umin):
                b = np.vstack([b, -umin * np.ones((self.udim, 1))])
            elif umin.shape == (self.udim, 1) or umin.shape == (self.udim,):
                b = np.vstack([b, -umin.reshape(-1, 1)])
            else:
                raise ValueError("params['u_min'] should be either a scalar or an (udim, 1) array")
            
        # Solve QP: min_u 0.5 * u^T P u + f^T u  subject to A u <= b
        P = np.eye(self.udim)
        f = -u_ref
        qp_sol = solve_qp(P, f, A, b, solver='quadprog')
        if qp_sol is None:
            raise ValueError("solve_qp returns None")
        u_qp = qp_sol.reshape(-1,1)

        return u_qp
    
    def ctrl_craccm(self, x, a_hat_ccm, x_d, u_d, geodesic_solver, use_qpsolvers=False, use_slack=True, verify_geodesic=False):
        """CRaCCM control law"""
        # x: current state
        # x_d: desired state
        # u_d: nominal control input; u_d.shape = (self.udim, 1)

        # Compute geodesic
        self.calc_geodesic(geodesic_solver, x, x_d, a_hat_ccm, verify_geodesic) # update gamma, gamma_s, and E_rem

        gamma_s1_M_x = self.gamma_s[:, -1].reshape(1,-1) @ np.linalg.inv(self.W_fcn(x, a_hat_ccm))
        gamma_s0_M_d = self.gamma_s[:, 0].reshape(1,-1) @ np.linalg.inv(self.W_fcn(x_d, a_hat_ccm))
        
        if self.use_adaptive: 
            Y_x_a = self.Y(x) @ a_hat_ccm
            Y_d_a = self.Y(x_d) @ a_hat_ccm
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
            + self.params["ccm"]["rate"] * self.Erem).item()

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
        dclfda = self.dclfda(x, a_hat_clf)
        dclfdx = self.dclfdx(x, a_hat_clf)

        # Projection operator to enforce bounds on a_hat_clf
        #TODO: check sign
        a_hat_clf_dot = self.nu_clf(rho_clf) * (self.Gamma_clf
                    @ projection_operator(a_hat_clf, self.Y(x).T @ dclfdx, self.a_hat_norm_max, self.epsilon))
        
        rho_clf_dot = -self.nu_clf(rho_clf)/(self.dnu_drho_clf(rho_clf) * (V + self.eta_clf)).item() * (dclfda.T @ a_hat_clf_dot).item()

        return a_hat_clf_dot, rho_clf_dot

    def adaptation_cracbf(self, x, a_hat_cbf, rho_cbf):
        """CRaCBF adaptation law"""
        h = self.cbf(x, a_hat_cbf)
        dcbfda = self.dcbfda(x, a_hat_cbf)
        dcbfdx = self.dcbfdx(x, a_hat_cbf)

        # Projection operator to enforce bounds on a_hat_cbf
        #TODO: check sign
        a_hat_cbf_dot = self.nu_cbf(rho_cbf) * (self.Gamma_cbf
                    @ projection_operator(a_hat_cbf, -self.Y(x).T @ dcbfdx, self.a_hat_norm_max, self.epsilon))
        
        rho_cbf_dot = -self.nu_cbf(rho_cbf)/(self.dnu_drho_cbf(rho_cbf) * (h + self.eta_cbf)).item() * (dcbfda.T @ a_hat_cbf_dot).item()

        return a_hat_cbf_dot, rho_cbf_dot

    def adaptation_craccm(self, x, x_d, a_hat_ccm, rho_ccm, geodesic_solver):
        """CRaCCM adaptation law"""

        # Make sure self.gamma_s and self.gamma are already updated by calc_geodesic
        gamma_s1_M_x = self.gamma_s[:, -1].reshape(1,-1) @ np.linalg.inv(self.W_fcn(x, a_hat_ccm))
        gamma_s0_M_d = self.gamma_s[:, 0].reshape(1,-1) @ np.linalg.inv(self.W_fcn(x_d, a_hat_ccm))

        # TODO: check correctness
        dErem_dai = np.zeros(self.adim)
        for i in range(self.adim):
            for k in range(geodesic_solver.N + 1):
                gk = self.gamma[:, k]
                gsk = self.gamma_s[:, k]
                dW_dai = self.dW_dai_fcn(i,gk,a_hat_ccm)
                M = np.linalg.inv(self.W_fcn(gk,a_hat_ccm)) # TODO: check correctness
                dM_dai = -M @ dW_dai @ M # TODO: check correctness
                dErem_dai[i] += (gsk.T @ dM_dai @ gsk) * geodesic_solver.w_cheby[k]

        a_hat_ccm_dot = np.linalg.inv(self.Gamma_ccm) @ projection_operator(a_hat_ccm, 
                                            self.nu_ccm(rho_ccm) * self.Y(x).T @ gamma_s1_M_x.T, 
                                            self.a_hat_norm_max, self.epsilon)
        #a_hat_dot = self.nu_ccm(rho_ccm) * np.linalg.inv(self.Gamma_ccm) @ self.Y(x).T @ self.gamma_s1_M_x.T
        
        c1 = (2 * gamma_s0_M_d @ self.Y(x_d) @ a_hat_ccm).item()
        c2 = (dErem_dai @ a_hat_ccm_dot).item()
        # Printing for debugging
        #print("dErem_dai = ", dErem_dai, "; a_hat_dot = ", a_hat_ccm_dot)
        #print("c1 = ", c1, "; c2 = ", c2)
        rho_ccm_dot = -(self.nu_ccm(rho_ccm) * (c1 + c2)) / (self.dnu_drho_ccm(rho_ccm) * (self.Erem + self.eta_ccm))

        return a_hat_ccm_dot, rho_ccm_dot

    # Scaling functions for unmatched adaptive controls
    def nu_clf(self, rho_clf):
        nu = np.arctan(rho_clf)/np.pi + 1.0
        return nu
    
    def dnu_drho_clf(self, rho_clf):
        dnu_drho = 1/(1+(rho_clf)**2)/np.pi
        return max(dnu_drho, 1e-20)

    def nu_cbf(self, rho_cbf):
        nu = np.arctan(rho_cbf)/np.pi + 1.0
        return nu
    
    def dnu_drho_cbf(self, rho_cbf):
        dnu_drho = 1/(1+(rho_cbf)**2)/np.pi
        return dnu_drho 
        #return max(dnu_drho, 1e-20)
    
    def nu_ccm(self, rho_ccm):
        nu = 0.9 * np.exp(rho_ccm) + 0.1 # must be bounded away from zero
        #nu = np.arctan(rho_ccm)/np.pi + 1.0
        return nu
    
    def dnu_drho_ccm(self, rho_ccm):
        dnu_drho = 0.9 * np.exp(rho_ccm)
        #dnu_drho = 1/(1+(rho_ccm)**2)/np.pi
        return max(dnu_drho, 1e-20)
        