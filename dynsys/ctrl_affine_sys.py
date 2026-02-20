import numpy as np
import sympy as sp
import cvxpy
from dynsys.geodesic_solver import GeodesicSolver
from dynsys.utils import *

class CtrlAffineSys:
    def __init__(self, params=None):
        # System parameters
        if params is None:
            params = {}
        elif not isinstance(params, dict):
            raise TypeError("Parameters must be a dictionary.")
        self.params = params

        self.use_cp = self.params.get("use_cp", 0)
        self.cp_quantile = self.params.get("cp_quantile", 0)
        self.use_adaptive = self.params.get("use_adaptive", 0)

        # Let subclass define symbolic system
        x_sym, f_sym, g_sym, Y_sym, a_sym = self.define_system_symbolic()
        self.xdim = x_sym.shape[0]
        self.udim = g_sym.shape[1]
        self.adim = Y_sym.shape[1]
        assert(f_sym.shape[0] == x_sym.shape[0])
        assert(g_sym.shape[0] == x_sym.shape[0])
        assert(Y_sym.shape[1] == a_sym.shape[0])

        # Uncertainty parameters
        self.a_true = np.copy(self.params["a_true"]) if "a_true" in self.params else np.zeros((self.adim,1))
        self.a_hat_b = np.copy(self.params["a_0"])  if "a_0" in self.params else np.zeros((self.adim,1)) # Initial guess for a_hat_b
        self.a_hat_L = np.copy(self.params["a_0"])  if "a_0" in self.params else np.zeros((self.adim,1)) # Initial guess for a_hat_L
        self.a_hat_ccm = np.copy(self.params["a_0"])  if "a_0" in self.params else np.zeros((self.adim,1)) # Initial guess for a_hat_ccm
        # WARNING: a_hat_b, a_hat_L, and a_hat_ccm should be initialized by copying, otherwise they will be references to the same array.

        clf_sym = self.define_clf_symbolic(x_sym)

        aclf_sym = None

        cbf_sym = self.define_cbf_symbolic(x_sym)

        acbf_sym = None

        if self.use_adaptive:
            # For uncertainty parameter estimates
            self.a_hat_norm_max = self.params["a_hat_norm_max"]
            self.a_err_max = np.ones((self.params["a_0"].shape[0], 1)) * self.a_hat_norm_max * 2 
            #TODO: check correctness: Are we considering box limits?
            
            # A small value for numerical stability of projection operator
            self.epsilon = self.params.get("epsilon", 1e-3)

            # For the scaling functions of the unmatched adapation laws
            self.eta_clf = self.params.get("eta_clf", 0.1)
            self.rho_clf = 0.0
            self.eta_cbf = self.params.get("eta_cbf", 0.1)
            self.rho_cbf = 0.0
            self.eta_ccm = self.params.get("eta_ccm", 0.1)
            self.rho_ccm = 0.0

            # Adaptive gain matrices
            self.Gamma_b = self.params.get("Gamma_b", np.eye(self.adim))
            self.Gamma_L = self.params.get("Gamma_L", np.eye(self.adim))
            self.Gamma_ccm = self.params.get("Gamma_ccm", np.eye(self.adim))

            aclf_sym = self.define_aclf_symbolic(x_sym, a_sym)
            acbf_sym = self.define_acbf_symbolic(x_sym, a_sym)

        # Convert symbolic functions into Python functions
        self.lambdify_symbolic_funcs(x_sym, f_sym, g_sym, Y_sym, a_sym, clf_sym, cbf_sym, aclf_sym, acbf_sym)

    def dynamics(self, x, u):
        return (self.f(x) + self.g(x) @ u + self.Y(x) @ self.a_true).ravel()

    def ctrl_nominal(self, x):
        raise NotImplementedError("Nominal control not implemented.")

    def define_system_symbolic(self):
        raise NotImplementedError("System definition not implemented.")

    def define_clf_symbolic(self, x_sym):
        pass

    def define_cbf_symbolic(self, x_sym):
        pass

    def define_aclf_symbolic(self, x_sym, a_hat_L=None):
        pass

    def define_acbf_symbolic(self, x_sym, a_hat_b=None):
        pass

    def lambdify_symbolic_funcs(self, x_sym, f_sym, g_sym, Y_sym, a_hat_sym,
                                clf_sym=None, cbf_sym=None, aclf_sym=None, acbf_sym=None):
        if x_sym is None or f_sym is None or g_sym is None:
            raise ValueError("Symbolic x, f, and g must be provided.")

        self.xdim = len(x_sym)
        self.udim = g_sym.shape[1] if isinstance(g_sym, sp.Matrix) else np.shape(g_sym)[1]

        self.f = sp.lambdify([x_sym], f_sym, modules='numpy')
        self.g = sp.lambdify([x_sym], g_sym, modules='numpy')
        self.Y = sp.lambdify([x_sym], Y_sym, modules='numpy')

        if cbf_sym is not None:
            dcbfdx = sp.simplify(sp.derive_by_array(cbf_sym, x_sym))
            dcbfdx = sp.Matrix(dcbfdx)  # Convert to Matrix for compatibility
            self.dcbfdx = sp.lambdify([x_sym], dcbfdx, modules='numpy')
            self.cbf = sp.lambdify([x_sym], cbf_sym, modules='numpy')
            self.lf_cbf = sp.lambdify([x_sym], dcbfdx.T @ f_sym, modules='numpy')
            self.lg_cbf = sp.lambdify([x_sym], dcbfdx.T @ g_sym, modules='numpy')
            self.ddcbfdx = sp.lambdify([x_sym], sp.hessian(cbf_sym, x_sym), modules='numpy')

        if clf_sym is not None:
            dclfdx = sp.simplify(sp.derive_by_array(clf_sym, x_sym))
            dclfdx = sp.Matrix(dclfdx)  # Convert to Matrix for compatibility
            self.dclfdx = sp.lambdify([x_sym], dclfdx, modules='numpy')
            self.clf = sp.lambdify([x_sym], clf_sym, modules='numpy')
            self.lf_clf = sp.lambdify([x_sym], dclfdx.T @ f_sym, modules='numpy')
            self.lg_clf = sp.lambdify([x_sym], dclfdx.T @ g_sym, modules='numpy')
            self.ddclfdx = sp.lambdify([x_sym], sp.hessian(clf_sym, x_sym), modules='numpy')

        # Adaptive control
        if acbf_sym is not None:
            self.acbf = sp.lambdify([x_sym, a_hat_sym], acbf_sym, modules='numpy')

            dacbfdx = sp.simplify(sp.derive_by_array(acbf_sym, x_sym))
            dacbfdx = sp.Matrix(dacbfdx)  # Convert to Matrix for compatibility
            self.dacbfdx = sp.lambdify([x_sym, a_hat_sym], dacbfdx, modules='numpy')
            self.lf_acbf = sp.lambdify([x_sym, a_hat_sym], dacbfdx.T @ f_sym, modules='numpy')
            self.lg_acbf = sp.lambdify([x_sym, a_hat_sym], dacbfdx.T @ g_sym, modules='numpy')
            self.lY_acbf = sp.lambdify([x_sym, a_hat_sym], dacbfdx.T @ Y_sym, modules='numpy') if Y_sym is not None else None

            dacbfda = sp.simplify(sp.derive_by_array(acbf_sym, a_hat_sym))
            dacbfda = sp.Matrix(dacbfda)  # Convert to Matrix for compatibility
            self.dacbfda = sp.lambdify([x_sym, a_hat_sym], dacbfda, modules='numpy')

        if aclf_sym is not None:
            self.aclf = sp.lambdify([x_sym, a_hat_sym], aclf_sym, modules='numpy')
            daclfdx = sp.simplify(sp.derive_by_array(aclf_sym, x_sym))
            daclfdx = sp.Matrix(daclfdx)  # Convert to Matrix for compatibility
            self.daclfdx = sp.lambdify([x_sym, a_hat_sym], daclfdx, modules='numpy')
            self.lf_aclf = sp.lambdify([x_sym, a_hat_sym], daclfdx.T @ f_sym, modules='numpy')
            self.lg_aclf = sp.lambdify([x_sym, a_hat_sym], daclfdx.T @ g_sym, modules='numpy')
            self.lY_aclf = sp.lambdify([x_sym, a_hat_sym], daclfdx.T @ Y_sym, modules='numpy') if Y_sym is not None else None
    
            daclfda = sp.simplify(sp.derive_by_array(aclf_sym, a_hat_sym))
            daclfda = sp.Matrix(daclfda)
            self.daclfda = sp.lambdify([x_sym, a_hat_sym], daclfda, modules='numpy')            

    # Control laws
    # CLFs
    # TODO: combine ctrl_cr_clf_qp and ctrl_cra_clf_qp
    def ctrl_cr_clf_qp(self, x, u_ref, cp_quantile, with_slack=True):
        """CR-CLF-QP Controller"""
        if self.clf is None:
            raise ValueError("CLF not defined.")
        if u_ref is None:
            u_ref = np.zeros((self.udim, 1))
        if u_ref.shape[0] != self.udim:
            raise ValueError("u_ref shape mismatch.")
        
        V = self.clf(x)
        LfV = self.lf_clf(x)
        LgV = self.lg_clf(x)
        dclfdx = self.dclfdx(x)

        cp_bound = cp_quantile * np.linalg.norm(dclfdx, 2)

        W = self.params["weight"]["input"] if "weight" in self.params and "input" in self.params["weight"] else 1.0
        W = W * np.eye(self.udim) if np.isscalar(W) else W

        if with_slack:
            # Constraints : A[u; slack] <= b
            A = np.hstack([LgV, np.array([[-1]])])
            b = -LfV - cp_bound - self.params["clf"]["rate"] * V
            if "u_max" in self.params:
                A = np.vstack([A, np.hstack([np.eye(self.udim), np.zeros((self.udim, 1))])])
                umax = self.params["u_max"]
                if np.isscalar(umax):
                    b = np.vstack([b, umax * np.ones((self.udim, 1))])
                elif umax.shape == (self.udim, 1) or umax.shape == (self.udim,):
                    b = np.vstack([b, umax.reshape(-1, 1)])
                else:
                    raise ValueError("params['u_max'] should be either a scalar or an (udim, 1) array")
            if "u_min" in self.params:
                A = np.vstack([A, np.hstack([-np.eye(self.udim), np.zeros((self.udim, 1))])])
                umin = self.params["u_min"]
                if np.isscalar(umin):
                    b = np.vstack([b, -umin * np.ones((self.udim, 1))])
                elif umin.shape == (self.udim, 1) or umin.shape == (self.udim,):
                    b = np.vstack([b, -umin.reshape(-1, 1)])
                else:
                    raise ValueError("params['u_min'] should be either a scalar or an (udim, 1) array")
            H = np.block([[W, np.zeros((self.udim, 1))],
                          [np.zeros((1, self.udim)), self.params["weight"]["slack"]]])
            f = np.concatenate([-W @ u_ref.flatten(), [0]])
            slack = cvxpy.Variable((1, 1))
            u = cvxpy.Variable((self.udim, 1))
            xvar = cvxpy.vstack([u, slack])
            prob = cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(xvar, H) + f @ xvar), [A @ xvar <= b, slack >= 0])
            prob.solve()
            feas = prob.status == cvxpy.OPTIMAL
            u_val = u.value if feas else np.array([self.params["u_min"] if LgV[i] > 0 else self.params["u_max"] for i in range(self.udim)])
            slack_val = slack.value.item()
        else:
            A = LgV
            b = -LfV - cp_bound - self.params["clf"]["rate"] * V
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
            f = -W @ u_ref.flatten()
            u = cvxpy.Variable((self.udim, 1))
            prob = cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(u, W) + f @ u), [A @ u <= b])
            prob.solve()
            feas = prob.status == cvxpy.OPTIMAL
            u_val = u.value if feas else np.array([self.params["u_min"] if LgV[i] > 0 else self.params["u_max"] for i in range(self.udim)])
            slack_val = []

        return u_val, V, slack_val, feas
    
    def ctrl_cra_clf_qp(self, x, u_ref, cp_quantile, dt, with_slack=True):
        """CRaCLF-QP Controller"""
        if self.aclf is None:
            raise ValueError("aCLF not defined.")
        if u_ref is None:
            u_ref = np.zeros((self.udim, 1))
        if u_ref.shape[0] != self.udim:
            raise ValueError("u_ref shape mismatch.")
        
        a_hat_L = self.a_hat_L

        V = self.aclf(x, a_hat_L)
        LfV = self.lf_aclf(x, a_hat_L)
        LgV = self.lg_aclf(x, a_hat_L)
        LYV = self.lY_aclf(x, a_hat_L)
        daclfdx = self.daclfdx(x, a_hat_L)
        daclfda = self.daclfda(x, a_hat_L)

        cp_bound = cp_quantile * np.linalg.norm(daclfdx, 2)

        W = self.params["weight"]["input"] if "weight" in self.params and "input" in self.params["weight"] else 1.0
        W = W * np.eye(self.udim) if np.isscalar(W) else W

        if with_slack:
            # Constraints : A[u; slack] <= b
            A = np.hstack([LgV, np.array([[-1]])])
            b = (
                -LfV 
                -cp_bound
                -LYV @ (a_hat_L + self.Gamma_L @ daclfda) #TODO: check sign
                - self.params["clf"]["rate"] * V
            )
            if "u_max" in self.params:
                A = np.vstack([A, np.hstack([np.eye(self.udim), np.zeros((self.udim, 1))])])
                umax = self.params["u_max"]
                if np.isscalar(umax):
                    b = np.vstack([b, umax * np.ones((self.udim, 1))])
                elif umax.shape == (self.udim, 1) or umax.shape == (self.udim,):
                    b = np.vstack([b, umax.reshape(-1, 1)])
                else:
                    raise ValueError("params['u_max'] should be either a scalar or an (udim, 1) array")
            if "u_min" in self.params:
                A = np.vstack([A, np.hstack([-np.eye(self.udim), np.zeros((self.udim, 1))])])
                umin = self.params["u_min"]
                if np.isscalar(umin):
                    b = np.vstack([b, -umin * np.ones((self.udim, 1))])
                elif umin.shape == (self.udim, 1) or umin.shape == (self.udim,):
                    b = np.vstack([b, -umin.reshape(-1, 1)])
                else:
                    raise ValueError("params['u_min'] should be either a scalar or an (udim, 1) array")
            H = np.block([[W, np.zeros((self.udim, 1))],
                          [np.zeros((1, self.udim)), self.params["weight"]["slack"]]])
            f = np.concatenate([-W @ u_ref.flatten(), [0]])
            slack = cvxpy.Variable((1, 1))
            u = cvxpy.Variable((self.udim, 1))
            xvar = cvxpy.vstack([u, slack])
            prob = cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(xvar, H) + f @ xvar), [A @ xvar <= b, slack >= 0])
            prob.solve()
            feas = prob.status == cvxpy.OPTIMAL
            u_val = u.value if feas else np.array([self.params["u_min"] if LgV[i] > 0 else self.params["u_max"] for i in range(self.udim)])
            slack_val = slack.value.item()
        else:
            A = LgV
            b = (
                -LfV
                -LYV @ (a_hat_L + self.Gamma_L @ daclfda) #TODO: check sign
                -cp_bound 
                -self.params["clf"]["rate"] * V
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
            f = -W @ u_ref.flatten()
            u = cvxpy.Variable((self.udim, 1))
            prob = cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(u, W) + f @ u), [A @ u <= b])
            prob.solve()
            feas = prob.status == cvxpy.OPTIMAL
            u_val = u.value if feas else np.array([self.params["u_min"] if LgV[i] > 0 else self.params["u_max"] for i in range(self.udim)])
            slack_val = None
        
        # Update a_hat_L
        self.update_a_hat_L(x, dt)

        return u_val, V, slack_val, feas
    
    # CBFs
    # TODO: combine ctrl_cr_cbf_qp and ctrl_cra_cbf_qp
    def ctrl_cr_cbf_qp(self, x, u_ref, cp_quantile):
        """CR-CBF-QP Controller"""
        if self.cbf is None:
            raise ValueError("CBF not defined.")
        if u_ref is None:
            u_ref = np.zeros((self.udim, 1))
        if u_ref.shape[0] != self.udim:
            raise ValueError("u_ref shape mismatch.")

        h = self.cbf(x)
        Lfh = self.lf_cbf(x)
        Lgh = self.lg_cbf(x)
        dcbfdx = self.dcbfdx(x)
        cp_bound = cp_quantile * np.linalg.norm(dcbfdx, 2)

        W = self.params["weight"]["input"] if "weight" in self.params and "input" in self.params["weight"] else 1.0
        W = W * np.eye(self.udim) if np.isscalar(W) else W

        A = -Lgh
        b = Lfh - cp_bound + self.params["cbf"]["rate"] * h
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
        u = cvxpy.Variable((self.udim, 1))
        f = -W @ u_ref.flatten()
        prob = cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(u, W) + f @ u), [A @ u <= b])
        prob.solve()

        feas = prob.status == cvxpy.OPTIMAL
        u_val = u.value if feas else np.zeros(self.udim)

        return u_val, h, feas

    def ctrl_cra_cbf_qp(self, x, u_ref, cp_quantile, dt):
        """CRaCBF-QP Controller"""
        if self.acbf is None:
            raise ValueError("aCBF not defined.")
        if u_ref is None:
            u_ref = np.zeros((self.udim, 1))
        if u_ref.shape[0] != self.udim:
            raise ValueError("u_ref shape mismatch.")
        
        a_hat_b = self.a_hat_b

        h = self.acbf(x, a_hat_b)
        Lfh = self.lf_acbf(x, a_hat_b)
        Lgh = self.lg_acbf(x, a_hat_b)
        LYh = self.lY_acbf(x, a_hat_b)
        dacbfdx = self.dacbfdx(x, a_hat_b)
        dacbfda = self.dacbfda(x, a_hat_b)

        cp_bound = cp_quantile * np.linalg.norm(dacbfdx, 2)

        W = self.params["weight"]["input"] if "weight" in self.params and "input" in self.params["weight"] else 1.0
        W = W * np.eye(self.udim) if np.isscalar(W) else W

        A = -Lgh
        b = (
            Lfh 
            + LYh @ (a_hat_b - self.Gamma_b @ dacbfda) #TODO: check sign
            - cp_bound
            + self.params["cbf"]["rate"] * (h - 0.5 * self.a_err_max.T @ np.linalg.inv(self.Gamma_b) @ self.a_err_max)
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
        u = cvxpy.Variable((self.udim, 1))
        f = -W @ u_ref.flatten()
        prob = cvxpy.Problem(cvxpy.Minimize(0.5 * cvxpy.quad_form(u, W) + f @ u), [A @ u <= b])
        prob.solve()

        feas = prob.status == cvxpy.OPTIMAL
        u_val = u.value if feas else np.zeros(self.udim)

        # Update a_hat_b
        self.update_a_hat_b(x, dt)

        return u_val, h, feas
    
    # CCMs
    def ctrl_cra_ccm(self, x, x_d, u_d):
        # x_d: Start point
        # x: End point
    
        # Formulate it as a min-norm CLF QP problem

        # TODO: avoid reshaping to make it faster
        u_d = u_d.reshape(-1, 1)
        
        if self.use_adaptive: 
            Y_x_a = self.Y(x) @ self.a_hat_ccm
            Y_d_a = self.Y(x_d) @ self.a_hat_ccm
        else:
            Y_x_a = 0.0
            Y_d_a = 0.0

        if self.use_cp:
            Theta = np.linalg.cholesky(self.M_x)
            sigma_max = np.max(np.linalg.svd(Theta, compute_uv=False))  # maximum singular value
            tightening = sigma_max * self.cp_quantile * np.sqrt(self.Erem)
        else:
            tightening = 0.0
        
        A = self.gamma_s1_M_x @ self.g(x)
        B = (self.gamma_s1_M_x @ (self.f(x) + self.g(x) @ u_d + Y_x_a)
            - self.gamma_s0_M_d @ (self.f(x_d) + self.g(x_d) @ u_d + Y_d_a)
            + self.params["ccm"]["rate"] * self.Erem).item()

        weight_slack = self.params["ccm"]["weight_slack"] if "weight_slack" in self.params["ccm"] else 100

        denom = (1 + weight_slack * A @ A.T).item()
        #tightening = (tightening * denom + B)/(denom-1)

        # Analytic solution
        if np.linalg.norm(A, 2) >= 1e-4:
            if B + tightening <= 0:
                u = 0.0
                slack = 0.0
            else:
                #denom = (1 + weight_slack * A @ A.T).item()
                u = (-weight_slack * (B + tightening) * A.T) / denom
                slack = (B + tightening) / denom
        else:
            if B + tightening <= 0:
                u = 0.0
                slack = 0.0
            else:
                u = 0.0
                slack = B + tightening

        uc = u_d + u

        return uc, slack #.item()

    # Solve for geodesics for CCM-based controllers
    def calc_geodesic(self, solver, x, x_d):
        
        # Initialize optimization variables and constraints internally
        c0, beq = solver.initialize_conditions(x_d, x)
        # Solve the geodesic optimization problem
        # TODO: check if self.a_hat_ccm is None?
        gamma, gamma_s, Erem = solver.solve_geodesic(c0, beq, self.a_hat_ccm)

        gamma_s_0 = gamma_s[:, 0]
        gamma_s_1 = gamma_s[:, -1]
        self.Erem = Erem.item()

        self.M_x = np.linalg.inv(self.W_fcn(x, self.a_hat_ccm))
        self.M_d = np.linalg.inv(self.W_fcn(x_d, self.a_hat_ccm))

        # TODO: make this faster
        self.gamma_s1_M_x = gamma_s_1.reshape(1,-1) @ self.M_x
        self.gamma_s0_M_d = gamma_s_0.reshape(1,-1) @ self.M_d

        if solver.dW_dai_fcn is not None and self.use_adaptive:
            dErem_dai = np.zeros(self.adim)
            # TODO: check correctness
            for i in range(self.adim):
                for k in range(solver.N + 1):
                    gk = gamma[:, k]
                    gsk = gamma_s[:, k]
                    dW_dai = solver.dW_dai_fcn(i,gk,self.a_hat_ccm)
                    dM_dai = -self.M_x @ dW_dai @ self.M_x
                    dErem_dai[i] += (gsk.T @ dM_dai @ gsk) * solver.w_cheby[k]
            self.dErem_dai = dErem_dai
    
    # Adaptation laws
    def adaptation_cra_clf(self, x, dt):
        """Update adaptive parameter a_hat_L for aCLF."""
        if self.Y is None or self.a_hat_L is None or self.a_hat_b is None:
            raise ValueError("Adaptive control parameters not defined.")

        daclfdx = self.daclfdx(x, self.a_hat_L)

        # Projection operator to enforce bounds on a_hat_L
        #TODO: check sign
        #self.a_hat_L += self.Gamma_L @ (daclfdx.T @ self.Y(x)).T, self.a_hat_norm_max
        self.a_hat_L += projection_operator(self.a_hat_L, self.Gamma_L @ (daclfdx.T @ self.Y(x)).T, self.a_hat_norm_max, self.epsilon) * dt

    def adaptation_cra_cbf(self, x, dt):
        """Update adaptive parameter a_hat_b for aCBF."""
        if self.Y is None or self.a_hat_L is None or self.a_hat_b is None:
            raise ValueError("Adaptive control parameters not defined.")

        dacbfdx = self.dacbfdx(x, self.a_hat_b)
        #self.a_hat_b += (-self.Gamma_b @ (dacbfdx.T @ self.Y(x)).T) * dt

        # Projection operator to enforce bounds on a_hat_b
        #TODO: check sign
        self.a_hat_b += projection_operator(self.a_hat_b, -self.Gamma_b @ (dacbfdx.T @ self.Y(x)).T, self.a_hat_norm_max, self.epsilon) * dt

    def adaptation_cra_ccm(self, x, x_d, dt):
        """Update adaptive parameter a_hat_ccm for tracking."""
        if self.Y is None or self.a_hat_L is None or self.a_hat_b is None:
            raise ValueError("Adaptive control parameters not defined.")

        # Update a_hat
        a_hat_dot = np.linalg.inv(self.Gamma_ccm) @ projection_operator(self.a_hat_ccm, 
                                            self.nu_ccm() * self.Y(x).T @ self.gamma_s1_M_x.T, 
                                            self.a_hat_norm_max, self.epsilon)
        #test_nonpositive = (self.a_hat_ccm - np.array([[0.1], [0.1], [0.1], [0.01]])).T @ (-self.nu_ccm()* self.Y(x).T @ self.gamma_s1_M_x.T + self.Gamma_ccm @ a_hat_dot)
        #a_hat_dot = self.nu_ccm() * np.linalg.inv(self.Gamma_ccm) @ self.Y(x).T @ self.gamma_s1_M_x.T
        self.a_hat_ccm += a_hat_dot * dt
        
        # Update rho
        rho_dot = -self.nu_ccm()/(self.dnu_drho_ccm() * (self.Erem+ self.eta_ccm)) * (2*self.gamma_s0_M_d @ self.Y(x_d) @ self.a_hat_ccm + self.dErem_dai @ a_hat_dot)
        self.rho_ccm += rho_dot * dt
    
    # Scaling functions for unmatched adaptive controls
    def nu_clf(self):
        pass
    
    def dnu_drho_clf(self):
        pass

    def nu_cbf(self):
        pass
    
    def dnu_drho_cbf(self):
        pass
    
    def nu_ccm(self):
        #TODO: pick a nu function of self.rho_ccm
        nu = 0.5*np.exp(self.rho_ccm/5) + 0.1
        return nu
    
    def dnu_drho_ccm(self):
        # TODO: take derivative of self.nu_ccm w.r.t. self.rho_ccm
        return 0.5*np.exp(-self.rho_ccm/5)/5