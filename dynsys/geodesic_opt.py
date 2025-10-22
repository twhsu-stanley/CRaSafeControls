import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds, BFGS, check_grad

class GeodesicSolver:
    def __init__(self, n, D, N, W_fcn, dW_fcn):
        """
        Initialize the geodesic solver.

        Parameters:
            n      : int, number of state variables.
            D      : int, degree of the Chebyshev polynomial basis.
            N      : int, one less than the number of collocation nodes (total nodes = N+1).
            W_fcn  : callable, function that returns the contraction metric matrix W(x) when given a state x.
            dW_fcn : callable, function that returns the derivative of the contraction metric,
                           dW/dxi, when given an index i and state x.
        """
        self.n = n
        self.D = D
        self.N = N
        self.W_fcn = W_fcn
        self.dW_fcn = dW_fcn

        # Compute Clenshaw-Curtis nodes and weights and Chebyshev basis/derivative
        self.s, self.w_cheby = self.clencurt(N)
        self.T, self.Tdot = self.compute_cheby(N, D, self.s)

        # Initialize cache variables for the gradient computation
        self._c_pre = None
        self._g_pre = None

    @staticmethod
    def clencurt(N):
        """
        Compute Clenshaw-Curtis nodes and weights.

        Parameters:
            N : int, number of subintervals (produces N+1 nodes).

        Returns:
            x : 1D np.array, nodes mapped to [0, 1].
            w : 1D np.array, corresponding quadrature weights.
        """
        theta = np.pi * np.arange(0, N + 1) / N
        x = np.cos(theta)
        w = np.zeros(N + 1)
        ii = np.arange(1, N)
        v = np.ones(len(ii))
        if N % 2 == 0:
            w[0] = 1.0 / (N ** 2 - 1)
            w[N] = w[0]
            for k in range(1, N // 2):
                v = v - 2 * np.cos(2 * k * theta[ii]) / (4 * k ** 2 - 1)
            v = v - np.cos(N * theta[ii]) / (N ** 2 - 1)
        else:
            w[0] = 1.0 / (N ** 2)
            w[N] = w[0]
            for k in range(1, (N + 1) // 2):
                v = v - 2 * np.cos(2 * k * theta[ii]) / (4 * k ** 2 - 1)
        w[ii] = 2 * v / N
        # Reverse the arrays to mimic MATLAB's flip
        x = np.flip(x)
        w = np.flip(w)
        # Map nodes from [-1, 1] to [0, 1] and adjust weights accordingly
        x = (x + 1) / 2
        w = w / 2
        return x, w

    @staticmethod
    def compute_cheby(N, D, t):
        """
        Compute Chebyshev basis and its derivative.

        Parameters:
            N : int, where there will be N+1 Chebyshev nodes.
            D : int, degree of Chebyshev polynomial.
            t : 1D np.array of Chebyshev nodes in [0,1].

        Returns:
            T    : 2D np.array of shape (D+1, N+1) containing the Chebyshev basis.
            Tdot : 2D np.array of shape (D+1, N+1) containing the derivative of the basis.
        """
        T = np.zeros((D + 1, N + 1))
        U = np.zeros((D + 1, N + 1))
        Tdot = np.zeros((D + 1, N + 1))
        T[0, :] = 1.0
        if D >= 1:
            T[1, :] = t  # first degree term
            U[0, :] = 1.0
            U[1, :] = 2 * t
            Tdot[1, :] = 1.0
            for n in range(1, D):
                T[n + 1, :] = 2.0 * t * T[n, :] - T[n - 1, :]
                U[n + 1, :] = 2.0 * t * U[n, :] - U[n - 1, :]
                Tdot[n + 1, :] = (n + 1) * U[n, :]
        else:
            U[0, :] = 1.0
        return T, Tdot

    def compute_riemann_energy(self, c):
        """
        Compute the Riemann Energy using a pseudospectral method.

        Parameters:
            c : 1D np.array, flattened vector of coefficients with length n*(D+1).

        Returns:
            E : float, the computed Riemann energy.
        """
        # Reshape c into a (n x (D+1)) coefficient matrix.
        c_mat = np.reshape(c, (self.D + 1, self.n), order='F').T
        # Evaluate the state along the nodes.
        gamma = c_mat.dot(self.T)  # shape: (n, N+1)
        # Evaluate its derivative.
        gamma_s = c_mat.dot(self.Tdot)  # shape: (n, N+1)

        E = 0.0
        # Loop over each collocation node.
        for k in range(self.N + 1):
            gk = gamma[:, k]
            gsk = gamma_s[:, k]
            W = self.W_fcn(gk)
            # Solve W * x = gamma_s(:,k)
            x_sol = np.linalg.solve(W, gsk)
            E += np.dot(gsk.T, x_sol) * self.w_cheby[k]
        return E

    def compute_energy_gradient(self, c):
        """
        Compute the gradient of the Riemann Energy.

        Parameters:
            c : 1D np.array, flattened coefficient vector.

        Returns:
            g : 1D np.array, flattened gradient.
        """
        # Reshape c to a (n x (D+1)) matrix.
        c_matrix = np.reshape(c, (self.D + 1, self.n), order='F').T

        # Check cache to avoid recomputation if c has not changed significantly.
        if self._c_pre is None or np.linalg.norm(c_matrix - self._c_pre) > 1e-5:
            gamma = c_matrix.dot(self.T)  # shape: (n, N+1)
            gamma_s = c_matrix.dot(self.Tdot)  # shape: (n, N+1)
            g = np.zeros((1, (self.D + 1) * self.n))
            # Loop through each spectral node.
            for k in range(self.N + 1):
                gamma_k = gamma[:, k]
                gamma_s_k = gamma_s[:, k]
                W_eval = self.W_fcn(gamma_k)
                M_x_gamma_sk = np.linalg.solve(W_eval, gamma_s_k)
                for i in range(self.n):
                    dW_dxi = self.dW_fcn(i, gamma_k)
                    # Prepare the contribution matrix.
                    temp = np.zeros((self.n, self.D + 1))
                    temp[i, :] = self.Tdot[:self.D + 1, k]
                    T_k = self.T[:self.D + 1, k]
                    # Compute the term as in the MATLAB code.
                    term = 2 * temp - np.outer(np.dot(dW_dxi, M_x_gamma_sk), T_k)
                    contribution = np.dot(M_x_gamma_sk, term) * self.w_cheby[k]
                    # Add contribution to the appropriate block of the gradient.
                    start = i * (self.D + 1)
                    end = (i + 1) * (self.D + 1)
                    g[0, start:end] += contribution
            # Cache the computed coefficient matrix and gradient.
            self._c_pre = c_matrix.copy()
            self._g_pre = g.copy()
        else:
            g = self._g_pre

        return g.reshape(-1)

    def initialize_conditions(self, x_nom, x_end):
        """
        Initialize the coefficient vector and the equality constraint vector from
        the boundary conditions.

        Parameters:
            x_nom : np.array of shape (n,)
                The starting state.
            x_end : np.array of shape (n,)
                The ending state.

        Returns:
            c0  : 1D np.array, initial guess for the coefficients (length n*(D+1)).
            beq : 1D np.array, equality constraints vector (concatenation of x_nom and x_end).
        """
        c0 = np.zeros(self.n * (self.D + 1))
        # Compute indices for the first and second coefficient in each block.
        indices_x_nom = np.arange(self.n) * (self.D + 1)
        indices_x_offset = indices_x_nom + 1
        c0[indices_x_nom] = x_nom
        c0[indices_x_offset] = x_end - x_nom  # Difference used for the second coefficient.
        # Equality constraints: first set equal to the start, second set to the end.
        beq = np.hstack((x_nom, x_end))
        return c0, beq

    def solve_geodesic(self, c0, beq):
        """
        Set up and solve the geodesic optimization problem.

        Parameters:
            c0  : 1D np.array, initial flattened coefficient vector.
            beq : 1D np.array, right-hand side for equality constraints.

        Returns:
            c_opt  : 2D np.array, optimized coefficients (shape n x (D+1)).
            energy : float, final Riemann energy.
        """
        # Construct equality constraint matrices:
        #   Aeq1 enforces the first Chebyshev node conditions.
        Aeq1 = np.kron(np.eye(self.n), self.T[:, 0].reshape(1, -1))
        #   Aeq2 enforces a row of ones (constant term).
        Aeq2 = np.kron(np.eye(self.n), np.ones((1, self.D + 1)))
        Aeq = np.vstack((Aeq1, Aeq2))

        # Set variable bounds (example values as in the MATLAB code).
        lb_matrix = -20 * np.ones((self.n, self.D + 1))
        ub_matrix = 20 * np.ones((self.n, self.D + 1))
        # For example, fix states 3 and 4 (indices 2 and 3) to be in [-5, 5].
        if self.n >= 4:
            lb_matrix[2:4, :] = -5
            ub_matrix[2:4, :] = 5
        lb = lb_matrix.T.flatten(order='F')
        ub = ub_matrix.T.flatten(order='F')
        bounds = Bounds(lb, ub)

        # Define the cost (energy) and its gradient.
        costf = lambda c: self.compute_riemann_energy(c)
        grad = lambda c: self.compute_energy_gradient(c)

        # Create the linear equality constraint.
        linear_constraint = LinearConstraint(Aeq, beq, beq)

        # Optimize using the trust-region constrained method.
        res = minimize(fun=costf, x0=c0, method='trust-constr',
                       jac=grad, bounds=bounds,
                       constraints=[linear_constraint],
                       options={'maxiter': 500, 'gtol': 1e-4, 'xtol': 1e-8, 'verbose': 0})

        # Reshape the result back into a (n x (D+1)) matrix.
        c_opt = np.reshape(res.x, (self.D + 1, self.n), order='F').T
        energy = res.fun

        return c_opt, energy

    def get_trajectory(self, c_opt):
        """
        Compute the geodesic matrix and endpoint derivatives from the optimized coefficients.

        Parameters:
            c_opt : 2D np.array of shape (n x (D+1)) - optimized coefficients.

        Returns:
            gamma     : Geodesic matrix computed as c_opt @ T.
            gamma_s_0 : Derivative at the initial point (c_opt @ Tdot[:, 0]).
            gamma_s_1 : Derivative at the final point (c_opt @ Tdot[:, -1]).
        """
        gamma = c_opt.dot(self.T)
        gamma_s_0 = c_opt.dot(self.Tdot)[:, 0]
        gamma_s_1 = c_opt.dot(self.Tdot)[:, -1]
        return gamma, gamma_s_0, gamma_s_1
