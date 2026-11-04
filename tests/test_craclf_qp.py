import unittest
from unittest.mock import patch

import numpy as np
from qpsolvers import available_solvers, solve_qp

from systems.control_affine_system import ControlAffineSystem


class QuadraticCLFSystem(ControlAffineSystem):
    xdim, udim, adim = 2, 2, 1

    def __init__(self, **overrides):
        params = {
            "a_lb": np.array([-1.0]), "a_ub": np.array([1.0]),
            "a_hat_norm_max": 1.01, "use_cp": True, "use_adaptive": False,
            "cp_quantile": 0.2, "clf_rate": 0.5, "weight_slack": 25.0,
            "clf_goal_radius": 0.1,
        }
        super().__init__({**params, **overrides})
        self.metric = np.array([[2.0, 0.3], [0.3, 1.5]])
        self.input_matrix = np.diag([1.0, 3.0])
        self.drift_rate = 0.2

    def f(self, x):
        return self.drift_rate * np.asarray(x)

    def g(self, x):
        return self.input_matrix

    def Y(self, x):
        return np.asarray(x).reshape(self.xdim, self.adim)

    def clf(self, x, a_hat):
        return np.asarray(0.5 * x @ self.metric @ x)

    def dclfdx(self, x, a_hat):
        return (self.metric @ x).reshape(self.xdim, 1)


class CRaCLFQPTests(unittest.TestCase):
    def setUp(self):
        self.system = QuadraticCLFSystem()
        self.a_hat = np.array([0.2])
        self.solver = next(
            (name for name in ["quadprog", "daqp", "proxqp"] if name in available_solvers),
            None,
        )

    def numerical_solver(self, P, q, G, h, solver):
        if self.solver is None:
            self.skipTest("No supported dense QP solver is available")
        options = {"primal_tol": 1e-12, "dual_tol": 1e-12} if self.solver == "daqp" else {}
        h = np.asarray(h).ravel()
        if self.solver == "quadprog":
            return solve_qp(P, q, G, h, solver=self.solver)
        # Precondition fallback solvers without changing the QP or its minimizer.
        # DAQP otherwise drops inverse-Hessian entries at the pendubot's slack weight.
        variable_scale = 1.0 / np.sqrt(np.diag(P))
        scaled_P = variable_scale[:, None] * P * variable_scale[None, :]
        scaled_G = G * variable_scale[None, :]
        row_scale = np.linalg.norm(scaled_G, axis=1)
        row_scale[row_scale == 0.0] = 1.0
        solution = solve_qp(scaled_P, q * variable_scale, scaled_G / row_scale[:, None],
                            h / row_scale, solver=self.solver, **options)
        return None if solution is None else variable_scale * solution

    def control(self, x, slack=True, u_ref=None):
        if u_ref is None:
            u_ref = np.zeros(self.system.udim)
        with patch("systems.control_affine_system.solve_qp", self.numerical_solver):
            u, xi = self.system.ctrl_craclf(x, self.a_hat, u_ref, use_slack=slack)
        return u.ravel(), xi

    def quantities(self, x):
        gradient = self.system.metric @ x
        A = gradient @ self.system.input_matrix
        V = 0.5 * x @ self.system.metric @ x
        base = gradient @ (self.system.f(x) + self.system.Y(x) @ self.a_hat)
        base += self.system.clf_rate * V
        r = min(np.linalg.norm(x) / self.system.clf_goal_radius, 1.0)
        sigma = r**3 * (10.0 - 15.0 * r + 6.0 * r**2)
        Q = self.system.cp_quantile if self.system.use_cp else 0.0
        C = base + sigma * Q * np.linalg.norm(gradient)
        return A, base, gradient, Q, sigma, C

    def oracle(self, x, slack=True, u_ref=None):
        if u_ref is None:
            u_ref = np.zeros(self.system.udim)
        A, _, _, _, _, C = self.quantities(x)
        violation = max(float(A @ u_ref + C), 0.0)
        authority = A @ A
        if slack:
            xi = violation / (1.0 + self.system.weight_slack * authority)
            return u_ref - self.system.weight_slack * xi * A, xi
        if violation == 0.0:
            return u_ref, 0.0
        return u_ref - violation * A / authority, 0.0

    def state_at_ratio(self, ratio):
        direction = np.array([0.6, -0.8])
        return direction * ratio * self.system.clf_goal_radius

    def capture_qp(self, x, slack=True):
        size = self.system.udim + int(slack)
        with patch("systems.control_affine_system.solve_qp", return_value=np.zeros(size)) as mock:
            self.system.ctrl_craclf(x, self.a_hat, np.zeros(2), use_slack=slack)
        return mock.call_args.kwargs

    def implemented_scale(self, ratio):
        x = self.state_at_ratio(ratio)
        qp = self.capture_qp(x)
        _, base, gradient, Q, _, _ = self.quantities(x)
        return (-float(qp["h"].ravel()[0]) - base) / (Q * np.linalg.norm(gradient))

    def test_soft_min_norm_solution_and_indicator_constraint(self):
        for use_cp in [False, True]:
            self.system.use_cp = use_cp
            for ratio in [0.0, 1e-3, 0.1, 0.5, 0.99, 1.0, 1.01, 2.0, 20.0]:
                with self.subTest(use_cp=use_cp, ratio=ratio):
                    x = self.state_at_ratio(ratio)
                    u, xi = self.control(x)
                    expected_u, expected_xi = self.oracle(x)
                    np.testing.assert_allclose(u, expected_u, rtol=1e-7, atol=1e-10)
                    self.assertAlmostEqual(xi, expected_xi, places=10)
                    A, base, gradient, Q, sigma, C = self.quantities(x)
                    self.assertGreaterEqual(xi, -1e-12)
                    self.assertLessEqual(A @ u + C - xi, 1e-9)
                    chi = float(np.linalg.norm(x) > self.system.clf_goal_radius)
                    self.assertGreaterEqual(sigma + 1e-14, chi)
                    self.assertLessEqual(A @ u + base + chi * Q * np.linalg.norm(gradient) - xi,
                                         1e-9)
                    np.testing.assert_allclose(u + self.system.weight_slack * xi * A,
                                               0.0, atol=1e-8)

    def test_no_slack_branch_uses_the_same_scaled_constraint(self):
        for use_cp in [False, True]:
            self.system.use_cp = use_cp
            for ratio in [0.1, 0.5, 1.0, 2.0]:
                with self.subTest(use_cp=use_cp, ratio=ratio):
                    x = self.state_at_ratio(ratio)
                    u, xi = self.control(x, slack=False)
                    expected_u, _ = self.oracle(x, slack=False)
                    np.testing.assert_allclose(u, expected_u, rtol=1e-7, atol=1e-10)
                    self.assertEqual(xi, 0.0)
                    A, _, _, _, _, C = self.quantities(x)
                    self.assertLessEqual(A @ u + C, 1e-9)

    def test_qp_has_only_clf_and_nonnegative_slack_constraints_without_bounds(self):
        x = self.state_at_ratio(0.5)
        soft = self.capture_qp(x)
        hard = self.capture_qp(x, slack=False)
        np.testing.assert_array_equal(soft["P"], np.diag([1.0, 1.0, 25.0]))
        np.testing.assert_array_equal(soft["q"], np.zeros(3))
        self.assertEqual(soft["G"].shape, (2, 3))
        np.testing.assert_array_equal(soft["G"][:, -1], -np.ones(2))
        np.testing.assert_array_equal(soft["G"][1, :2], np.zeros(2))
        np.testing.assert_array_equal(soft["G"][:1, :2], hard["G"])
        np.testing.assert_array_equal(soft["h"].ravel()[:1], hard["h"].ravel())
        np.testing.assert_array_equal(hard["P"], np.eye(2))
        self.assertEqual(hard["G"].shape, (1, 2))

    def test_scaling_is_monotone_and_full_outside_goal_ball(self):
        ratios = np.linspace(0.01, 2.0, 100)
        scales = np.array([self.implemented_scale(ratio) for ratio in ratios])
        self.assertTrue(np.all(np.diff(scales) >= -1e-13))
        self.assertTrue(np.all(scales >= 0.0))
        self.assertTrue(np.all(scales <= 1.0 + 1e-13))
        np.testing.assert_allclose(scales[ratios >= 1.0], 1.0, atol=1e-13)
        self.assertAlmostEqual(self.implemented_scale(0.5), 0.5, places=13)

    def test_first_and_second_derivatives_match_at_goal_boundary_and_origin(self):
        for h in [1e-3, 1e-4]:
            below = self.implemented_scale(1.0 - h)
            at_boundary = self.implemented_scale(1.0)
            above = self.implemented_scale(1.0 + h)
            first = (above - below) / (2.0 * h)
            second = (above - 2.0 * at_boundary + below) / h**2
            self.assertLess(abs(first), 6.0 * h**2)
            self.assertLess(abs(second), 12.0 * h)
            positive = self.implemented_scale(h)
            negative = self.implemented_scale(-h)
            self.assertAlmostEqual((positive - negative) / (2.0 * h), 0.0, places=10)
            self.assertLess(abs((positive + negative) / h**2), 21.0 * h)

    def test_origin_has_zero_input_and_slack(self):
        u, xi = self.control(np.zeros(2))
        np.testing.assert_allclose(u, np.zeros(2), atol=1e-12)
        self.assertAlmostEqual(xi, 0.0, places=12)
        np.testing.assert_array_equal(self.capture_qp(np.zeros(2))["h"], np.zeros((2, 1)))

    def test_example_slack_weights_and_small_or_zero_control_authority(self):
        x = self.state_at_ratio(2.0)
        for weight in [1e3, 4e11]:
            self.system.weight_slack = weight
            for scale in [0.0, 1e-6, 1e-3]:
                with self.subTest(weight=weight, scale=scale):
                    self.system.input_matrix = scale * np.diag([1.0, 3.0])
                    u, xi = self.control(x)
                    expected_u, expected_xi = self.oracle(x)
                    np.testing.assert_allclose(u, expected_u, rtol=1e-7, atol=1e-9)
                    self.assertAlmostEqual(xi, expected_xi, places=9)
                    A, _, _, _, _, C = self.quantities(x)
                    self.assertLessEqual(A @ u + C - xi, 1e-9)

    def test_no_cp_solution_is_independent_of_goal_radius(self):
        self.system.use_cp = False
        x = self.state_at_ratio(0.5)
        expected = self.control(x)
        for radius in [0.001, 0.1, 100.0]:
            self.system.clf_goal_radius = radius
            u, xi = self.control(x)
            np.testing.assert_allclose(u, expected[0], atol=1e-12)
            self.assertAlmostEqual(xi, expected[1], places=12)

    def test_nonzero_reference_interface_is_preserved(self):
        x = self.state_at_ratio(0.5)
        for u_ref in [np.array([0.4, -0.2]), np.array([-1.0, 1.0])]:
            with self.subTest(u_ref=u_ref):
                u, xi = self.control(x, u_ref=u_ref)
                expected_u, expected_xi = self.oracle(x, u_ref=u_ref)
                np.testing.assert_allclose(u, expected_u, rtol=1e-7, atol=1e-10)
                self.assertAlmostEqual(xi, expected_xi, places=10)

    def test_optional_bounds_are_preserved_for_other_callers(self):
        self.system.params.update({"u_min": -0.01, "u_max": 0.01})
        x = self.state_at_ratio(2.0)
        u, xi = self.control(x)
        self.assertTrue(np.all(np.abs(u) <= 0.01 + 1e-12))
        A, _, _, _, _, C = self.quantities(x)
        self.assertLessEqual(A @ u + C - xi, 1e-9)

    def test_invalid_goal_radii_are_rejected(self):
        for radius in [0.0, -0.1, np.inf, np.nan]:
            with self.subTest(radius=radius):
                with self.assertRaisesRegex(ValueError, "clf_goal_radius"):
                    QuadraticCLFSystem(clf_goal_radius=radius)

    def test_invalid_cp_quantiles_are_rejected_only_when_cp_is_used(self):
        for quantile in [-0.1, np.inf, np.nan]:
            self.system.cp_quantile = quantile
            with self.subTest(quantile=quantile):
                with self.assertRaisesRegex(ValueError, "conformal quantile"):
                    self.capture_qp(self.state_at_ratio(0.5))
                self.system.use_cp = False
                self.capture_qp(self.state_at_ratio(0.5))
                self.system.use_cp = True


if __name__ == "__main__":
    unittest.main()
