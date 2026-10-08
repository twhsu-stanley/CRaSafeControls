import unittest
from unittest.mock import patch

import numpy as np
from qpsolvers import available_solvers, solve_qp

from systems.control_affine_system import ControlAffineSystem


class ConstantMetricSystem(ControlAffineSystem):
    xdim, udim, adim = 2, 2, 1

    def __init__(self, **overrides):
        params = {
            "a_lb": np.array([-1.0]), "a_ub": np.array([1.0]),
            "a_hat_norm_max": 1.01, "Gamma_ccm": np.eye(1),
            "use_cp": True, "use_adaptive": False, "cp_quantile": 0.2,
            "ccm_rate": 0.2, "weight_slack": 2000.0 / 3.0,
            "ccm_authority_threshold": 0.01,
        }
        super().__init__({**params, **overrides})
        self.metric = np.array([[2.0, 0.3], [0.3, 1.5]])
        self.input_matrix = np.diag([1.0, 12.0])
        self.drift_rate = 0.0

    def f(self, x):
        return -self.drift_rate * np.asarray(x)

    def g(self, x):
        return self.input_matrix

    def Y(self, x):
        return np.asarray(x).reshape(self.xdim, self.adim)

    def W_fcn(self, x, a_hat):
        return np.linalg.inv(self.metric)

    def calc_geodesic(self, solver, x, x_d, a_hat=None, verify_geodesic=False):
        error = x - x_d
        self.gamma = np.column_stack((x_d, x))
        self.gamma_s = np.repeat(error[:, None], 2, axis=1)
        self.Erem = float(error @ self.metric @ error)


class CRaCCMQPTests(unittest.TestCase):
    def setUp(self):
        self.system = ConstantMetricSystem()
        self.x_d = np.array([0.1, -0.2])
        self.a_hat = np.array([0.2])
        self.solver = next(
            (name for name in ["quadprog", "daqp", "proxqp"] if name in available_solvers), None,
        )

    def numerical_solver(self, P, q, G, h, solver):
        if self.solver is None:
            self.skipTest("No supported dense QP solver is available")
        options = {"primal_tol": 1e-12, "dual_tol": 1e-12} if self.solver == "daqp" else {}
        return solve_qp(P, q, G, h, solver=self.solver, **options)

    def control(self, system, x, slack=True):
        x_d = self.x_d if system.xdim == 2 else np.zeros(system.xdim)
        u_d = np.linspace(0.1, -0.1, system.udim)
        with patch("systems.control_affine_system.solve_qp", self.numerical_solver):
            u, xi = system.ctrl_craccm(
                x, self.a_hat, x_d, u_d, None, use_qpsolvers=True, use_slack=slack,
            )
        return u.ravel() - u_d, xi

    def quantities(self, system, x):
        x_d = self.x_d if system.xdim == 2 else np.zeros(system.xdim)
        error = x - x_d
        p = error @ system.metric
        A = p @ system.input_matrix
        rate = system.ccm_rate - system.drift_rate
        if system.use_adaptive:
            rate += self.a_hat.item()
        B = rate * float(error @ system.metric @ error)
        Q = system.cp_quantile if system.use_cp else 0.0
        r = min(np.linalg.norm(A) / system.ccm_authority_threshold, 1.0)
        sigma = r**3 * (10.0 - 15.0 * r + 6.0 * r**2)
        C = B + sigma * Q * np.linalg.norm(p)
        H = system.input_matrix.T @ system.input_matrix
        return A, B, p, Q, sigma, C, H

    def oracle(self, system, x, slack=True):
        A, _, _, _, _, C, H = self.quantities(system, x)
        if C <= 0.0:
            return np.zeros(system.udim), 0.0
        direction = np.linalg.solve(H, A)
        authority = A @ direction
        if slack:
            xi = C / (1.0 + system.weight_slack * authority)
            return -system.weight_slack * xi * direction, xi
        if authority == 0.0:
            raise ValueError("Infeasible hard constraint")
        return -C * direction / authority, 0.0

    def state_at_authority(self, authority):
        direction = np.array([1.0, 0.0])
        scale = np.linalg.norm(direction @ self.system.metric @ self.system.input_matrix)
        return self.x_d + direction * authority / scale

    def test_soft_solution_and_equation_55_across_authority_threshold(self):
        for use_cp in [False, True]:
            self.system.use_cp = use_cp
            for ratio in [0.0, 0.1, 0.5, 0.99, 1.0, 1.01, 2.0, 100.0]:
                with self.subTest(use_cp=use_cp, ratio=ratio):
                    x = self.state_at_authority(ratio * self.system.ccm_authority_threshold)
                    du, xi = self.control(self.system, x)
                    expected_du, expected_xi = self.oracle(self.system, x)
                    np.testing.assert_allclose(du, expected_du, rtol=1e-7, atol=1e-9)
                    self.assertAlmostEqual(xi, expected_xi, places=9)
                    A, B, p, Q, sigma, C, H = self.quantities(self.system, x)
                    self.assertGreaterEqual(xi, -1e-12)
                    self.assertLessEqual(A @ du + C - xi, 1e-9)
                    chi = float(np.linalg.norm(A) > self.system.ccm_authority_threshold)
                    self.assertGreaterEqual(sigma + 1e-14, chi)
                    self.assertLessEqual(A @ du + B + chi * Q * np.linalg.norm(p) - xi, 1e-9)
                    np.testing.assert_allclose(H @ du + self.system.weight_slack * xi * A,
                                               0.0, atol=1e-8)

    def test_hard_solution_uses_the_same_input_cost_and_tightening(self):
        for use_cp in [False, True]:
            self.system.use_cp = use_cp
            for ratio in [0.5, 1.0, 2.0, 100.0]:
                with self.subTest(use_cp=use_cp, ratio=ratio):
                    x = self.state_at_authority(ratio * self.system.ccm_authority_threshold)
                    du, xi = self.control(self.system, x, slack=False)
                    expected_du, _ = self.oracle(self.system, x, slack=False)
                    np.testing.assert_allclose(du, expected_du, rtol=1e-7, atol=1e-9)
                    self.assertEqual(xi, 0.0)
                    A, _, _, _, _, C, _ = self.quantities(self.system, x)
                    self.assertLessEqual(A @ du + C, 1e-9)

    def test_no_slack_qp_only_removes_the_slack_cost_and_constraints(self):
        x = self.x_d + np.array([0.2, -0.1])
        captured = []
        for slack in [True, False]:
            du, xi = self.oracle(self.system, x, slack=slack)
            solution = np.r_[du, xi] if slack else du
            with patch("systems.control_affine_system.solve_qp", return_value=solution) as mock:
                self.system.ctrl_craccm(
                    x, self.a_hat, self.x_d, np.zeros(2), None, use_slack=slack,
                )
            captured.append(mock.call_args.args)
        soft_P, soft_q, soft_G, soft_h = captured[0]
        hard_P, hard_q, hard_G, hard_h = captured[1]
        _, _, _, _, _, _, H = self.quantities(self.system, x)
        np.testing.assert_array_equal(soft_P[:2, :2], H)
        np.testing.assert_array_equal(hard_P, H)
        self.assertEqual(soft_P[-1, -1], self.system.weight_slack)
        np.testing.assert_array_equal(soft_P[:2, -1], np.zeros(2))
        np.testing.assert_array_equal(soft_q[:2], hard_q)
        np.testing.assert_array_equal(soft_G[:1, :2], hard_G)
        np.testing.assert_array_equal(soft_G[:, -1], -np.ones(2))
        np.testing.assert_array_equal(soft_G[1, :2], np.zeros(2))
        np.testing.assert_array_equal(soft_h[:1], hard_h)
        self.assertEqual(soft_h[-1], 0.0)

    def test_normalized_weight_preserves_the_previously_tuned_solution(self):
        x = self.x_d + np.array([0.2, -0.1])
        du, xi = self.control(self.system, x)
        A, _, _, _, _, C, H = self.quantities(self.system, x)
        P = np.zeros((3, 3))
        P[:2, :2] = 150.0 * H
        P[-1, -1] = 1e5
        G = np.array([[*A, -1.0], [0.0, 0.0, -1.0]])
        expected = self.numerical_solver(P, np.zeros(3), G, np.array([-C, 0.0]), None)
        np.testing.assert_allclose(np.r_[du, xi], expected, rtol=1e-8, atol=1e-10)

    def test_zero_tracking_error_returns_desired_input_without_slack(self):
        for slack in [True, False]:
            du, xi = self.control(self.system, self.x_d, slack=slack)
            np.testing.assert_allclose(du, np.zeros(2), atol=1e-12)
            self.assertAlmostEqual(xi, 0.0)

    def test_zero_authority_has_slack_but_may_be_infeasible_without_it(self):
        system = ConstantMetricSystem(udim=1)
        system.metric = np.eye(2)
        system.input_matrix = np.array([[0.0], [1.0]])
        x = self.x_d + np.array([0.3, 0.0])
        du, xi = self.control(system, x)
        np.testing.assert_allclose(du, np.zeros(1), atol=1e-12)
        self.assertAlmostEqual(xi, system.ccm_rate * 0.3**2)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            self.control(system, x, slack=False)

    def test_inactive_constraint_returns_zero_correction_and_slack(self):
        self.system.use_cp = False
        self.system.drift_rate = 0.5
        for slack in [True, False]:
            du, xi = self.control(self.system, self.x_d + np.array([0.2, -0.1]), slack=slack)
            np.testing.assert_allclose(du, np.zeros(2), atol=1e-12)
            self.assertAlmostEqual(xi, 0.0)

    def test_no_cp_ignores_the_quantile(self):
        self.system.use_cp = False
        x = self.x_d + np.array([0.2, -0.1])
        for slack in [True, False]:
            first = self.control(self.system, x, slack=slack)
            self.system.cp_quantile = np.nan
            second = self.control(self.system, x, slack=slack)
            np.testing.assert_allclose(first[0], second[0])
            self.assertEqual(first[1], second[1])

    def test_certainty_equivalent_terms_are_preserved(self):
        system = ConstantMetricSystem(use_adaptive=True)
        x = self.x_d + np.array([0.2, -0.1])
        for slack in [True, False]:
            actual_du, actual_xi = self.control(system, x, slack=slack)
            expected_du, expected_xi = self.oracle(system, x, slack=slack)
            np.testing.assert_allclose(actual_du, expected_du, atol=1e-10)
            self.assertAlmostEqual(actual_xi, expected_xi)

    def test_control_is_continuous_at_authority_threshold(self):
        threshold = self.system.ccm_authority_threshold
        for slack in [True, False]:
            changes = []
            for width in [1e-4, 1e-7]:
                left = self.control(self.system, self.state_at_authority(threshold - width), slack)
                right = self.control(self.system, self.state_at_authority(threshold + width), slack)
                changes.append(np.linalg.norm(np.r_[right[0] - left[0], right[1] - left[1]]))
            self.assertLess(changes[1], changes[0] * 0.002)

    def test_general_input_dimensions_are_supported(self):
        system = ConstantMetricSystem(xdim=3, udim=3)
        system.metric = np.eye(3)
        system.input_matrix = np.diag([1.0, 2.0, 4.0])
        x = np.array([0.2, -0.1, 0.3])
        for slack in [True, False]:
            actual_du, actual_xi = self.control(system, x, slack=slack)
            expected_du, expected_xi = self.oracle(system, x, slack=slack)
            self.assertEqual(actual_du.shape, (3,))
            np.testing.assert_allclose(actual_du, expected_du, atol=1e-10)
            self.assertAlmostEqual(actual_xi, expected_xi)

    def test_cost_does_not_depend_on_the_sampling_time(self):
        x = self.x_d + np.array([0.2, -0.1])
        for slack in [True, False]:
            first = self.control(ConstantMetricSystem(dt=None), x, slack=slack)
            second = self.control(ConstantMetricSystem(dt=0.5), x, slack=slack)
            np.testing.assert_array_equal(first[0], second[0])
            self.assertEqual(first[1], second[1])

    def test_no_slack_does_not_use_the_slack_weight(self):
        system = ConstantMetricSystem(weight_slack=np.nan)
        x = self.x_d + np.array([0.2, -0.1])
        actual_du, actual_xi = self.control(system, x, slack=False)
        expected_du, _ = self.oracle(system, x, slack=False)
        np.testing.assert_allclose(actual_du, expected_du, atol=1e-10)
        self.assertEqual(actual_xi, 0.0)

    def test_invalid_parameters_are_rejected(self):
        for value in [0.0, -0.1, np.nan, np.inf]:
            with self.subTest(threshold=value), self.assertRaisesRegex(ValueError, "threshold"):
                ConstantMetricSystem(ccm_authority_threshold=value)
            with self.subTest(weight=value), self.assertRaisesRegex(ValueError, "weight_slack"):
                self.control(ConstantMetricSystem(weight_slack=value), self.x_d)

    def test_invalid_cp_quantile_is_rejected(self):
        for value in [-0.1, np.nan, np.inf]:
            with self.subTest(quantile=value), self.assertRaisesRegex(ValueError, "quantile"):
                self.control(ConstantMetricSystem(cp_quantile=value), self.x_d)

    def test_failed_numerical_solves_have_explicit_errors(self):
        for slack in [True, False]:
            with patch("systems.control_affine_system.solve_qp", return_value=None):
                with self.assertRaisesRegex(ValueError, "solution"):
                    self.system.ctrl_craccm(
                        self.x_d, self.a_hat, self.x_d, np.zeros(2), None, use_slack=slack,
                    )

    def test_analytical_path_remains_unsupported(self):
        for slack in [True, False]:
            with self.assertRaisesRegex(ValueError, "not supported"):
                self.system.ctrl_craccm(
                    self.x_d, self.a_hat, self.x_d, np.zeros(2), None,
                    use_qpsolvers=False, use_slack=slack,
                )


if __name__ == "__main__":
    unittest.main()
