import unittest
from unittest.mock import patch

import numpy as np

from simulations.acc.sim_acc_cracbf_olacp import (
    THETA_TRUE,
    drag_force,
    environment_coefficients,
    latent_parameter,
    run_simulation,
)
from systems.acc.acc import ACC


class TestACCRepresentationAndCertificate(unittest.TestCase):
    def setUp(self):
        self.a_lb = np.array([-0.2] * 6 + [20.0])
        self.a_ub = np.array([0.2] * 6 + [24.0])
        self.true_uncertainty = lambda x, t: np.array(
            [0.0, -0.25 - 0.01 * t, 22.0]
        )
        self.params = {
            "Theta_init": np.array([0.08]),
            "true_uncertainty": self.true_uncertainty,
            "m": 1650.0,
            "vd": 25.0,
            "Kp": 400.0,
            "z_min": 10.0,
            "T_h": 1.8,
            "cbf_smoothing_epsilon": 0.1,
            "a_lb": self.a_lb,
            "a_ub": self.a_ub,
            "a_hat_norm_max": 0.1,
            "use_adaptive": False,
            "use_cp": False,
            "cbf": {"rate": 0.5},
        }
        self.system = ACC(self.params)

    def test_representation_matches_section_vb_structure(self):
        x = np.array([5.0, 23.0, 18.0])
        theta = np.array([0.06])
        wake = np.exp(-theta.item() * x[2])
        expected = np.zeros((3, 7))
        expected[1, :6] = [
            1.0,
            x[1],
            x[1] ** 2,
            wake,
            x[1] * wake,
            x[1] ** 2 * wake,
        ]
        expected[2, 6] = 1.0

        np.testing.assert_allclose(self.system.Y_theta(x, theta), expected)

    def test_representation_gradient_matches_finite_difference(self):
        x = np.array([0.0, 21.0, 16.0])
        theta = np.array([0.07])
        a = np.array([-0.12, 0.004, -0.0003, 0.015, -0.002, 0.0002, 22.0])
        w = np.array([0.0, -0.31, 21.8])
        analytic = self.system.representation_loss_gradient(x, theta, a, w)

        step = 1e-6
        residual_plus = self.system.Y_theta(x, theta + step) @ a - w
        residual_minus = self.system.Y_theta(x, theta - step) @ a - w
        finite_difference = (
            residual_plus @ residual_plus - residual_minus @ residual_minus
        ) / (2.0 * step)

        np.testing.assert_allclose(
            analytic, np.array([finite_difference]), rtol=1e-6, atol=1e-9
        )

    def test_installing_representation_does_not_change_true_plant(self):
        x = np.array([2.0, 22.0, 19.0])
        u = np.array([350.0])
        a_hat = 0.5 * (self.a_lb + self.a_ub)
        dynamics_before = self.system.dynamics(x, u, t=0.4)
        Y_before = self.system.Y(x)
        lie_before = self.system.lY_cbf(x, a_hat)

        self.system.set_representation(np.array([0.04]))

        np.testing.assert_allclose(
            self.system.dynamics(x, u, t=0.4), dynamics_before
        )
        self.assertFalse(np.allclose(self.system.Y(x), Y_before))
        self.assertFalse(np.allclose(self.system.lY_cbf(x, a_hat), lie_before))

    def test_cbf_gradients_match_finite_differences(self):
        x = np.array([4.0, 24.0, 25.0])
        a_hat = np.array([0.0] * 6 + [22.0])
        step = 1e-6

        state_gradient = np.zeros(3)
        for index in range(3):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[index] += step
            x_minus[index] -= step
            state_gradient[index] = (
                self.system.cbf(x_plus, a_hat)
                - self.system.cbf(x_minus, a_hat)
            ) / (2.0 * step)

        parameter_gradient = np.zeros(7)
        for index in range(7):
            a_plus = a_hat.copy()
            a_minus = a_hat.copy()
            a_plus[index] += step
            a_minus[index] -= step
            parameter_gradient[index] = (
                self.system.cbf(x, a_plus)
                - self.system.cbf(x, a_minus)
            ) / (2.0 * step)

        np.testing.assert_allclose(
            self.system.dcbfdx(x, a_hat).ravel(),
            state_gradient,
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            self.system.dcbfda(x, a_hat).ravel(),
            parameter_gradient,
            rtol=1e-6,
            atol=1e-8,
        )

    def test_cbf_set_is_contained_in_physical_safe_set(self):
        a_hat = np.array([0.0] * 6 + [22.0])
        x = np.array([0.0, 24.0, 15.0])
        self.assertGreaterEqual(float(self.system.cbf(x, a_hat)), 0.0)
        self.assertGreaterEqual(x[2], self.system.z_min)
        self.assertGreater(
            float(
                self.system.smooth_positive_part(
                    x[1] - a_hat[6], self.system.cbf_smoothing_epsilon
                )
            ),
            0.0,
        )

    def test_dynamics_include_only_nominal_terms_and_true_uncertainty(self):
        x = np.array([1.0, 20.0, 30.0])
        u = np.array([500.0])
        expected = (
            self.system.f(x)
            + self.system.g(x) @ u
            + self.true_uncertainty(x, 0.3)
        )
        np.testing.assert_allclose(self.system.dynamics(x, u, 0.3), expected)


class TestCRaCBFController(unittest.TestCase):
    def adaptive_system(self):
        a_lb = np.array([-0.2] * 6 + [20.0])
        a_ub = np.array([0.2] * 6 + [24.0])
        return ACC(
            {
                "Theta_init": np.array([0.08]),
                "m": 1650.0,
                "vd": 25.0,
                "Kp": 400.0,
                "z_min": 10.0,
                "T_h": 1.8,
                "cbf_smoothing_epsilon": 0.1,
                "a_lb": a_lb,
                "a_ub": a_ub,
                "a_hat_norm_max": 0.1,
                "use_adaptive": True,
                "use_cp": True,
                "cp_quantile": 0.03,
                "Gamma_cbf": 2.0 * np.eye(7),
                "epsilon": 0.01,
                "eta_cbf": 3.0,
                "cbf": {"rate": 0.5},
            }
        )

    def test_qp_uses_equation_35_correction_with_eta(self):
        system = self.adaptive_system()
        x = np.array([0.0, 23.0, 25.0])
        a_hat = system.a_center.copy()
        rho = 0.2
        _, rho_dot = system.adaptation_cracbf(x, a_hat, rho)
        h = float(system.cbf(x, a_hat))
        expected_correction = (
            system.eta_cbf
            * system.dnu_drho_cbf(rho)
            * rho_dot
            / system.nu_cbf(rho)
        )

        a_dot, _ = system.adaptation_cracbf(x, a_hat, rho)
        substituted_correction = (
            -system.eta_cbf
            / (h + system.eta_cbf)
            * (system.dcbfda(x, a_hat).T @ a_dot).item()
        )
        self.assertAlmostEqual(expected_correction, substituted_correction)

        with patch("systems.control_affine_system.solve_qp") as solve_qp_mock:
            solve_qp_mock.return_value = np.zeros(1)
            system.ctrl_cracbf(x, a_hat, np.zeros(1), rho)

        expected_b = (
            system.lf_cbf(x, a_hat)
            + system.lY_cbf(x, a_hat) @ a_hat
            - system.cp_quantile * np.linalg.norm(system.dcbfdx(x, a_hat))
            + system.params["cbf"]["rate"]
            * (
                h
                - 0.5
                / system.nu_cbf(rho)
                * system.safe_set_tightening
            )
            - expected_correction
        )
        np.testing.assert_allclose(
            solve_qp_mock.call_args.kwargs["h"],
            np.asarray(expected_b).reshape(-1),
        )

    def test_returned_control_satisfies_equation_35(self):
        system = self.adaptive_system()
        x = np.array([0.0, 23.0, 25.0])
        a_hat = system.a_center.copy()
        rho = 0.2
        control = system.ctrl_cracbf(x, a_hat, np.zeros(1), rho).reshape(-1)
        _, rho_dot = system.adaptation_cracbf(x, a_hat, rho)

        gradient = system.dcbfdx(x, a_hat).reshape(-1)
        lhs = (
            gradient
            @ (
                system.f(x)
                + system.g(x) @ control
                + system.Y(x) @ a_hat
            )
            - np.linalg.norm(gradient) * system.cp_quantile
        )
        h = float(system.cbf(x, a_hat))
        rhs = (
            -system.params["cbf"]["rate"]
            * (
                h
                - 0.5
                / system.nu_cbf(rho)
                * system.safe_set_tightening
            )
            + system.dnu_drho_cbf(rho)
            * rho_dot
            / system.nu_cbf(rho)
            * system.eta_cbf
        )
        self.assertGreaterEqual(lhs + 1e-7, rhs)

    def test_nonadaptive_controller_requires_no_adaptation_configuration(self):
        a_lb = np.array([-0.2] * 6 + [20.0])
        a_ub = np.array([0.2] * 6 + [24.0])
        system = ACC(
            {
                "Theta_init": [0.08],
                "a_lb": a_lb,
                "a_ub": a_ub,
                "a_hat_norm_max": 0.1,
                "use_adaptive": False,
                "use_cp": False,
                "cbf": {"rate": 0.5},
            }
        )
        control = system.ctrl_cracbf(
            np.array([0.0, 23.0, 25.0]),
            system.a_center,
            np.zeros(1),
            0.0,
        )
        self.assertEqual(control.shape, (1, 1))
        self.assertEqual(system.safe_set_tightening, 0.0)


class TestACCExperiment(unittest.TestCase):
    def test_physical_drag_matches_paper_representation(self):
        mass = 1650.0
        wind_speed = 5.0
        environment = environment_coefficients(3)
        a_true = latent_parameter(environment, mass, wind_speed)
        system = ACC(
            {
                "Theta_init": [THETA_TRUE],
                "m": mass,
                "a_lb": a_true - 1.0,
                "a_ub": a_true + 1.0,
                "a_hat_norm_max": 1.0,
            }
        )
        x = np.array([0.0, 24.0, 20.0])
        expected = np.array(
            [
                0.0,
                -drag_force(x, environment, wind_speed) / mass,
                environment["lead_velocity"],
            ]
        )
        np.testing.assert_allclose(
            system.Y_theta(x, [THETA_TRUE]) @ a_true,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_short_algorithm_1_simulation_is_safe(self):
        results = run_simulation(
            plot=False,
            k_intervals=1,
            representation_period=2,
            dt=0.02,
            interval_duration=0.2,
            n_cal=100,
            verbose=False,
        )
        self.assertEqual(results["score"].shape, (1,))
        self.assertTrue(np.all(np.isfinite(results["x"])))
        self.assertGreaterEqual(np.min(results["h"]), 0.0)
        self.assertGreaterEqual(np.min(results["physical_safety_margin"]), 0.0)
        self.assertGreaterEqual(np.min(results["tightened_cbf_margin"]), 0.0)


if __name__ == "__main__":
    unittest.main()
