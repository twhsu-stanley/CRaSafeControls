import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp

from motion_planner import MotionPlanner
from olacp import OLACP
from simulations.planar_quad.sim_planar_quad_craccm_olacp import (
    build_waypoint_cost_guide, run_craccm_simulation,
)
from systems.planar_quad.planar_quad import PLANAR_QUAD


class CertaintyEquivalentMPCTests(unittest.TestCase):
    def setUp(self):
        self.theta = np.array([[0.3, 0.1], [0.1, 0.2], [0.2, -0.1], [0.1, 0.15]])
        self.params = {
            "Theta_init": self.theta, "a_lb": -np.ones(2), "a_ub": np.ones(2),
            "a_hat_norm_max": 2.0, "Gamma_ccm": np.eye(2), "eta_ccm": 500.0,
            "ccm_rate": 0.2, "u_min": np.zeros(2), "u_max": 6.0 * np.ones(2),
        }
        self.system = PLANAR_QUAD(self.params)
        self.planner = MotionPlanner(self.system, 0.01, np.eye(6), np.eye(2), np.eye(6),
                                     np.zeros(2), 6.0 * np.ones(2))
        self.waypoints = np.array([
            [0.0, 5.0, -0.03, 0.0, 0.0, 0.0],
            [0.5, 5.5, 0.02, 0.4, 0.2, 0.01],
            [1.0, 5.0, 0.03, 0.0, 0.0, 0.0],
        ])
        self.guide = build_waypoint_cost_guide(self.waypoints, 2.0)
        self.hover = self.system.mass * self.system.grav / 2 * np.ones(2)

    def plan(self, start_step=0, x_init=None, horizon=4.0, a_hat=None):
        return self.planner.plan_certainty_equivalent_through_waypoints(
            self.waypoints[0] if x_init is None else x_init, self.waypoints, 200, start_step,
            self.theta, np.zeros(2) if a_hat is None else a_hat, self.hover,
            horizon_duration=horizon, prediction_dt=0.05, state_reference=self.guide,
        )

    def test_guide_hits_all_waypoint_states_and_pads_final_state(self):
        np.testing.assert_allclose(self.guide([0.0, 2.0, 4.0]).T, self.waypoints, atol=1e-14)
        np.testing.assert_allclose(self.guide([4.0, 5.0, 8.0]).T,
                                   np.repeat(self.waypoints[-1:], 3, axis=0), atol=1e-14)

    def test_guide_velocities_and_angle_rate_are_kinematically_consistent(self):
        times = np.array([0.3, 1.2, 2.3, 3.5])
        state = self.guide(times)
        derivative = (self.guide(times + 1e-6) - self.guide(times - 1e-6)) / 2e-6
        inertial_vx = state[3] * np.cos(state[2]) - state[4] * np.sin(state[2])
        inertial_vz = state[3] * np.sin(state[2]) + state[4] * np.cos(state[2])
        np.testing.assert_allclose(derivative[:2], [inertial_vx, inertial_vz], atol=1e-8)
        np.testing.assert_allclose(derivative[2], state[5], atol=1e-8)

    def test_fixed_size_problem_is_reused_and_all_upcoming_waypoints_are_penalized(self):
        X, U = self.plan()
        problem = self.planner._mpc_problem
        self.assertEqual(X.shape, (6, 81))
        self.assertEqual(U.shape, (2, 80))
        self.assertEqual(self.planner.last_mpc_stats["waypoint_indices"], [1, 2])
        self.plan(20, X[:, 4], a_hat=np.array([0.1, -0.05]))
        self.assertIs(self.planner._mpc_problem, problem)
        mask = np.asarray(problem["opti"].value(problem["parameters"]["waypoint_mask"]))
        np.testing.assert_array_equal(np.flatnonzero(mask), [35, 75])

    def test_past_waypoints_are_excluded_and_size_stays_fixed_near_mission_end(self):
        self.plan()
        problem = self.planner._mpc_problem
        X, U = self.plan(220, self.waypoints[1])
        self.assertIs(self.planner._mpc_problem, problem)
        self.assertEqual(self.planner.last_mpc_stats["waypoint_indices"], [2])
        X, U = self.plan(390, self.waypoints[-1])
        self.assertEqual(X.shape, (6, 81))
        self.assertEqual(U.shape, (2, 80))
        reference = problem["opti"].value(problem["parameters"]["state_reference"])
        np.testing.assert_allclose(reference[:, -1], self.waypoints[-1], atol=1e-14)

    def test_off_grid_waypoint_cost_interpolates_at_the_original_deadline(self):
        self.plan(1)
        problem = self.planner._mpc_problem
        fraction = problem["opti"].value(problem["parameters"]["waypoint_fraction"])
        np.testing.assert_allclose(np.asarray(fraction).ravel()[[39, 79]], [0.8, 0.8])

    def test_changed_horizon_rebuilds_cache_and_model_constraints_are_satisfied(self):
        self.plan()
        problem = self.planner._mpc_problem
        a_hat = np.array([0.2, -0.1])
        X, U = self.plan(horizon=2.0, a_hat=a_hat)
        self.assertIsNot(self.planner._mpc_problem, problem)
        self.assertEqual(U.shape, (2, 40))
        for k in range(U.shape[1]):
            predicted = self.planner._certainty_equivalent_rk4(X[:, k], U[:, k], self.theta,
                                                              a_hat)
            np.testing.assert_allclose(X[:, k + 1], np.asarray(predicted).ravel(), atol=1e-7)
        self.assertTrue(np.all(U >= -1e-7) and np.all(U <= 6.0 + 1e-7))

    def test_nominal_planning_still_works(self):
        X, U = self.planner.plan(self.waypoints[0], self.waypoints[0], 2,
                                 np.repeat(self.waypoints[0, :, None], 3, axis=1),
                                 np.repeat(self.hover[:, None], 2, axis=1), self.hover)
        self.assertEqual(X.shape, (6, 3))
        self.assertEqual(U.shape, (2, 2))

    def test_invalid_prediction_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            self.plan(horizon=0.12)
        with self.assertRaisesRegex(ValueError, "last waypoint"):
            self.plan(start_step=400)

    def test_planner_uses_whole_nonlinear_representation_without_psi(self):
        class NonlinearRepresentation:
            xdim, udim, adim = 2, 1, 1
            theta_shape = (2, 1)

            def f(self, x):
                return np.array([x[1], -0.5 * x[0]])

            def g(self, x):
                return np.array([[0.0], [1.0]])

            def Y_Theta(self, x, theta):
                return np.array([[np.sin(theta[0, 0] * x[0]) + theta[1, 0]**2],
                                 [x[1] * theta[0, 0] / (1.0 + x[0]**2)]])

        system = NonlinearRepresentation()
        planner = MotionPlanner(system, 0.01, np.eye(2), np.eye(1), np.eye(2))
        theta, a_hat = np.array([[0.2], [-0.1]]), np.array([0.3])
        waypoints = np.array([[0.1, 0.0], [0.2, 0.1]])
        X, U = planner.plan_certainty_equivalent_through_waypoints(
            waypoints[0], waypoints, 200, 0, theta, a_hat,
            horizon_duration=2.0, prediction_dt=0.05,
        )
        expected = system.f(X[:, 0]) + system.g(X[:, 0]) @ U[:, 0]
        expected += system.Y_Theta(X[:, 0], theta) @ a_hat
        actual = planner._certainty_equivalent_dynamics(X[:, 0], U[:, 0], theta, a_hat)
        np.testing.assert_allclose(np.asarray(actual).ravel(), expected)

    def test_simulation_executes_input_prefix_and_integrates_live_adaptive_reference(self):
        params = {**self.params, "use_cp": False, "use_adaptive": True}
        system = PLANAR_QUAD(params)
        waypoints = np.repeat(np.array([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0]]), 3, axis=0)
        learner = OLACP(np.full(80, 0.2), N_cal=80, Theta_init=self.theta,
                        representation_period=5, representation_lr=1e-3,
                        Y_Theta=system.Y_Theta,
                        representation_loss_gradient=system.representation_loss_gradient)
        config = {
            "K": 2, "dt": 0.01, "I_length": 20, "interval_duration": 0.2,
            "nominal_waypoints": waypoints, "motion_planner_Q": np.eye(6),
            "motion_planner_R": np.eye(2), "motion_planner_Q_f": np.eye(6),
            "mpc_horizon_duration": 0.4, "mpc_prediction_dt": 0.05,
            "mpc_replan_interval": 0.1, "tracking_initial_offset": np.zeros(6),
            "rho_divergence_threshold": 1e6, "x_norm_divergence_threshold": 10.0,
            "geodesic_degree": 2, "geodesic_nodes": 8,
            "use_qpsolvers": False, "verify_geodesic": False,
        }
        calls = []
        input_sequence = self.hover[:, None] + np.array([1.0, -1.0])[:, None] * (
            0.01 * np.arange(8)
        )

        def fake_plan(planner, x_init, points, interval_steps, start_step, theta, a_hat,
                      u_reference, **kwargs):
            calls.append((start_step, x_init.copy(), a_hat.copy()))
            planner.last_mpc_stats = {"horizon_steps": 8, "iterations": 0}
            return np.zeros((6, 9)), input_sequence.copy()

        adaptation = np.array([0.1, -0.02])
        with (
            patch.object(MotionPlanner, "plan_certainty_equivalent_through_waypoints", fake_plan),
            patch.object(PLANAR_QUAD, "ctrl_craccm", return_value=(self.hover, 0.0)),
            patch.object(PLANAR_QUAD, "adaptation_craccm", return_value=(adaptation, 0.0)),
        ):
            # The controller is mocked only to isolate planning/execution scheduling.
            system.Erem = 0.0
            result = run_craccm_simulation(system, learner, config, False, True, "test")
        self.assertEqual([call[0] for call in calls], [0, 10, 20, 30])
        self.assertEqual(result["metrics"]["mpc_replan_count"], 4)
        np.testing.assert_array_equal(np.flatnonzero(result["mpc_replan_hist"]), [0, 10, 20, 30])
        for step, initial_state, initial_a_hat in calls:
            np.testing.assert_allclose(initial_state, result["x_d_hist"][step])
            np.testing.assert_allclose(initial_a_hat, adaptation * step * 0.01, atol=1e-12)
        for step in range(40):
            expected_u = input_sequence[:, (step % 10) // 5]
            np.testing.assert_allclose(result["u_d_hist"][step], expected_u)
        reference = result["desired_trajectory"]["x_d"].T
        self.assertEqual(reference.shape, (41, 6))
        for step in range(40):
            t = step * 0.01
            expected_u = result["u_d_hist"][step]
            def desired_dynamics(tau, x):
                return system.dynamics_certainty_equivalent(x, expected_u, adaptation * tau)
            solution = solve_ivp(
                desired_dynamics, (t, t + 0.01), reference[step], rtol=1e-9, atol=1e-11,
            )
            np.testing.assert_allclose(reference[step + 1], solution.y[:, -1], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
