import unittest
from unittest.mock import patch

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

from motion_planner import MotionPlanner
from olacp import OLACP
from simulations.planar_quad.sim_planar_quad_craccm_olacp import (
    build_waypoint_cost_guide, plan_nominal_trajectory, run_craccm_simulation,
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
        self.output_matrix = np.eye(2, 6)
        self.planner = MotionPlanner(
            self.system, 0.01, 5.0 * np.eye(2), 5.0 * np.eye(2),
            u_min=np.zeros(2), u_max=6.0 * np.ones(2), output_matrix=self.output_matrix,
            Q_terminal=np.eye(2),
        )
        self.x_init = np.array([0.0, 5.0, -0.03, 0.0, 0.0, 0.0])
        self.waypoints = np.array([[0.0, 5.0], [0.5, 5.5], [1.0, 5.0]])
        self.guide = build_waypoint_cost_guide(self.waypoints, 2.0)
        self.hover = self.system.mass * self.system.grav / 2 * np.ones(2)

    def plan(self, start_step=0, x_init=None, horizon=4.0, a_hat=None, u_previous=None):
        return self.planner.plan_certainty_equivalent_through_waypoints(
            self.x_init if x_init is None else x_init, self.waypoints, 200, start_step,
            self.theta, np.zeros(2) if a_hat is None else a_hat, self.hover,
            horizon_duration=horizon, prediction_dt=0.05, output_reference=self.guide,
            u_previous=u_previous,
        )

    def test_guide_hits_position_waypoints_and_pads_endpoints(self):
        np.testing.assert_allclose(self.guide([0.0, 2.0, 4.0]).T, self.waypoints, atol=1e-14)
        np.testing.assert_allclose(self.guide([-1.0]).ravel(), self.waypoints[0], atol=1e-14)
        np.testing.assert_allclose(self.guide([4.0, 5.0, 8.0]).T,
                                   np.repeat(self.waypoints[-1:], 3, axis=0), atol=1e-14)

    def test_guide_is_smooth_at_intermediate_waypoints_without_forcing_a_stop(self):
        step = 1e-4
        position = self.guide([2.0]).ravel()
        left_velocity = (position - self.guide([2.0 - step]).ravel()) / step
        right_velocity = (self.guide([2.0 + step]).ravel() - position) / step
        np.testing.assert_allclose(left_velocity, right_velocity, atol=1e-4)
        self.assertGreater(np.linalg.norm(left_velocity), 0.2)
        self.assertEqual(self.guide([0.3, 1.2, 2.3]).shape, (2, 3))

    def test_two_waypoint_guide_is_linear(self):
        guide = build_waypoint_cost_guide(np.array([[0.0, 5.0], [1.0, 6.0]]), 2.0)
        np.testing.assert_allclose(guide([0.0, 1.0, 2.0]).T,
                                   [[0.0, 5.0], [0.5, 5.5], [1.0, 6.0]])

    def test_fixed_size_problem_is_reused_with_only_model_and_guide_parameters(self):
        X, U = self.plan()
        problem = self.planner._mpc_problem
        self.assertEqual(X.shape, (6, 81))
        self.assertEqual(U.shape, (2, 80))
        self.assertEqual(set(problem["parameters"]), {
            "x_init", "theta", "a_hat", "u_reference", "u_previous", "output_reference",
        })
        self.plan(20, X[:, 4], a_hat=np.array([0.1, -0.05]))
        self.assertIs(self.planner._mpc_problem, problem)

    def test_size_stays_fixed_near_mission_end_and_guide_is_padded(self):
        self.plan()
        problem = self.planner._mpc_problem
        x_init = self.x_init.copy()
        x_init[:2] = self.waypoints[1]
        X, U = self.plan(220, x_init)
        self.assertIs(self.planner._mpc_problem, problem)
        x_init[:2] = self.waypoints[-1]
        X, U = self.plan(390, x_init)
        self.assertEqual(X.shape, (6, 81))
        self.assertEqual(U.shape, (2, 80))
        reference = problem["opti"].value(problem["parameters"]["output_reference"])
        np.testing.assert_allclose(reference[:, -1], self.waypoints[-1], atol=1e-14)

    def test_off_grid_replan_samples_the_output_guide(self):
        self.plan(1)
        problem = self.planner._mpc_problem
        reference = problem["opti"].value(problem["parameters"]["output_reference"])
        np.testing.assert_allclose(reference, self.guide(0.01 + 0.05 * np.arange(81)))

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
        np.testing.assert_allclose(X[:, 0], self.x_init, atol=1e-12)

    def test_cost_uses_discrete_output_weights_and_input_increments(self):
        self.planner.S = np.diag([2.0, 3.0])
        u_previous = self.hover + np.array([0.2, -0.1])
        X, U = self.plan(u_previous=u_previous)
        problem = self.planner._mpc_problem
        errors = self.output_matrix @ X - self.guide(0.05 * np.arange(81))
        input_errors = U - self.hover[:, None]
        increments = U - np.column_stack((u_previous, U[:, :-1]))
        expected = np.sum(errors[:, :-1] * (self.planner.Q @ errors[:, :-1]))
        expected += np.sum(input_errors * (self.planner.R @ input_errors))
        expected += np.sum(increments * (self.planner.S @ increments))
        expected += errors[:, -1] @ self.planner.Q_terminal @ errors[:, -1]
        self.assertAlmostEqual(float(problem["opti"].value(problem["opti"].f)), expected, places=9)
        objective = ca.Function(
            "output_cost", [problem["X"], problem["U"], problem["opti"].p], [problem["opti"].f]
        )
        changed_states = X.copy()
        changed_states[2:, :] += np.arange(4)[:, None] + 1.0
        changed_cost = objective(changed_states, U, problem["opti"].value(problem["opti"].p))
        self.assertAlmostEqual(float(changed_cost), expected, places=9)
        self.plan(20, X[:, 4], u_previous=U[:, 3])
        self.assertIs(self.planner._mpc_problem, problem)
        previous = problem["opti"].value(problem["parameters"]["u_previous"])
        np.testing.assert_allclose(previous, U[:, 3])

    def test_nominal_and_certainty_equivalent_waypoint_plans_match_when_a_hat_is_zero(self):
        planner = MotionPlanner(
            self.system, 0.05, self.planner.Q, self.planner.R,
            u_min=np.zeros(2), u_max=6.0 * np.ones(2), output_matrix=self.output_matrix,
            Q_terminal=np.diag([3.0, 7.0]), S=np.diag([2.0, 3.0]),
        )
        u_previous = self.hover + np.array([0.2, -0.1])
        for guide in (None, self.guide):
            with self.subTest(custom_guide=guide is not None):
                X_ce, U_ce = planner.plan_certainty_equivalent_through_waypoints(
                    self.x_init, self.waypoints, 40, 0, self.theta, np.zeros(2), self.hover,
                    horizon_duration=4.0, prediction_dt=0.05, output_reference=guide,
                    u_previous=u_previous,
                )
                X_nominal, U_nominal = planner.plan_nominal_through_waypoints(
                    self.waypoints, 40, X_ce, U_ce, self.hover, x_init=self.x_init,
                    output_reference=guide, u_previous=u_previous,
                )
                np.testing.assert_allclose(X_nominal, X_ce, atol=1e-6)
                np.testing.assert_allclose(U_nominal, U_ce, atol=1e-6)

    def test_nominal_goal_planner_uses_the_terminal_weight(self):
        class Integrator:
            xdim, udim = 1, 1

            def f(self, x):
                return np.zeros(1)

            def g(self, x):
                return np.ones((1, 1))

        planner = MotionPlanner(
            Integrator(), 1.0, np.array([[2.0]]), np.array([[3.0]]),
            Q_terminal=np.array([[12.0]]),
        )
        X, U = planner.plan_nominal_towards_goal(
            np.array([0.0]), np.array([1.0]), 1, np.array([[0.0, 1.0]]), np.array([[0.8]]),
        )
        # The one-step optimum is u = 12 / (3 + 12), independently of the running Q.
        np.testing.assert_allclose(U, [[0.8]], atol=1e-8)
        np.testing.assert_allclose(X, [[0.0, 0.8]], atol=1e-8)

    def test_full_state_nominal_planning_still_works(self):
        planner = MotionPlanner(self.system, 0.01, np.eye(6), np.eye(2), Q_terminal=np.eye(6))
        x_guess = np.repeat(self.x_init[:, None], 3, axis=1)
        u_guess = np.repeat(self.hover[:, None], 2, axis=1)
        X, U = planner.plan_nominal_towards_goal(
            self.x_init, self.x_init, 2, x_guess, u_guess, self.hover,
        )
        self.assertEqual(X.shape, (6, 3))
        self.assertEqual(U.shape, (2, 2))
        X, U = planner.plan_nominal_through_waypoints(
            np.array([self.x_init, self.x_init]), 2, x_guess, u_guess, self.hover,
        )
        self.assertEqual(X.shape, (6, 3))
        self.assertEqual(U.shape, (2, 2))

    def test_position_only_nominal_helper_uses_a_full_initial_state(self):
        config = {
            "K": 1, "I_length": 20, "dt": 0.01, "interval_duration": 0.2,
            "nominal_waypoints": np.array([[0.0, 5.0], [0.02, 5.01]]),
            "reference_initial_state": self.x_init,
            "motion_planner_Q": np.eye(2), "motion_planner_R": np.eye(2),
            "motion_planner_Q_terminal": np.eye(2),
        }
        trajectory = plan_nominal_trajectory(self.system, config, plot=False)
        self.assertEqual(trajectory["x_d"].shape, (6, 21))
        self.assertEqual(trajectory["waypoints"].shape, (2, 2))
        np.testing.assert_allclose(trajectory["x_d"][:, 0], self.x_init, atol=1e-12)

    def test_invalid_output_shapes_and_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "output_matrix"):
            MotionPlanner(self.system, 0.01, np.eye(2), np.eye(2), np.eye(2),
                          output_matrix=np.eye(2))
        with self.assertRaisesRegex(ValueError, "Q must be"):
            MotionPlanner(self.system, 0.01, np.eye(6), np.eye(2), np.eye(2),
                          output_matrix=self.output_matrix)
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            MotionPlanner(self.system, 0.01, np.eye(2), np.eye(2), np.eye(2),
                          output_matrix=self.output_matrix, S=-np.eye(2))
        with self.assertRaisesRegex(ValueError, "ydim"):
            self.planner.plan_certainty_equivalent_through_waypoints(
                self.x_init, np.zeros((3, 6)), 200, 0, self.theta, np.zeros(2),
            )
        with self.assertRaisesRegex(ValueError, "x_init is required"):
            self.planner.plan_nominal_through_waypoints(self.waypoints, 2, None, None)
        with self.assertRaisesRegex(ValueError, "output_reference"):
            self.planner.plan_certainty_equivalent_through_waypoints(
                self.x_init, self.waypoints, 200, 0, self.theta, np.zeros(2),
                output_reference=lambda times: np.zeros((6, len(times))),
            )
        with self.assertRaisesRegex(ValueError, "output_reference"):
            self.planner.plan_nominal_through_waypoints(
                self.waypoints, 2, None, None, x_init=self.x_init,
                output_reference=lambda times: np.zeros((6, len(times))),
            )

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
        planner = MotionPlanner(system, 0.01, np.eye(2), np.eye(1))
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

    def test_simulation_rejects_misaligned_representation_and_replanning_periods(self):
        system = PLANAR_QUAD({**self.params, "use_cp": False, "use_adaptive": True})
        learner = OLACP(
            np.full(80, 0.2), N_cal=80, Theta_init=self.theta, representation_period=5,
            Y_Theta=system.Y_Theta,
            representation_loss_gradient=system.representation_loss_gradient,
        )
        config = {
            "K": 2, "dt": 0.01, "I_length": 200, "interval_duration": 2.0,
            "nominal_waypoints": self.waypoints, "reference_initial_state": self.x_init,
            "mpc_prediction_dt": 0.05, "mpc_replan_interval": 0.3,
            "B": 3,  # This would align; the learner's actual period of five does not.
        }
        with patch.object(MotionPlanner, "plan_certainty_equivalent_through_waypoints") as plan:
            with self.assertRaisesRegex(ValueError, "representation update period"):
                run_craccm_simulation(system, learner, config, False, True, "test")
            plan.assert_not_called()

    def test_simulation_replans_after_representation_updates_and_integrates_live_reference(self):
        params = {**self.params, "use_cp": False, "use_adaptive": True}
        system = PLANAR_QUAD(params)
        waypoints = np.repeat(np.array([[0.0, 5.0]]), 3, axis=0)
        x_init = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        learner = OLACP(np.full(80, 0.2), N_cal=80, Theta_init=self.theta,
                        representation_period=1, representation_lr=1e-3,
                        Y_Theta=system.Y_Theta,
                        representation_loss_gradient=system.representation_loss_gradient)
        config = {
            "K": 2, "dt": 0.01, "I_length": 20, "interval_duration": 0.2,
            "nominal_waypoints": waypoints, "motion_planner_Q": np.eye(2),
            "motion_planner_R": np.eye(2),
            "motion_planner_Q_terminal": np.eye(2), "motion_planner_S": np.zeros((2, 2)),
            "reference_initial_state": x_init,
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
            calls.append((
                start_step, x_init.copy(), a_hat.copy(), kwargs["u_previous"].copy(), theta.copy(),
            ))
            self.assertEqual(points.shape, (3, 2))
            self.assertEqual(kwargs["output_reference"]([0.0, 0.1]).shape, (2, 2))
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
        for step, initial_state, initial_a_hat, u_previous, theta in calls:
            np.testing.assert_allclose(initial_state, result["x_d_hist"][step])
            np.testing.assert_allclose(initial_a_hat, adaptation * step * 0.01, atol=1e-12)
            np.testing.assert_allclose(theta, result["theta_hist"][step])
            expected_previous = result["u_d_hist"][step - 1] if step else self.hover
            np.testing.assert_allclose(u_previous, expected_previous)
        np.testing.assert_allclose(calls[1][4], self.theta)
        self.assertGreater(np.linalg.norm(calls[2][4] - calls[1][4]), 1e-12)
        for step in range(40):
            expected_u = input_sequence[:, (step % 10) // 5]
            np.testing.assert_allclose(result["u_d_hist"][step], expected_u)
        reference = result["desired_trajectory"]["x_d"].T
        self.assertEqual(reference.shape, (41, 6))
        for step in range(40):
            t = step * 0.01
            expected_u = result["u_d_hist"][step]
            expected_theta = result["theta_hist"][step]
            def desired_dynamics(tau, x):
                nominal = system.dynamics_nominal(x, expected_u)
                return nominal + system.Y_Theta(x, expected_theta) @ (adaptation * tau)
            solution = solve_ivp(
                desired_dynamics, (t, t + 0.01), reference[step], rtol=1e-9, atol=1e-11,
            )
            np.testing.assert_allclose(reference[step + 1], solution.y[:, -1], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
