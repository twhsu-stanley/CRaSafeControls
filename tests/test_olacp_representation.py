import unittest

import numpy as np

from olacp import OLACP
from systems.strict_feedback.strict_feedback import StrictFeedbackSystem


class TestOLACPRepresentation(unittest.TestCase):
    @staticmethod
    def calibration_scores():
        return np.linspace(0.01, 1.0, 100)

    def test_linear_feature_representation_matches_analytic_update(self):
        theta = np.array([[0.8, -0.2], [0.3, 0.6]])
        a = np.array([0.4, -0.7])
        learning_rate = 1e-2

        def psi(x):
            return np.array([[x[0], 1.0], [0.0, x[1]]])

        def Y_theta(x, theta_value):
            return psi(x) @ theta_value

        def loss_gradient(x, theta_value, a_value, w_value):
            residual = Y_theta(x, theta_value) @ a_value - w_value
            return 2.0 * np.outer(psi(x).T @ residual, a_value)

        olacp = OLACP(
            self.calibration_scores(),
            N_cal=100,
            buffer_maxlen=10,
            theta_init=theta,
            representation_lr=learning_rate,
            Y_theta=Y_theta,
            representation_loss_gradient=loss_gradient,
        )

        expected_gradient = np.zeros_like(theta)
        for index in range(10):
            x = np.array([0.1 * index, -0.05 * index])
            w = Y_theta(x, theta) @ a + np.array([0.02, -0.01])
            olacp.add_data_to_buffers(x, np.zeros(2), xdot=w)
            expected_gradient += loss_gradient(x, theta, a, w)

        olacp.estimate_uncertainty(dt=0.1)
        update = olacp.update_representation(a)

        np.testing.assert_allclose(update["gradient"], expected_gradient)
        np.testing.assert_allclose(
            update["Theta"], theta - learning_rate * expected_gradient
        )

    def test_neural_representation_accepts_flat_parameter_vector(self):
        # One hidden layer with two tanh units. The packed parameters are
        # [input weights, hidden biases, output weights, output bias].
        theta = np.array([0.4, -0.3, 0.1, -0.2, 0.7, -0.5, 0.05])
        a = np.array([0.8])

        def forward(x, theta_value):
            hidden = np.tanh(theta_value[:2] * x[0] + theta_value[2:4])
            output = theta_value[4:6] @ hidden + theta_value[6]
            return np.array([[output]])

        def backprop(x, theta_value, a_value, w_value):
            hidden = np.tanh(theta_value[:2] * x[0] + theta_value[2:4])
            residual = (forward(x, theta_value) @ a_value - w_value).item()
            output_sensitivity = 2.0 * residual * a_value.item()
            hidden_sensitivity = theta_value[4:6] * (1.0 - hidden**2)
            return output_sensitivity * np.concatenate(
                [
                    hidden_sensitivity * x[0],
                    hidden_sensitivity,
                    hidden,
                    np.ones(1),
                ]
            )

        olacp = OLACP(
            self.calibration_scores(),
            N_cal=100,
            buffer_maxlen=10,
            theta_init=theta,
            representation_lr=5e-3,
            Y_theta=forward,
            representation_loss_gradient=backprop,
        )

        for x_value in np.linspace(-1.0, 1.0, 10):
            x = np.array([x_value])
            w = forward(x, theta) @ a + np.array([0.03])
            olacp.add_data_to_buffers(x, np.zeros(1), xdot=w)

            analytic = backprop(x, theta, a, w)
            finite_difference = np.zeros_like(theta)
            epsilon = 1e-6
            for index in range(theta.size):
                theta_plus = theta.copy()
                theta_minus = theta.copy()
                theta_plus[index] += epsilon
                theta_minus[index] -= epsilon
                residual_plus = forward(x, theta_plus) @ a - w
                residual_minus = forward(x, theta_minus) @ a - w
                finite_difference[index] = (
                    residual_plus @ residual_plus
                    - residual_minus @ residual_minus
                ) / (2.0 * epsilon)
            np.testing.assert_allclose(
                analytic, finite_difference, rtol=1e-6, atol=1e-8
            )

        olacp.estimate_uncertainty(dt=0.1)
        update = olacp.update_representation(a)

        self.assertEqual(update["gradient"].shape, theta.shape)
        self.assertFalse(np.allclose(update["Theta"], theta))

    def test_gradient_shape_must_match_theta(self):
        olacp = OLACP(
            self.calibration_scores(),
            N_cal=100,
            buffer_maxlen=10,
            theta_init=np.ones(3),
            Y_theta=lambda x, theta: np.ones((1, 1)),
            representation_loss_gradient=lambda x, theta, a, w: np.ones((3, 1)),
        )

        for index in range(10):
            olacp.add_data_to_buffers(
                np.array([float(index)]), np.zeros(1), xdot=np.ones(1)
            )
        olacp.estimate_uncertainty(dt=0.1)

        with self.assertRaisesRegex(ValueError, "must return an array with shape"):
            olacp.update_representation(np.ones(1))


class TestStrictFeedbackRepresentation(unittest.TestCase):
    def setUp(self):
        self.theta = np.array(
            [[1.0, 0.1], [0.2, 0.8], [-1.0 / 6.0, -0.05]]
        )
        self.system = StrictFeedbackSystem(
            {
                "Theta_init": self.theta,
                "a_true": np.zeros(2),
                "a_lb": np.array([-0.5, -0.5]),
                "a_ub": np.array([0.5, 0.5]),
                "a_hat_norm_max": 1.0,
            }
        )

    def test_strict_feedback_gradient_matches_finite_differences(self):
        x = np.array([0.6, -0.2, 0.1])
        a = np.array([0.3, -0.4])
        w = np.array([0.15, 0.0, 0.0])
        analytic = self.system.representation_loss_gradient(
            x, self.theta, a, w
        )

        finite_difference = np.zeros_like(self.theta)
        epsilon = 1e-6
        for index in np.ndindex(self.theta.shape):
            theta_plus = self.theta.copy()
            theta_minus = self.theta.copy()
            theta_plus[index] += epsilon
            theta_minus[index] -= epsilon
            residual_plus = self.system.Y_theta(x, theta_plus) @ a - w
            residual_minus = self.system.Y_theta(x, theta_minus) @ a - w
            finite_difference[index] = (
                residual_plus @ residual_plus
                - residual_minus @ residual_minus
            ) / (2.0 * epsilon)

        np.testing.assert_allclose(analytic, finite_difference, rtol=1e-6, atol=1e-8)

    def test_installing_representation_updates_runtime_regressor(self):
        x = np.array([0.4, 0.1, -0.2])
        a = np.array([0.2, -0.1])
        theta_next = self.theta + 0.05

        self.system.set_representation(theta_next)

        np.testing.assert_allclose(
            self.system.Y(x), self.system.Y_theta(x, theta_next)
        )
        expected_lie_derivative = (
            np.asarray(self.system.dclfdx(x, a), dtype=float).reshape(3, 1).T
            @ self.system.Y(x)
        )
        np.testing.assert_allclose(
            self.system.lY_clf(x, a), expected_lie_derivative
        )


if __name__ == "__main__":
    unittest.main()
