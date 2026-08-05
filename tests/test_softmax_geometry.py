from __future__ import annotations

import unittest

import numpy as np

from predictive_geometry.softmax import (
    SoftmaxHessianGeometry,
    softmax_fisher_metric,
    softmax_sectional_curvature,
)


class SoftmaxGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        # Saturated natural coordinates for a three-category distribution:
        # logits are (h_1, h_2, 0).  This is the full Fisher simplex, so K=1/4.
        self.weights = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        self.p = np.array([0.23, 0.41, 0.36])
        self.u = np.array([0.8, -0.2])
        self.v = np.array([0.1, 1.1])

    def test_metric_matches_categorical_covariance(self) -> None:
        expected = self.weights.T @ (
            np.diag(self.p) - np.outer(self.p, self.p)
        ) @ self.weights
        observed = softmax_fisher_metric(self.weights, self.p)
        np.testing.assert_allclose(observed, expected, atol=1e-14)

    def test_predictively_null_decoder_direction_is_exactly_quotiented(self) -> None:
        # The second hidden coordinate adds the same scalar to every logit.
        weights = np.array(
            [
                [1.0, 1.0],
                [0.0, 1.0],
                [-1.0, 1.0],
            ]
        )
        hidden = np.array([0.4, -2.0])
        null_direction = np.array([0.0, 1.0])

        def probabilities(point: np.ndarray) -> np.ndarray:
            logits = weights @ point
            unnormalized = np.exp(logits - np.max(logits))
            return unnormalized / np.sum(unnormalized)

        p = probabilities(hidden)
        shifted_p = probabilities(hidden + 17.0 * null_direction)
        metric = softmax_fisher_metric(weights, p)

        np.testing.assert_allclose(shifted_p, p, atol=1e-15)
        np.testing.assert_allclose(metric @ null_direction, 0.0, atol=1e-15)
        self.assertEqual(np.linalg.matrix_rank(metric, tol=1e-12), 1)

    def test_saturated_three_category_family_has_curvature_one_quarter(self) -> None:
        result = softmax_sectional_curvature(
            self.weights,
            self.p,
            self.u,
            self.v,
        )
        self.assertAlmostEqual(result.sectional_curvature, 0.25, places=12)
        self.assertLess(result.solve_relative_residual, 1e-14)

    def test_near_boundary_saturated_family_still_has_curvature_one_quarter(self) -> None:
        for epsilon in (1e-1, 1e-3, 1e-6, 1e-9):
            with self.subTest(epsilon=epsilon):
                p = np.array([1.0 - 2.0 * epsilon, epsilon, epsilon])
                result = softmax_sectional_curvature(
                    self.weights,
                    p,
                    np.array([1.0, 0.0]),
                    np.array([0.0, 1.0]),
                )
                self.assertAlmostEqual(
                    result.sectional_curvature,
                    0.25,
                    places=7,
                )

    def test_curvature_is_invariant_to_hidden_linear_reparameterization(self) -> None:
        transform = np.array([[1.7, 0.4], [-0.3, 0.8]])
        transformed_weights = self.weights @ np.linalg.inv(transform)
        transformed_u = transform @ self.u
        transformed_v = transform @ self.v
        original = softmax_sectional_curvature(
            self.weights,
            self.p,
            self.u,
            self.v,
        )
        transformed = softmax_sectional_curvature(
            transformed_weights,
            self.p,
            transformed_u,
            transformed_v,
        )
        self.assertAlmostEqual(
            original.sectional_curvature,
            transformed.sectional_curvature,
            places=12,
        )

    def test_curvature_is_cubic_operator_commutator(self) -> None:
        geometry = SoftmaxHessianGeometry(self.weights, self.p)
        operator_u = geometry.cubic_operator(self.u)
        operator_v = geometry.cubic_operator(self.v)
        np.testing.assert_allclose(
            geometry.metric @ operator_u,
            operator_u.T @ geometry.metric,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            geometry.metric @ operator_v,
            operator_v.T @ geometry.metric,
            atol=1e-13,
        )

        curvature_operator = geometry.curvature_operator(self.u, self.v)
        numerator = float(
            self.u @ geometry.metric @ curvature_operator @ self.v
        )
        direct = geometry.sectional_curvature(self.u, self.v)
        self.assertAlmostEqual(
            numerator,
            direct.curvature_numerator,
            places=13,
        )

    def test_cubic_tensor_is_metric_derivative_under_hidden_translation(self) -> None:
        geometry = SoftmaxHessianGeometry(self.weights, self.p)
        base_h = np.log(self.p[:2] / self.p[2])
        epsilon = 1e-5

        def probabilities(hidden: np.ndarray) -> np.ndarray:
            logits = self.weights @ hidden
            shifted = logits - np.max(logits)
            unnormalized = np.exp(shifted)
            return unnormalized / np.sum(unnormalized)

        metric_plus = softmax_fisher_metric(
            self.weights,
            probabilities(base_h + epsilon * self.u),
        )
        metric_minus = softmax_fisher_metric(
            self.weights,
            probabilities(base_h - epsilon * self.u),
        )
        finite_difference = (metric_plus - metric_minus) / (2.0 * epsilon)
        cubic_matrix = geometry.metric @ geometry.cubic_operator(self.u)
        np.testing.assert_allclose(
            finite_difference,
            cubic_matrix,
            atol=2e-11,
            rtol=2e-9,
        )

    def test_degenerate_decoder_is_rejected(self) -> None:
        degenerate_weights = np.column_stack(
            (self.weights[:, 0], self.weights[:, 0])
        )
        with self.assertRaisesRegex(ValueError, "metric is numerically degenerate"):
            softmax_sectional_curvature(
                degenerate_weights,
                self.p,
                self.u,
                self.v,
            )

    def test_random_plane_is_fisher_orthonormal(self) -> None:
        geometry = SoftmaxHessianGeometry(self.weights, self.p)
        u, v = geometry.random_fisher_orthonormal_plane(
            np.random.default_rng(17)
        )
        gram = np.array(
            [
                [u @ geometry.metric @ u, u @ geometry.metric @ v],
                [v @ geometry.metric @ u, v @ geometry.metric @ v],
            ]
        )
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-12)
        self.assertAlmostEqual(
            geometry.sectional_curvature(u, v).sectional_curvature,
            0.25,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
