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
        self.assertGreaterEqual(result.normalized_gram_determinant, 1e-4)
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

    def test_square_root_second_derivative_is_fourth_score_moment(self) -> None:
        geometry = SoftmaxHessianGeometry(self.weights, self.p)
        score_u = geometry.centered @ self.u
        score_v = geometry.centered @ self.v
        guv = float(self.u @ geometry.metric @ self.v)
        second_derivative = np.sqrt(self.p) * (
            0.5 * score_u * score_v - guv
        )
        observed = geometry.square_root_second_derivative_norm(
            self.u,
            self.v,
        )
        self.assertAlmostEqual(
            observed,
            np.linalg.norm(second_derivative),
            places=14,
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

    def test_same_plane_fisher_orthonormalization_preserves_curvature(self) -> None:
        geometry = SoftmaxHessianGeometry(self.weights, self.p)
        original = geometry.sectional_curvature(self.u, self.v)
        u_normalized, v_normalized = geometry.fisher_orthonormalize_plane(
            self.u,
            self.v,
        )
        normalized = geometry.sectional_curvature(u_normalized, v_normalized)
        self.assertAlmostEqual(
            original.sectional_curvature,
            normalized.sectional_curvature,
            places=12,
        )
        self.assertAlmostEqual(normalized.normalized_gram_determinant, 1.0, places=12)

    def test_vocabulary_chunking_reproduces_full_curvature(self) -> None:
        rng = np.random.default_rng(2048)
        weights = rng.normal(size=(37, 4))
        probabilities = rng.dirichlet(np.linspace(0.7, 2.1, 37))
        u = rng.normal(size=4)
        v = rng.normal(size=4)
        geometry = SoftmaxHessianGeometry(weights, probabilities)
        full = geometry.sectional_curvature(u, v)
        for chunk_size in (1, 5, 16, 64):
            with self.subTest(chunk_size=chunk_size):
                chunked = geometry.sectional_curvature_chunked(
                    u,
                    v,
                    chunk_size=chunk_size,
                )
                self.assertAlmostEqual(
                    chunked.sectional_curvature,
                    full.sectional_curvature,
                    places=11,
                )
                np.testing.assert_allclose(
                    (
                        chunked.metric_min_eigenvalue,
                        chunked.metric_max_eigenvalue,
                    ),
                    (
                        full.metric_min_eigenvalue,
                        full.metric_max_eigenvalue,
                    ),
                    rtol=2e-13,
                    atol=2e-14,
                )

    def test_vocabulary_chunking_rejects_invalid_chunk_size(self) -> None:
        geometry = SoftmaxHessianGeometry(self.weights, self.p)
        for chunk_size in (0, -1, 1.5):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    geometry.sectional_curvature_chunked(
                        self.u,
                        self.v,
                        chunk_size=chunk_size,  # type: ignore[arg-type]
                    )

    def test_spectrum_matched_control_preserves_eigen_leverage(self) -> None:
        geometry = SoftmaxHessianGeometry(self.weights, self.p)
        source_u, source_v = geometry.fisher_orthonormalize_plane(
            self.u,
            self.v,
        )
        control_u, control_v = geometry.spectrum_matched_control_plane(
            self.u,
            self.v,
            np.random.default_rng(91),
        )
        source = np.column_stack((source_u, source_v))
        control = np.column_stack((control_u, control_v))
        source_whitened = (
            np.sqrt(geometry.eigenvalues)[:, None]
            * (geometry.eigenvectors.T @ source)
        )
        control_whitened = (
            np.sqrt(geometry.eigenvalues)[:, None]
            * (geometry.eigenvectors.T @ control)
        )
        for source_row, control_row in zip(source_whitened, control_whitened):
            np.testing.assert_allclose(
                np.outer(source_row, source_row),
                np.outer(control_row, control_row),
                atol=1e-12,
            )
        np.testing.assert_allclose(
            control.T @ geometry.metric @ control,
            np.eye(2),
            atol=1e-12,
        )

    def test_repeated_eigenspace_uses_basis_invariant_haar_block(self) -> None:
        scale = np.sqrt(3.0)
        weights = np.vstack(
            [
                scale * np.eye(3),
                -scale * np.eye(3),
            ]
        )
        probabilities = np.full(6, 1.0 / 6.0)
        geometry = SoftmaxHessianGeometry(weights, probabilities)
        np.testing.assert_allclose(geometry.metric, np.eye(3), atol=2e-15)
        self.assertEqual(
            geometry.eigenvalue_bands(eigenspace_rtol=1e-12),
            ((0, 3),),
        )
        source_u = np.array([1.0, 0.0, 0.0])
        source_v = np.array([0.0, 1.0, 0.0])
        control_u, control_v = geometry.spectrum_matched_control_plane(
            source_u,
            source_v,
            np.random.default_rng(41),
            eigenspace_rtol=1e-12,
        )
        control = np.column_stack((control_u, control_v))
        np.testing.assert_allclose(
            control.T @ geometry.metric @ control,
            np.eye(2),
            atol=1e-12,
        )
        self.assertFalse(
            np.allclose(np.abs(control), np.column_stack((source_u, source_v)))
        )

    def test_nonfinite_numerical_thresholds_are_rejected(self) -> None:
        for keyword in (
            "eigenvalue_rtol",
            "plane_gram_rtol",
            "solve_residual_rtol",
            "solve_forward_error_rtol",
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaises(ValueError):
                    SoftmaxHessianGeometry(
                        self.weights,
                        self.p,
                        **{keyword: float("nan")},
                    )
        geometry = SoftmaxHessianGeometry(self.weights, self.p)
        with self.assertRaisesRegex(ValueError, "eigenspace_rtol"):
            geometry.spectrum_matched_control_plane(
                self.u,
                self.v,
                np.random.default_rng(0),
                eigenspace_rtol=float("nan"),
            )

    def test_plane_gate_is_distinct_from_metric_rank_gate(self) -> None:
        geometry = SoftmaxHessianGeometry(
            self.weights,
            self.p,
            eigenvalue_rtol=1e-14,
            plane_gram_rtol=1e-4,
        )
        nearly_collinear = self.u + 1e-3 * self.v
        with self.assertRaisesRegex(ValueError, "normalized Fisher-Gram gate"):
            geometry.sectional_curvature(self.u, nearly_collinear)

    def test_linear_solve_residual_gate_is_enforced(self) -> None:
        geometry = SoftmaxHessianGeometry(
            self.weights,
            self.p,
            solve_residual_rtol=1e-30,
        )
        with self.assertRaisesRegex(ValueError, "relative-residual gate"):
            geometry.sectional_curvature(self.u, self.v)
        with self.assertRaisesRegex(ValueError, "relative-residual gate"):
            geometry.cubic_operator(self.u)


if __name__ == "__main__":
    unittest.main()
