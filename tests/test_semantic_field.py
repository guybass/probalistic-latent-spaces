from __future__ import annotations

import unittest

import numpy as np

from predictive_geometry.field import (
    alpha_parallel_transport,
    first_order_alpha_transport,
    fit_semantic_alpha,
    local_transport_defect_scalar,
    softmax_probabilities,
)
from predictive_geometry.softmax import SoftmaxHessianGeometry


class SemanticFieldTests(unittest.TestCase):
    def setUp(self) -> None:
        # Saturated natural coordinates for a three-category distribution.
        self.weights = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        self.bias = np.zeros(3)
        self.start = np.array([0.2, -0.4])
        self.end = np.array([0.45, 0.1])
        self.vector = np.array([0.7, -0.2])

    def geometry(self, hidden: np.ndarray) -> SoftmaxHessianGeometry:
        return SoftmaxHessianGeometry(
            self.weights,
            softmax_probabilities(self.weights, self.bias, hidden),
        )

    def test_exponential_transport_keeps_natural_components_constant(self) -> None:
        observed = alpha_parallel_transport(
            self.weights,
            self.bias,
            self.start,
            self.end,
            self.vector,
            alpha=1.0,
        )
        np.testing.assert_array_equal(observed, self.vector)

    def test_hidden_analogy_equals_exponential_distribution_composition(self) -> None:
        hidden_p = np.array([0.2, -0.4])
        hidden_q = np.array([-0.3, 0.7])
        hidden_r = np.array([0.5, 0.1])
        p = softmax_probabilities(self.weights, self.bias, hidden_p)
        q = softmax_probabilities(self.weights, self.bias, hidden_q)
        r = softmax_probabilities(self.weights, self.bias, hidden_r)

        expected = r * q / p
        expected /= np.sum(expected)
        observed = softmax_probabilities(
            self.weights,
            self.bias,
            hidden_r + hidden_q - hidden_p,
        )
        np.testing.assert_allclose(observed, expected, atol=1e-14)

    def test_mixture_transport_keeps_metric_lowered_components_constant(self) -> None:
        transported = alpha_parallel_transport(
            self.weights,
            self.bias,
            self.start,
            self.end,
            self.vector,
            alpha=-1.0,
        )
        start_metric = self.geometry(self.start).metric
        end_metric = self.geometry(self.end).metric
        np.testing.assert_allclose(
            end_metric @ transported,
            start_metric @ self.vector,
            atol=1e-13,
        )

    def test_levi_civita_transport_preserves_fisher_length(self) -> None:
        transported = alpha_parallel_transport(
            self.weights,
            self.bias,
            self.start,
            self.end,
            self.vector,
            alpha=0.0,
            steps=128,
        )
        initial_squared = float(
            self.vector @ self.geometry(self.start).metric @ self.vector
        )
        final_squared = float(
            transported @ self.geometry(self.end).metric @ transported
        )
        self.assertAlmostEqual(initial_squared, final_squared, places=11)

    def test_first_order_transport_has_quadratic_local_error(self) -> None:
        geometry = self.geometry(self.start)
        direction = np.array([0.3, -0.5])
        errors = []
        for epsilon in (1e-2, 5e-3):
            exact = alpha_parallel_transport(
                self.weights,
                self.bias,
                self.start,
                self.start + epsilon * direction,
                self.vector,
                alpha=0.0,
                steps=32,
            )
            approximate = first_order_alpha_transport(
                geometry,
                epsilon * direction,
                self.vector,
                alpha=0.0,
            )
            errors.append(np.linalg.norm(exact - approximate))
        # Halving an edge should quarter the leading second-order error.
        self.assertGreater(errors[0] / errors[1], 3.8)
        self.assertLess(errors[0] / errors[1], 4.2)

    def test_closed_form_fit_recovers_its_first_order_fixture(self) -> None:
        geometry = self.geometry(self.start)
        alpha_true = 0.35
        displacements = [
            np.array([0.01, -0.02]),
            np.array([-0.015, 0.005]),
            np.array([0.012, 0.009]),
        ]
        source_vectors = [
            np.array([0.7, -0.2]),
            np.array([-0.1, 0.8]),
            np.array([0.4, 0.3]),
        ]
        target_vectors = [
            first_order_alpha_transport(
                geometry,
                displacement,
                source,
                alpha=alpha_true,
            )
            for displacement, source in zip(displacements, source_vectors)
        ]
        result = fit_semantic_alpha(
            [geometry] * len(displacements),
            displacements,
            source_vectors,
            target_vectors,
        )
        self.assertAlmostEqual(result.alpha, alpha_true, places=12)
        self.assertAlmostEqual(result.residual_squared, 0.0, places=15)
        self.assertAlmostEqual(result.explained_fraction, 1.0, places=12)
        self.assertGreater(result.max_edge_fisher_length, 0.0)
        self.assertEqual(len(result.edge_fisher_lengths), len(displacements))
        self.assertGreater(result.excitation_ratio, 0.0)
        self.assertLess(
            result.residual_squared,
            min(
                result.exponential_residual_squared,
                result.levi_civita_residual_squared,
                result.mixture_residual_squared,
            ),
        )

    def test_weak_alpha_excitation_is_rejected(self) -> None:
        geometry = self.geometry(self.start)
        displacement = np.array([0.01, -0.02])
        source = np.array([0.7, -0.2])
        target = first_order_alpha_transport(
            geometry,
            displacement,
            source,
            alpha=0.35,
        )
        with self.assertRaisesRegex(ValueError, "weakly identified"):
            fit_semantic_alpha(
                [geometry],
                [displacement],
                [source],
                [target],
                excitation_rtol=1e6,
            )

    def test_integrated_transport_bias_decreases_with_edge_length(self) -> None:
        geometry = self.geometry(self.start)
        alpha_true = -1.0
        directions = [
            np.array([0.3, -0.5]),
            np.array([-0.4, 0.2]),
            np.array([0.25, 0.35]),
        ]
        source_vectors = [
            np.array([0.7, -0.2]),
            np.array([-0.1, 0.8]),
            np.array([0.4, 0.3]),
        ]

        def fitted(scale: float):
            displacements = [scale * direction for direction in directions]
            target_vectors = [
                alpha_parallel_transport(
                    self.weights,
                    self.bias,
                    self.start,
                    self.start + displacement,
                    source,
                    alpha=alpha_true,
                )
                for displacement, source in zip(displacements, source_vectors)
            ]
            return fit_semantic_alpha(
                [geometry] * len(displacements),
                displacements,
                source_vectors,
                target_vectors,
            )

        coarse = fitted(0.05)
        fine = fitted(0.025)
        self.assertLess(
            abs(fine.alpha - alpha_true),
            0.6 * abs(coarse.alpha - alpha_true),
        )
        self.assertAlmostEqual(
            fine.max_edge_fisher_length,
            0.5 * coarse.max_edge_fisher_length,
            places=14,
        )

    def test_alpha_identifiability_guard_is_field_scale_invariant(self) -> None:
        geometry = self.geometry(self.start)
        displacement = np.array([0.01, -0.02])
        source = np.array([0.7, -0.2])
        target = first_order_alpha_transport(
            geometry,
            displacement,
            source,
            alpha=0.35,
        )
        ordinary = fit_semantic_alpha(
            [geometry],
            [displacement],
            [source],
            [target],
        )
        tiny_scale = 1e-9
        rescaled = fit_semantic_alpha(
            [geometry],
            [displacement],
            [tiny_scale * source],
            [tiny_scale * target],
        )
        self.assertAlmostEqual(ordinary.alpha, rescaled.alpha, places=12)

    def test_transport_defect_is_scale_invariant(self) -> None:
        geometry = self.geometry(self.start)
        direction = self.end - self.start
        original = local_transport_defect_scalar(
            geometry,
            direction,
            self.vector,
        )
        rescaled = local_transport_defect_scalar(
            geometry,
            -3.0 * direction,
            2.5 * self.vector,
        )
        self.assertAlmostEqual(original, rescaled, places=13)


if __name__ == "__main__":
    unittest.main()
