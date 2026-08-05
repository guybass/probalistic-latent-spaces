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

    def test_closed_form_fit_recovers_generating_alpha(self) -> None:
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
