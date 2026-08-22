from __future__ import annotations

import unittest

import numpy as np

from predictive_geometry.simplex import (
    analogy,
    exponential_exp,
    exponential_log,
    exponential_parallel_transport,
    fisher_distance,
    fisher_exp,
    fisher_inner,
    fisher_log,
    fisher_parallel_transport,
    mixture_parallel_transport,
    transport_around_loop,
    triangle_holonomy_angle,
)


class SimplexGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.p = np.array([0.50, 0.30, 0.20])
        self.q = np.array([0.42, 0.38, 0.20])
        self.r = np.array([0.44, 0.29, 0.27])
        self.u = np.array([0.03, -0.01, -0.02])

    def test_fisher_log_exp_round_trip(self) -> None:
        reconstructed = fisher_exp(self.p, fisher_log(self.p, self.q))
        np.testing.assert_allclose(reconstructed, self.q, atol=1e-12)

    def test_exponential_log_exp_round_trip(self) -> None:
        reconstructed = exponential_exp(
            self.p,
            exponential_log(self.p, self.q),
        )
        np.testing.assert_allclose(reconstructed, self.q, atol=1e-12)

    def test_fisher_small_distance_does_not_collapse_to_zero(self) -> None:
        p = np.array([0.5, 0.5])
        q = np.array([0.5 + 1e-8, 0.5 - 1e-8])
        distance = fisher_distance(p, q)
        self.assertGreater(distance, 0.0)
        self.assertAlmostEqual(distance, 2e-8, places=15)
        np.testing.assert_allclose(fisher_exp(p, fisher_log(p, q)), q, atol=1e-15)
        self.assertEqual(fisher_distance(p, p), 0.0)

    def test_exponential_log_exp_round_trip_near_boundary(self) -> None:
        epsilon = 1e-20
        p = np.array([1.0 - epsilon, epsilon])
        q = np.array([epsilon, 1.0 - epsilon])
        reconstructed = exponential_exp(p, exponential_log(p, q))
        np.testing.assert_allclose(reconstructed, q, rtol=1e-14, atol=0.0)

    def test_fisher_log_exp_round_trip_near_boundary(self) -> None:
        epsilon = 1e-20
        p = np.array([1.0 - epsilon, epsilon])
        q = np.array([epsilon, 1.0 - epsilon])
        reconstructed = fisher_exp(p, fisher_log(p, q))
        np.testing.assert_allclose(reconstructed, q, rtol=2e-6, atol=0.0)

    def test_unrepresentable_exponential_scores_fail_closed(self) -> None:
        p = np.array([np.nextafter(0.0, 1.0), 1.0])
        u = np.array([1e-3, -1e-3])
        with self.assertRaisesRegex(ValueError, "not representable"):
            exponential_exp(p, u)
        with self.assertRaisesRegex(ValueError, "not representable"):
            exponential_parallel_transport(p, p, u)

    def test_unrepresentable_fisher_tangent_fails_closed(self) -> None:
        p = np.array([np.nextafter(0.0, 1.0), 1.0])
        u = np.array([1e150, -1e150])
        with self.assertRaisesRegex(ValueError, "not representable"):
            fisher_exp(p, u)
        with self.assertRaisesRegex(ValueError, "not representable"):
            fisher_parallel_transport(p, p, u)

    def test_unrepresentable_exponential_endpoint_fails_closed(self) -> None:
        p = np.array([0.5, 0.5])
        u = np.array([500.0, -500.0])
        with self.assertRaisesRegex(ValueError, "endpoint.*not representable"):
            exponential_exp(p, u)

    def test_fisher_transport_is_metric_compatible(self) -> None:
        transported = fisher_parallel_transport(self.p, self.q, self.u)
        before = fisher_inner(self.p, self.u, self.u)
        after = fisher_inner(self.q, transported, transported)
        self.assertAlmostEqual(before, after, places=12)
        self.assertAlmostEqual(float(np.sum(transported)), 0.0, places=12)

    def test_exponential_and_mixture_transports_are_dual(self) -> None:
        v = np.array([-0.02, 0.04, -0.02])
        mixture_u = mixture_parallel_transport(self.p, self.q, self.u)
        exponential_v = exponential_parallel_transport(self.p, self.q, v)
        before = fisher_inner(self.p, self.u, v)
        after = fisher_inner(self.q, mixture_u, exponential_v)
        self.assertAlmostEqual(before, after, places=12)

    def test_flat_transports_have_trivial_loop_holonomy(self) -> None:
        loop = [self.p, self.q, self.r, self.p]
        for connection in ("mixture", "exponential"):
            returned = transport_around_loop(loop, self.u, connection)
            np.testing.assert_allclose(returned, self.u, atol=1e-12)

    def test_fisher_triangle_holonomy_matches_spherical_excess(self) -> None:
        initial = fisher_log(self.p, self.q)
        returned = transport_around_loop(
            [self.p, self.q, self.r, self.p],
            initial,
            "fisher",
        )
        cosine = fisher_inner(self.p, initial, returned) / np.sqrt(
            fisher_inner(self.p, initial, initial)
            * fisher_inner(self.p, returned, returned)
        )
        observed_angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
        predicted_angle = triangle_holonomy_angle(self.p, self.q, self.r)
        self.assertAlmostEqual(observed_angle, predicted_angle, places=10)
        self.assertGreater(observed_angle, 0.0)

    def test_three_connections_give_distinct_analogy_predictions(self) -> None:
        predictions = {
            name: analogy(self.p, self.q, self.r, name)
            for name in ("mixture", "exponential", "fisher")
        }
        for prediction in predictions.values():
            self.assertAlmostEqual(float(np.sum(prediction)), 1.0, places=12)
            self.assertTrue(np.all(prediction > 0.0))
        self.assertGreater(
            fisher_distance(predictions["mixture"], predictions["exponential"]),
            1e-6,
        )
        self.assertGreater(
            fisher_distance(predictions["mixture"], predictions["fisher"]),
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()
