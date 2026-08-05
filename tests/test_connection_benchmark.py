from __future__ import annotations

import unittest

import numpy as np

from predictive_geometry.benchmark import (
    evaluate_quadrilateral,
    predict_composition,
)
from predictive_geometry.simplex import fisher_distance


class ConnectionBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.p00 = np.array([0.50, 0.30, 0.20])
        self.p10 = np.array([0.43, 0.37, 0.20])
        self.p01 = np.array([0.46, 0.28, 0.26])

    def test_recovers_exact_mixture_composition(self) -> None:
        p11 = self.p01 + self.p10 - self.p00
        report = evaluate_quadrilateral(self.p00, self.p10, self.p01, p11)
        mixture_rows = [
            row
            for row in report["predictions"]
            if row["connection"] == "mixture"
        ]
        self.assertTrue(all(row["feasible"] for row in mixture_rows))
        self.assertTrue(all(row["fisher_distance"] < 1e-12 for row in mixture_rows))
        self.assertLess(report["orientation_closure"]["mixture"], 1e-12)

    def test_recovers_exact_exponential_composition(self) -> None:
        weights = self.p01 * self.p10 / self.p00
        p11 = weights / np.sum(weights)
        for orientation in ("A_along_B", "B_along_A"):
            prediction = predict_composition(
                self.p00,
                self.p10,
                self.p01,
                connection="exponential",
                orientation=orientation,
            )
            self.assertLess(fisher_distance(prediction, p11), 1e-12)

    def test_flat_connections_agree_between_orientations(self) -> None:
        for connection in ("mixture", "exponential"):
            left = predict_composition(
                self.p00,
                self.p10,
                self.p01,
                connection=connection,
                orientation="A_along_B",
            )
            right = predict_composition(
                self.p00,
                self.p10,
                self.p01,
                connection=connection,
                orientation="B_along_A",
            )
            np.testing.assert_allclose(left, right, atol=1e-14)

    def test_infeasible_mixture_is_reported_not_projected(self) -> None:
        p00 = np.array([0.80, 0.10, 0.10])
        p10 = np.array([0.10, 0.80, 0.10])
        p01 = np.array([0.10, 0.10, 0.80])
        p11 = np.array([0.20, 0.30, 0.50])
        report = evaluate_quadrilateral(p00, p10, p01, p11)
        mixture_rows = [
            row
            for row in report["predictions"]
            if row["connection"] == "mixture"
        ]
        self.assertTrue(all(not row["feasible"] for row in mixture_rows))
        self.assertTrue(all(row["fisher_distance"] is None for row in mixture_rows))
        self.assertTrue(
            all(row["transport_log_length_ratio"] is not None for row in mixture_rows)
        )

    def test_fisher_transport_has_zero_metric_distortion(self) -> None:
        p11 = self.p01 * self.p10 / self.p00
        p11 /= np.sum(p11)
        report = evaluate_quadrilateral(self.p00, self.p10, self.p01, p11)
        fisher_rows = [
            row
            for row in report["predictions"]
            if row["connection"] == "fisher"
        ]
        self.assertTrue(
            all(abs(row["transport_log_length_ratio"]) < 1e-12 for row in fisher_rows)
        )


if __name__ == "__main__":
    unittest.main()
