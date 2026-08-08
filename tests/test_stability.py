from __future__ import annotations

import math
import unittest

import numpy as np

from predictive_geometry.stability import (
    levi_civita_stability_bounds,
    square_root_cubic_tensor,
)


class StabilityDiagnosticsTests(unittest.TestCase):
    def test_square_root_cubic_matches_probability_coordinates(self) -> None:
        probabilities = np.array([0.2, 0.3, 0.5])
        probability_jacobian = np.array(
            [
                [0.10, -0.03],
                [-0.04, 0.05],
                [-0.06, -0.02],
            ]
        )
        psi = 2.0 * np.sqrt(probabilities)
        psi_jacobian = probability_jacobian / np.sqrt(probabilities)[:, None]
        expected = np.einsum(
            "ai,aj,ak,a->ijk",
            probability_jacobian,
            probability_jacobian,
            probability_jacobian,
            1.0 / (probabilities * probabilities),
            optimize=True,
        )
        observed = square_root_cubic_tensor(psi, psi_jacobian)
        np.testing.assert_allclose(observed, expected, atol=2e-16)

    def test_bernoulli_boundary_cubic_is_two_cotangent(self) -> None:
        theta = 0.17
        psi = 2.0 * np.array(
            [np.sin(theta / 2.0), np.cos(theta / 2.0)]
        )
        jacobian = np.array(
            [[np.cos(theta / 2.0)], [-np.sin(theta / 2.0)]]
        )
        tensor = square_root_cubic_tensor(psi, jacobian)
        self.assertAlmostEqual(
            float(tensor[0, 0, 0]),
            2.0 / np.tan(theta),
            places=13,
        )

    def test_paired_bernoulli_sqrt_agreement_has_diverging_alpha_defect(
        self,
    ) -> None:
        def square_root_jet(theta: float) -> tuple[np.ndarray, ...]:
            psi = 2.0 * np.array(
                [np.sin(theta / 2.0), np.cos(theta / 2.0)]
            )
            first = np.array(
                [[np.cos(theta / 2.0)], [-np.sin(theta / 2.0)]]
            )
            second = -0.5 * np.array(
                [np.sin(theta / 2.0), np.cos(theta / 2.0)]
            )
            cubic = float(square_root_cubic_tensor(psi, first)[0, 0, 0])
            return psi, first[:, 0], second, np.array([cubic])

        agreements = []
        connection_defects = []
        for epsilon in (0.1, 0.05, 0.025):
            first_jet = square_root_jet(epsilon)
            second_jet = square_root_jet(2.0 * epsilon)
            agreements.append(
                max(
                    np.linalg.norm(left - right)
                    for left, right in zip(first_jet[:3], second_jet[:3])
                )
            )
            # For alpha=1 and g=1, Gamma^(1)=-C/2.
            connection_defects.append(
                0.5 * abs(float(first_jet[3][0] - second_jet[3][0]))
            )
            self.assertAlmostEqual(
                float(first_jet[1] @ first_jet[1]),
                1.0,
                places=14,
            )
            self.assertAlmostEqual(
                float(second_jet[1] @ second_jet[1]),
                1.0,
                places=14,
            )

        self.assertGreater(agreements[0], agreements[1])
        self.assertGreater(agreements[1], agreements[2])
        self.assertLess(connection_defects[0], connection_defects[1])
        self.assertLess(connection_defects[1], connection_defects[2])

    def test_lc_bounds_use_sharper_isometric_branch(self) -> None:
        bounds = levi_civita_stability_bounds(
            first_derivative_bound=2.0,
            second_derivative_bound=3.0,
            metric_eigenvalue_floor=0.5,
            first_derivative_defect=0.1,
            second_derivative_defect=0.2,
            path_length=0.4,
        )
        self.assertAlmostEqual(bounds.metric_defect_bound, 0.4)
        self.assertAlmostEqual(bounds.metric_derivative_defect_bound, 1.4)
        self.assertAlmostEqual(bounds.connection_defect_bound, 11.0)
        self.assertAlmostEqual(bounds.dimensionless_connection_length, 4.8)
        self.assertAlmostEqual(bounds.condition_envelope, 8.0)
        self.assertAlmostEqual(bounds.isometric_transport_defect_bound, 35.2)
        self.assertEqual(
            bounds.transport_defect_bound,
            bounds.isometric_transport_defect_bound,
        )

    def test_lc_bounds_retain_gronwall_branch_for_very_short_paths(self) -> None:
        bounds = levi_civita_stability_bounds(
            first_derivative_bound=2.0,
            second_derivative_bound=3.0,
            metric_eigenvalue_floor=0.5,
            first_derivative_defect=0.1,
            second_derivative_defect=0.2,
            path_length=1e-5,
        )
        self.assertLess(
            bounds.gronwall_transport_defect_bound,
            bounds.isometric_transport_defect_bound,
        )
        self.assertEqual(
            bounds.transport_defect_bound,
            bounds.gronwall_transport_defect_bound,
        )

    def test_invalid_stability_inputs_are_rejected(self) -> None:
        valid = dict(
            first_derivative_bound=2.0,
            second_derivative_bound=3.0,
            metric_eigenvalue_floor=0.5,
            first_derivative_defect=0.1,
            second_derivative_defect=0.2,
            path_length=0.4,
        )
        for name, value in (
            ("first_derivative_bound", -1.0),
            ("second_derivative_bound", math.nan),
            ("metric_eigenvalue_floor", 0.0),
            ("path_length", math.inf),
        ):
            with self.subTest(name=name):
                inputs = {**valid, name: value}
                with self.assertRaises(ValueError):
                    levi_civita_stability_bounds(**inputs)

        inconsistent = {
            **valid,
            "first_derivative_bound": 0.5,
            "metric_eigenvalue_floor": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            levi_civita_stability_bounds(**inconsistent)


if __name__ == "__main__":
    unittest.main()
