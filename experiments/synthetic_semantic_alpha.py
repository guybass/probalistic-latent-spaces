"""Recover the connection generating a synthetic semantic vector field.

This is a numerical identifiability check, not empirical language-model
evidence. It runs with NumPy only.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from predictive_geometry.field import (
    alpha_parallel_transport,
    fit_semantic_alpha,
    softmax_probabilities,
)
from predictive_geometry.softmax import SoftmaxHessianGeometry


def main() -> None:
    rng = np.random.default_rng(20260804)
    vocabulary = 12
    dimension = 3
    weights = rng.normal(size=(vocabulary, dimension))
    bias = rng.normal(scale=0.2, size=vocabulary)

    report: dict[str, Any] = {}
    for label, true_alpha in {
        "exponential": 1.0,
        "levi_civita": 0.0,
        "mixture": -1.0,
        "intermediate": 0.35,
    }.items():
        base_samples = []
        for _ in range(24):
            hidden = rng.normal(scale=0.35, size=dimension)
            displacement_direction = rng.normal(size=dimension)
            source = rng.normal(size=dimension)
            base_samples.append((hidden, displacement_direction, source))

        scale_reports = []
        for edge_scale in (0.08, 0.04, 0.02, 0.01):
            geometries = []
            displacements = []
            source_vectors = []
            target_vectors = []
            refinement_errors = []
            for sample_index, (
                hidden,
                displacement_direction,
                source,
            ) in enumerate(base_samples):
                displacement = edge_scale * displacement_direction
                probabilities = softmax_probabilities(weights, bias, hidden)
                geometry = SoftmaxHessianGeometry(weights, probabilities)
                target = alpha_parallel_transport(
                    weights,
                    bias,
                    hidden,
                    hidden + displacement,
                    source,
                    alpha=true_alpha,
                    steps=64,
                )
                if true_alpha not in (-1.0, 1.0) and sample_index < 3:
                    coarse_target = alpha_parallel_transport(
                        weights,
                        bias,
                        hidden,
                        hidden + displacement,
                        source,
                        alpha=true_alpha,
                        steps=32,
                    )
                    refinement_errors.append(
                        float(np.linalg.norm(target - coarse_target))
                    )
                geometries.append(geometry)
                displacements.append(displacement)
                source_vectors.append(source)
                target_vectors.append(target)

            fit = fit_semantic_alpha(
                geometries,
                displacements,
                source_vectors,
                target_vectors,
            )
            scale_reports.append(
                {
                    "edge_scale": edge_scale,
                    "max_edge_fisher_length": fit.max_edge_fisher_length,
                    "mean_edge_fisher_length": float(
                        np.mean(fit.edge_fisher_lengths)
                    ),
                    "estimated_alpha": fit.alpha,
                    "absolute_error": abs(fit.alpha - true_alpha),
                    "excitation_ratio": fit.excitation_ratio,
                    "rk4_32_to_64_max_euclidean_difference": (
                        max(refinement_errors) if refinement_errors else 0.0
                    ),
                    "unrestricted_residual_squared": fit.residual_squared,
                    "exponential_residual_squared": (
                        fit.exponential_residual_squared
                    ),
                    "levi_civita_residual_squared": (
                        fit.levi_civita_residual_squared
                    ),
                    "mixture_residual_squared": fit.mixture_residual_squared,
                }
            )
        report[label] = {
            "true_alpha": true_alpha,
            "targets": "integrated alpha transport",
            "length_scale_reports": scale_reports,
        }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
