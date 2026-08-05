"""Recover the connection generating a synthetic semantic vector field.

This is a numerical identifiability check, not empirical language-model
evidence. It runs with NumPy only.
"""

from __future__ import annotations

import json

import numpy as np

from predictive_geometry.field import (
    first_order_alpha_transport,
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

    report: dict[str, dict[str, float]] = {}
    for label, true_alpha in {
        "exponential": 1.0,
        "levi_civita": 0.0,
        "mixture": -1.0,
        "intermediate": 0.35,
    }.items():
        geometries = []
        displacements = []
        source_vectors = []
        target_vectors = []
        for _ in range(80):
            hidden = rng.normal(scale=0.35, size=dimension)
            probabilities = softmax_probabilities(weights, bias, hidden)
            geometry = SoftmaxHessianGeometry(weights, probabilities)
            displacement = rng.normal(scale=0.02, size=dimension)
            source = rng.normal(size=dimension)
            target = first_order_alpha_transport(
                geometry,
                displacement,
                source,
                alpha=true_alpha,
            )
            target += rng.normal(scale=2e-5, size=dimension)
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
        report[label] = {
            "true_alpha": true_alpha,
            "estimated_alpha": fit.alpha,
            "absolute_error": abs(fit.alpha - true_alpha),
            "explained_fraction": fit.explained_fraction,
        }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
