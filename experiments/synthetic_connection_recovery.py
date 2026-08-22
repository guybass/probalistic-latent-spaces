"""Independent closed-form smoke fixtures for connection recovery.

Targets are generated from direct mixture, exponential, and radius-two sphere
formulas rather than the production ``analogy`` routine used by the evaluator.
This remains a synthetic regression test, not empirical semantic validation.
"""

from __future__ import annotations

import json

import numpy as np

from predictive_geometry.benchmark import evaluate_quadrilateral


def normalize(weights: np.ndarray) -> np.ndarray:
    if np.any(weights <= 0.0):
        raise ValueError("closed-form target left the open simplex")
    return weights / weights.sum()


def fisher_target_closed(p00: np.ndarray, p10: np.ndarray, p01: np.ndarray) -> np.ndarray:
    radius = 2.0
    x = radius * np.sqrt(p00)
    y = radius * np.sqrt(p10)
    z = radius * np.sqrt(p01)
    chord = np.linalg.norm(x - y)
    theta = 2.0 * np.arcsin(np.clip(chord / (2.0 * radius), 0.0, 1.0))
    cosine = 1.0 - chord**2 / (2.0 * radius**2)
    sphere_log = (theta / np.sin(theta)) * (y - cosine * x)
    transported = sphere_log - (
        np.dot(sphere_log, z) / (radius**2 + np.dot(x, z))
    ) * (x + z)
    norm = np.linalg.norm(transported)
    endpoint = np.cos(norm / radius) * z + radius * np.sin(norm / radius) * (
        transported / norm
    )
    return normalize(np.square(endpoint / radius))


def compact(report: dict[str, object]) -> dict[str, object]:
    rows = report["predictions"]
    assert isinstance(rows, list)
    return {
        "fisher_distance": {
            f"{row['connection']}:{row['orientation']}": row["fisher_distance"]
            for row in rows
        },
        "orientation_closure": report["orientation_closure"],
    }


def main() -> None:
    p00 = np.array([0.50, 0.30, 0.20])
    p10 = np.array([0.43, 0.37, 0.20])
    p01 = np.array([0.46, 0.28, 0.26])

    targets = {
        "mixture_generated": normalize(p10 + p01 - p00),
        "exponential_generated": normalize(p10 * p01 / p00),
        "fisher_generated_A_along_B": fisher_target_closed(p00, p10, p01),
    }
    output = {
        "validation_scope": (
            "independent closed-form synthetic regression; not semantic evidence"
        ),
        "results": {
            name: compact(evaluate_quadrilateral(p00, p10, p01, target))
            for name, target in targets.items()
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
