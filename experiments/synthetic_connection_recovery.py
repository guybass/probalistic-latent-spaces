"""Show that held-out quadrilaterals can identify their generating connection."""

from __future__ import annotations

import json

import numpy as np

from predictive_geometry.benchmark import evaluate_quadrilateral
from predictive_geometry.simplex import analogy


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
        "mixture_generated": analogy(p00, p10, p01, "mixture"),
        "exponential_generated": analogy(p00, p10, p01, "exponential"),
        "fisher_generated_A_along_B": analogy(p00, p10, p01, "fisher"),
    }
    output = {
        name: compact(evaluate_quadrilateral(p00, p10, p01, target))
        for name, target in targets.items()
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
