"""Run a small, model-free validation of the three simplex connections."""

from __future__ import annotations

import numpy as np

from predictive_geometry.simplex import (
    analogy,
    fisher_distance,
    fisher_inner,
    fisher_log,
    transport_around_loop,
    triangle_holonomy_angle,
)


def main() -> None:
    p = np.array([0.50, 0.30, 0.20])
    q = np.array([0.42, 0.38, 0.20])
    r = np.array([0.44, 0.29, 0.27])

    print("Analogy: apply p -> q at r")
    predictions: dict[str, np.ndarray] = {}
    for connection in ("mixture", "exponential", "fisher"):
        predictions[connection] = analogy(p, q, r, connection)
        print(f"  {connection:11s}: {predictions[connection]}")

    print("\nPairwise Fisher distances between predictions")
    names = list(predictions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            distance = fisher_distance(predictions[left], predictions[right])
            print(f"  {left:11s} vs {right:11s}: {distance:.8f}")

    initial = fisher_log(p, q)
    returned = transport_around_loop([p, q, r, p], initial, "fisher")
    cosine = fisher_inner(p, initial, returned) / np.sqrt(
        fisher_inner(p, initial, initial) * fisher_inner(p, returned, returned)
    )
    observed = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    predicted = triangle_holonomy_angle(p, q, r)

    print("\nFisher triangle holonomy")
    print(f"  observed transport angle : {observed:.12f}")
    print(f"  spherical-excess formula : {predicted:.12f}")
    print(f"  absolute error           : {abs(observed - predicted):.3e}")


if __name__ == "__main__":
    main()
