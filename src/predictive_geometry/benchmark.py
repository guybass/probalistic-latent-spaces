"""Connection-selection benchmark for semantic probability quadrilaterals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .simplex import (
    analogy,
    exponential_log,
    exponential_parallel_transport,
    fisher_distance,
    fisher_inner,
    fisher_log,
    fisher_parallel_transport,
    mixture_log,
    mixture_parallel_transport,
)

Vector = NDArray[np.float64]
CONNECTIONS = ("mixture", "exponential", "fisher")


@dataclass(frozen=True)
class CompositionResult:
    """One connection's prediction for one orientation of a 2x2 design."""

    connection: str
    orientation: str
    feasible: bool
    failure_reason: str | None
    fisher_distance: float | None
    kl_target_to_prediction: float | None
    kl_prediction_to_target: float | None
    jensen_shannon: float | None
    top1_match: bool | None
    transport_log_length_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probability(value: ArrayLike, name: str) -> Vector:
    p = np.asarray(value, dtype=np.float64)
    if p.ndim != 1 or not np.all(np.isfinite(p)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    if np.any(p <= 0.0):
        raise ValueError(f"{name} must lie in the open simplex")
    if not np.isclose(float(np.sum(p)), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must sum to one")
    return p


def kl_divergence(p: ArrayLike, q: ArrayLike) -> float:
    """Return ``D_KL(p || q)`` for two strictly positive distributions."""

    p_vec = _probability(p, "p")
    q_vec = _probability(q, "q")
    if p_vec.shape != q_vec.shape:
        raise ValueError("p and q must have the same shape")
    return float(np.dot(p_vec, np.log(p_vec) - np.log(q_vec)))


def jensen_shannon_divergence(p: ArrayLike, q: ArrayLike) -> float:
    """Return the equal-weight Jensen--Shannon divergence in nats."""

    p_vec = _probability(p, "p")
    q_vec = _probability(q, "q")
    if p_vec.shape != q_vec.shape:
        raise ValueError("p and q must have the same shape")
    midpoint = 0.5 * (p_vec + q_vec)
    return 0.5 * (
        kl_divergence(p_vec, midpoint) + kl_divergence(q_vec, midpoint)
    )


def predict_composition(
    p00: ArrayLike,
    p10: ArrayLike,
    p01: ArrayLike,
    *,
    connection: str,
    orientation: str = "A_along_B",
) -> Vector:
    """Predict ``p11`` from three corners of a semantic factorial design.

    ``A_along_B`` transports the observed effect ``p00 -> p10`` to ``p01``.
    ``B_along_A`` transports ``p00 -> p01`` to ``p10``.
    """

    if orientation == "A_along_B":
        return analogy(p00, p10, p01, connection)
    if orientation == "B_along_A":
        return analogy(p00, p01, p10, connection)
    raise ValueError("orientation must be 'A_along_B' or 'B_along_A'")


def evaluate_composition(
    p00: ArrayLike,
    p10: ArrayLike,
    p01: ArrayLike,
    p11: ArrayLike,
    *,
    connection: str,
    orientation: str,
) -> tuple[CompositionResult, Vector | None]:
    """Predict the held-out corner and score it without silently projecting failures."""

    base = _probability(p00, "p00")
    effect_a = _probability(p10, "p10")
    effect_b = _probability(p01, "p01")
    target = _probability(p11, "p11")
    if not (
        base.shape == effect_a.shape == effect_b.shape == target.shape
    ):
        raise ValueError("all four corners must have the same shape")

    if orientation == "A_along_B":
        effect, new_base = effect_a, effect_b
    elif orientation == "B_along_A":
        effect, new_base = effect_b, effect_a
    else:
        raise ValueError("orientation must be 'A_along_B' or 'B_along_A'")
    transport_operations = {
        "mixture": (mixture_log, mixture_parallel_transport),
        "exponential": (exponential_log, exponential_parallel_transport),
        "fisher": (fisher_log, fisher_parallel_transport),
    }
    try:
        log_map, transport = transport_operations[connection]
    except KeyError as error:
        choices = ", ".join(sorted(transport_operations))
        raise ValueError(f"connection must be one of: {choices}") from error
    displacement = log_map(base, effect)
    moved = transport(base, new_base, displacement)
    source_length_squared = fisher_inner(base, displacement, displacement)
    target_length_squared = fisher_inner(new_base, moved, moved)
    length_distortion = (
        None
        if source_length_squared <= np.finfo(np.float64).tiny
        else 0.5 * float(
            np.log(target_length_squared / source_length_squared)
        )
    )
    try:
        prediction = predict_composition(
            p00,
            p10,
            p01,
            connection=connection,
            orientation=orientation,
        )
    except ValueError as error:
        return (
            CompositionResult(
                connection=connection,
                orientation=orientation,
                feasible=False,
                failure_reason=str(error),
                fisher_distance=None,
                kl_target_to_prediction=None,
                kl_prediction_to_target=None,
                jensen_shannon=None,
                top1_match=None,
                transport_log_length_ratio=length_distortion,
            ),
            None,
        )

    result = CompositionResult(
        connection=connection,
        orientation=orientation,
        feasible=True,
        failure_reason=None,
        fisher_distance=fisher_distance(target, prediction),
        kl_target_to_prediction=kl_divergence(target, prediction),
        kl_prediction_to_target=kl_divergence(prediction, target),
        jensen_shannon=jensen_shannon_divergence(target, prediction),
        top1_match=bool(np.argmax(target) == np.argmax(prediction)),
        transport_log_length_ratio=length_distortion,
    )
    return result, prediction


def evaluate_quadrilateral(
    p00: ArrayLike,
    p10: ArrayLike,
    p01: ArrayLike,
    p11: ArrayLike,
) -> dict[str, Any]:
    """Evaluate all three connections and both semantic path orientations."""

    results: list[dict[str, Any]] = []
    closure: dict[str, float | None] = {}
    for connection in CONNECTIONS:
        predictions: dict[str, Vector | None] = {}
        for orientation in ("A_along_B", "B_along_A"):
            result, prediction = evaluate_composition(
                p00,
                p10,
                p01,
                p11,
                connection=connection,
                orientation=orientation,
            )
            results.append(result.to_dict())
            predictions[orientation] = prediction
        left = predictions["A_along_B"]
        right = predictions["B_along_A"]
        closure[connection] = (
            None if left is None or right is None else fisher_distance(left, right)
        )
    return {"predictions": results, "orientation_closure": closure}
