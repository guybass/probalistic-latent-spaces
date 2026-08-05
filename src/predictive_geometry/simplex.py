"""Exact information-geometric operations on the open probability simplex.

The Fisher--Rao metric on categorical distributions is isometric, under
``p -> 2 * sqrt(p)``, to the positive orthant of a sphere of radius two.
This module implements the resulting Levi--Civita operations as well as the
flat mixture and exponential connections.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vector = NDArray[np.float64]
FISHER_RADIUS = 2.0
_ATOL = 1e-10


def _as_vector(value: ArrayLike, name: str) -> Vector:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _probability(value: ArrayLike, name: str = "p") -> Vector:
    p = _as_vector(value, name)
    if np.any(p <= 0.0):
        raise ValueError(f"{name} must lie in the open probability simplex")
    if not np.isclose(float(np.sum(p)), 1.0, atol=_ATOL, rtol=0.0):
        raise ValueError(f"{name} must sum to one")
    return p


def _tangent(p: Vector, value: ArrayLike, name: str = "u") -> Vector:
    u = _as_vector(value, name)
    if u.shape != p.shape:
        raise ValueError(f"{name} must have the same shape as the base point")
    if not np.isclose(float(np.sum(u)), 0.0, atol=_ATOL, rtol=0.0):
        raise ValueError(f"{name} must sum to zero")
    return u


def sqrt_embed(p: ArrayLike) -> Vector:
    """Return the Fisher-isometric radius-two sphere representation."""

    p_vec = _probability(p)
    return FISHER_RADIUS * np.sqrt(p_vec)


def _sphere_to_probability(x: ArrayLike) -> Vector:
    sphere_point = _as_vector(x, "x")
    if not np.isclose(
        float(np.linalg.norm(sphere_point)),
        FISHER_RADIUS,
        atol=1e-8,
        rtol=1e-8,
    ):
        raise ValueError("x must lie on the radius-two sphere")
    if np.any(sphere_point < -1e-9):
        raise ValueError("the geodesic left the positive square-root orthant")
    clipped = np.maximum(sphere_point, 0.0)
    p = (clipped / FISHER_RADIUS) ** 2
    return p / np.sum(p)


def bhattacharyya_coefficient(p: ArrayLike, q: ArrayLike) -> float:
    """Return ``sum_i sqrt(p_i q_i)``."""

    p_vec = _probability(p, "p")
    q_vec = _probability(q, "q")
    if p_vec.shape != q_vec.shape:
        raise ValueError("p and q must have the same shape")
    return float(np.dot(np.sqrt(p_vec), np.sqrt(q_vec)))


def fisher_inner(p: ArrayLike, u: ArrayLike, v: ArrayLike) -> float:
    """Fisher--Rao tangent inner product at ``p``."""

    p_vec = _probability(p)
    u_vec = _tangent(p_vec, u, "u")
    v_vec = _tangent(p_vec, v, "v")
    return float(np.dot(u_vec / p_vec, v_vec))


def fisher_distance(p: ArrayLike, q: ArrayLike) -> float:
    """Fisher--Rao geodesic distance between categorical distributions."""

    coefficient = np.clip(bhattacharyya_coefficient(p, q), -1.0, 1.0)
    return float(FISHER_RADIUS * np.arccos(coefficient))


def _sphere_log(x: Vector, y: Vector) -> Vector:
    radius_squared = FISHER_RADIUS**2
    cosine = float(np.clip(np.dot(x, y) / radius_squared, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    if theta < 1e-14:
        return np.zeros_like(x)
    sine = float(np.sin(theta))
    if abs(sine) < 1e-14:
        raise ValueError("the spherical logarithm is undefined at antipodes")
    return (theta / sine) * (y - cosine * x)


def _sphere_exp(x: Vector, v: Vector) -> Vector:
    norm = float(np.linalg.norm(v))
    if norm < 1e-14:
        return x.copy()
    angle = norm / FISHER_RADIUS
    return (
        np.cos(angle) * x
        + FISHER_RADIUS * np.sin(angle) * (v / norm)
    )


def _sphere_parallel_transport(x: Vector, y: Vector, v: Vector) -> Vector:
    denominator = FISHER_RADIUS**2 + float(np.dot(x, y))
    if abs(denominator) < 1e-14:
        raise ValueError("minimal-geodesic transport is undefined at antipodes")
    return v - (float(np.dot(v, y)) / denominator) * (x + y)


def fisher_log(p: ArrayLike, q: ArrayLike) -> Vector:
    """Levi--Civita logarithmic map ``Log_p(q)`` in probability coordinates."""

    p_vec = _probability(p, "p")
    q_vec = _probability(q, "q")
    if p_vec.shape != q_vec.shape:
        raise ValueError("p and q must have the same shape")
    sphere_tangent = _sphere_log(sqrt_embed(p_vec), sqrt_embed(q_vec))
    tangent = np.sqrt(p_vec) * sphere_tangent
    tangent -= np.sum(tangent) / tangent.size
    return tangent


def fisher_exp(p: ArrayLike, u: ArrayLike) -> Vector:
    """Levi--Civita exponential map ``Exp_p(u)``."""

    p_vec = _probability(p)
    u_vec = _tangent(p_vec, u)
    sphere_tangent = u_vec / np.sqrt(p_vec)
    endpoint = _sphere_exp(sqrt_embed(p_vec), sphere_tangent)
    return _sphere_to_probability(endpoint)


def fisher_parallel_transport(
    p: ArrayLike,
    q: ArrayLike,
    u: ArrayLike,
) -> Vector:
    """Levi--Civita transport along the minimal Fisher geodesic ``p -> q``."""

    p_vec = _probability(p, "p")
    q_vec = _probability(q, "q")
    if p_vec.shape != q_vec.shape:
        raise ValueError("p and q must have the same shape")
    u_vec = _tangent(p_vec, u)
    transported_sphere = _sphere_parallel_transport(
        sqrt_embed(p_vec),
        sqrt_embed(q_vec),
        u_vec / np.sqrt(p_vec),
    )
    transported = np.sqrt(q_vec) * transported_sphere
    transported -= np.sum(transported) / transported.size
    return transported


def mixture_log(p: ArrayLike, q: ArrayLike) -> Vector:
    """Logarithmic map for the flat mixture connection."""

    p_vec = _probability(p, "p")
    q_vec = _probability(q, "q")
    if p_vec.shape != q_vec.shape:
        raise ValueError("p and q must have the same shape")
    return q_vec - p_vec


def mixture_exp(p: ArrayLike, u: ArrayLike) -> Vector:
    """Exponential map for the mixture connection, when ``p + u`` is positive."""

    p_vec = _probability(p)
    u_vec = _tangent(p_vec, u)
    return _probability(p_vec + u_vec, "p + u")


def mixture_parallel_transport(
    p: ArrayLike,
    q: ArrayLike,
    u: ArrayLike,
) -> Vector:
    """Path-independent mixture transport in probability coordinates."""

    p_vec = _probability(p, "p")
    q_vec = _probability(q, "q")
    if p_vec.shape != q_vec.shape:
        raise ValueError("p and q must have the same shape")
    return _tangent(p_vec, u).copy()


def exponential_log(p: ArrayLike, q: ArrayLike) -> Vector:
    """Logarithmic map for the flat exponential connection."""

    p_vec = _probability(p, "p")
    q_vec = _probability(q, "q")
    if p_vec.shape != q_vec.shape:
        raise ValueError("p and q must have the same shape")
    log_ratio = np.log(q_vec) - np.log(p_vec)
    centered_score = log_ratio - float(np.dot(p_vec, log_ratio))
    tangent = p_vec * centered_score
    tangent -= np.sum(tangent) / tangent.size
    return tangent


def exponential_exp(p: ArrayLike, u: ArrayLike) -> Vector:
    """Exponential map for the exponential connection."""

    p_vec = _probability(p)
    u_vec = _tangent(p_vec, u)
    score = u_vec / p_vec
    logits = np.log(p_vec) + score
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / np.sum(weights)


def exponential_parallel_transport(
    p: ArrayLike,
    q: ArrayLike,
    u: ArrayLike,
) -> Vector:
    """Path-independent exponential transport using centered score coordinates."""

    p_vec = _probability(p, "p")
    q_vec = _probability(q, "q")
    if p_vec.shape != q_vec.shape:
        raise ValueError("p and q must have the same shape")
    u_vec = _tangent(p_vec, u)
    score = u_vec / p_vec
    transported = q_vec * (score - float(np.dot(q_vec, score)))
    transported -= np.sum(transported) / transported.size
    return transported


def analogy(
    p: ArrayLike,
    q: ArrayLike,
    r: ArrayLike,
    connection: str = "fisher",
) -> Vector:
    """Apply the displacement ``p -> q`` at base point ``r``.

    ``connection`` may be ``"mixture"``, ``"exponential"``, or ``"fisher"``.
    The mixture result is the familiar ``r + q - p`` when it remains positive.
    """

    operations = {
        "mixture": (mixture_log, mixture_parallel_transport, mixture_exp),
        "exponential": (
            exponential_log,
            exponential_parallel_transport,
            exponential_exp,
        ),
        "fisher": (fisher_log, fisher_parallel_transport, fisher_exp),
    }
    try:
        log_map, transport, exp_map = operations[connection]
    except KeyError as error:
        choices = ", ".join(sorted(operations))
        raise ValueError(f"connection must be one of: {choices}") from error
    displacement = log_map(p, q)
    moved_displacement = transport(p, r, displacement)
    return exp_map(r, moved_displacement)


def transport_around_loop(
    points: Iterable[ArrayLike],
    u: ArrayLike,
    connection: str = "fisher",
) -> Vector:
    """Transport ``u`` around a closed sequence whose final point equals its first."""

    path = [_probability(point, f"points[{index}]") for index, point in enumerate(points)]
    if len(path) < 2:
        raise ValueError("a loop needs at least two points")
    if not np.allclose(path[0], path[-1], atol=_ATOL, rtol=0.0):
        raise ValueError("the final loop point must equal the initial point")
    if any(point.shape != path[0].shape for point in path[1:]):
        raise ValueError("all loop points must have the same shape")

    transports = {
        "mixture": mixture_parallel_transport,
        "exponential": exponential_parallel_transport,
        "fisher": fisher_parallel_transport,
    }
    try:
        transport = transports[connection]
    except KeyError as error:
        choices = ", ".join(sorted(transports))
        raise ValueError(f"connection must be one of: {choices}") from error

    vector = _tangent(path[0], u).copy()
    for start, end in zip(path[:-1], path[1:], strict=True):
        vector = transport(start, end, vector)
    return vector


def triangle_holonomy_angle(
    p: ArrayLike,
    q: ArrayLike,
    r: ArrayLike,
) -> float:
    """Unsigned Levi--Civita holonomy angle of a minimal geodesic triangle.

    The three square-root points span a great two-sphere.  Its spherical excess,
    and therefore its holonomy angle, is determined by the Gram matrix of the
    three unit square-root vectors.
    """

    p_vec = _probability(p, "p")
    q_vec = _probability(q, "q")
    r_vec = _probability(r, "r")
    if p_vec.shape != q_vec.shape or p_vec.shape != r_vec.shape:
        raise ValueError("p, q, and r must have the same shape")

    a = bhattacharyya_coefficient(p_vec, q_vec)
    b = bhattacharyya_coefficient(q_vec, r_vec)
    c = bhattacharyya_coefficient(r_vec, p_vec)
    gram_determinant = 1.0 + 2.0 * a * b * c - a * a - b * b - c * c
    numerator = np.sqrt(max(0.0, gram_determinant))
    denominator = 1.0 + a + b + c
    return float(2.0 * np.arctan2(numerator, denominator))
