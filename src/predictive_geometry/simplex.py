"""Exact information-geometric operations on the open probability simplex.

The Fisher--Rao metric on categorical distributions is isometric, under
``p -> 2 * sqrt(p)``, to the positive orthant of a sphere of radius two.
This module implements the resulting Levi--Civita operations as well as the
flat mixture and exponential connections.
"""

from __future__ import annotations

import math
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
    if np.any(sphere_point <= 0.0):
        raise ValueError("the geodesic endpoint left the open square-root orthant")
    p = (sphere_point / FISHER_RADIUS) ** 2
    if np.any(p <= 0.0) or not np.all(np.isfinite(p)):
        raise ValueError("the Fisher endpoint is not representable in the open simplex")
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
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        terms = (u_vec * v_vec) / p_vec
    value = float(np.sum(terms))
    if not math.isfinite(value):
        raise OverflowError("Fisher inner product is not representable in float64")
    return value


def fisher_distance(p: ArrayLike, q: ArrayLike) -> float:
    """Fisher--Rao geodesic distance between categorical distributions."""

    p_vec = _probability(p, "p")
    q_vec = _probability(q, "q")
    if p_vec.shape != q_vec.shape:
        raise ValueError("p and q must have the same shape")
    chord = float(np.linalg.norm(sqrt_embed(p_vec) - sqrt_embed(q_vec)))
    half_angle_sine = np.clip(chord / (2.0 * FISHER_RADIUS), 0.0, 1.0)
    return float(2.0 * FISHER_RADIUS * np.arcsin(half_angle_sine))


def _sphere_log(x: Vector, y: Vector) -> Vector:
    radius_squared = FISHER_RADIUS**2
    chord = float(np.linalg.norm(x - y))
    half_angle_sine = float(
        np.clip(chord / (2.0 * FISHER_RADIUS), 0.0, 1.0)
    )
    theta = float(2.0 * np.arcsin(half_angle_sine))
    if theta == 0.0:
        return np.zeros_like(x)
    cosine = float(np.clip(1.0 - chord * chord / (2.0 * radius_squared), -1.0, 1.0))
    sine = float(np.sin(theta))
    if abs(sine) < 1e-14:
        raise ValueError("the spherical logarithm is undefined at antipodes")
    return (theta / sine) * (y - cosine * x)


def _enforce_zero_sum(vector: Vector, reference: Vector) -> Vector:
    """Remove roundoff from a tangent without perturbing every rare component."""

    corrected = np.array(vector, dtype=np.float64, copy=True)
    corrected[int(np.argmax(reference))] -= float(np.sum(corrected))
    return corrected


def _sphere_exp(x: Vector, v: Vector) -> Vector:
    with np.errstate(invalid="ignore", over="ignore"):
        norm = float(np.linalg.norm(v))
    if not math.isfinite(norm):
        raise ValueError("Fisher tangent norm is not representable in float64")
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
    with np.errstate(invalid="ignore", over="ignore"):
        transported = v - (float(np.dot(v, y)) / denominator) * (x + y)
    if not np.all(np.isfinite(transported)):
        raise ValueError("Fisher transport is not representable in float64")
    return transported


def fisher_log(p: ArrayLike, q: ArrayLike) -> Vector:
    """Levi--Civita logarithmic map ``Log_p(q)`` in probability coordinates."""

    p_vec = _probability(p, "p")
    q_vec = _probability(q, "q")
    if p_vec.shape != q_vec.shape:
        raise ValueError("p and q must have the same shape")
    sphere_tangent = _sphere_log(sqrt_embed(p_vec), sqrt_embed(q_vec))
    tangent = np.sqrt(p_vec) * sphere_tangent
    return _enforce_zero_sum(tangent, p_vec)


def fisher_exp(p: ArrayLike, u: ArrayLike) -> Vector:
    """Levi--Civita exponential map ``Exp_p(u)``."""

    p_vec = _probability(p)
    u_vec = _tangent(p_vec, u)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sphere_tangent = u_vec / np.sqrt(p_vec)
    if not np.all(np.isfinite(sphere_tangent)):
        raise ValueError("Fisher tangent is not representable in float64 sphere coordinates")
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
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sphere_tangent = u_vec / np.sqrt(p_vec)
    if not np.all(np.isfinite(sphere_tangent)):
        raise ValueError("Fisher tangent is not representable in float64 sphere coordinates")
    transported_sphere = _sphere_parallel_transport(
        sqrt_embed(p_vec), sqrt_embed(q_vec), sphere_tangent
    )
    transported = np.sqrt(q_vec) * transported_sphere
    return _enforce_zero_sum(transported, q_vec)


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
    anchor = int(np.argmax(p_vec))
    shifted_score = log_ratio - log_ratio[anchor]
    centered_score = shifted_score - float(np.dot(p_vec, shifted_score))
    tangent = p_vec * centered_score
    return _enforce_zero_sum(tangent, p_vec)


def exponential_exp(p: ArrayLike, u: ArrayLike) -> Vector:
    """Exponential map for the exponential connection."""

    p_vec = _probability(p)
    u_vec = _tangent(p_vec, u)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        score = u_vec / p_vec
    if not np.all(np.isfinite(score)):
        raise ValueError("exponential score is not representable in float64")
    logits = np.log(p_vec) + score
    logits -= np.max(logits)
    weights = np.exp(logits)
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("exponential endpoint is not representable in the open simplex")
    endpoint = weights / np.sum(weights)
    if np.any(endpoint <= 0.0):
        raise ValueError("exponential endpoint is not representable in the open simplex")
    return endpoint


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
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        score = u_vec / p_vec
    if not np.all(np.isfinite(score)):
        raise ValueError("exponential score is not representable in float64")
    anchor = int(np.argmax(q_vec))
    shifted_score = score - score[anchor]
    centered_score = shifted_score - float(np.dot(q_vec, shifted_score))
    transported = q_vec * centered_score
    return _enforce_zero_sum(transported, q_vec)


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
