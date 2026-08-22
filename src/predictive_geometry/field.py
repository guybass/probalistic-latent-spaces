"""Connection diagnostics for vector fields in a linear-softmax family.

The hidden coordinates of ``softmax(W @ h + b)`` are natural parameters.
Consequently, the exponential connection has zero coefficients, while the
Amari alpha-connection has

    Gamma^(alpha)(a, v) = (1 - alpha) / 2 * G^{-1} C(a, v, .).

This module supplies the small-edge estimator used by the MSc proposal and a
reference RK4 implementation of transport along a straight natural-coordinate
path.  It deliberately uses dense float64 linear algebra: the intended primary
model is Pythia-14M, whose decoded hidden dimension is only 128.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .softmax import SoftmaxHessianGeometry

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class SemanticConnectionFit:
    """Least-squares alpha-connection fit to local vector-field edges.

    ``alpha=1`` is the exponential connection, ``alpha=0`` is Levi--Civita,
    and ``alpha=-1`` is the mixture connection.  The fit is not clipped to this
    interval; an estimate outside it is useful evidence that none of the three
    canonical connections describes the observed field locally.
    """

    alpha: float
    beta: float
    residual_squared: float
    exponential_residual_squared: float
    levi_civita_residual_squared: float
    mixture_residual_squared: float
    explained_fraction: float
    denominator: float
    excitation_ratio: float
    edge_count: int
    edge_fisher_lengths: tuple[float, ...]
    max_edge_fisher_length: float


def softmax_probabilities(
    unembedding: ArrayLike,
    bias: ArrayLike,
    hidden: ArrayLike,
) -> Vector:
    """Evaluate a linear softmax decoder in stable float64 arithmetic."""

    weights, bias_vec, hidden_vec = _decoder_inputs(unembedding, bias, hidden)
    logits = weights @ hidden_vec + bias_vec
    logits -= np.max(logits)
    unnormalized = np.exp(logits)
    return unnormalized / np.sum(unnormalized)


def first_order_alpha_transport(
    geometry: SoftmaxHessianGeometry,
    displacement: ArrayLike,
    vector: ArrayLike,
    *,
    alpha: float,
) -> Vector:
    """First-order transport along ``h -> h + displacement``.

    The returned vector is ``v - beta * A_displacement v``, where
    ``beta=(1-alpha)/2``.  Its error relative to integrated transport is second
    order in the displacement.
    """

    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    delta = _vector(displacement, geometry.hidden_dim, "displacement")
    value = _vector(vector, geometry.hidden_dim, "vector")
    beta = 0.5 * (1.0 - float(alpha))
    return value - beta * geometry.cubic_action(delta, value)


def local_transport_defect_scalar(
    geometry: SoftmaxHessianGeometry,
    context_direction: ArrayLike,
    semantic_direction: ArrayLike,
) -> float:
    """Return the dimensionless local exponential-vs-LC transport defect.

    This is

    ``0.5 ||G^{-1} C(a,v,.)||_G / (||a||_G ||v||_G)``.

    It is the first-order relative discrepancy per unit Fisher context distance
    between keeping a hidden vector constant and Levi--Civita transporting it.
    """

    a = _vector(context_direction, geometry.hidden_dim, "context_direction")
    v = _vector(semantic_direction, geometry.hidden_dim, "semantic_direction")
    ga = float(a @ geometry.metric @ a)
    gv = float(v @ geometry.metric @ v)
    if ga <= 0.0 or gv <= 0.0:
        raise ValueError("directions must have positive Fisher norm")
    covector = geometry.cubic_contraction(a, v)
    raised = geometry.solve_metric(covector)
    squared = float(covector @ raised)
    # Roundoff can produce a tiny negative value for a theoretically nonnegative
    # quadratic form.
    squared = max(squared, 0.0)
    return 0.5 * np.sqrt(squared / (ga * gv))


def fit_semantic_alpha(
    source_geometries: Sequence[SoftmaxHessianGeometry],
    displacements: Sequence[ArrayLike],
    source_vectors: Sequence[ArrayLike],
    target_vectors: Sequence[ArrayLike],
    *,
    excitation_rtol: float = 1e-8,
) -> SemanticConnectionFit:
    """Fit the alpha-connection that makes a sampled field most parallel.

    On edge ``i -> j``, put ``Delta V = V_j - V_i`` and
    ``B = A_(h_j-h_i) V_i``.  The first-order transport residual is

    ``Delta V + beta B``,  ``beta=(1-alpha)/2``.

    This function minimizes the pooled source-Fisher squared residual and has
    the closed-form solution

    ``beta_hat = -sum <Delta V,B>_G / sum ||B||_G^2``.

    This is an infinitesimal estimator.  Relative to integrated finite-edge
    transport its alpha error is generically first order in edge length.  The
    returned maximum source-Fisher edge length must therefore be reported, and
    scientific use requires a length-to-zero extrapolation or an integrated fit.
    """

    if not np.isfinite(excitation_rtol) or excitation_rtol <= 0.0:
        raise ValueError("excitation_rtol must be finite and positive")
    count = len(source_geometries)
    if count == 0:
        raise ValueError("at least one edge is required")
    if not (
        len(displacements)
        == len(source_vectors)
        == len(target_vectors)
        == count
    ):
        raise ValueError("all edge collections must have the same length")

    numerator = 0.0
    denominator = 0.0
    baseline = 0.0
    excitation_scale = 0.0
    edge_fisher_lengths: list[float] = []
    edge_terms: list[tuple[Matrix, Vector, Vector]] = []
    for geometry, displacement, source, target in zip(
        source_geometries,
        displacements,
        source_vectors,
        target_vectors,
    ):
        delta = _vector(displacement, geometry.hidden_dim, "displacement")
        source_vec = _vector(source, geometry.hidden_dim, "source_vector")
        target_vec = _vector(target, geometry.hidden_dim, "target_vector")
        field_change = target_vec - source_vec
        connection_term = geometry.cubic_action(delta, source_vec)
        metric = geometry.metric
        edge_squared = float(delta @ metric @ delta)
        edge_fisher_lengths.append(float(np.sqrt(max(edge_squared, 0.0))))
        source_squared = float(source_vec @ metric @ source_vec)
        numerator += float(field_change @ metric @ connection_term)
        denominator += float(connection_term @ metric @ connection_term)
        baseline += float(field_change @ metric @ field_change)
        excitation_scale += max(edge_squared, 0.0) * max(source_squared, 0.0)
        edge_terms.append((metric, field_change, connection_term))

    if (
        not np.isfinite(denominator)
        or not np.isfinite(excitation_scale)
        or denominator <= np.finfo(np.float64).tiny
        or excitation_scale <= np.finfo(np.float64).tiny
    ):
        raise ValueError(
            "alpha is not identifiable: all cubic connection terms vanish"
        )
    excitation_ratio = denominator / excitation_scale
    if not np.isfinite(excitation_ratio) or excitation_ratio < excitation_rtol:
        raise ValueError(
            "alpha is weakly identified: dimensionless cubic excitation "
            f"{excitation_ratio:.3e} is below {excitation_rtol:.3e}"
        )
    beta = -numerator / denominator

    def pooled_residual(candidate_beta: float) -> float:
        residual = 0.0
        for metric, field_change, connection_term in edge_terms:
            error = field_change + candidate_beta * connection_term
            residual += float(error @ metric @ error)
        return residual

    residual = pooled_residual(beta)
    levi_civita_residual = pooled_residual(0.5)
    mixture_residual = pooled_residual(1.0)
    explained = (
        1.0 - residual / baseline
        if baseline > np.finfo(np.float64).tiny
        else float("nan")
    )
    return SemanticConnectionFit(
        alpha=1.0 - 2.0 * beta,
        beta=beta,
        residual_squared=residual,
        exponential_residual_squared=baseline,
        levi_civita_residual_squared=levi_civita_residual,
        mixture_residual_squared=mixture_residual,
        explained_fraction=explained,
        denominator=denominator,
        excitation_ratio=excitation_ratio,
        edge_count=count,
        edge_fisher_lengths=tuple(edge_fisher_lengths),
        max_edge_fisher_length=max(edge_fisher_lengths),
    )


def alpha_parallel_transport(
    unembedding: ArrayLike,
    bias: ArrayLike,
    start: ArrayLike,
    end: ArrayLike,
    vector: ArrayLike,
    *,
    alpha: float,
    steps: int = 32,
    eigenvalue_rtol: float = 1e-12,
) -> Vector:
    """Transport a vector along the straight natural-coordinate segment.

    Exponential transport (``alpha=1``) and mixture transport (``alpha=-1``)
    use their exact closed forms.  Other alpha values use fixed-step RK4 on

    ``dV/dt = -(1-alpha)/2 A_(end-start) V``.

    The path is part of the definition: non-flat alpha-connections can produce
    different answers along different paths.
    """

    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    weights, bias_vec, start_vec = _decoder_inputs(unembedding, bias, start)
    end_vec = _vector(end, weights.shape[1], "end")
    value = _vector(vector, weights.shape[1], "vector").copy()
    displacement = end_vec - start_vec

    if float(alpha) == 1.0:
        return value

    start_geometry = SoftmaxHessianGeometry(
        weights,
        softmax_probabilities(weights, bias_vec, start_vec),
        eigenvalue_rtol=eigenvalue_rtol,
    )
    if float(alpha) == -1.0:
        end_geometry = SoftmaxHessianGeometry(
            weights,
            softmax_probabilities(weights, bias_vec, end_vec),
            eigenvalue_rtol=eigenvalue_rtol,
        )
        return end_geometry.solve_metric(start_geometry.metric @ value)

    beta = 0.5 * (1.0 - float(alpha))
    step_size = 1.0 / steps

    def derivative(time: float, transported: Vector) -> Vector:
        hidden = start_vec + time * displacement
        geometry = SoftmaxHessianGeometry(
            weights,
            softmax_probabilities(weights, bias_vec, hidden),
            eigenvalue_rtol=eigenvalue_rtol,
        )
        return -beta * geometry.cubic_action(displacement, transported)

    for index in range(steps):
        time = index * step_size
        k1 = derivative(time, value)
        k2 = derivative(time + 0.5 * step_size, value + 0.5 * step_size * k1)
        k3 = derivative(time + 0.5 * step_size, value + 0.5 * step_size * k2)
        k4 = derivative(time + step_size, value + step_size * k3)
        value = value + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return value


def _decoder_inputs(
    unembedding: ArrayLike,
    bias: ArrayLike,
    hidden: ArrayLike,
) -> tuple[Matrix, Vector, Vector]:
    weights = np.asarray(unembedding, dtype=np.float64)
    bias_vec = np.asarray(bias, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("unembedding must have shape (vocabulary, hidden_dim)")
    if bias_vec.ndim != 1 or bias_vec.shape[0] != weights.shape[0]:
        raise ValueError("bias must have one entry per vocabulary row")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias_vec)):
        raise ValueError("decoder parameters must be finite")
    hidden_vec = _vector(hidden, weights.shape[1], "hidden")
    return weights, bias_vec, hidden_vec


def _vector(value: ArrayLike, dimension: int, name: str) -> Vector:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.shape[0] != dimension:
        raise ValueError(f"{name} must have shape ({dimension},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result
