"""Shared-chart connection packets for predictive distillation.

Implements the finite-dimensional core of
``PREDICTIVE_CONNECTION_DISTILLATION.md``: packet assembly ``(q, G, L, C)``
on a declared intervention chart, the distillation losses, acceptance gates,
checksummed serialization, and the algebraic packet-to-connection bound
(Section 11.2) together with the proven packet-to-transport bound
(Section 11.3).

The module is model-free. The compatibility entry point accepts a callable
``z -> q(z)`` and strictly validates the finite open simplex. The preferred
entry point accepts logits produced in float64 and optionally an exact logit
Jacobian assembled from JVPs. The cubic tensor uses the score-moment identity:
scores are logit derivatives recentered by the ``q``-weighted mean, which
removes the softmax gauge exactly, carries no inverse-probability factor, and
is exact for affine (log-linear) families. The probability-difference form is
retained only as an independent audit; it divides by ``q**2`` and may become
non-finite at numerical extremes without corrupting the primary estimator.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


SCHEMA_VERSION = "pcd-packet-2"

_TRAPEZOID = getattr(np, "trapezoid", None) or getattr(np, "trapz")

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
Tensor3 = NDArray[np.float64]

PredictiveMap = Callable[[Vector], Vector]
LogitMap = Callable[[Vector], Vector]
LogitJacobianMap = Callable[[Vector], Matrix]


@dataclass(frozen=True)
class ConnectionPacket:
    """Local predictive-connection packet at one chart point.

    ``first_kind`` stores the first-kind Levi--Civita coefficient
    ``L[i, j, k] = <d_ij psi, d_k psi>`` with ``psi = 2 sqrt(q)``; it is a
    pointwise function of the metric derivatives, so its independent content
    relative to a dense metric field is sampling density only.  ``cubic`` is
    the score-moment Amari--Chentsov tensor (primary estimator);
    ``cubic_audit`` is the probability-difference estimator retained as an
    independent audit and may be non-finite when the map was evaluated in a
    quantized dtype.
    """

    z: Vector
    step: float
    q: Vector
    metric: Matrix
    first_kind: Tensor3
    cubic: Tensor3
    cubic_audit: Tensor3
    metric_eigenvalues: Vector
    refinement_error: float
    excluded_outcomes: int
    accepted: bool
    rejection_reason: str
    serialization_quantization_error: float = 0.0
    schema_version: str = SCHEMA_VERSION


def _as_point(z: Sequence[float] | Vector) -> Vector:
    point = np.atleast_1d(np.asarray(z, dtype=np.float64))
    if point.ndim != 1:
        raise ValueError("chart point must be one-dimensional")
    return point


def _evaluate_probability_vector(
    q_fn: PredictiveMap,
    z: Vector,
    expected_size: int | None = None,
) -> Vector:
    """Evaluate and validate a map into the finite open simplex."""

    q = np.asarray(q_fn(z), dtype=np.float64)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("predictive map must return a nonempty one-dimensional vector")
    if expected_size is not None and q.size != expected_size:
        raise ValueError("predictive-map outcome dimension changed across the stencil")
    if not np.all(np.isfinite(q)):
        raise ValueError("predictive probabilities must be finite")
    if np.any(q <= 0.0):
        raise ValueError(
            "predictive probabilities must lie in the open simplex; use "
            "build_packet_from_logits to avoid silent probability underflow"
        )
    if not np.isclose(float(q.sum()), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("predictive probabilities must sum to one")
    return q


def _evaluate_logits(
    logits_fn: LogitMap,
    z: Vector,
    expected_size: int | None = None,
) -> Vector:
    """Evaluate declared float64 logits without silently upgrading a lower dtype."""

    raw = np.asarray(logits_fn(z))
    if raw.dtype != np.dtype(np.float64):
        raise TypeError(
            "logits_fn must return float64 logits from the model evaluation itself; "
            "casting lower-precision outputs afterward does not restore precision"
        )
    logits = np.asarray(raw, dtype=np.float64)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("logits_fn must return a nonempty one-dimensional vector")
    if expected_size is not None and logits.size != expected_size:
        raise ValueError("logit outcome dimension changed across the stencil")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must be finite")
    return logits


def _softmax_float64(logits: Vector) -> Vector:
    shifted = logits - float(np.max(logits))
    weights = np.exp(shifted)
    probabilities = weights / float(weights.sum())
    if np.any(probabilities <= 0.0):
        raise ValueError(
            "float64 softmax underflowed; reduce the outcome space with a declared "
            "error bound or supply a higher-precision implementation"
        )
    return probabilities


def _sqrt_map(q: Vector) -> Vector:
    return 2.0 * np.sqrt(q)


def _jet_tensors(
    q_fn: PredictiveMap,
    z: Vector,
    h: float,
    score_logits_fn: LogitMap | None = None,
    score_logit_jacobian_fn: LogitJacobianMap | None = None,
) -> tuple[Vector, Matrix, Tensor3, Tensor3, Tensor3, int]:
    """Return ``(q, G, L, C_score, C_audit, excluded)`` at step ``h``.

    ``G`` and ``L`` come from central differences of ``psi = 2 sqrt(q)``.
    ``C_score`` uses an exact logit Jacobian when supplied, otherwise central
    differences of logits or, for the probability-only compatibility path,
    central differences of ``log q``. Invalid or underflowed simplex values
    are rejected rather than silently excluded.
    """

    m = z.shape[0]
    q0 = _evaluate_probability_vector(q_fn, z)
    n = q0.shape[0]

    q_plus = np.empty((m, n))
    q_minus = np.empty((m, n))
    for i in range(m):
        unit = np.zeros(m)
        unit[i] = h
        q_plus[i] = _evaluate_probability_vector(q_fn, z + unit, n)
        q_minus[i] = _evaluate_probability_vector(q_fn, z - unit, n)

    psi0 = _sqrt_map(q0)
    d_psi = (_sqrt_map(q_plus) - _sqrt_map(q_minus)) / (2.0 * h)

    d2_psi = np.empty((m, m, n))
    for i in range(m):
        d2_psi[i, i] = (
            _sqrt_map(q_plus[i]) - 2.0 * psi0 + _sqrt_map(q_minus[i])
        ) / (h * h)
    for i in range(m):
        for j in range(i + 1, m):
            unit_i = np.zeros(m)
            unit_j = np.zeros(m)
            unit_i[i] = h
            unit_j[j] = h
            pp = _sqrt_map(_evaluate_probability_vector(q_fn, z + unit_i + unit_j, n))
            pm = _sqrt_map(_evaluate_probability_vector(q_fn, z + unit_i - unit_j, n))
            mp = _sqrt_map(_evaluate_probability_vector(q_fn, z - unit_i + unit_j, n))
            mm = _sqrt_map(_evaluate_probability_vector(q_fn, z - unit_i - unit_j, n))
            d2_psi[i, j] = (pp - pm - mp + mm) / (4.0 * h * h)
            d2_psi[j, i] = d2_psi[i, j]

    metric = np.einsum("ia,ja->ij", d_psi, d_psi)
    first_kind = np.einsum("ija,ka->ijk", d2_psi, d_psi)

    if score_logit_jacobian_fn is not None:
        raw_source = np.asarray(score_logit_jacobian_fn(z))
        if raw_source.dtype != np.dtype(np.float64):
            raise TypeError("score_logit_jacobian_fn must return float64 values")
        raw = np.asarray(raw_source, dtype=np.float64)
        if raw.shape != (m, n) or not np.all(np.isfinite(raw)):
            raise ValueError("logit Jacobian must be a finite (chart_dim, outcomes) array")
    elif score_logits_fn is not None:
        logit_plus = np.empty((m, n))
        logit_minus = np.empty((m, n))
        for i in range(m):
            unit = np.zeros(m)
            unit[i] = h
            logit_plus[i] = _evaluate_logits(score_logits_fn, z + unit, n)
            logit_minus[i] = _evaluate_logits(score_logits_fn, z - unit, n)
        raw = (logit_plus - logit_minus) / (2.0 * h)
    else:
        raw = (np.log(q_plus) - np.log(q_minus)) / (2.0 * h)

    gauge = raw @ q0
    scores = raw - gauge[:, None]
    cubic = np.einsum("a,ia,ja,ka->ijk", q0, scores, scores, scores)

    d_q = (q_plus - q_minus) / (2.0 * h)
    with np.errstate(divide="ignore", invalid="ignore"):
        cubic_audit = np.einsum(
            "ia,ja,ka,a->ijk", d_q, d_q, d_q, 1.0 / np.square(q0)
        )

    return q0, metric, first_kind, cubic, cubic_audit, 0


def _max_relative_difference(
    first: Sequence[NDArray[np.float64]],
    second: Sequence[NDArray[np.float64]],
) -> float:
    worst = 0.0
    for a, b in zip(first, second):
        scale = max(1.0, float(np.linalg.norm(b)))
        worst = max(worst, float(np.linalg.norm(a - b)) / scale)
    return worst


def build_packet(
    q_fn: PredictiveMap,
    z: Sequence[float] | Vector,
    step: float,
    *,
    metric_floor: float = 1e-10,
    refinement_rtol: float = 1e-3,
    score_logits_fn: LogitMap | None = None,
    score_logit_jacobian_fn: LogitJacobianMap | None = None,
) -> ConnectionPacket:
    """Assemble and gate a connection packet at one chart point.

    Tensors are computed at steps ``(h, h/2, h/4)`` and combined by
    Richardson extrapolation of the ``O(h^2)`` estimators; the packet is
    accepted only when the two extrapolants agree to ``refinement_rtol`` in
    relative norm (a formal acceptance rule, not a visually judged plateau),
    all primary tensors are finite, and the metric clears ``metric_floor``.
    Rejected packets retain their values and a ``rejection_reason`` so the
    audit log can report acceptance coverage.
    """

    point = _as_point(z)
    h = float(step)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("step must be finite and positive")
    if not math.isfinite(metric_floor) or metric_floor <= 0.0:
        raise ValueError("metric_floor must be finite and positive")
    if not math.isfinite(refinement_rtol) or refinement_rtol < 0.0:
        raise ValueError("refinement_rtol must be finite and nonnegative")

    levels = [
        _jet_tensors(
            q_fn,
            point,
            h / factor,
            score_logits_fn=score_logits_fn,
            score_logit_jacobian_fn=score_logit_jacobian_fn,
        )
        for factor in (1.0, 2.0, 4.0)
    ]
    q0 = levels[0][0]

    def extrapolate(coarse_index: int, fine_index: int) -> list[Tensor3]:
        coarse = levels[coarse_index]
        fine = levels[fine_index]
        return [
            (4.0 * fine[slot] - coarse[slot]) / 3.0
            for slot in (1, 2, 3)
        ]

    first_ext = extrapolate(0, 1)
    second_ext = extrapolate(1, 2)
    refinement_error = _max_relative_difference(first_ext, second_ext)

    metric = 0.5 * (second_ext[0] + second_ext[0].T)
    first_kind = second_ext[1]
    cubic = second_ext[2]
    cubic_audit = levels[2][4]
    excluded = max(level[5] for level in levels)

    primary_finite = all(
        np.all(np.isfinite(block)) for block in (metric, first_kind, cubic)
    )
    eigenvalues = (
        np.linalg.eigvalsh(metric)
        if primary_finite
        else np.full(point.size, np.nan, dtype=np.float64)
    )

    accepted = True
    reason = ""
    if not primary_finite:
        accepted = False
        reason = "nonfinite"
    elif float(eigenvalues.min()) < metric_floor:
        accepted = False
        reason = "rank"
    elif refinement_error > refinement_rtol:
        accepted = False
        reason = "refinement"

    return ConnectionPacket(
        z=point,
        step=h,
        q=q0,
        metric=metric,
        first_kind=first_kind,
        cubic=cubic,
        cubic_audit=cubic_audit,
        metric_eigenvalues=eigenvalues,
        refinement_error=refinement_error,
        excluded_outcomes=excluded,
        accepted=accepted,
        rejection_reason=reason,
    )


def build_packet_from_logits(
    logits_fn: LogitMap,
    z: Sequence[float] | Vector,
    step: float,
    *,
    logit_jacobian_fn: LogitJacobianMap | None = None,
    metric_floor: float = 1e-10,
    refinement_rtol: float = 1e-3,
) -> ConnectionPacket:
    """Build a packet from float64 logits, with optional exact logit JVPs.

    This is the preferred precision-safe entry point. ``logits_fn`` must
    return logits produced in float64; a lower-precision result cast to
    float64 before returning cannot be detected and violates the contract.
    ``logit_jacobian_fn(z)`` returns a ``(chart_dim, outcomes)`` array.
    """

    def q_fn(point: Vector) -> Vector:
        return _softmax_float64(_evaluate_logits(logits_fn, point))

    return build_packet(
        q_fn,
        z,
        step,
        metric_floor=metric_floor,
        refinement_rtol=refinement_rtol,
        score_logits_fn=logits_fn,
        score_logit_jacobian_fn=logit_jacobian_fn,
    )


def alpha_connection(
    metric: Matrix,
    first_kind: Tensor3,
    cubic: Tensor3,
    alpha: float,
) -> Tensor3:
    """Return ``Gamma[l, i, j] = G^{-1}[l, k] (L[i, j, k] - alpha/2 C[i, j, k])``."""

    lower = first_kind - 0.5 * alpha * cubic
    inverse_metric = np.linalg.inv(metric)
    return np.einsum("lk,ijk->lij", inverse_metric, lower)


def tensor_operator_norm(tensor: Tensor3, output_axis: int = 0) -> float:
    """Spectral norm after unfolding the index acted on by a matrix.

    Raised connection tensors ``Gamma[l, i, j]`` use ``output_axis=0``.
    Lowered tensors ``L[i, j, k]`` and ``C[i, j, k]`` are raised by acting on
    their final index and therefore require ``output_axis=2``. With this
    convention ``||A T|| <= ||A||_2 ||T||`` holds for the contraction used by
    :func:`alpha_connection`.
    """

    array = np.asarray(tensor, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("tensor must have three indices")
    axis = int(output_axis)
    if output_axis != axis or axis not in (0, 1, 2):
        raise ValueError("output_axis must be 0, 1, or 2")
    unfolded = np.moveaxis(array, axis, 0)
    return float(np.linalg.norm(unfolded.reshape(unfolded.shape[0], -1), 2))


def packet_connection_bound(
    metric_floor: float,
    metric_defect: float,
    first_kind_defect: float,
    cubic_defect: float,
    alpha: float,
    first_kind_bound: float,
    cubic_bound: float,
) -> float:
    """Section 11.2 algebraic packet-to-connection bound.

    The first-kind and cubic inputs must use a norm compatible with raising
    their final covariant index, such as
    ``tensor_operator_norm(tensor, output_axis=2)``. Connection defects use
    ``output_axis=0`` after the raised index has moved to the front.
    """

    scalars = (
        metric_floor,
        metric_defect,
        first_kind_defect,
        cubic_defect,
        first_kind_bound,
        cubic_bound,
    )
    if not all(math.isfinite(float(value)) for value in scalars) or not math.isfinite(alpha):
        raise ValueError("bound inputs must be finite")
    if metric_floor <= 0.0 or any(value < 0.0 for value in scalars[1:]):
        raise ValueError("metric floor must be positive and norm bounds nonnegative")

    lowered = first_kind_defect + 0.5 * abs(alpha) * cubic_defect
    inflation = first_kind_bound + 0.5 * abs(alpha) * cubic_bound
    return lowered / metric_floor + metric_defect * inflation / metric_floor**2


def packet_transport_bound(
    path_length: float,
    sup_teacher: float,
    sup_student: float,
    packet_defect: float,
    holder_teacher: float,
    holder_student: float,
    fill_distance: float,
    holder_exponent: float = 1.0,
) -> float:
    """Section 11.3 proven packet-to-transport bound.

    ``L_gamma * exp((M_T + M_S) L_gamma) * (delta_pack + (H_T + H_S) h^rho)``
    with continuum constants supplied by a certificate, or clearly labeled
    empirical grid estimates when no such certificate is available.
    """

    nonnegative = (
        path_length,
        sup_teacher,
        sup_student,
        packet_defect,
        holder_teacher,
        holder_student,
        fill_distance,
    )
    if not all(math.isfinite(float(value)) and value >= 0.0 for value in nonnegative):
        raise ValueError("transport-bound inputs must be finite and nonnegative")
    if not math.isfinite(holder_exponent) or not 0.0 < holder_exponent <= 1.0:
        raise ValueError("holder_exponent must lie in (0, 1]")

    interpolation = (holder_teacher + holder_student) * fill_distance**holder_exponent
    growth = math.exp((sup_teacher + sup_student) * path_length)
    return path_length * growth * (packet_defect + interpolation)


def metric_relative_loss(student_metric: Matrix, teacher_metric: Matrix) -> float:
    """``|| G_T^{-1/2} (G_S - G_T) G_T^{-1/2} ||_F^2`` (Section 4.4)."""

    eigenvalues, eigenvectors = np.linalg.eigh(teacher_metric)
    if float(eigenvalues.min()) <= 0.0:
        raise ValueError("teacher metric must be positive definite")
    inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    normalized = inv_sqrt @ (student_metric - teacher_metric) @ inv_sqrt
    return float(np.sum(normalized * normalized))


def _teacher_fisher_tensor_norm_sq(defect: Tensor3, teacher_metric: Matrix) -> float:
    inverse = np.linalg.inv(teacher_metric)
    return float(
        np.einsum(
            "lr,ia,jb,lij,rab->",
            teacher_metric,
            inverse,
            inverse,
            defect,
            defect,
        )
    )


def levi_civita_loss(student: ConnectionPacket, teacher: ConnectionPacket) -> float:
    """Teacher-Fisher squared norm of the Levi--Civita defect (Section 4.5)."""

    defect = alpha_connection(
        student.metric, student.first_kind, student.cubic, 0.0
    ) - alpha_connection(teacher.metric, teacher.first_kind, teacher.cubic, 0.0)
    return _teacher_fisher_tensor_norm_sq(defect, teacher.metric)


def raised_cubic_loss(student: ConnectionPacket, teacher: ConnectionPacket) -> float:
    """Teacher-Fisher squared norm of the raised-cubic defect (Section 4.6)."""

    student_raised = np.einsum(
        "lk,ijk->lij", np.linalg.inv(student.metric), student.cubic
    )
    teacher_raised = np.einsum(
        "lk,ijk->lij", np.linalg.inv(teacher.metric), teacher.cubic
    )
    return _teacher_fisher_tensor_norm_sq(
        student_raised - teacher_raised, teacher.metric
    )


def centered_logit_jacobian_loss(
    student_jacobian: Matrix, teacher_jacobian: Matrix
) -> float:
    """Squared defect of outcome-centered logit Jacobians (Section 4.2)."""

    student_centered = student_jacobian - student_jacobian.mean(axis=0, keepdims=True)
    teacher_centered = teacher_jacobian - teacher_jacobian.mean(axis=0, keepdims=True)
    difference = student_centered - teacher_centered
    return float(np.sum(difference * difference))


def sqrt_jacobian_loss(student_jacobian: Matrix, teacher_jacobian: Matrix) -> float:
    """Squared defect of square-root-output Jacobians (Section 4.2)."""

    difference = student_jacobian - teacher_jacobian
    return float(np.sum(difference * difference))


def sobolev_grid_audit(values: Matrix, spacing: float, order: int) -> float:
    """Finite-difference squared ``H^order`` estimate on a uniform 1-D grid.

    ``values[t, a]`` samples a curve in ``ell_2`` at grid points with the
    given spacing.  Derivatives are forward differences, integrals are
    trapezoidal on the surviving points.  This is a numerical audit of a
    surrogate, not a certificate for the underlying map (Section 11.4).
    """

    if values.ndim != 2:
        raise ValueError("values must be a (grid, outcomes) array")
    if order < 0:
        raise ValueError("order must be nonnegative")
    total = 0.0
    current = np.asarray(values, dtype=np.float64)
    for _ in range(order + 1):
        squared = np.sum(current * current, axis=1)
        total += float(_TRAPEZOID(squared, dx=spacing))
        if current.shape[0] < 2:
            break
        current = np.diff(current, axis=0) / spacing
    return total


def sufficiency_decomposition(
    fiber_ids: Sequence[int],
    effects: Matrix,
    transferred: Matrix,
    metric: Matrix,
) -> tuple[float, float, float]:
    """Exact conditional-variance split of Section 9.4 on discrete fibers.

    Returns ``(insufficiency, mismatch, total)`` where ``total`` equals
    their sum exactly whenever ``transferred`` is constant on each fiber and
    the metric is fiber-independent.
    """

    labels = np.asarray(fiber_ids)
    effect_array = np.asarray(effects, dtype=np.float64)
    transfer_array = np.asarray(transferred, dtype=np.float64)
    for fiber in np.unique(labels):
        block = transfer_array[labels == fiber]
        if not np.allclose(block, block[0]):
            raise ValueError("transferred field must be constant on each fiber")

    def norm_sq(rows: Matrix) -> Vector:
        return np.einsum("nd,de,ne->n", rows, metric, rows)

    conditional_mean = np.empty_like(effect_array)
    for fiber in np.unique(labels):
        mask = labels == fiber
        conditional_mean[mask] = effect_array[mask].mean(axis=0)

    insufficiency = float(np.mean(norm_sq(effect_array - conditional_mean)))
    mismatch = float(np.mean(norm_sq(conditional_mean - transfer_array)))
    total = float(np.mean(norm_sq(effect_array - transfer_array)))
    return insufficiency, mismatch, total


def context_shuffled_cubics(
    cubics: Sequence[Tensor3],
    strata: Sequence[Hashable],
    context_ids: Sequence[Hashable],
    chart_keys: Sequence[Hashable],
    *,
    seed: int = 0,
) -> list[Tensor3]:
    """Shuffle complete donor-context cubic fields within conditioning strata.

    All packets belonging to one context are assigned to one *different*
    donor context in the same stratum, and tensors are copied at matching
    ``chart_keys``. Context blocks must have identical chart-key sets. This
    preserves the donor field as a coherent sampled field instead of creating
    the discontinuous per-packet scramble rejected by Section 7.

    The resulting cubic field is realized by the teacher in the donor context.
    Its combination with the recipient's retained ``(G, L)`` is not guaranteed
    to be the jet of any single predictive map; achieved loss floors must still
    be compared before interpreting the control.
    """

    count = len(cubics)
    if not (len(strata) == len(context_ids) == len(chart_keys) == count):
        raise ValueError("cubics, strata, context_ids, and chart_keys must align")
    if count == 0:
        return []
    reference_shape = np.asarray(cubics[0]).shape
    if len(reference_shape) != 3 or any(
        np.asarray(cubic).shape != reference_shape
        or not np.all(np.isfinite(np.asarray(cubic)))
        for cubic in cubics
    ):
        raise ValueError("cubics must be finite three-tensors with one shared shape")

    context_indices: dict[Hashable, list[int]] = {}
    context_strata: dict[Hashable, Hashable] = {}
    for index, (stratum, context, key) in enumerate(
        zip(strata, context_ids, chart_keys)
    ):
        try:
            hash(stratum)
            hash(context)
            hash(key)
        except TypeError as error:
            raise ValueError("strata, context IDs, and chart keys must be hashable") from error
        if context in context_strata and context_strata[context] != stratum:
            raise ValueError("every context must belong to exactly one stratum")
        context_strata[context] = stratum
        context_indices.setdefault(context, []).append(index)

    key_maps: dict[Hashable, dict[Hashable, int]] = {}
    for context, indices in context_indices.items():
        mapping: dict[Hashable, int] = {}
        for index in indices:
            key = chart_keys[index]
            if key in mapping:
                raise ValueError("chart keys must be unique within each context")
            mapping[key] = index
        key_maps[context] = mapping

    contexts_by_stratum: dict[Hashable, list[Hashable]] = {}
    for context in context_indices:
        contexts_by_stratum.setdefault(context_strata[context], []).append(context)

    shuffled: list[Tensor3] = [np.array(cubic, copy=True) for cubic in cubics]
    rng = np.random.default_rng(seed)
    for stratum, contexts in contexts_by_stratum.items():
        if len(contexts) < 2:
            raise ValueError(
                f"stratum {stratum!r} has fewer than two contexts; "
                "a non-self donor control is impossible"
            )
        order = [contexts[i] for i in rng.permutation(len(contexts))]
        donors = order[-1:] + order[:-1]
        reference_keys = set(key_maps[order[0]])
        if any(set(key_maps[context]) != reference_keys for context in order[1:]):
            raise ValueError("all context blocks in a stratum must share chart keys")
        for recipient, donor in zip(order, donors):
            for key, target_index in key_maps[recipient].items():
                donor_index = key_maps[donor][key]
                shuffled[target_index] = np.array(cubics[donor_index], copy=True)
    return shuffled


def _payload_arrays(packet: ConnectionPacket) -> list[NDArray]:
    return [
        packet.z.astype(np.float32),
        # The geometric sidecar is float32. Keep the output anchor in float64
        # so rare positive probabilities cannot become exact zeros on disk.
        packet.q.astype(np.float64),
        packet.metric.astype(np.float32),
        packet.first_kind.astype(np.float32),
        packet.cubic.astype(np.float32),
        packet.cubic_audit.astype(np.float32),
    ]


def _packet_checksum(body: dict) -> str:
    """Checksum the complete canonical packet body, including audit metadata."""

    digest = hashlib.blake2b(digest_size=16, person=b"pcdpack")
    canonical = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    digest.update(canonical)
    return digest.hexdigest()


def packet_to_dict(
    packet: ConnectionPacket,
    *,
    max_quantization_error: float = 1e-6,
) -> dict:
    """Serialize float32 geometry, a float64 output anchor, and all-field checksum."""

    if packet.schema_version != SCHEMA_VERSION:
        raise ValueError("packet schema version does not match this writer")
    if not math.isfinite(max_quantization_error) or max_quantization_error < 0.0:
        raise ValueError("max_quantization_error must be finite and nonnegative")
    measured_quantization_error = quantization_error(packet)
    if measured_quantization_error > max_quantization_error:
        raise ValueError("float32 packet quantization exceeds the declared tolerance")
    payload = _payload_arrays(packet)
    body = {
        "schema_version": packet.schema_version,
        "step": packet.step,
        "z": payload[0].tolist(),
        "q": payload[1].tolist(),
        "metric": payload[2].tolist(),
        "first_kind": payload[3].tolist(),
        "cubic": payload[4].tolist(),
        "cubic_audit": payload[5].tolist(),
        "metric_eigenvalues": packet.metric_eigenvalues.tolist(),
        "refinement_error": packet.refinement_error,
        "excluded_outcomes": packet.excluded_outcomes,
        "accepted": packet.accepted,
        "rejection_reason": packet.rejection_reason,
        "serialization_quantization_error": measured_quantization_error,
    }
    return {**body, "checksum": _packet_checksum(body)}


def packet_from_dict(data: dict) -> ConnectionPacket:
    """Deserialize, validating the checksum, shapes, and packet invariants."""

    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported packet schema version")
    body = {key: value for key, value in data.items() if key != "checksum"}
    if _packet_checksum(body) != data.get("checksum"):
        raise ValueError("packet checksum mismatch")
    arrays = [
        np.asarray(data["z"], dtype=np.float32),
        np.asarray(data["q"], dtype=np.float64),
        np.asarray(data["metric"], dtype=np.float32),
        np.asarray(data["first_kind"], dtype=np.float32),
        np.asarray(data["cubic"], dtype=np.float32),
        np.asarray(data["cubic_audit"], dtype=np.float32),
    ]
    z, q, metric, first_kind, cubic, cubic_audit = arrays
    if z.ndim != 1 or z.size == 0 or q.ndim != 1 or q.size == 0:
        raise ValueError("invalid serialized point or probability shape")
    m = z.size
    if metric.shape != (m, m) or any(
        block.shape != (m, m, m) for block in (first_kind, cubic, cubic_audit)
    ):
        raise ValueError("serialized geometric tensor shapes are inconsistent")
    accepted = bool(data["accepted"])
    rejection_reason = str(data["rejection_reason"])
    primary_finite = all(
        np.all(np.isfinite(block)) for block in (z, q, metric, first_kind, cubic)
    )
    if accepted and not primary_finite:
        raise ValueError("an accepted packet must have finite primary values")
    if not primary_finite and rejection_reason != "nonfinite":
        raise ValueError("nonfinite packet values require the nonfinite rejection reason")
    if np.any(q <= 0.0) or not np.isclose(float(q.sum()), 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError("serialized probabilities must lie in the open simplex")

    step = float(data["step"])
    refinement_error = float(data["refinement_error"])
    excluded_outcomes = int(data["excluded_outcomes"])
    serialization_quantization_error = float(data["serialization_quantization_error"])
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("serialized step must be finite and positive")
    if not math.isfinite(refinement_error) or refinement_error < 0.0:
        raise ValueError("serialized refinement error must be finite and nonnegative")
    if excluded_outcomes < 0:
        raise ValueError("serialized excluded-outcome count must be nonnegative")
    if (
        not math.isfinite(serialization_quantization_error)
        or serialization_quantization_error < 0.0
    ):
        raise ValueError("serialized quantization error must be finite and nonnegative")

    eigenvalues = np.asarray(data["metric_eigenvalues"], dtype=np.float64)
    recomputed = (
        np.linalg.eigvalsh(metric.astype(np.float64))
        if np.all(np.isfinite(metric))
        else np.full(m, np.nan)
    )
    if eigenvalues.shape != (m,):
        raise ValueError("serialized metric eigenvalues are invalid")
    if not np.allclose(eigenvalues, recomputed, rtol=1e-5, atol=1e-7, equal_nan=True):
        raise ValueError("serialized metric eigenvalues disagree with the metric")

    if accepted and rejection_reason:
        raise ValueError("an accepted packet cannot have a rejection reason")
    if not accepted and not rejection_reason:
        raise ValueError("a rejected packet must retain its rejection reason")

    return ConnectionPacket(
        z=z.astype(np.float64),
        step=step,
        q=q.astype(np.float64),
        metric=metric.astype(np.float64),
        first_kind=first_kind.astype(np.float64),
        cubic=cubic.astype(np.float64),
        cubic_audit=cubic_audit.astype(np.float64),
        metric_eigenvalues=eigenvalues,
        refinement_error=refinement_error,
        excluded_outcomes=excluded_outcomes,
        accepted=accepted,
        rejection_reason=rejection_reason,
        serialization_quantization_error=serialization_quantization_error,
    )


def quantization_error(packet: ConnectionPacket) -> float:
    """Relative float32 round-trip error over the geometric payload."""

    worst = 0.0
    for block in (packet.metric, packet.first_kind, packet.cubic):
        roundtrip = block.astype(np.float32).astype(np.float64)
        scale = max(1.0, float(np.linalg.norm(block)))
        worst = max(worst, float(np.linalg.norm(block - roundtrip)) / scale)
    return worst
