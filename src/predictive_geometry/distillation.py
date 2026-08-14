"""Shared-chart connection packets for predictive distillation.

Implements the finite-dimensional core of
``PREDICTIVE_CONNECTION_DISTILLATION.md``: packet assembly ``(q, G, L, C)``
on a declared intervention chart, the distillation losses, acceptance gates,
checksummed serialization, and the algebraic packet-to-connection bound
(Section 11.2) together with the proven packet-to-transport bound
(Section 11.3).

The module is model-free: a predictive map is any callable ``z -> q(z)``
returning a categorical distribution on the open simplex, evaluated in
float64.  The cubic tensor uses the score-moment identity as its primary
estimator: scores are central differences of ``log q`` recentered by the
``q``-weighted mean, which removes the softmax gauge exactly, carries no
inverse-probability factor, and is exact for affine (log-linear) families.
The probability-difference form is retained only as an independent audit; it
divides by ``q**2`` and is deliberately left unguarded so that float32
underflow shows up as a non-finite audit value instead of a silent error.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


SCHEMA_VERSION = "pcd-packet-1"

_TRAPEZOID = getattr(np, "trapezoid", None) or getattr(np, "trapz")

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
Tensor3 = NDArray[np.float64]

PredictiveMap = Callable[[Vector], Vector]


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
    schema_version: str = SCHEMA_VERSION


def _as_point(z: Sequence[float] | Vector) -> Vector:
    point = np.atleast_1d(np.asarray(z, dtype=np.float64))
    if point.ndim != 1:
        raise ValueError("chart point must be one-dimensional")
    return point


def _sqrt_map(q: Vector) -> Vector:
    return 2.0 * np.sqrt(np.maximum(q, 0.0))


def _jet_tensors(
    q_fn: PredictiveMap,
    z: Vector,
    h: float,
) -> tuple[Vector, Matrix, Tensor3, Tensor3, Tensor3, int]:
    """Return ``(q, G, L, C_score, C_audit, excluded)`` at step ``h``.

    ``G`` and ``L`` come from central differences of ``psi = 2 sqrt(q)``.
    ``C_score`` uses recentered central differences of ``log q``; outcomes
    with a nonpositive probability at any stencil evaluation are excluded
    from the score sums (their score-weighted contribution is negligible by
    construction) and counted in ``excluded``.
    """

    m = z.shape[0]
    q0 = np.asarray(q_fn(z), dtype=np.float64)
    n = q0.shape[0]

    q_plus = np.empty((m, n))
    q_minus = np.empty((m, n))
    for i in range(m):
        unit = np.zeros(m)
        unit[i] = h
        q_plus[i] = q_fn(z + unit)
        q_minus[i] = q_fn(z - unit)

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
            pp = _sqrt_map(np.asarray(q_fn(z + unit_i + unit_j), dtype=np.float64))
            pm = _sqrt_map(np.asarray(q_fn(z + unit_i - unit_j), dtype=np.float64))
            mp = _sqrt_map(np.asarray(q_fn(z - unit_i + unit_j), dtype=np.float64))
            mm = _sqrt_map(np.asarray(q_fn(z - unit_i - unit_j), dtype=np.float64))
            d2_psi[i, j] = (pp - pm - mp + mm) / (4.0 * h * h)
            d2_psi[j, i] = d2_psi[i, j]

    metric = np.einsum("ia,ja->ij", d_psi, d_psi)
    first_kind = np.einsum("ija,ka->ijk", d2_psi, d_psi)

    included = (q0 > 0.0) & np.all(q_plus > 0.0, axis=0) & np.all(q_minus > 0.0, axis=0)
    excluded = int(n - np.count_nonzero(included))

    scores = np.zeros((m, n))
    if np.any(included):
        log_plus = np.log(q_plus[:, included])
        log_minus = np.log(q_minus[:, included])
        raw = (log_plus - log_minus) / (2.0 * h)
        weights = q0[included]
        gauge = raw @ weights / np.sum(weights)
        scores[:, included] = raw - gauge[:, None]
    cubic = np.einsum("a,ia,ja,ka->ijk", q0, scores, scores, scores)

    d_q = (q_plus - q_minus) / (2.0 * h)
    with np.errstate(divide="ignore", invalid="ignore"):
        cubic_audit = np.einsum(
            "ia,ja,ka,a->ijk", d_q, d_q, d_q, 1.0 / np.square(q0)
        )

    return q0, metric, first_kind, cubic, cubic_audit, excluded


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

    levels = [_jet_tensors(q_fn, point, h / factor) for factor in (1.0, 2.0, 4.0)]
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

    eigenvalues = np.linalg.eigvalsh(metric)

    accepted = True
    reason = ""
    primary_finite = all(
        np.all(np.isfinite(block)) for block in (metric, first_kind, cubic)
    )
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


def tensor_operator_norm(tensor: Tensor3) -> float:
    """Spectral norm of the first-index flattening, compatible with
    contraction by a matrix on the first index."""

    m = tensor.shape[0]
    return float(np.linalg.norm(tensor.reshape(m, -1), 2))


def packet_connection_bound(
    metric_floor: float,
    metric_defect: float,
    first_kind_defect: float,
    cubic_defect: float,
    alpha: float,
    first_kind_bound: float,
    cubic_bound: float,
) -> float:
    """Section 11.2 algebraic packet-to-connection bound."""

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
    with all constants audited on the declared chart.
    """

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
    strata: Sequence[int],
) -> list[Tensor3]:
    """Section 7 negative control: roll cubic targets within each stratum.

    Every packet in a stratum of size at least two receives a donor cubic
    from a different packet of the same stratum; singleton strata keep their
    own tensor and should be avoided by the stratification design.
    """

    labels = np.asarray(strata)
    shuffled: list[Tensor3] = [np.array(c, copy=True) for c in cubics]
    for stratum in np.unique(labels):
        indices = np.flatnonzero(labels == stratum)
        if indices.size < 2:
            continue
        rolled = np.roll(indices, 1)
        for target, donor in zip(indices, rolled):
            shuffled[target] = np.array(cubics[donor], copy=True)
    return shuffled


def _payload_arrays(packet: ConnectionPacket) -> list[NDArray[np.float32]]:
    return [
        packet.z.astype(np.float32),
        packet.q.astype(np.float32),
        packet.metric.astype(np.float32),
        packet.first_kind.astype(np.float32),
        packet.cubic.astype(np.float32),
    ]


def _payload_checksum(arrays: Sequence[NDArray[np.float32]]) -> str:
    digest = hashlib.blake2b(digest_size=16, person=b"pcdpack")
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def packet_to_dict(packet: ConnectionPacket) -> dict:
    """Serialize with float32 payload, float64 audit summary, and checksum."""

    payload = _payload_arrays(packet)
    return {
        "schema_version": packet.schema_version,
        "step": packet.step,
        "z": payload[0].tolist(),
        "q": payload[1].tolist(),
        "metric": payload[2].tolist(),
        "first_kind": payload[3].tolist(),
        "cubic": payload[4].tolist(),
        "metric_eigenvalues": packet.metric_eigenvalues.tolist(),
        "refinement_error": packet.refinement_error,
        "excluded_outcomes": packet.excluded_outcomes,
        "accepted": packet.accepted,
        "rejection_reason": packet.rejection_reason,
        "checksum": _payload_checksum(payload),
    }


def packet_from_dict(data: dict) -> ConnectionPacket:
    """Deserialize, validating schema version and payload checksum."""

    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported packet schema version")
    arrays = [
        np.asarray(data["z"], dtype=np.float32),
        np.asarray(data["q"], dtype=np.float32),
        np.asarray(data["metric"], dtype=np.float32),
        np.asarray(data["first_kind"], dtype=np.float32),
        np.asarray(data["cubic"], dtype=np.float32),
    ]
    if _payload_checksum(arrays) != data.get("checksum"):
        raise ValueError("packet checksum mismatch")
    return ConnectionPacket(
        z=arrays[0].astype(np.float64),
        step=float(data["step"]),
        q=arrays[1].astype(np.float64),
        metric=arrays[2].astype(np.float64),
        first_kind=arrays[3].astype(np.float64),
        cubic=arrays[4].astype(np.float64),
        cubic_audit=np.full_like(arrays[4].astype(np.float64), np.nan),
        metric_eigenvalues=np.asarray(data["metric_eigenvalues"], dtype=np.float64),
        refinement_error=float(data["refinement_error"]),
        excluded_outcomes=int(data["excluded_outcomes"]),
        accepted=bool(data["accepted"]),
        rejection_reason=str(data["rejection_reason"]),
    )


def quantization_error(packet: ConnectionPacket) -> float:
    """Relative float32 round-trip error over the geometric payload."""

    worst = 0.0
    for block in (packet.metric, packet.first_kind, packet.cubic):
        roundtrip = block.astype(np.float32).astype(np.float64)
        scale = max(1.0, float(np.linalg.norm(block)))
        worst = max(worst, float(np.linalg.norm(block - roundtrip)) / scale)
    return worst
