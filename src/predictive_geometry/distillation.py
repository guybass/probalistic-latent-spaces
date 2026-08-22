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

import hashlib
import math
import struct
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

SCHEMA_VERSION = "pcd-packet-6"

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
Tensor3 = NDArray[np.float64]

PredictiveMap = Callable[[Vector], Vector]
LogitMap = Callable[[Vector], Vector]
LogitJacobianMap = Callable[[Vector], Matrix]


@dataclass(frozen=True)
class PacketProvenance:
    """Immutable provenance needed to interpret a serialized packet.

    ``reproducible`` must be true before the default serializer accepts a
    packet. Model-free and exploratory callers may carry an explicit untracked
    record in memory, but must opt out deliberately when serializing it.
    """

    teacher_model_id: str = "untracked:model-free-callable"
    teacher_revision: str = "untracked"
    tokenizer_hash: str = "not-applicable"
    outcome_map_hash: str = "untracked"
    chart_id: str = "untracked"
    chart_bounds: tuple[tuple[float, float], ...] = ()
    context_id: str = "untracked"
    inference_dtype: str = "float64"
    reproducible: bool = False


@dataclass(frozen=True)
class TransportBoundAudit:
    """A transport bound bundled with its evidentiary status and source."""

    value: float
    evidence_status: str
    evidence_source: str


@dataclass(frozen=True)
class ConnectionPacket:
    """Local predictive-connection packet at one chart point.

    ``first_kind`` stores the first-kind Levi--Civita coefficient
    ``L[i, j, k] = <d_ij psi, d_k psi>`` with ``psi = 2 sqrt(q)``; it is a
    pointwise function of the metric derivatives, so its independent content
    relative to a dense metric field is sampling density only.  ``cubic`` is
    the score-moment Amari--Chentsov tensor (primary estimator);
    ``cubic_audit`` is the probability-difference estimator retained as an
    independent audit and may be non-finite even for strictly positive inputs
    when squaring a tiny float64 probability underflows.  This does not
    invalidate the score-moment primary estimator.
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
    accepted: bool
    rejection_reason: str
    metric_floor: float = 1e-10
    metric_relative_floor: float = 1e-12
    refinement_rtol: float = 1e-3
    refinement_atol: float = 1e-6
    refinement_absolute_error: float = 0.0
    refinement_tolerance_ratio: float = 0.0
    jvp_consistency_relative_error: float | None = None
    jvp_consistency_absolute_error: float | None = None
    jvp_consistency_tolerance_ratio: float | None = None
    jvp_rtol: float = 1e-5
    jvp_atol: float = 0.0
    provenance: PacketProvenance = field(default_factory=PacketProvenance)
    serialization_quantization_error: float = 0.0
    serialization_metric_eigenvalue_error_bound: float = 0.0
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
) -> tuple[
    Vector,
    Matrix,
    Tensor3,
    Tensor3,
    Tensor3,
    tuple[float, float] | None,
]:
    """Return ``(q, G, L, C_score, C_audit, jvp_audit)`` at step ``h``.

    ``G`` and ``L`` come from central differences of ``psi = 2 sqrt(q)``. When
    an exact logit Jacobian is supplied, its exact square-root derivative is
    used for ``G`` and the first-derivative leg of ``L``. ``C_score`` uses that
    same Jacobian when supplied, otherwise central differences of logits or,
    for the probability-only compatibility path, central differences of
    ``log q``. Invalid or underflowed simplex values are rejected rather than
    silently excluded.
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

    jvp_audit: tuple[float, float] | None = None
    if score_logit_jacobian_fn is not None:
        if score_logits_fn is None:
            raise ValueError(
                "an exact logit Jacobian requires score_logits_fn for an "
                "independent finite-difference consistency audit"
            )
        raw_source = np.asarray(score_logit_jacobian_fn(z))
        if raw_source.dtype != np.dtype(np.float64):
            raise TypeError("score_logit_jacobian_fn must return float64 values")
        raw = np.asarray(raw_source, dtype=np.float64)
        if raw.shape != (m, n) or not np.all(np.isfinite(raw)):
            raise ValueError("logit Jacobian must be a finite (chart_dim, outcomes) array")
        logit_plus = np.empty((m, n))
        logit_minus = np.empty((m, n))
        for i in range(m):
            unit = np.zeros(m)
            unit[i] = h
            logit_plus[i] = _evaluate_logits(score_logits_fn, z + unit, n)
            logit_minus[i] = _evaluate_logits(score_logits_fn, z - unit, n)
        finite_difference_raw = (logit_plus - logit_minus) / (2.0 * h)
        # Compare modulo the rowwise softmax gauge. The declared Jacobian layout
        # is (chart_dim, outcomes), so outcome centering is along axis 1.
        centered_raw = raw - raw.mean(axis=1, keepdims=True)
        centered_finite_difference = finite_difference_raw - finite_difference_raw.mean(
            axis=1, keepdims=True
        )
        jvp_audit = (
            float(np.linalg.norm(centered_raw - centered_finite_difference)),
            max(
                float(np.linalg.norm(centered_raw)),
                float(np.linalg.norm(centered_finite_difference)),
            ),
        )
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
    if score_logit_jacobian_fn is not None:
        d_psi = np.sqrt(q0)[None, :] * scores
        metric = np.einsum("ia,ja->ij", d_psi, d_psi)
        first_kind = np.einsum("ija,ka->ijk", d2_psi, d_psi)
    cubic = np.einsum("a,ia,ja,ka->ijk", q0, scores, scores, scores)

    d_q = (q_plus - q_minus) / (2.0 * h)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cubic_audit = np.einsum(
            "ia,ja,ka,a->ijk", d_q, d_q, d_q, 1.0 / np.square(q0)
        )

    return q0, metric, first_kind, cubic, cubic_audit, jvp_audit


def _fisher_norm(tensor: Tensor3, metric: Matrix, inverse: Matrix) -> float:
    """Fisher norm of a ``(1, 2)``-tensor ``T[l, i, j]`` at one chart point."""

    squared = float(
        np.einsum(
            "lr,ia,jb,lij,rab->",
            metric,
            inverse,
            inverse,
            tensor,
            tensor,
        )
    )
    return math.sqrt(max(squared, 0.0))


def _difference_diagnostics(
    first: Sequence[NDArray[np.float64]],
    second: Sequence[NDArray[np.float64]],
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float, float]:
    """Return worst chart-invariant relative, absolute, and tolerance errors.

    The two Richardson extrapolants are compared through quantities that are
    invariant under a linear change of the intervention chart, rather than
    through raw Frobenius norms of ``(G, L, C)``:

    * the metric through the dimensionless relative defect
      ``||G_2^{-1/2}(G_1-G_2)G_2^{-1/2}||_F``, whose own scale is
      ``||I||_F = sqrt(m)``;
    * ``L`` and ``C`` through their raised counterparts ``G^{-1}L`` and
      ``G^{-1}C`` measured in the Fisher norm, each extrapolant raised by its
      own metric exactly as :func:`alpha_connection` later raises it.

    Two properties follow, and both are the point of the change.  First, the
    verdict no longer depends on chart units: ``z -> sz`` rescales ``G, L, C``
    by ``s^2, s^3, s^3`` and leaves every quantity above fixed.  Second, a
    structurally zero ``L`` or ``C`` -- for example the first-kind coefficient
    of the paper's boundary Bernoulli family, where ``<d_ij psi, d_k psi>``
    vanishes identically -- has a well-defined zero Fisher scale, so its
    roundoff-level difference is compared against ``atol`` rather than against
    itself.  A pure relative criterion on the raw tensor would instead divide
    roundoff by roundoff and report a spurious order-one failure.

    ``atol`` is therefore dimensionless: it is the Fisher-norm magnitude below
    which this numerical gate treats a connection or raised-cubic coefficient
    as indistinguishable from zero.  It does not prove that the exact
    coefficient vanishes.  The reported relative error is regularized by the
    same constant, so it remains well defined at zero scale and stays monotone
    with the acceptance ratio.  Non-invertible or non-finite metrics return
    infinite diagnostics; the rank and conditioning gates report those cases
    with their own reasons.
    """

    metric_coarse = 0.5 * (first[0] + first[0].T)
    metric_fine = 0.5 * (second[0] + second[0].T)
    dimension = metric_fine.shape[0]
    if not (
        np.all(np.isfinite(metric_coarse)) and np.all(np.isfinite(metric_fine))
    ):
        return math.inf, math.inf, math.inf
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(metric_fine)
        if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues.min()) <= 0.0:
            return math.inf, math.inf, math.inf
        inverse_fine = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T
        inverse_sqrt = (
            eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        )
        inverse_coarse = np.linalg.inv(metric_coarse)
    except np.linalg.LinAlgError:
        return math.inf, math.inf, math.inf

    normalized_metric_defect = float(
        np.linalg.norm(inverse_sqrt @ (metric_coarse - metric_fine) @ inverse_sqrt)
    )
    entries: list[tuple[float, float]] = [
        (normalized_metric_defect, math.sqrt(dimension))
    ]
    for slot in (1, 2):
        raised_coarse = np.einsum("lk,ijk->lij", inverse_coarse, first[slot])
        raised_fine = np.einsum("lk,ijk->lij", inverse_fine, second[slot])
        entries.append(
            (
                _fisher_norm(raised_coarse - raised_fine, metric_fine, inverse_fine),
                max(
                    _fisher_norm(raised_coarse, metric_fine, inverse_fine),
                    _fisher_norm(raised_fine, metric_fine, inverse_fine),
                ),
            )
        )

    worst_relative = 0.0
    worst_absolute = 0.0
    worst_tolerance_ratio = 0.0
    for difference, scale in entries:
        if not (math.isfinite(difference) and math.isfinite(scale)):
            return math.inf, math.inf, math.inf
        # Regularize by ``atol`` so the reported error stays well posed, and
        # monotone with the verdict, when the tensor's own scale is zero.
        denominator = max(scale, atol)
        relative = (
            0.0
            if difference == 0.0
            else math.inf
            if denominator == 0.0
            else difference / denominator
        )
        tolerance = atol + rtol * scale
        tolerance_ratio = (
            0.0
            if difference == 0.0
            else math.inf
            if tolerance == 0.0
            else difference / tolerance
        )
        worst_relative = max(worst_relative, relative)
        worst_absolute = max(worst_absolute, difference)
        worst_tolerance_ratio = max(worst_tolerance_ratio, tolerance_ratio)
    return worst_relative, worst_absolute, worst_tolerance_ratio


def _maximum_diagnostics(
    *diagnostics: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Combine independent refinement checks without hiding the worst one."""

    return tuple(max(values) for values in zip(*diagnostics, strict=True))


def _validate_stencil_bounds(
    point: Vector,
    step: float,
    provenance: PacketProvenance,
) -> None:
    """Fail before evaluation when the full central stencil leaves the chart."""

    bounds = provenance.chart_bounds
    if not bounds:
        return
    if len(bounds) != point.size:
        raise ValueError("chart bounds dimension does not match the packet chart")
    for coordinate, raw_bounds in zip(point, bounds, strict=True):
        if len(raw_bounds) != 2:
            raise ValueError("each chart bound must contain lower and upper values")
        lower, upper = (float(raw_bounds[0]), float(raw_bounds[1]))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("chart bounds must be finite and strictly ordered")
        if coordinate - step < lower or coordinate + step > upper:
            raise ValueError(
                "central-difference stencil extends outside declared chart bounds"
            )


def _gate_reason(
    metric: Matrix,
    *,
    metric_floor: float,
    metric_relative_floor: float,
    refinement_tolerance_ratio: float,
    jvp_consistency_tolerance_ratio: float | None,
) -> tuple[str, Vector]:
    """Recompute the ordered packet verdict from values carried by the packet."""

    if not np.all(np.isfinite(metric)):
        return "nonfinite", np.full(metric.shape[0], np.nan, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(0.5 * (metric + metric.T))
    if (
        jvp_consistency_tolerance_ratio is not None
        and jvp_consistency_tolerance_ratio > 1.0
    ):
        return "jvp_consistency", eigenvalues
    if float(eigenvalues.min()) <= 0.0:
        return "rank", eigenvalues
    if float(eigenvalues.min() / eigenvalues.max()) < metric_relative_floor:
        return "conditioning", eigenvalues
    if float(eigenvalues.min()) < metric_floor:
        return "metric_scale", eigenvalues
    if refinement_tolerance_ratio > 1.0:
        return "refinement", eigenvalues
    return "", eigenvalues


def build_packet(
    q_fn: PredictiveMap,
    z: Sequence[float] | Vector,
    step: float,
    *,
    metric_floor: float = 1e-10,
    metric_relative_floor: float = 1e-12,
    refinement_rtol: float = 1e-3,
    refinement_atol: float = 1e-6,
    jvp_rtol: float = 1e-5,
    jvp_atol: float = 0.0,
    score_logits_fn: LogitMap | None = None,
    score_logit_jacobian_fn: LogitJacobianMap | None = None,
    provenance: PacketProvenance | None = None,
) -> ConnectionPacket:
    """Assemble and gate a connection packet at one chart point.

    Tensors are computed on the nested ladder ``(h, h/2, h/4)`` and the
    incommensurate ladder ``(h/sqrt(2), h/(2sqrt(2)), h/(4sqrt(2)))``. Each is
    combined by Richardson extrapolation of the ``O(h^2)`` estimators; the
    packet is accepted only when both ladders converge internally and their
    fine extrapolants agree under the declared
    ``refinement_atol + refinement_rtol * scale`` rule (not a visually judged
    plateau). Both the difference and the scale are the chart-invariant
    quantities of :func:`_difference_diagnostics`: the relative metric defect
    and the Fisher norms of the raised ``G^{-1}L`` and ``G^{-1}C``. The
    verdict is therefore unchanged by a linear change of chart units, and
    ``refinement_atol`` is dimensionless -- it is the Fisher-norm magnitude
    below which this numerical gate treats a coefficient as indistinguishable
    from zero. This prevents an analytically vanishing ``L`` or ``C`` from
    being compared with its own roundoff; it does not certify exact vanishing.
    All primary tensors must be finite, the metric must
    clear both the absolute chart-scale and relative-conditioning floors, and
    any supplied exact JVP agrees with a gauge-centered finite-difference
    audit. Rejected packets retain their values and a ``rejection_reason`` so
    the audit log can report acceptance coverage.
    """

    point = _as_point(z)
    h = float(step)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("step must be finite and positive")
    if not math.isfinite(metric_floor) or metric_floor <= 0.0:
        raise ValueError("metric_floor must be finite and positive")
    if (
        not math.isfinite(metric_relative_floor)
        or not 0.0 < metric_relative_floor <= 1.0
    ):
        raise ValueError("metric_relative_floor must lie in (0, 1]")
    if not math.isfinite(refinement_rtol) or refinement_rtol < 0.0:
        raise ValueError("refinement_rtol must be finite and nonnegative")
    if not math.isfinite(refinement_atol) or refinement_atol < 0.0:
        raise ValueError("refinement_atol must be finite and nonnegative")
    if not math.isfinite(jvp_rtol) or jvp_rtol < 0.0:
        raise ValueError("jvp_rtol must be finite and nonnegative")
    if not math.isfinite(jvp_atol) or jvp_atol < 0.0:
        raise ValueError("jvp_atol must be finite and nonnegative")
    if score_logit_jacobian_fn is not None and score_logits_fn is None:
        raise ValueError(
            "score_logit_jacobian_fn requires score_logits_fn for consistency audit"
        )
    packet_provenance = provenance if provenance is not None else PacketProvenance()
    if not isinstance(packet_provenance, PacketProvenance):
        raise TypeError("provenance must be a PacketProvenance instance")
    _validate_stencil_bounds(point, h, packet_provenance)

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
    incommensurate_levels = [
        _jet_tensors(
            q_fn,
            point,
            h / (math.sqrt(2.0) * factor),
            score_logits_fn=score_logits_fn,
            score_logit_jacobian_fn=score_logit_jacobian_fn,
        )
        for factor in (1.0, 2.0, 4.0)
    ]
    q0 = levels[0][0]

    def extrapolate(
        source_levels: Sequence[tuple], coarse_index: int, fine_index: int
    ) -> list[Tensor3]:
        coarse = source_levels[coarse_index]
        fine = source_levels[fine_index]
        return [
            (4.0 * fine[slot] - coarse[slot]) / 3.0
            for slot in (1, 2, 3)
        ]

    first_ext = extrapolate(levels, 0, 1)
    second_ext = extrapolate(levels, 1, 2)
    incommensurate_first_ext = extrapolate(incommensurate_levels, 0, 1)
    incommensurate_second_ext = extrapolate(incommensurate_levels, 1, 2)
    refinement_diagnostics = _maximum_diagnostics(
        _difference_diagnostics(
            first_ext,
            second_ext,
            atol=refinement_atol,
            rtol=refinement_rtol,
        ),
        _difference_diagnostics(
            incommensurate_first_ext,
            incommensurate_second_ext,
            atol=refinement_atol,
            rtol=refinement_rtol,
        ),
        _difference_diagnostics(
            second_ext,
            incommensurate_second_ext,
            atol=refinement_atol,
            rtol=refinement_rtol,
        ),
    )
    (
        refinement_error,
        refinement_absolute_error,
        refinement_tolerance_ratio,
    ) = refinement_diagnostics

    metric = 0.5 * (second_ext[0] + second_ext[0].T)
    first_kind = second_ext[1]
    cubic = second_ext[2]
    cubic_audit = levels[2][4]
    finest_jvp_audits = (levels[2][5], incommensurate_levels[2][5])
    if all(audit is None for audit in finest_jvp_audits):
        jvp_consistency_relative_error = None
        jvp_consistency_absolute_error = None
        jvp_consistency_tolerance_ratio = None
    else:
        if any(audit is None for audit in finest_jvp_audits):
            raise RuntimeError("inconsistent exact-JVP audit construction")
        relative_errors = []
        absolute_errors = []
        tolerance_ratios = []
        for audit in finest_jvp_audits:
            assert audit is not None
            jvp_difference, jvp_scale = audit
            relative_errors.append(
                0.0
                if jvp_difference == 0.0
                else math.inf
                if jvp_scale == 0.0
                else jvp_difference / jvp_scale
            )
            absolute_errors.append(jvp_difference)
            jvp_tolerance = jvp_atol + jvp_rtol * jvp_scale
            tolerance_ratios.append(
                0.0
                if jvp_difference == 0.0
                else math.inf
                if jvp_tolerance == 0.0
                else jvp_difference / jvp_tolerance
            )
        jvp_consistency_relative_error = max(relative_errors)
        jvp_consistency_absolute_error = max(absolute_errors)
        jvp_consistency_tolerance_ratio = max(tolerance_ratios)

    primary_finite = all(
        np.all(np.isfinite(block)) for block in (metric, first_kind, cubic)
    )
    reason, eigenvalues = _gate_reason(
        metric,
        metric_floor=metric_floor,
        metric_relative_floor=metric_relative_floor,
        refinement_tolerance_ratio=refinement_tolerance_ratio,
        jvp_consistency_tolerance_ratio=jvp_consistency_tolerance_ratio,
    )
    if not primary_finite:
        reason = "nonfinite"
    accepted = reason == ""

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
        accepted=accepted,
        rejection_reason=reason,
        metric_floor=metric_floor,
        metric_relative_floor=metric_relative_floor,
        refinement_rtol=refinement_rtol,
        refinement_atol=refinement_atol,
        refinement_absolute_error=refinement_absolute_error,
        refinement_tolerance_ratio=refinement_tolerance_ratio,
        jvp_consistency_relative_error=jvp_consistency_relative_error,
        jvp_consistency_absolute_error=jvp_consistency_absolute_error,
        jvp_consistency_tolerance_ratio=jvp_consistency_tolerance_ratio,
        jvp_rtol=jvp_rtol,
        jvp_atol=jvp_atol,
        provenance=packet_provenance,
    )


def build_packet_from_logits(
    logits_fn: LogitMap,
    z: Sequence[float] | Vector,
    step: float,
    *,
    logit_jacobian_fn: LogitJacobianMap | None = None,
    metric_floor: float = 1e-10,
    metric_relative_floor: float = 1e-12,
    refinement_rtol: float = 1e-3,
    refinement_atol: float = 1e-6,
    jvp_rtol: float = 1e-5,
    jvp_atol: float = 0.0,
    provenance: PacketProvenance | None = None,
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
        metric_relative_floor=metric_relative_floor,
        refinement_rtol=refinement_rtol,
        refinement_atol=refinement_atol,
        jvp_rtol=jvp_rtol,
        jvp_atol=jvp_atol,
        score_logits_fn=logits_fn,
        score_logit_jacobian_fn=logit_jacobian_fn,
        provenance=provenance,
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
    *,
    evidence_status: str,
    evidence_source: str,
) -> TransportBoundAudit:
    """Section 11.3 proven packet-to-transport bound.

    ``L_gamma * exp((M_T + M_S) L_gamma) * (delta_pack + (H_T + H_S) h^rho)``
    The result cannot be constructed without declaring whether the continuum
    constants are ``certified`` or ``sampled`` and giving a nonempty source.
    This prevents an empirical grid modulus from being serialized or reported
    as a mathematical certificate by omission.
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
    if evidence_status not in {"certified", "sampled"}:
        raise ValueError("evidence_status must be 'certified' or 'sampled'")
    if not isinstance(evidence_source, str) or not evidence_source.strip():
        raise ValueError("evidence_source must be a nonempty string")

    interpolation = (holder_teacher + holder_student) * fill_distance**holder_exponent
    try:
        growth = math.exp((sup_teacher + sup_student) * path_length)
    except OverflowError:
        value = math.inf
    else:
        value = path_length * growth * (packet_defect + interpolation)
    return TransportBoundAudit(
        value=value,
        evidence_status=evidence_status,
        evidence_source=evidence_source.strip(),
    )


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
    """Squared defect for ``(chart_dim, outcomes)`` logit Jacobians."""

    student = np.asarray(student_jacobian, dtype=np.float64)
    teacher = np.asarray(teacher_jacobian, dtype=np.float64)
    if student.ndim != 2 or teacher.ndim != 2 or student.shape != teacher.shape:
        raise ValueError(
            "student and teacher Jacobians must share (chart_dim, outcomes) shape"
        )
    if student.shape[1] < 2:
        raise ValueError("a logit Jacobian requires at least two outcomes")
    if not np.all(np.isfinite(student)) or not np.all(np.isfinite(teacher)):
        raise ValueError("logit Jacobians must be finite")
    student_centered = student - student.mean(axis=1, keepdims=True)
    teacher_centered = teacher - teacher.mean(axis=1, keepdims=True)
    difference = student_centered - teacher_centered
    return float(np.sum(difference * difference))


def sqrt_jacobian_loss(student_jacobian: Matrix, teacher_jacobian: Matrix) -> float:
    """Squared defect of square-root-output Jacobians (Section 4.2)."""

    difference = student_jacobian - teacher_jacobian
    return float(np.sum(difference * difference))


def sobolev_grid_audit(values: Matrix, spacing: float, order: int) -> float:
    """Finite-difference squared ``H^order`` estimate on a uniform 1-D grid.

    ``values[t, a]`` samples a curve in ``ell_2`` at grid points with the
    given spacing. Derivatives use second-order edge-aware gradients on the
    full grid and integrals use composite Simpson rules (with a 3/8 tail when
    needed). This is a numerical audit of a surrogate, not a certificate for
    the underlying map (Section 11.4).
    """

    value_array = np.asarray(values, dtype=np.float64)
    if value_array.ndim != 2:
        raise ValueError("values must be a (grid, outcomes) array")
    if not np.all(np.isfinite(value_array)):
        raise ValueError("values must be finite")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    if isinstance(order, bool) or int(order) != order or order < 0:
        raise ValueError("order must be a nonnegative integer")
    order = int(order)
    minimum_points = 2 if order == 0 else max(3, order + 2)
    if value_array.shape[0] < minimum_points:
        raise ValueError(
            f"order {order} requires at least {minimum_points} grid points"
        )

    def integrate_uniform(samples: Vector) -> float:
        count = samples.size
        if count == 2:
            return 0.5 * spacing * float(samples[0] + samples[1])

        def simpson_one_third(block: Vector) -> float:
            return (spacing / 3.0) * float(
                block[0]
                + block[-1]
                + 4.0 * np.sum(block[1:-1:2])
                + 2.0 * np.sum(block[2:-1:2])
            )

        if count % 2 == 1:
            return simpson_one_third(samples)
        if count == 4:
            return (3.0 * spacing / 8.0) * float(
                samples[0] + 3.0 * samples[1] + 3.0 * samples[2] + samples[3]
            )
        prefix = simpson_one_third(samples[:-3])
        tail = (3.0 * spacing / 8.0) * float(
            samples[-4]
            + 3.0 * samples[-3]
            + 3.0 * samples[-2]
            + samples[-1]
        )
        return prefix + tail

    total = 0.0
    current = value_array
    for derivative_order in range(order + 1):
        squared = np.sum(current * current, axis=1)
        total += integrate_uniform(squared)
        if derivative_order < order:
            current = np.gradient(
                current,
                spacing,
                axis=0,
                edge_order=2,
            )
    return total


def sufficiency_decomposition(
    fiber_ids: Sequence[int],
    effects: Matrix,
    transferred: Matrix,
    metric: Matrix,
    *,
    estimand: str,
) -> tuple[float, float, float]:
    """Conditional-variance split of Section 9.4 on sampled discrete fibers.

    ``estimand='empirical'`` returns the exact decomposition of the empirical
    distribution. ``estimand='population_unbiased'`` applies the finite-fiber
    ANOVA correction under IID sampling within each fiber; the corrected
    mismatch component can be negative in a finite sample even though its
    population target is nonnegative. In both modes the components sum to the
    direct empirical total exactly.
    """

    labels = np.asarray(fiber_ids)
    effect_array = np.asarray(effects, dtype=np.float64)
    transfer_array = np.asarray(transferred, dtype=np.float64)
    metric_array = np.asarray(metric, dtype=np.float64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("fiber_ids must be a nonempty one-dimensional sequence")
    if effect_array.ndim != 2 or transfer_array.shape != effect_array.shape:
        raise ValueError("effects and transferred must share a two-dimensional shape")
    if effect_array.shape[0] != labels.size:
        raise ValueError("fiber_ids length must equal the number of effect rows")
    dimension = effect_array.shape[1]
    if metric_array.shape != (dimension, dimension):
        raise ValueError("metric shape must match the effect dimension")
    if not all(
        np.all(np.isfinite(block))
        for block in (effect_array, transfer_array, metric_array)
    ):
        raise ValueError("effects, transferred values, and metric must be finite")
    if not np.allclose(metric_array, metric_array.T, rtol=0.0, atol=1e-12):
        raise ValueError("metric must be symmetric")
    if float(np.linalg.eigvalsh(metric_array).min()) <= 0.0:
        raise ValueError("metric must be positive definite")
    if estimand not in {"empirical", "population_unbiased"}:
        raise ValueError("estimand must be 'empirical' or 'population_unbiased'")
    for fiber in np.unique(labels):
        block = transfer_array[labels == fiber]
        if not np.allclose(block, block[0]):
            raise ValueError("transferred field must be constant on each fiber")

    def norm_sq(rows: Matrix) -> Vector:
        return np.einsum("nd,de,ne->n", rows, metric_array, rows)

    conditional_mean = np.empty_like(effect_array)
    for fiber in np.unique(labels):
        mask = labels == fiber
        conditional_mean[mask] = effect_array[mask].mean(axis=0)

    total = float(np.mean(norm_sq(effect_array - transfer_array)))
    if estimand == "empirical":
        insufficiency = float(np.mean(norm_sq(effect_array - conditional_mean)))
        mismatch = float(np.mean(norm_sq(conditional_mean - transfer_array)))
        return insufficiency, mismatch, total

    insufficiency = 0.0
    mismatch = 0.0
    sample_count = labels.size
    for fiber in np.unique(labels):
        mask = labels == fiber
        fiber_count = int(np.count_nonzero(mask))
        if fiber_count < 2:
            raise ValueError(
                "population_unbiased estimand requires at least two rows per fiber"
            )
        residuals = effect_array[mask] - conditional_mean[mask]
        sample_variance = float(np.sum(norm_sq(residuals)) / (fiber_count - 1))
        weight = fiber_count / sample_count
        fiber_mismatch = float(
            norm_sq((conditional_mean[mask][0] - transfer_array[mask][0])[None, :])[0]
        )
        insufficiency += weight * sample_variance
        mismatch += weight * (fiber_mismatch - sample_variance / fiber_count)
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


_PACKET_BODY_FIELDS = frozenset(
    {
        "schema_version",
        "step",
        "z",
        "q",
        "metric",
        "first_kind",
        "cubic",
        "cubic_audit",
        "cubic_audit_finite",
        "metric_eigenvalues",
        "refinement_error",
        "metric_floor",
        "metric_relative_floor",
        "refinement_rtol",
        "refinement_atol",
        "refinement_absolute_error",
        "refinement_tolerance_ratio",
        "jvp_consistency_checked",
        "jvp_consistency_relative_error",
        "jvp_consistency_absolute_error",
        "jvp_consistency_tolerance_ratio",
        "jvp_rtol",
        "jvp_atol",
        "provenance",
        "accepted",
        "rejection_reason",
        "serialization_quantization_error",
        "serialization_metric_eigenvalue_error_bound",
    }
)

_PROVENANCE_FIELDS = frozenset(
    {
        "teacher_model_id",
        "teacher_revision",
        "tokenizer_hash",
        "outcome_map_hash",
        "chart_id",
        "chart_bounds",
        "context_id",
        "inference_dtype",
        "reproducible",
    }
)


def _provenance_to_dict(provenance: PacketProvenance) -> dict:
    return {
        "teacher_model_id": provenance.teacher_model_id,
        "teacher_revision": provenance.teacher_revision,
        "tokenizer_hash": provenance.tokenizer_hash,
        "outcome_map_hash": provenance.outcome_map_hash,
        "chart_id": provenance.chart_id,
        "chart_bounds": [list(bounds) for bounds in provenance.chart_bounds],
        "context_id": provenance.context_id,
        "inference_dtype": provenance.inference_dtype,
        "reproducible": provenance.reproducible,
    }


def _provenance_from_dict(data: object, chart_dimension: int) -> PacketProvenance:
    if not isinstance(data, dict) or set(data) != _PROVENANCE_FIELDS:
        raise ValueError("packet provenance fields do not match the declared schema")
    string_fields = (
        "teacher_model_id",
        "teacher_revision",
        "tokenizer_hash",
        "outcome_map_hash",
        "chart_id",
        "context_id",
        "inference_dtype",
    )
    if any(not isinstance(data[key], str) or not data[key] for key in string_fields):
        raise ValueError("packet provenance string fields must be nonempty")
    if not isinstance(data["reproducible"], bool):
        raise ValueError("packet provenance reproducible flag must be boolean")
    raw_bounds = data["chart_bounds"]
    if not isinstance(raw_bounds, list):
        raise ValueError("packet chart_bounds must be a list")
    bounds: list[tuple[float, float]] = []
    for raw_pair in raw_bounds:
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            raise ValueError("each chart bound must be a [lower, upper] pair")
        lower, upper = (float(raw_pair[0]), float(raw_pair[1]))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("chart bounds must be finite and strictly ordered")
        bounds.append((lower, upper))
    if bounds and len(bounds) != chart_dimension:
        raise ValueError("chart bounds dimension does not match the packet chart")
    if data["reproducible"]:
        if len(bounds) != chart_dimension:
            raise ValueError("reproducible provenance requires complete chart bounds")
        if any("untracked" in data[key].lower() for key in string_fields):
            raise ValueError("reproducible provenance cannot contain untracked fields")
    return PacketProvenance(
        teacher_model_id=data["teacher_model_id"],
        teacher_revision=data["teacher_revision"],
        tokenizer_hash=data["tokenizer_hash"],
        outcome_map_hash=data["outcome_map_hash"],
        chart_id=data["chart_id"],
        chart_bounds=tuple(bounds),
        context_id=data["context_id"],
        inference_dtype=data["inference_dtype"],
        reproducible=data["reproducible"],
    )


def _packet_checksum(body: dict) -> str:
    """Checksum a language-independent binary encoding of the packet body.

    Every field is labeled and length-prefixed. Numeric arrays use C-order
    little-endian IEEE-754 bytes with explicit dimensions; scalar floats use
    little-endian binary64. This avoids dependence on a JSON implementation's
    float spelling while the surrounding packet remains strict JSON.
    """

    if set(body) != _PACKET_BODY_FIELDS:
        raise ValueError("packet body fields do not match the declared schema")
    digest = hashlib.blake2b(digest_size=16, person=b"pcdpack")
    digest.update(b"pcd-packet-6-binary-v1\x00")

    def add_bytes(label: str, payload: bytes) -> None:
        encoded_label = label.encode("utf-8")
        digest.update(struct.pack("<H", len(encoded_label)))
        digest.update(encoded_label)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)

    def add_float(label: str) -> None:
        add_bytes(label, struct.pack("<d", float(body[label])))

    def add_string(label: str) -> None:
        value = body[label]
        if not isinstance(value, str):
            raise ValueError(f"packet field {label!r} must be a string")
        add_bytes(label, value.encode("utf-8"))

    def add_bool(label: str) -> None:
        value = body[label]
        if not isinstance(value, bool):
            raise ValueError(f"packet field {label!r} must be boolean")
        add_bytes(label, struct.pack("<B", int(value)))

    def add_array(label: str, dtype: str) -> None:
        array = np.ascontiguousarray(np.asarray(body[label], dtype=np.dtype(dtype)))
        shape = struct.pack("<I", array.ndim) + b"".join(
            struct.pack("<Q", dimension) for dimension in array.shape
        )
        add_bytes(f"{label}:shape", shape)
        add_bytes(f"{label}:data", array.tobytes(order="C"))

    add_string("schema_version")
    add_float("step")
    add_array("z", "<f4")
    add_array("q", "<f8")
    add_array("metric", "<f4")
    add_array("first_kind", "<f4")
    add_array("cubic", "<f4")
    add_bool("cubic_audit_finite")
    if body["cubic_audit_finite"]:
        add_array("cubic_audit", "<f4")
    elif body["cubic_audit"] is not None:
        raise ValueError("non-finite cubic audit must use a null payload")
    add_array("metric_eigenvalues", "<f8")
    add_float("refinement_error")
    add_float("metric_floor")
    add_float("metric_relative_floor")
    add_float("refinement_rtol")
    add_float("refinement_atol")
    add_float("refinement_absolute_error")
    add_float("refinement_tolerance_ratio")
    add_bool("jvp_consistency_checked")
    if body["jvp_consistency_checked"]:
        add_float("jvp_consistency_relative_error")
        add_float("jvp_consistency_absolute_error")
        add_float("jvp_consistency_tolerance_ratio")
    elif any(
        body[key] is not None
        for key in (
            "jvp_consistency_relative_error",
            "jvp_consistency_absolute_error",
            "jvp_consistency_tolerance_ratio",
        )
    ):
        raise ValueError("unchecked JVP diagnostics must use null payloads")
    add_float("jvp_rtol")
    add_float("jvp_atol")
    provenance = body["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("packet provenance fields do not match the declared schema")
    for label in (
        "teacher_model_id",
        "teacher_revision",
        "tokenizer_hash",
        "outcome_map_hash",
        "chart_id",
        "context_id",
        "inference_dtype",
    ):
        value = provenance[label]
        if not isinstance(value, str):
            raise ValueError(f"packet provenance field {label!r} must be a string")
        add_bytes(f"provenance:{label}", value.encode("utf-8"))
    add_bytes(
        "provenance:chart_bounds",
        np.ascontiguousarray(
            np.asarray(provenance["chart_bounds"], dtype=np.dtype("<f8"))
        ).tobytes(order="C"),
    )
    if not isinstance(provenance["reproducible"], bool):
        raise ValueError("packet provenance reproducible flag must be boolean")
    add_bytes(
        "provenance:reproducible",
        struct.pack("<B", int(provenance["reproducible"])),
    )
    add_bool("accepted")
    add_string("rejection_reason")
    add_float("serialization_quantization_error")
    add_float("serialization_metric_eigenvalue_error_bound")
    return digest.hexdigest()


def _validate_packet_semantics(
    packet: ConnectionPacket,
    *,
    require_reproducible_provenance: bool,
) -> None:
    """Validate shapes, gate metadata, and acceptance as semantic invariants."""

    z = np.asarray(packet.z)
    q = np.asarray(packet.q)
    metric = np.asarray(packet.metric)
    first_kind = np.asarray(packet.first_kind)
    cubic = np.asarray(packet.cubic)
    if z.ndim != 1 or z.size == 0 or q.ndim != 1 or q.size == 0:
        raise ValueError("invalid packet point or probability shape")
    dimension = z.size
    if metric.shape != (dimension, dimension) or any(
        block.shape != (dimension, dimension, dimension)
        for block in (first_kind, cubic)
    ):
        raise ValueError("packet geometric tensor shapes are inconsistent")
    if not np.allclose(metric, metric.T, rtol=1e-7, atol=1e-12):
        raise ValueError("packet metric must be symmetric")
    if not np.allclose(first_kind, first_kind.swapaxes(0, 1), rtol=1e-6, atol=1e-10):
        raise ValueError("packet first-kind tensor must be symmetric in derivative indices")
    cubic_permutations = (
        cubic.swapaxes(0, 1),
        cubic.swapaxes(0, 2),
        cubic.swapaxes(1, 2),
    )
    if any(
        not np.allclose(cubic, permutation, rtol=1e-6, atol=1e-10)
        for permutation in cubic_permutations
    ):
        raise ValueError("packet cubic tensor must be fully symmetric")
    if np.any(q <= 0.0) or not np.isclose(float(q.sum()), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("packet probabilities must lie in the open simplex")

    scalars = (
        packet.step,
        packet.refinement_error,
        packet.metric_floor,
        packet.metric_relative_floor,
        packet.refinement_rtol,
        packet.refinement_atol,
        packet.refinement_absolute_error,
        packet.refinement_tolerance_ratio,
        packet.jvp_rtol,
        packet.jvp_atol,
    )
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("packet gate metadata must be finite")
    if packet.step <= 0.0 or packet.metric_floor <= 0.0:
        raise ValueError("packet step and absolute metric floor must be positive")
    if not 0.0 < packet.metric_relative_floor <= 1.0:
        raise ValueError("packet relative metric floor must lie in (0, 1]")
    if any(
        value < 0.0
        for value in (
            packet.refinement_error,
            packet.refinement_rtol,
            packet.refinement_atol,
            packet.refinement_absolute_error,
            packet.refinement_tolerance_ratio,
            packet.jvp_rtol,
            packet.jvp_atol,
        )
    ):
        raise ValueError("packet errors and tolerances must be nonnegative")

    jvp_values = (
        packet.jvp_consistency_relative_error,
        packet.jvp_consistency_absolute_error,
        packet.jvp_consistency_tolerance_ratio,
    )
    jvp_checked = all(value is not None for value in jvp_values)
    if any(value is not None for value in jvp_values) and not jvp_checked:
        raise ValueError("JVP consistency diagnostics must be all present or all absent")
    if jvp_checked and any(
        not math.isfinite(float(value)) or float(value) < 0.0 for value in jvp_values
    ):
        raise ValueError("JVP consistency diagnostics must be finite and nonnegative")

    provenance = _provenance_from_dict(
        _provenance_to_dict(packet.provenance),
        dimension,
    )
    if require_reproducible_provenance and not provenance.reproducible:
        raise ValueError("serialization requires reproducible packet provenance")
    if provenance.chart_bounds and any(
        not lower <= coordinate <= upper
        for coordinate, (lower, upper) in zip(packet.z, provenance.chart_bounds)
    ):
        raise ValueError("packet chart point lies outside its provenance bounds")

    allowed_reasons = {
        "nonfinite",
        "rank",
        "conditioning",
        "metric_scale",
        "refinement",
        "jvp_consistency",
    }
    if packet.accepted and packet.rejection_reason:
        raise ValueError("an accepted packet cannot have a rejection reason")
    if not packet.accepted and packet.rejection_reason not in allowed_reasons:
        raise ValueError("a rejected packet must retain a recognized rejection reason")

    stored_eigenvalues = np.asarray(packet.metric_eigenvalues, dtype=np.float64)
    if stored_eigenvalues.shape != (dimension,) or not np.all(
        np.isfinite(stored_eigenvalues)
    ):
        raise ValueError("packet metric eigenvalues are invalid")
    expected_reason, recomputed_eigenvalues = _gate_reason(
        metric,
        metric_floor=packet.metric_floor,
        metric_relative_floor=packet.metric_relative_floor,
        refinement_tolerance_ratio=packet.refinement_tolerance_ratio,
        jvp_consistency_tolerance_ratio=(
            float(packet.jvp_consistency_tolerance_ratio) if jvp_checked else None
        ),
    )
    eigenvalue_scale = max(1.0, float(np.linalg.norm(metric, 2)))
    eigenvalue_slack = 32.0 * np.finfo(np.float64).eps * eigenvalue_scale
    if not np.allclose(
        stored_eigenvalues,
        recomputed_eigenvalues,
        rtol=0.0,
        atol=eigenvalue_slack,
    ):
        raise ValueError("packet metric eigenvalues disagree with the metric")

    if packet.accepted and expected_reason:
        accepted_errors = {
            "nonfinite": "accepted packet contains a non-finite metric",
            "jvp_consistency": "accepted packet violates its JVP consistency gate",
            "rank": "accepted packet metric must be positive definite",
            "conditioning": "accepted packet violates its relative conditioning gate",
            "metric_scale": "accepted packet violates its absolute metric-scale gate",
            "refinement": "accepted packet violates its refinement gate",
        }
        raise ValueError(accepted_errors[expected_reason])
    if not packet.accepted and packet.rejection_reason != expected_reason:
        expected_label = expected_reason or "accepted"
        raise ValueError(
            "packet rejection reason does not match the recomputed gate: "
            f"stored {packet.rejection_reason!r}, expected {expected_label!r}"
        )


def packet_to_dict(
    packet: ConnectionPacket,
    *,
    max_quantization_error: float = 1e-6,
    require_reproducible_provenance: bool = True,
) -> dict:
    """Serialize float32 geometry, a float64 output anchor, and all-field checksum.

    The portable schema is strict JSON. A non-finite secondary cubic audit is
    represented by ``null`` plus ``cubic_audit_finite=false``; non-finite
    primary tensors still require a separate rejection-audit channel.
    """

    if packet.schema_version != SCHEMA_VERSION:
        raise ValueError("packet schema version does not match this writer")
    _validate_packet_semantics(
        packet,
        require_reproducible_provenance=require_reproducible_provenance,
    )
    if not math.isfinite(max_quantization_error) or max_quantization_error < 0.0:
        raise ValueError("max_quantization_error must be finite and nonnegative")
    primary_finite = all(
        np.all(np.isfinite(block))
        for block in (
            packet.z,
            packet.q,
            packet.metric,
            packet.first_kind,
            packet.cubic,
            packet.metric_eigenvalues,
        )
    ) and math.isfinite(packet.refinement_error)
    if not primary_finite:
        raise ValueError(
            "packets with non-finite primary values cannot be serialized under the "
            "strict-JSON schema; log them through a separate audit channel"
        )
    measured_quantization_error = quantization_error(packet)
    if measured_quantization_error > max_quantization_error:
        raise ValueError("float32 packet quantization exceeds the declared tolerance")
    payload = _payload_arrays(packet)
    serialized_metric = payload[2].astype(np.float64)
    serialized_reason, serialized_eigenvalues = _gate_reason(
        serialized_metric,
        metric_floor=packet.metric_floor,
        metric_relative_floor=packet.metric_relative_floor,
        refinement_tolerance_ratio=packet.refinement_tolerance_ratio,
        jvp_consistency_tolerance_ratio=packet.jvp_consistency_tolerance_ratio,
    )
    source_reason = "" if packet.accepted else packet.rejection_reason
    if serialized_reason != source_reason:
        serialized_label = serialized_reason or "accepted"
        source_label = source_reason or "accepted"
        raise ValueError(
            "float32 serialization changes the packet gate from "
            f"{source_label!r} to {serialized_label!r}"
        )
    metric_eigenvalue_error_bound = float(
        np.linalg.norm(packet.metric - serialized_metric, 2)
    )
    cubic_audit_finite = bool(np.all(np.isfinite(payload[5])))
    body = {
        "schema_version": packet.schema_version,
        "step": packet.step,
        "z": payload[0].tolist(),
        "q": payload[1].tolist(),
        "metric": payload[2].tolist(),
        "first_kind": payload[3].tolist(),
        "cubic": payload[4].tolist(),
        "cubic_audit": payload[5].tolist() if cubic_audit_finite else None,
        "cubic_audit_finite": cubic_audit_finite,
        "metric_eigenvalues": serialized_eigenvalues.tolist(),
        "refinement_error": packet.refinement_error,
        "metric_floor": packet.metric_floor,
        "metric_relative_floor": packet.metric_relative_floor,
        "refinement_rtol": packet.refinement_rtol,
        "refinement_atol": packet.refinement_atol,
        "refinement_absolute_error": packet.refinement_absolute_error,
        "refinement_tolerance_ratio": packet.refinement_tolerance_ratio,
        "jvp_consistency_checked": packet.jvp_consistency_relative_error is not None,
        "jvp_consistency_relative_error": packet.jvp_consistency_relative_error,
        "jvp_consistency_absolute_error": packet.jvp_consistency_absolute_error,
        "jvp_consistency_tolerance_ratio": packet.jvp_consistency_tolerance_ratio,
        "jvp_rtol": packet.jvp_rtol,
        "jvp_atol": packet.jvp_atol,
        "provenance": _provenance_to_dict(packet.provenance),
        "accepted": packet.accepted,
        "rejection_reason": packet.rejection_reason,
        "serialization_quantization_error": measured_quantization_error,
        "serialization_metric_eigenvalue_error_bound": metric_eigenvalue_error_bound,
    }
    return {**body, "checksum": _packet_checksum(body)}


def packet_from_dict(
    data: dict,
    *,
    require_reproducible_provenance: bool = True,
) -> ConnectionPacket:
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
    ]
    z, q, metric, first_kind, cubic = arrays
    if z.ndim != 1 or z.size == 0 or q.ndim != 1 or q.size == 0:
        raise ValueError("invalid serialized point or probability shape")
    m = z.size
    if metric.shape != (m, m) or any(
        block.shape != (m, m, m) for block in (first_kind, cubic)
    ):
        raise ValueError("serialized geometric tensor shapes are inconsistent")
    cubic_audit_finite = data.get("cubic_audit_finite")
    if not isinstance(cubic_audit_finite, bool):
        raise ValueError("serialized cubic-audit status must be boolean")
    if cubic_audit_finite:
        cubic_audit = np.asarray(data["cubic_audit"], dtype=np.float32)
        if cubic_audit.shape != (m, m, m) or not np.all(np.isfinite(cubic_audit)):
            raise ValueError("finite cubic audit has invalid values or shape")
    else:
        if data.get("cubic_audit") is not None:
            raise ValueError("non-finite cubic audit must be represented by null")
        cubic_audit = np.full((m, m, m), np.nan, dtype=np.float32)
    if not isinstance(data["accepted"], bool):
        raise ValueError("serialized accepted flag must be boolean")
    accepted = data["accepted"]
    if not isinstance(data["rejection_reason"], str):
        raise ValueError("serialized rejection reason must be a string")
    rejection_reason = data["rejection_reason"]
    primary_finite = all(
        np.all(np.isfinite(block)) for block in (z, q, metric, first_kind, cubic)
    )
    if not primary_finite:
        raise ValueError("strict-JSON packet primary values must be finite")
    if np.any(q <= 0.0) or not np.isclose(float(q.sum()), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("serialized probabilities must lie in the open simplex")

    step = float(data["step"])
    refinement_error = float(data["refinement_error"])
    metric_floor = float(data["metric_floor"])
    metric_relative_floor = float(data["metric_relative_floor"])
    refinement_rtol = float(data["refinement_rtol"])
    refinement_atol = float(data["refinement_atol"])
    refinement_absolute_error = float(data["refinement_absolute_error"])
    refinement_tolerance_ratio = float(data["refinement_tolerance_ratio"])
    jvp_rtol = float(data["jvp_rtol"])
    jvp_atol = float(data["jvp_atol"])
    jvp_checked = data.get("jvp_consistency_checked")
    if not isinstance(jvp_checked, bool):
        raise ValueError("serialized JVP consistency status must be boolean")
    jvp_keys = (
        "jvp_consistency_relative_error",
        "jvp_consistency_absolute_error",
        "jvp_consistency_tolerance_ratio",
    )
    if jvp_checked:
        jvp_values = tuple(float(data[key]) for key in jvp_keys)
    else:
        if any(data.get(key) is not None for key in jvp_keys):
            raise ValueError("unchecked JVP diagnostics must use null payloads")
        jvp_values = (None, None, None)
    provenance = _provenance_from_dict(data.get("provenance"), m)
    serialization_quantization_error = float(data["serialization_quantization_error"])
    metric_eigenvalue_error_bound = float(
        data["serialization_metric_eigenvalue_error_bound"]
    )
    if (
        not math.isfinite(serialization_quantization_error)
        or serialization_quantization_error < 0.0
    ):
        raise ValueError("serialized quantization error must be finite and nonnegative")
    if (
        not math.isfinite(metric_eigenvalue_error_bound)
        or metric_eigenvalue_error_bound < 0.0
    ):
        raise ValueError("serialized metric eigenvalue error bound is invalid")

    eigenvalues = np.asarray(data["metric_eigenvalues"], dtype=np.float64)
    recomputed = (
        np.linalg.eigvalsh(metric.astype(np.float64))
        if np.all(np.isfinite(metric))
        else np.full(m, np.nan)
    )
    if eigenvalues.shape != (m,):
        raise ValueError("serialized metric eigenvalues are invalid")
    eigenvalue_scale = max(1.0, float(np.linalg.norm(metric, 2)))
    numeric_slack = 32.0 * np.finfo(np.float64).eps * eigenvalue_scale
    if not np.allclose(eigenvalues, recomputed, rtol=0.0, atol=numeric_slack):
        raise ValueError("serialized metric eigenvalues disagree with the metric")

    packet = ConnectionPacket(
        z=z.astype(np.float64),
        step=step,
        q=q.astype(np.float64),
        metric=metric.astype(np.float64),
        first_kind=first_kind.astype(np.float64),
        cubic=cubic.astype(np.float64),
        cubic_audit=cubic_audit.astype(np.float64),
        metric_eigenvalues=eigenvalues,
        refinement_error=refinement_error,
        accepted=accepted,
        rejection_reason=rejection_reason,
        metric_floor=metric_floor,
        metric_relative_floor=metric_relative_floor,
        refinement_rtol=refinement_rtol,
        refinement_atol=refinement_atol,
        refinement_absolute_error=refinement_absolute_error,
        refinement_tolerance_ratio=refinement_tolerance_ratio,
        jvp_consistency_relative_error=jvp_values[0],
        jvp_consistency_absolute_error=jvp_values[1],
        jvp_consistency_tolerance_ratio=jvp_values[2],
        jvp_rtol=jvp_rtol,
        jvp_atol=jvp_atol,
        provenance=provenance,
        serialization_quantization_error=serialization_quantization_error,
        serialization_metric_eigenvalue_error_bound=metric_eigenvalue_error_bound,
    )
    _validate_packet_semantics(
        packet,
        require_reproducible_provenance=require_reproducible_provenance,
    )
    return packet


def quantization_error(packet: ConnectionPacket) -> float:
    """Relative float32 round-trip error over the geometric payload."""

    worst = 0.0
    for block in (packet.metric, packet.first_kind, packet.cubic):
        roundtrip = block.astype(np.float32).astype(np.float64)
        difference = float(np.linalg.norm(block - roundtrip))
        scale = float(np.linalg.norm(block))
        relative_error = 0.0 if difference == 0.0 else difference / scale
        worst = max(worst, relative_error)
    return worst
