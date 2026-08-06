"""Exact Fisher geometry of a linear softmax decoder.

For logits ``W @ h + b``, the Fisher pullback metric on decoded hidden space is
the Hessian of the log-partition function.  The Riemann curvature of a Hessian
metric depends only on its third derivatives, which here are third cumulants of
the rows of ``W`` under the predicted categorical distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class SoftmaxCurvature:
    """Diagnostics for one decoded-hidden sectional-curvature calculation."""

    sectional_curvature: float
    curvature_numerator: float
    gram_determinant: float
    normalized_gram_determinant: float
    metric_condition_number: float
    metric_min_eigenvalue: float
    metric_max_eigenvalue: float
    solve_relative_residual: float
    solve_condition_weighted_residual: float


class SoftmaxHessianGeometry:
    """Reusable geometry at one point of a linear-softmax decoder family.

    Constructing the metric costs ``O(vocabulary * hidden_dim**2)``.  Reuse this
    object when evaluating several semantic or random planes at the same context.
    """

    def __init__(
        self,
        unembedding: ArrayLike,
        probabilities: ArrayLike,
        *,
        eigenvalue_rtol: float = 1e-12,
        plane_gram_rtol: float = 1e-4,
        solve_residual_rtol: float = 1e-10,
        solve_forward_error_rtol: float = 1e-4,
    ) -> None:
        if not np.isfinite(eigenvalue_rtol) or eigenvalue_rtol <= 0.0:
            raise ValueError("eigenvalue_rtol must be finite and positive")
        if not np.isfinite(plane_gram_rtol) or not 0.0 < plane_gram_rtol < 1.0:
            raise ValueError("plane_gram_rtol must lie between zero and one")
        if not np.isfinite(solve_residual_rtol) or solve_residual_rtol <= 0.0:
            raise ValueError("solve_residual_rtol must be finite and positive")
        if (
            not np.isfinite(solve_forward_error_rtol)
            or solve_forward_error_rtol <= 0.0
        ):
            raise ValueError(
                "solve_forward_error_rtol must be finite and positive"
            )
        weights, p = _inputs(unembedding, probabilities)
        self.hidden_dim = weights.shape[1]
        self.unembedding = weights
        self.probabilities = p
        self.centered = weights - p @ weights
        metric = self.centered.T @ (p[:, None] * self.centered)
        self.metric = 0.5 * (metric + metric.T)
        if not np.all(np.isfinite(self.metric)):
            raise ValueError("the decoded-hidden Fisher metric is not finite")
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.metric)
        self.metric_max_eigenvalue = float(self.eigenvalues[-1])
        self.metric_min_eigenvalue = float(self.eigenvalues[0])
        threshold = eigenvalue_rtol * self.metric_max_eigenvalue
        if (
            self.metric_max_eigenvalue <= 0.0
            or self.metric_min_eigenvalue <= threshold
        ):
            raise ValueError(
                "the decoded-hidden Fisher metric is numerically degenerate; "
                "quotient null directions before computing intrinsic curvature"
            )
        self.metric_condition_number = (
            self.metric_max_eigenvalue / self.metric_min_eigenvalue
        )
        self.eigenvalue_rtol = eigenvalue_rtol
        self.plane_gram_rtol = plane_gram_rtol
        self.solve_residual_rtol = solve_residual_rtol
        self.solve_forward_error_rtol = solve_forward_error_rtol

    def cubic_contraction(self, u: ArrayLike, v: ArrayLike) -> Vector:
        """Return ``C(u, v, .)`` while reusing the centered unembedding."""

        u_vec = _direction(u, self.hidden_dim, "u")
        v_vec = _direction(v, self.hidden_dim, "v")
        score_u = self.centered @ u_vec
        score_v = self.centered @ v_vec
        return self.centered.T @ (
            self.probabilities * score_u * score_v
        )

    def cubic_operator(self, u: ArrayLike) -> Matrix:
        """Return ``A_u`` defined by ``G(A_u v, w) = C(u, v, w)``.

        The operator is self-adjoint with respect to the Fisher metric, although
        it need not be symmetric in the ambient Euclidean coordinates.
        """

        u_vec = _direction(u, self.hidden_dim, "u")
        score_u = self.centered @ u_vec
        lowered = self.centered.T @ (
            (self.probabilities * score_u)[:, None] * self.centered
        )
        lowered = 0.5 * (lowered + lowered.T)
        return self.solve_metric(lowered)

    def solve_metric(self, right_hand_side: ArrayLike) -> Matrix:
        """Solve a Fisher linear system subject to both numerical gates."""

        rhs = np.asarray(right_hand_side, dtype=np.float64)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.hidden_dim:
            raise ValueError(
                "right_hand_side must have the Fisher metric's leading dimension"
            )
        if not np.all(np.isfinite(rhs)):
            raise ValueError("right_hand_side must contain only finite values")
        solution, _, _ = self._checked_metric_solve(
            self.metric,
            rhs,
            self.metric_condition_number,
        )
        return solution

    def curvature_operator(self, u: ArrayLike, v: ArrayLike) -> Matrix:
        """Return the Levi--Civita operator ``R(u, v)``.

        For this Hessian metric, ``R(u, v) = -[A_u, A_v] / 4`` under the
        convention for which the saturated categorical simplex has curvature
        ``+1/4``.
        """

        operator_u = self.cubic_operator(u)
        operator_v = self.cubic_operator(v)
        return -0.25 * (
            operator_u @ operator_v - operator_v @ operator_u
        )

    def sectional_curvature(self, u: ArrayLike, v: ArrayLike) -> SoftmaxCurvature:
        """Compute exact Levi--Civita sectional curvature in ``span(u, v)``."""

        u_vec = _direction(u, self.hidden_dim, "u")
        v_vec = _direction(v, self.hidden_dim, "v")
        score_u = self.centered @ u_vec
        score_v = self.centered @ v_vec
        weighted = self.probabilities
        c_uu = self.centered.T @ (weighted * score_u * score_u)
        c_uv = self.centered.T @ (weighted * score_u * score_v)
        c_vv = self.centered.T @ (weighted * score_v * score_v)
        return self._sectional_curvature_from_moments(
            u_vec,
            v_vec,
            self.metric,
            c_uu,
            c_uv,
            c_vv,
        )

    def sectional_curvature_chunked(
        self,
        u: ArrayLike,
        v: ArrayLike,
        *,
        chunk_size: int,
    ) -> SoftmaxCurvature:
        """Recompute curvature by independent vocabulary-block accumulation.

        This is a numerical audit path rather than a new estimator.  It changes
        the floating-point summation order for the metric and all three cubic
        contractions, so agreement with :meth:`sectional_curvature` checks that
        a reported value is not an artifact of one full-vocabulary matrix
        contraction.
        """

        if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        u_vec = _direction(u, self.hidden_dim, "u")
        v_vec = _direction(v, self.hidden_dim, "v")
        metric = np.zeros((self.hidden_dim, self.hidden_dim), dtype=np.float64)
        c_uu = np.zeros(self.hidden_dim, dtype=np.float64)
        c_uv = np.zeros(self.hidden_dim, dtype=np.float64)
        c_vv = np.zeros(self.hidden_dim, dtype=np.float64)
        vocabulary_size = self.unembedding.shape[0]
        weighted_mean = np.zeros(self.hidden_dim, dtype=np.float64)
        for start in range(0, vocabulary_size, int(chunk_size)):
            stop = min(start + int(chunk_size), vocabulary_size)
            weighted_mean += (
                self.probabilities[start:stop]
                @ self.unembedding[start:stop]
            )
        for start in range(0, vocabulary_size, int(chunk_size)):
            stop = min(start + int(chunk_size), vocabulary_size)
            centered = self.unembedding[start:stop] - weighted_mean
            probabilities = self.probabilities[start:stop]
            score_u = centered @ u_vec
            score_v = centered @ v_vec
            metric += centered.T @ (probabilities[:, None] * centered)
            c_uu += centered.T @ (probabilities * score_u * score_u)
            c_uv += centered.T @ (probabilities * score_u * score_v)
            c_vv += centered.T @ (probabilities * score_v * score_v)
        metric = 0.5 * (metric + metric.T)
        return self._sectional_curvature_from_moments(
            u_vec,
            v_vec,
            metric,
            c_uu,
            c_uv,
            c_vv,
        )

    def _sectional_curvature_from_moments(
        self,
        u_vec: Vector,
        v_vec: Vector,
        metric: Matrix,
        c_uu: Vector,
        c_uv: Vector,
        c_vv: Vector,
    ) -> SoftmaxCurvature:
        """Assemble sectional curvature from a metric and cubic contractions."""

        if not (
            np.all(np.isfinite(metric))
            and np.all(np.isfinite(c_uu))
            and np.all(np.isfinite(c_uv))
            and np.all(np.isfinite(c_vv))
        ):
            raise ValueError("curvature moments must contain only finite values")
        eigenvalues = np.linalg.eigvalsh(metric)
        metric_max_eigenvalue = float(eigenvalues[-1])
        metric_min_eigenvalue = float(eigenvalues[0])
        if (
            metric_max_eigenvalue <= 0.0
            or metric_min_eigenvalue
            <= self.eigenvalue_rtol * metric_max_eigenvalue
        ):
            raise ValueError(
                "the decoded-hidden Fisher metric is numerically degenerate; "
                "quotient null directions before computing intrinsic curvature"
            )
        metric_condition_number = (
            metric_max_eigenvalue / metric_min_eigenvalue
        )
        guu = float(u_vec @ metric @ u_vec)
        guv = float(u_vec @ metric @ v_vec)
        gvv = float(v_vec @ metric @ v_vec)
        gram_determinant = guu * gvv - guv * guv
        plane_scale = max(guu * gvv, np.finfo(np.float64).tiny)
        normalized_gram_determinant = gram_determinant / plane_scale
        if normalized_gram_determinant < self.plane_gram_rtol:
            raise ValueError(
                "u and v fail the pre-specified normalized Fisher-Gram gate"
            )

        covectors = np.column_stack((c_uu, c_uv, c_vv))
        raised, solve_residual, condition_weighted_residual = (
            self._checked_metric_solve(
                metric,
                covectors,
                metric_condition_number,
            )
        )
        uv_square = float(c_uv @ raised[:, 1])
        uu_vv = 0.5 * float(c_uu @ raised[:, 2] + c_vv @ raised[:, 0])
        numerator = 0.25 * (uv_square - uu_vv)
        sectional_curvature = numerator / gram_determinant
        if not np.isfinite(sectional_curvature):
            raise ValueError("sectional curvature is not finite")

        return SoftmaxCurvature(
            sectional_curvature=sectional_curvature,
            curvature_numerator=numerator,
            gram_determinant=gram_determinant,
            normalized_gram_determinant=normalized_gram_determinant,
            metric_condition_number=metric_condition_number,
            metric_min_eigenvalue=metric_min_eigenvalue,
            metric_max_eigenvalue=metric_max_eigenvalue,
            solve_relative_residual=solve_residual,
            solve_condition_weighted_residual=condition_weighted_residual,
        )

    def _checked_metric_solve(
        self,
        metric: Matrix,
        right_hand_side: Matrix,
        condition_number: float,
    ) -> tuple[Matrix, float, float]:
        """Return a solve and backward/condition-weighted residual diagnostics."""

        solution = np.linalg.solve(metric, right_hand_side)
        residual = metric @ solution - right_hand_side
        residual_denominator = (
            np.linalg.norm(metric, ord=2)
            * np.linalg.norm(solution, ord=2)
            + np.linalg.norm(right_hand_side, ord=2)
        )
        relative_residual = float(
            np.linalg.norm(residual, ord=2)
            / max(residual_denominator, np.finfo(np.float64).tiny)
        )
        condition_weighted_residual = condition_number * relative_residual
        if not np.isfinite(relative_residual):
            raise ValueError("the Fisher linear-solve residual is not finite")
        if relative_residual > self.solve_residual_rtol:
            raise ValueError(
                "the Fisher linear solve failed its relative-residual gate: "
                f"{relative_residual:.3e} > {self.solve_residual_rtol:.3e}"
            )
        if (
            not np.isfinite(condition_weighted_residual)
            or condition_weighted_residual > self.solve_forward_error_rtol
        ):
            raise ValueError(
                "the Fisher linear solve failed its condition-weighted "
                "residual gate: "
                f"{condition_weighted_residual:.3e} > "
                f"{self.solve_forward_error_rtol:.3e}"
            )
        return solution, relative_residual, condition_weighted_residual

    def random_fisher_orthonormal_plane(
        self,
        rng: np.random.Generator,
    ) -> tuple[Vector, Vector]:
        """Sample a coordinate-invariant isotropic plane and Fisher-orthonormalize it."""

        gaussian = rng.normal(size=(self.hidden_dim, 2))
        inverse_square_root = (
            self.eigenvectors
            @ np.diag(1.0 / np.sqrt(self.eigenvalues))
            @ self.eigenvectors.T
        )
        directions = inverse_square_root @ gaussian
        gram = directions.T @ self.metric @ directions
        values, vectors = np.linalg.eigh(gram)
        if float(values[0]) <= self.eigenvalue_rtol * float(values[-1]):
            raise ValueError("random directions produced a degenerate plane")
        gram_inverse_square_root = (
            vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
        )
        normalized = directions @ gram_inverse_square_root
        return normalized[:, 0], normalized[:, 1]

    def fisher_orthonormalize_plane(
        self,
        u: ArrayLike,
        v: ArrayLike,
    ) -> tuple[Vector, Vector]:
        """Return a Fisher-orthonormal basis for the same accepted plane."""

        directions = np.column_stack(
            (
                _direction(u, self.hidden_dim, "u"),
                _direction(v, self.hidden_dim, "v"),
            )
        )
        gram = directions.T @ self.metric @ directions
        scale = max(float(gram[0, 0] * gram[1, 1]), np.finfo(float).tiny)
        normalized_determinant = float(np.linalg.det(gram)) / scale
        if normalized_determinant < self.plane_gram_rtol:
            raise ValueError(
                "u and v fail the pre-specified normalized Fisher-Gram gate"
            )
        values, vectors = np.linalg.eigh(gram)
        if float(values[0]) <= 0.0:
            raise ValueError("u and v do not span a positive Fisher plane")
        inverse_square_root = (
            vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
        )
        normalized = directions @ inverse_square_root
        return normalized[:, 0], normalized[:, 1]

    def spectrum_matched_control_plane(
        self,
        u: ArrayLike,
        v: ArrayLike,
        rng: np.random.Generator,
        *,
        eigenspace_rtol: float = 1e-6,
    ) -> tuple[Vector, Vector]:
        """Randomize a plane while matching its Fisher spectral leverage.

        The source plane is Fisher-orthonormalized and whitened.  Eigenvalues are
        partitioned at relative eigengaps larger than ``eigenspace_rtol``.  A
        Haar orthogonal rotation is applied inside every multi-axis band; a
        singleton receives a random sign.  This preserves each band's 2-by-2
        leverage contribution and is invariant to the arbitrary basis chosen
        inside repeated eigenspaces.  It is therefore a block-spectrum-matched
        control, unlike a Fisher-Haar plane.
        """

        bands = self.eigenvalue_bands(eigenspace_rtol=eigenspace_rtol)
        source_u, source_v = self.fisher_orthonormalize_plane(u, v)
        source = np.column_stack((source_u, source_v))
        whitened = (
            np.sqrt(self.eigenvalues)[:, None]
            * (self.eigenvectors.T @ source)
        )
        randomized = np.empty_like(whitened)
        for start, stop in bands:
            width = stop - start
            if width == 1:
                randomized[start:stop] = (
                    rng.choice(np.array([-1.0, 1.0]))
                    * whitened[start:stop]
                )
                continue
            gaussian = rng.normal(size=(width, width))
            rotation, triangular = np.linalg.qr(gaussian)
            diagonal_signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
            rotation = rotation * diagonal_signs
            randomized[start:stop] = rotation @ whitened[start:stop]
        reflected = self.eigenvectors @ (
            randomized / np.sqrt(self.eigenvalues)[:, None]
        )
        return reflected[:, 0], reflected[:, 1]

    def eigenvalue_bands(
        self,
        *,
        eigenspace_rtol: float = 1e-6,
    ) -> tuple[tuple[int, int], ...]:
        """Return half-open eigengap bands used by matched controls."""

        if (
            not np.isfinite(eigenspace_rtol)
            or eigenspace_rtol < 0.0
            or eigenspace_rtol >= 1.0
        ):
            raise ValueError("eigenspace_rtol must be finite and in [0, 1)")
        boundaries: list[tuple[int, int]] = []
        start = 0
        for index in range(1, self.hidden_dim):
            lower = float(self.eigenvalues[index - 1])
            upper = float(self.eigenvalues[index])
            relative_gap = (upper - lower) / max(upper, np.finfo(float).tiny)
            if relative_gap > eigenspace_rtol:
                boundaries.append((start, index))
                start = index
        boundaries.append((start, self.hidden_dim))
        return tuple(boundaries)


def _inputs(
    unembedding: ArrayLike,
    probabilities: ArrayLike,
) -> tuple[Matrix, Vector]:
    weights = np.asarray(unembedding, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("unembedding must have shape (vocabulary, hidden_dim)")
    if p.ndim != 1 or p.shape[0] != weights.shape[0]:
        raise ValueError("probabilities must have one entry per vocabulary row")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(p)):
        raise ValueError("inputs must contain only finite values")
    if np.any(p <= 0.0):
        raise ValueError("probabilities must lie in the open simplex")
    if not np.isclose(float(np.sum(p)), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("probabilities must sum to one")
    return weights, p


def _direction(value: ArrayLike, hidden_dim: int, name: str) -> Vector:
    direction = np.asarray(value, dtype=np.float64)
    if direction.ndim != 1 or direction.shape[0] != hidden_dim:
        raise ValueError(f"{name} must have shape ({hidden_dim},)")
    if not np.all(np.isfinite(direction)):
        raise ValueError(f"{name} must contain only finite values")
    return direction


def centered_unembedding(
    unembedding: ArrayLike,
    probabilities: ArrayLike,
) -> Matrix:
    """Center token-row features by their probability-weighted mean."""

    weights, p = _inputs(unembedding, probabilities)
    return weights - p @ weights


def softmax_fisher_metric(
    unembedding: ArrayLike,
    probabilities: ArrayLike,
) -> Matrix:
    """Return ``W.T @ (diag(p) - p p.T) @ W`` without forming ``diag(p)``."""

    weights, p = _inputs(unembedding, probabilities)
    centered = weights - p @ weights
    metric = centered.T @ (p[:, None] * centered)
    return 0.5 * (metric + metric.T)


def softmax_cubic_contraction(
    unembedding: ArrayLike,
    probabilities: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
) -> Vector:
    """Return the covector ``C(u, v, .)`` for the third cumulant tensor."""

    weights, p = _inputs(unembedding, probabilities)
    hidden_dim = weights.shape[1]
    u_vec = _direction(u, hidden_dim, "u")
    v_vec = _direction(v, hidden_dim, "v")
    centered = weights - p @ weights
    score_u = centered @ u_vec
    score_v = centered @ v_vec
    return centered.T @ (p * score_u * score_v)


def softmax_sectional_curvature(
    unembedding: ArrayLike,
    probabilities: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    *,
    eigenvalue_rtol: float = 1e-12,
    plane_gram_rtol: float = 1e-4,
    solve_residual_rtol: float = 1e-10,
    solve_forward_error_rtol: float = 1e-4,
) -> SoftmaxCurvature:
    """Compute exact Levi--Civita sectional curvature in ``span(u, v)``.

    The sign convention is the one used by Totaro (2004), for which the full
    categorical Fisher simplex has sectional curvature ``+1/4``.  The function
    requires a nondegenerate Fisher metric and a nondegenerate two-plane; it
    deliberately does not regularize either because ridge regularization changes
    the geometry being measured.
    """

    geometry = SoftmaxHessianGeometry(
        unembedding,
        probabilities,
        eigenvalue_rtol=eigenvalue_rtol,
        plane_gram_rtol=plane_gram_rtol,
        solve_residual_rtol=solve_residual_rtol,
        solve_forward_error_rtol=solve_forward_error_rtol,
    )
    return geometry.sectional_curvature(u, v)
