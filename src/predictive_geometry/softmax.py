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
    metric_condition_number: float
    metric_min_eigenvalue: float
    metric_max_eigenvalue: float
    solve_relative_residual: float


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
    ) -> None:
        if eigenvalue_rtol <= 0.0:
            raise ValueError("eigenvalue_rtol must be positive")
        weights, p = _inputs(unembedding, probabilities)
        self.hidden_dim = weights.shape[1]
        self.probabilities = p
        self.centered = weights - p @ weights
        metric = self.centered.T @ (p[:, None] * self.centered)
        self.metric = 0.5 * (metric + metric.T)
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
        return np.linalg.solve(self.metric, lowered)

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
        guu = float(u_vec @ self.metric @ u_vec)
        guv = float(u_vec @ self.metric @ v_vec)
        gvv = float(v_vec @ self.metric @ v_vec)
        gram_determinant = guu * gvv - guv * guv
        plane_scale = max(guu * gvv, np.finfo(np.float64).tiny)
        if gram_determinant <= self.eigenvalue_rtol * plane_scale:
            raise ValueError(
                "u and v do not span a numerically nondegenerate Fisher plane"
            )

        score_u = self.centered @ u_vec
        score_v = self.centered @ v_vec
        weighted = self.probabilities
        c_uu = self.centered.T @ (weighted * score_u * score_u)
        c_uv = self.centered.T @ (weighted * score_u * score_v)
        c_vv = self.centered.T @ (weighted * score_v * score_v)
        covectors = np.column_stack((c_uu, c_uv, c_vv))
        raised = np.linalg.solve(self.metric, covectors)
        residual = self.metric @ raised - covectors
        residual_denominator = (
            np.linalg.norm(self.metric, ord=2)
            * np.linalg.norm(raised, ord=2)
            + np.linalg.norm(covectors, ord=2)
        )
        solve_residual = float(
            np.linalg.norm(residual, ord=2)
            / max(residual_denominator, np.finfo(np.float64).tiny)
        )
        uv_square = float(c_uv @ raised[:, 1])
        uu_vv = 0.5 * float(c_uu @ raised[:, 2] + c_vv @ raised[:, 0])
        numerator = 0.25 * (uv_square - uu_vv)

        return SoftmaxCurvature(
            sectional_curvature=numerator / gram_determinant,
            curvature_numerator=numerator,
            gram_determinant=gram_determinant,
            metric_condition_number=self.metric_condition_number,
            metric_min_eigenvalue=self.metric_min_eigenvalue,
            metric_max_eigenvalue=self.metric_max_eigenvalue,
            solve_relative_residual=solve_residual,
        )

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
    )
    return geometry.sectional_curvature(u, v)
