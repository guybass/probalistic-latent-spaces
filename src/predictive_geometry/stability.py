"""Finite-scale diagnostics for square-root Fisher stability.

For the categorical square-root map ``psi = 2 * sqrt(p)``, the Fisher metric
is ``g = Dpsi.T @ Dpsi``.  Its Levi--Civita connection can therefore be
controlled without a vocabulary-size factor or a tokenwise probability floor.
The same square-root regularity does *not* automatically control a nonzero
Amari alpha-connection.  Paired phase-shifted Bernoulli maps can converge in
square-root ``C2`` with identical unit Fisher metrics while their fixed
nonzero-alpha connection defect diverges.  The cubic tensor contains

``C_ijk = 2 * sum_a D_i psi_a D_j psi_a D_k psi_a / psi_a``.

The apparent boundary denominator can cancel in structured families, so a
probability floor is sufficient but not necessary.  Direct control of the
raised cubic tensor (equivalently, the third score moment under a spectral
floor) is the relevant extra assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Tensor3 = NDArray[np.float64]


@dataclass(frozen=True)
class LeviCivitaStabilityBounds:
    """Conditional bounds from the square-root Levi--Civita theorem.

    All inputs to :func:`levi_civita_stability_bounds` must be uniform on the
    declared common chart and along the entire path.  Pointwise metric
    eigenvalues alone do not instantiate these bounds.
    """

    metric_defect_bound: float
    metric_derivative_defect_bound: float
    connection_defect_bound: float
    dimensionless_connection_length: float
    condition_envelope: float
    gronwall_transport_defect_bound: float
    isometric_transport_defect_bound: float
    transport_defect_bound: float


def square_root_cubic_tensor(
    square_root_coordinates: ArrayLike,
    jacobian: ArrayLike,
) -> Tensor3:
    """Return the Amari--Chentsov tensor from ``psi=2*sqrt(p)``.

    ``jacobian[a, i]`` is ``D_i psi_a``.  The explicit division by ``psi``
    records why bounded square-root derivatives alone do not control a general
    nonzero-alpha connection near the simplex boundary.  It does not imply
    that the tensor must diverge: derivatives or probability-weighted score
    moments can cancel the factor.
    """

    psi = np.asarray(square_root_coordinates, dtype=np.float64)
    derivative = np.asarray(jacobian, dtype=np.float64)
    if psi.ndim != 1:
        raise ValueError("square_root_coordinates must be one-dimensional")
    if derivative.ndim != 2 or derivative.shape[0] != psi.shape[0]:
        raise ValueError(
            "jacobian must have shape (outcomes, chart_dimension)"
        )
    if not np.all(np.isfinite(psi)) or not np.all(np.isfinite(derivative)):
        raise ValueError("square-root inputs must contain only finite values")
    if np.any(psi <= 0.0):
        raise ValueError("square_root_coordinates must lie in the open orthant")
    tensor = 2.0 * np.einsum(
        "ai,aj,ak,a->ijk",
        derivative,
        derivative,
        derivative,
        1.0 / psi,
        optimize=True,
    )
    if not np.all(np.isfinite(tensor)):
        raise ValueError("the square-root cubic tensor is not finite")
    return tensor


def levi_civita_stability_bounds(
    *,
    first_derivative_bound: float,
    second_derivative_bound: float,
    metric_eigenvalue_floor: float,
    first_derivative_defect: float,
    second_derivative_defect: float,
    path_length: float,
) -> LeviCivitaStabilityBounds:
    """Evaluate the theorem's metric, connection, and transport bounds.

    Write the inputs as ``B1, B2, lambda, delta1, delta2, L``.  The connection
    defect is

    ``D_LC = (B1*delta2 + B2*delta1)/lambda
             + 2*B1**2*B2*delta1/lambda**2``.

    Generic Gronwall gives ``L*exp(B1*B2*L/lambda)*D_LC``.  Levi--Civita
    metric compatibility also gives ``L*(B1**2/lambda)*D_LC`` because its
    parallel transports are Fisher isometries.  The returned certified bound
    is the smaller of the two.  This distinction prevents a large generic
    Gronwall exponent from being misread as an intrinsic short-path barrier.
    """

    b1 = _finite_nonnegative(first_derivative_bound, "first_derivative_bound")
    b2 = _finite_nonnegative(second_derivative_bound, "second_derivative_bound")
    delta1 = _finite_nonnegative(first_derivative_defect, "first_derivative_defect")
    delta2 = _finite_nonnegative(second_derivative_defect, "second_derivative_defect")
    length = _finite_nonnegative(path_length, "path_length")
    eigenvalue_floor = float(metric_eigenvalue_floor)
    if not math.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0.0:
        raise ValueError("metric_eigenvalue_floor must be finite and positive")
    if b1 * b1 < eigenvalue_floor * (1.0 - 1e-12):
        raise ValueError(
            "first_derivative_bound is inconsistent with the metric floor"
        )

    metric_defect = 2.0 * b1 * delta1
    metric_derivative_defect = 2.0 * (b1 * delta2 + b2 * delta1)
    connection_defect = (
        (b1 * delta2 + b2 * delta1) / eigenvalue_floor
        + 2.0
        * b1
        * b1
        * b2
        * delta1
        / (eigenvalue_floor * eigenvalue_floor)
    )
    connection_length = b1 * b2 * length / eigenvalue_floor
    condition_envelope = b1 * b1 / eigenvalue_floor

    if length == 0.0 or connection_defect == 0.0:
        gronwall_transport = 0.0
        isometric_transport = 0.0
    else:
        log_max = math.log(np.finfo(np.float64).max)
        gronwall_factor = (
            math.exp(connection_length)
            if connection_length <= log_max
            else math.inf
        )
        gronwall_transport = length * gronwall_factor * connection_defect
        isometric_transport = (
            length * condition_envelope * connection_defect
        )
    transport_defect = min(gronwall_transport, isometric_transport)

    return LeviCivitaStabilityBounds(
        metric_defect_bound=metric_defect,
        metric_derivative_defect_bound=metric_derivative_defect,
        connection_defect_bound=connection_defect,
        dimensionless_connection_length=connection_length,
        condition_envelope=condition_envelope,
        gronwall_transport_defect_bound=gronwall_transport,
        isometric_transport_defect_bound=isometric_transport,
        transport_defect_bound=transport_defect,
    )


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result
