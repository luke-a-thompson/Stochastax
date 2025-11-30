import jax
import jax.numpy as jnp
from stochastax.integrators.integrator_types import ButcherSeries, LieSeries, LieButcherSeries
from stochastax.vector_field_lifts.vector_field_lift_types import (
    ButcherDifferentials,
    LieButcherDifferentials,
)


def form_butcher_series(
    differentials: ButcherDifferentials,
    coefficients: list[jax.Array],
) -> ButcherSeries:
    """Form a Butcher series from differentials and coefficients.

    Args:
        differentials: ButcherDifferentials to contract with coefficients
        coefficients: List of coefficient arrays to contract with differentials

    Returns:
        ButcherSeries formed by contracting coefficients with differentials
    """
    coefficients_flat: jax.Array = (
        jnp.concatenate(coefficients, axis=0) if coefficients else jnp.zeros((0,))
    )

    if differentials.shape[0] != coefficients_flat.shape[0]:
        raise ValueError(
            f"Coefficient count {coefficients_flat.shape[0]} does not match number of basis terms {differentials.shape[0]}."
        )

    series = jnp.tensordot(coefficients_flat, differentials, axes=1)
    return ButcherSeries(series)


def form_lie_butcher_series(
    differentials: LieButcherDifferentials,
    coefficients: list[jax.Array],
) -> LieButcherSeries:
    """Form a Lie-Butcher series from differentials and coefficients.

    Args:
        differentials: LieButcherDifferentials to contract with coefficients
        coefficients: List of coefficient arrays to contract with differentials

    Returns:
        LieButcherSeries formed by contracting coefficients with differentials
    """
    coefficients_flat: jax.Array = (
        jnp.concatenate(coefficients, axis=0) if coefficients else jnp.zeros((0,))
    )

    if differentials.shape[0] != coefficients_flat.shape[0]:
        raise ValueError(
            f"Coefficient count {coefficients_flat.shape[0]} does not match number of basis terms {differentials.shape[0]}."
        )

    series = jnp.tensordot(coefficients_flat, differentials, axes=1)
    return LieButcherSeries(series)


def form_lie_series(
    basis_terms: jax.Array,
    lam_by_len: list[jax.Array],
) -> LieSeries:
    """Form the Lie series matrix: C = sum_w lam_w * W[w]."""
    # Include non-empty degrees
    lams = [lam for lam in lam_by_len if lam.size != 0]
    coefficients_flat: jax.Array = jnp.concatenate(lams, axis=0) if lams else jnp.zeros((0,))

    if basis_terms.shape[0] != coefficients_flat.shape[0]:
        raise ValueError(
            f"Signature coefficient count {coefficients_flat.shape[0]} does not match number of basis terms {basis_terms.shape[0]} in the vector field brackets."
        )
    series = jnp.tensordot(coefficients_flat, basis_terms, axes=1)
    return LieSeries(series)
