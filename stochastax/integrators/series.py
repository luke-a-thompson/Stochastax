import jax
import jax.numpy as jnp
from stochastax.integrators.integrator_types import LieSeries


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
