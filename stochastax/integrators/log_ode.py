import jax
import jax.numpy as jnp
from typing import overload
from jax.scipy.linalg import expm as jexpm
from stochastax.integrators.series import form_lie_series
from stochastax.vector_field_lifts.vector_field_lift_types import (
    LyndonBrackets,
    BCKBrackets,
    MKWBrackets,
)
from stochastax.control_lifts.signature_types import LogSignature, BCKLogSignature, MKWLogSignature


@overload
def log_ode(
    vector_field_brackets: LyndonBrackets,
    primitive_signature: LogSignature,
    curr_state: jax.Array,
) -> jax.Array: ...


@overload
def log_ode(
    vector_field_brackets: BCKBrackets,
    primitive_signature: BCKLogSignature,
    curr_state: jax.Array,
) -> jax.Array: ...


@overload
def log_ode(
    vector_field_brackets: MKWBrackets,
    primitive_signature: MKWLogSignature,
    curr_state: jax.Array,
) -> jax.Array: ...


@jax.jit
def log_ode(
    vector_field_brackets: LyndonBrackets | BCKBrackets | MKWBrackets,
    primitive_signature: LogSignature | BCKLogSignature | MKWLogSignature,
    curr_state: jax.Array,
) -> jax.Array:
    # Keep degrees that have non-empty coefficients
    W_levels = [
        Wk for Wk, lam in zip(vector_field_brackets, primitive_signature.coeffs) if lam.size != 0
    ]
    W_flat = jnp.concatenate(W_levels, axis=0) if W_levels else jnp.zeros((0, 1, 1))
    polynomial = form_lie_series(W_flat, primitive_signature.coeffs)
    exp_polynomial = jexpm(polynomial)
    return (exp_polynomial @ curr_state) / jnp.linalg.norm(curr_state)

