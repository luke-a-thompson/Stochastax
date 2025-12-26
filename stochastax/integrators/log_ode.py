import jax
import jax.numpy as jnp
import diffrax
from typing import overload
from functools import partial
from jax.scipy.linalg import expm as jexpm

from stochastax.integrators.series import form_lie_series
from stochastax.vector_field_lifts.vector_field_lift_types import (
    LyndonBrackets,
    BCKBrackets,
    MKWBrackets,
    LyndonBracketFunctions,
    BCKBracketFunctions,
    MKWBracketFunctions,
    VectorFieldBracketFunctions,
)
from stochastax.control_lifts.signature_types import (
    LogSignature,
    BCKLogSignature,
    MKWLogSignature,
    PrimitiveSignature,
)
from stochastax.manifolds.manifolds import Manifold, EuclideanSpace


@overload
def log_ode_homogeneous(
    vector_field_brackets: LyndonBrackets,
    primitive_signature: LogSignature,
    curr_state: jax.Array,
    manifold: Manifold = ...,
) -> jax.Array: ...


@overload
def log_ode_homogeneous(
    vector_field_brackets: BCKBrackets,
    primitive_signature: BCKLogSignature,
    curr_state: jax.Array,
    manifold: Manifold = ...,
) -> jax.Array: ...


@overload
def log_ode_homogeneous(
    vector_field_brackets: MKWBrackets,
    primitive_signature: MKWLogSignature,
    curr_state: jax.Array,
    manifold: Manifold = ...,
) -> jax.Array: ...


@partial(jax.jit, static_argnames=["manifold"])
def log_ode_homogeneous(
    vector_field_brackets: LyndonBrackets | BCKBrackets | MKWBrackets,
    primitive_signature: LogSignature | BCKLogSignature | MKWLogSignature,
    curr_state: jax.Array,
    manifold: Manifold = EuclideanSpace(),
) -> jax.Array:
    """Log-ODE step for linear/homogeneous dynamics using matrix exponential."""
    # Keep degrees that have non-empty coefficients
    W_levels = [
        Wk for Wk, lam in zip(vector_field_brackets, primitive_signature.coeffs) if lam.size != 0
    ]
    W_flat = jnp.concatenate(W_levels, axis=0) if W_levels else jnp.zeros((0, 1, 1))
    polynomial = form_lie_series(W_flat, primitive_signature.coeffs)
    exp_polynomial = jexpm(polynomial)
    return manifold.retract(exp_polynomial @ curr_state)


def _evaluate_bracket_sum(
    bracket_functions: VectorFieldBracketFunctions,
    coeffs: list[jax.Array],
    y: jax.Array,
) -> jax.Array:
    """Evaluate Σ_w λ_w V_w(y) for functional brackets."""
    dy = jnp.zeros_like(y)

    for level_fns, level_coeffs in zip(bracket_functions, coeffs):
        if not level_fns or level_coeffs.size == 0:
            continue
        for i, fn in enumerate(level_fns):
            lam = level_coeffs[i]
            dy = dy + lam * fn(y)

    return dy


@overload
def log_ode(
    bracket_functions: LyndonBracketFunctions,
    primitive_signature: LogSignature,
    curr_state: jax.Array,
    manifold: Manifold = ...,
    *,
    solver: diffrax.AbstractSolver = ...,
    dt0: float = ...,
    rtol: float = ...,
    atol: float = ...,
    max_steps: int | None = ...,
) -> jax.Array: ...


@overload
def log_ode(
    bracket_functions: BCKBracketFunctions,
    primitive_signature: BCKLogSignature,
    curr_state: jax.Array,
    manifold: Manifold = ...,
    *,
    solver: diffrax.AbstractSolver = ...,
    dt0: float = ...,
    rtol: float = ...,
    atol: float = ...,
    max_steps: int | None = ...,
) -> jax.Array: ...


@overload
def log_ode(
    bracket_functions: MKWBracketFunctions,
    primitive_signature: MKWLogSignature,
    curr_state: jax.Array,
    manifold: Manifold = ...,
    *,
    solver: diffrax.AbstractSolver = ...,
    dt0: float = ...,
    rtol: float = ...,
    atol: float = ...,
    max_steps: int | None = ...,
) -> jax.Array: ...


def log_ode(
    bracket_functions: VectorFieldBracketFunctions,
    primitive_signature: PrimitiveSignature,
    curr_state: jax.Array,
    manifold: Manifold = EuclideanSpace(),
    *,
    solver: diffrax.AbstractSolver = diffrax.Heun(),
    dt0: float = 0.1,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    max_steps: int | None = None,
) -> jax.Array:
    """
    Log-ODE step for state-dependent NCDEs using Diffrax on dummy time u ∈ [0, 1].

    Requires bracket_functions that evaluate the bracket vector fields at the
    current state during integration.
    """

    if len(bracket_functions) != len(primitive_signature.coeffs):
        raise ValueError(
            f"Bracket levels {len(bracket_functions)} do not match signature levels {len(primitive_signature.coeffs)}."
        )

    def rhs(_u, y: jax.Array, _args) -> jax.Array:
        y_on = manifold.retract(y)
        dy = _evaluate_bracket_sum(bracket_functions, primitive_signature.coeffs, y_on)
        return manifold.project_to_tangent(y_on, dy)

    term = diffrax.ODETerm(rhs)
    sol = diffrax.diffeqsolve(
        term,
        solver=solver,
        t0=0.0,
        t1=1.0,
        dt0=dt0,
        y0=curr_state,
        args=None,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=max_steps,
    )
    saved_y = sol.ys
    if saved_y is None:
        raise RuntimeError("Diffrax solve did not return any saved states.")

    y1 = saved_y[0]
    return manifold.retract(y1)
