import jax
import jax.numpy as jnp
import pytest
from jax.scipy.linalg import expm as jexpm

from stochastax.integrators.log_ode import log_ode, log_ode_homogeneous
from stochastax.vector_field_lifts.bck_lift import form_bck_lift, form_bck_bracket_functions
from stochastax.hopf_algebras.hopf_algebras import GLHopfAlgebra
from stochastax.control_lifts.branched_signature_ito import compute_nonplanar_branched_signature

from tests.test_integrators.conftest import (
    _linear_vector_fields,
    build_block_initial_state,
    build_bck_log_ode_inputs,
    build_block_rotation_generators,
)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("dim", [1, 3, 5])
def test_bck_log_ode_euclidean(
    rotation_matrix_2d: jax.Array,
    euclidean_initial_state: jax.Array,
    dim: int,
    depth: int,
) -> None:
    """Depth-1 BCK log-ODE in Euclidean space matches the corresponding matrix exponential."""
    delta = 0.35
    bck_brackets, logsig, y0 = build_bck_log_ode_inputs(depth=depth, dim=dim, delta=delta)
    y_next = log_ode_homogeneous(bck_brackets, logsig, y0)
    block_generators = build_block_rotation_generators(dim)
    combined_generator = jnp.sum(block_generators, axis=0)
    expected = jexpm(delta * combined_generator) @ y0
    expected = expected / jnp.linalg.norm(y0)
    assert jnp.allclose(y_next, expected, rtol=1e-6, atol=1e-6)


def test_bck_log_ode_state_dependent_matches_linear_homogeneous() -> None:
    """Callable BCK bracket functions match homogeneous path for linear fields."""
    dim = 2
    depth = 1
    delta = 0.15

    hopf = GLHopfAlgebra.build(dim, depth)
    generators = build_block_rotation_generators(dim)
    vector_fields = _linear_vector_fields(generators)
    bracket_funcs = form_bck_bracket_functions(vector_fields, hopf)
    base_point = build_block_initial_state(dim)
    bracket_mats = form_bck_lift(vector_fields, base_point, hopf)

    path = jnp.stack(
        [jnp.zeros((dim,), dtype=jnp.float32), jnp.full((dim,), delta, dtype=jnp.float32)]
    )
    logsig = build_bck_log_ode_inputs(depth=depth, dim=dim, delta=delta)[1]
    y0 = build_block_initial_state(dim)

    with jax.disable_jit():
        y_fun = log_ode(bracket_funcs, logsig, y0)
    y_mat = log_ode_homogeneous(bracket_mats, logsig, y0)

    assert jnp.allclose(y_fun, y_mat, rtol=1e-5, atol=1e-6)


def test_form_bck_lift_jittable() -> None:
    depth = 2
    dim = 2
    hopf = GLHopfAlgebra.build(dim, depth)
    generators = build_block_rotation_generators(dim)
    vector_fields = _linear_vector_fields(generators)
    y0 = build_block_initial_state(dim)

    compiled = jax.jit(lambda bp: form_bck_lift(vector_fields, bp, hopf))
    lift = compiled(y0)

    assert len(lift) == depth
