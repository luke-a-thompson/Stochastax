import jax
import jax.numpy as jnp
import pytest
from jax.scipy.linalg import expm as jexpm

from stochastax.integrators.log_ode import log_ode
from stochastax.vector_field_lifts.bck_lift import form_bck_lift
from stochastax.hopf_algebras.hopf_algebras import GLHopfAlgebra

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
    y_next = log_ode(bck_brackets, logsig, y0)
    block_generators = build_block_rotation_generators(dim)
    combined_generator = jnp.sum(block_generators, axis=0)
    expected = jexpm(delta * combined_generator) @ y0
    expected = expected / jnp.linalg.norm(y0)
    assert jnp.allclose(y_next, expected, rtol=1e-6, atol=1e-6)


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
