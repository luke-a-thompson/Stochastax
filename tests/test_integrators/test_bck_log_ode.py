import jax
import jax.numpy as jnp
import pytest
from jax.scipy.linalg import expm as jexpm

from stochastax.integrators.log_ode import log_ode
from stochastax.vector_field_lifts.gl_lift import form_gl_bracket_functions
from stochastax.hopf_algebras.hopf_algebras import GLHopfAlgebra
from stochastax.control_lifts.branched_signature_ito import compute_nonplanar_branched_signature

from tests.test_integrators.conftest import (
    _linear_vector_fields,
    build_block_initial_state,
    build_block_rotation_generators,
    build_two_point_path,
)


@pytest.mark.parametrize("depth", [1])
@pytest.mark.parametrize("dim", [1, 3, 5])
def test_bck_log_ode_euclidean(
    rotation_matrix_2d: jax.Array,
    euclidean_initial_state: jax.Array,
    dim: int,
    depth: int,
) -> None:
    """Depth-1 BCK log-ODE (state-dependent path) matches matrix exponential for linear fields."""
    delta = 0.35

    hopf = GLHopfAlgebra.build(dim, depth)
    generators = build_block_rotation_generators(dim)
    def batched_field(y: jax.Array) -> jax.Array:
        return jnp.stack([M @ y for M in generators], axis=0)

    bracket_functions = form_gl_bracket_functions(batched_field, hopf)

    y0 = build_block_initial_state(dim)
    path = build_two_point_path(delta, dim)
    steps = path.shape[0] - 1
    cov = jnp.zeros((steps, dim, dim), dtype=jnp.float32)
    sig = compute_nonplanar_branched_signature(
        path=path,
        depth=depth,
        hopf=hopf,
        mode="full",
        cov_increments=cov,
    )
    logsig = sig.log()

    y_next = log_ode(bracket_functions, logsig, y0)
    block_generators = build_block_rotation_generators(dim)
    combined_generator = jnp.sum(block_generators, axis=0)
    expected = jexpm(delta * combined_generator) @ y0
    assert jnp.allclose(y_next, expected, rtol=1e-6, atol=1e-6)
