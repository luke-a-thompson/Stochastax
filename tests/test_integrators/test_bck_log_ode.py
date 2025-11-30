import jax
import jax.numpy as jnp
import pytest
from jax.scipy.linalg import expm as jexpm
from pytest_benchmark.fixture import BenchmarkFixture

from stochastax.controls.drivers import bm_driver
from stochastax.control_lifts.branched_signature_ito import compute_nonplanar_branched_signature
from stochastax.integrators.log_ode import log_ode
from stochastax.vector_field_lifts.bck_lift import form_bck_brackets
from stochastax.hopf_algebras import enumerate_bck_trees
from stochastax.hopf_algebras.hopf_algebra_types import GLHopfAlgebra

from tests.test_integrators.conftest import (
    _linear_vector_fields,
    benchmark_wrapper,
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


BCK_BENCH_CASES = [
    pytest.param(1, 1, 0.35, id="depth1-dim1"),
    pytest.param(2, 2, 0.25, id="depth1-dim2"),
    pytest.param(2, 5, 0.25, id="depth2-dim5"),
]


@pytest.mark.benchmark(group="log_ode_bck")
@pytest.mark.parametrize("depth,dim,delta", BCK_BENCH_CASES)
def test_bck_log_ode_benchmark_euclidean(
    benchmark: BenchmarkFixture,
    depth: int,
    dim: int,
    delta: float,
) -> None:
    """Benchmark the BCK variant of log-ODE in Euclidean space."""
    bck_brackets, logsig, y0 = build_bck_log_ode_inputs(depth, dim, delta)
    result = benchmark_wrapper(benchmark, log_ode, bck_brackets, logsig, y0)
    assert result.shape == y0.shape
