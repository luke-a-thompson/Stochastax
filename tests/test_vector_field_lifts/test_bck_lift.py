import jax
import jax.numpy as jnp
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from stochastax.hopf_algebras.hopf_algebra_types import GLHopfAlgebra
from stochastax.vector_field_lifts.bck_lift import form_bck_brackets
from tests.test_integrators.conftest import (
    _linear_vector_fields,
    benchmark_wrapper,
    build_block_rotation_generators,
)


BCK_LIFT_BENCH_CASES: list = [
    pytest.param(2, 2, id="dim-2-depth-2"),
    pytest.param(8, 2, id="dim-2-depth-2"),
]


@pytest.mark.benchmark(group="bck_lift")
@pytest.mark.parametrize("dim,depth", BCK_LIFT_BENCH_CASES)
def test_bck_lift_benchmark_linear_block_rotation(
    benchmark: BenchmarkFixture,
    dim: int,
    depth: int,
) -> None:
    """Benchmark BCK lift build for simple linear block-rotation vector fields.

    This mirrors the Euclidean log-ODE helpers and isolates ``form_bck_brackets`` so
    changes to its implementation show up clearly in benchmark results.
    """
    generators = build_block_rotation_generators(dim)
    n_state: int = int(generators.shape[-1])
    vector_fields = _linear_vector_fields(generators)
    base_point = jnp.linspace(0.1, 0.2, num=n_state, dtype=jnp.float32)

    hopf = GLHopfAlgebra.build(dim, depth)

    compiled = jax.jit(lambda y: form_bck_brackets(vector_fields, y, hopf))
    brackets = benchmark_wrapper(benchmark, compiled, base_point)

    # Keep this a real test with light sanity checks.
    assert isinstance(brackets, list)
    assert len(brackets) == depth
