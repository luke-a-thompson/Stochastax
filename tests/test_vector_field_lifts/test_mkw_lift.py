import jax
import jax.numpy as jnp
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from stochastax.hopf_algebras.hopf_algebra_types import MKWHopfAlgebra
from stochastax.vector_field_lifts.mkw_lift import form_mkw_brackets
from tests.conftest import _so3_generators
from tests.test_integrators.conftest import (
    _linear_vector_fields,
    _project_to_tangent,
    benchmark_wrapper,
)


MKW_LIFT_BENCH_CASES: list = [
    pytest.param(2, id="depth-2"),
    pytest.param(3, id="depth-3"),
]


@pytest.mark.benchmark(group="mkw_lift")
@pytest.mark.parametrize("depth", MKW_LIFT_BENCH_CASES)
def test_mkw_lift_benchmark_so3_manifold(
    benchmark: BenchmarkFixture,
    depth: int,
) -> None:
    """Benchmark MKW lift build for simple SO(3) manifold vector fields.

    Uses the standard ``_so3_generators`` and projection to tangent space so
    this aligns with existing manifold log-ODE tests.
    """
    A: jax.Array = _so3_generators()
    dim: int = int(A.shape[0])
    hopf = MKWHopfAlgebra.build(dim, depth)

    vector_fields = _linear_vector_fields(A)
    base_point = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    compiled = jax.jit(lambda y: form_mkw_brackets(vector_fields, y, hopf, _project_to_tangent))
    brackets = benchmark_wrapper(benchmark, compiled, base_point)

    # Keep this a real test with light sanity checks.
    assert isinstance(brackets, list)
    assert len(brackets) == depth
