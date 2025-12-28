import jax
import jax.numpy as jnp
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from stochastax.hopf_algebras.hopf_algebras import GLHopfAlgebra, MKWHopfAlgebra
from stochastax.vector_field_lifts.bck_lift import form_bck_lift
from stochastax.vector_field_lifts.lie_lift import form_lyndon_lift
from stochastax.vector_field_lifts.mkw_lift import form_mkw_lift
from tests.conftest import BENCH_GL_CASES, BENCH_MKW_CASES, BENCH_SHUFFLE_CASES, _so3_generators
from tests.test_integrators.conftest import (
    _linear_vector_fields,
    benchmark_wrapper,
    build_block_rotation_generators,
)
from stochastax.hopf_algebras.hopf_algebras import ShuffleHopfAlgebra
from stochastax.manifolds import Sphere


@pytest.mark.benchmark(group="lyndon_lift")
@pytest.mark.parametrize("dim,depth", BENCH_SHUFFLE_CASES)
def test_lyndon_lift_benchmark_linear_block_rotation(
    benchmark: BenchmarkFixture,
    dim: int,
    depth: int,
) -> None:
    """Benchmark nonlinear Lyndon lift build for simple linear block-rotation fields.

    This isolates the overhead of ``form_lyndon_lift`` so micro-optimisations
    (e.g. batched per-level Jacobians) are visible outside integrator tests.
    """
    generators = build_block_rotation_generators(dim)
    n_state = int(generators.shape[-1])
    vector_fields = _linear_vector_fields(generators)
    base_point = jnp.linspace(0.1, 0.2, num=n_state, dtype=jnp.float32)
    hopf = ShuffleHopfAlgebra.build(ambient_dim=dim, depth=depth)

    compiled = jax.jit(lambda y: form_lyndon_lift(vector_fields, y, hopf))
    brackets = benchmark_wrapper(benchmark, compiled, base_point)

    # Basic sanity checks to keep this a proper test.
    assert isinstance(brackets, list)
    assert len(brackets) == depth


@pytest.mark.benchmark(group="bck_lift")
@pytest.mark.parametrize("dim,depth", BENCH_GL_CASES)
def test_bck_lift_benchmark_linear_block_rotation(
    benchmark: BenchmarkFixture,
    dim: int,
    depth: int,
) -> None:
    """Benchmark BCK lift build for simple linear block-rotation vector fields.

    This mirrors the Euclidean log-ODE helpers and isolates ``form_bck_lift`` so
    changes to its implementation show up clearly in benchmark results.
    """
    generators = build_block_rotation_generators(dim)
    n_state: int = int(generators.shape[-1])
    vector_fields = _linear_vector_fields(generators)
    base_point = jnp.linspace(0.1, 0.2, num=n_state, dtype=jnp.float32)

    hopf = GLHopfAlgebra.build(dim, depth)

    compiled = jax.jit(lambda y: form_bck_lift(vector_fields, y, hopf))
    lift = benchmark_wrapper(benchmark, compiled, base_point)

    # Keep this a real test with light sanity checks.
    assert isinstance(lift, list)
    assert len(lift) == depth


@pytest.mark.benchmark(group="mkw_lift")
@pytest.mark.parametrize("_dim,depth", BENCH_MKW_CASES)
def test_mkw_lift_benchmark_so3_manifold(
    benchmark: BenchmarkFixture,
    _dim: int,
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

    compiled = jax.jit(lambda y: form_mkw_lift(vector_fields, y, hopf, Sphere()))
    brackets = benchmark_wrapper(benchmark, compiled, base_point)

    # Keep this a real test with light sanity checks.
    assert isinstance(brackets, list)
    assert len(brackets) == depth
