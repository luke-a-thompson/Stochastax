import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from stochastax.manifolds import SO3
from tests.conftest import benchmark_wrapper
from functools import partial


RETRACTION_BENCH_CASES: list = [
    # pytest.param(1, id="single-matrix"),
    pytest.param(8, id="batch-8"),
    pytest.param(32, id="batch-32"),
    # pytest.param(128, id="batch-128"),
]


@pytest.mark.benchmark(group="so3_retractions")
@pytest.mark.parametrize("batch_size", RETRACTION_BENCH_CASES)
def test_so3_retract_svd_benchmark(
    benchmark: BenchmarkFixture,
    batch_size: int,
) -> None:
    """Benchmark SVD retraction across batch sizes."""
    key = jrandom.PRNGKey(0)

    x = (
        jnp.eye(3, dtype=jnp.float32)[None, ...]
        + jrandom.normal(key, (batch_size, 3, 3), dtype=jnp.float32) * 0.05
    )

    retract_svd = partial(SO3.retract, method="svd")

    result = benchmark_wrapper(benchmark, retract_svd, x)
    assert result.shape == (batch_size, 3, 3), (
        f"SVD retraction bench shape mismatch: got {result.shape}, expected {batch_size, 3, 3}"
    )


@pytest.mark.benchmark(group="so3_retractions")
@pytest.mark.parametrize("batch_size", RETRACTION_BENCH_CASES)
def test_so3_retract_gram_schmidt_benchmark(
    benchmark: BenchmarkFixture,
    batch_size: int,
) -> None:
    """Benchmark Gram-Schmidt retraction across batch sizes."""
    key = jrandom.PRNGKey(1)

    # Gram–Schmidt retraction uses the common 6D representation in R^6.
    # See `SO3.so3_from_6d`.
    x = jrandom.normal(key, (batch_size, 6), dtype=jnp.float32)

    retract_gram_schmidt = partial(SO3.retract, method="gram_schmidt")
    result = benchmark_wrapper(benchmark, retract_gram_schmidt, x)
    assert result.shape == (
        batch_size,
        3,
        3,
    ), (
        f"Gram-Schmidt retraction bench shape mismatch: got {result.shape}, expected {batch_size, 3, 3}"
    )


@pytest.mark.benchmark(group="so3_retractions")
@pytest.mark.parametrize("batch_size", RETRACTION_BENCH_CASES)
def test_so3_retract_polar_express_benchmark(
    benchmark: BenchmarkFixture,
    batch_size: int,
) -> None:
    """Benchmark polar express retraction across batch sizes."""
    key = jrandom.PRNGKey(1)

    x = (
        jnp.eye(3, dtype=jnp.float32)[None, ...]
        + jrandom.normal(key, (batch_size, 3, 3), dtype=jnp.float32) * 0.05
    )

    retract_polar = partial(SO3.retract, method="polar_express")

    result = benchmark_wrapper(benchmark, retract_polar, x)
    assert result.shape == (batch_size, 3, 3), (
        f"Polar express retraction bench shape mismatch: got {result.shape}, expected {batch_size, 3, 3}"
    )


POLAR_STEPS_BENCH_CASES: list = [
    pytest.param(1, id="steps-1"),
    pytest.param(3, id="steps-3"),
    pytest.param(6, id="steps-6"),
    pytest.param(10, id="steps-10"),
]


@pytest.mark.benchmark(group="so3_polar_steps")
@pytest.mark.parametrize("polar_steps", POLAR_STEPS_BENCH_CASES)
def test_so3_polar_express_steps_benchmark(
    benchmark: BenchmarkFixture,
    polar_steps: int,
) -> None:
    """Benchmark polar express with varying number of iteration steps."""
    manifold = SO3(polar_steps=polar_steps)
    key = jrandom.PRNGKey(2)

    batch_size = 32
    x = (
        jnp.eye(3, dtype=jnp.float32)[None, ...]
        + jrandom.normal(key, (batch_size, 3, 3), dtype=jnp.float32) * 0.05
    )

    retract_polar = partial(manifold.retract, method="polar_express")

    result = benchmark_wrapper(benchmark, retract_polar, x)
    assert result.shape == (batch_size, 3, 3), (
        f"Polar express steps bench shape mismatch: got {result.shape}, expected {batch_size, 3, 3}"
    )


@pytest.mark.benchmark(group="so3_tangent_projection")
@pytest.mark.parametrize("batch_size", RETRACTION_BENCH_CASES)
def test_so3_project_to_tangent_benchmark(
    benchmark: BenchmarkFixture,
    batch_size: int,
) -> None:
    """Benchmark tangent projection across batch sizes."""
    key = jrandom.PRNGKey(3)

    y = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None, ...], (batch_size, 1, 1))
    v = jrandom.normal(key, (batch_size, 3, 3), dtype=jnp.float32)

    result = benchmark_wrapper(benchmark, SO3.project_to_tangent, y, v)
    assert result.shape == (batch_size, 3, 3), (
        f"Project to tangent bench shape mismatch: got {result.shape}, expected {batch_size, 3, 3}"
    )
