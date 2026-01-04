import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from stochastax.manifolds import SO3


def benchmark_wrapper(benchmark: BenchmarkFixture, func, *args, **kwargs) -> jax.Array:
    """Wrapper for JAX benchmark that ensures device synchronization."""
    warmed = jax.block_until_ready(func(*args, **kwargs))

    def run() -> None:
        jax.block_until_ready(func(*args, **kwargs))
        return None

    benchmark(run)
    return warmed


RETRACTION_BENCH_CASES: list = [
    # pytest.param(1, id="single-matrix"),
    # pytest.param(8, id="batch-8"),
    pytest.param(32, id="batch-32"),
    # pytest.param(128, id="batch-128"),
]


@pytest.mark.benchmark(group="so3_retract_svd")
@pytest.mark.parametrize("batch_size", RETRACTION_BENCH_CASES)
def test_so3_retract_svd_benchmark(
    benchmark: BenchmarkFixture,
    batch_size: int,
) -> None:
    """Benchmark SVD retraction across batch sizes."""
    manifold = SO3()
    key = jrandom.PRNGKey(0)

    # Create batch of near-identity matrices with small perturbation
    if batch_size == 1:
        x = jnp.eye(3, dtype=jnp.float32) + jrandom.normal(key, (3, 3), dtype=jnp.float32) * 0.05
    else:
        x = (
            jnp.eye(3, dtype=jnp.float32)[None, ...]
            + jrandom.normal(key, (batch_size, 3, 3), dtype=jnp.float32) * 0.05
        )

    @jax.jit
    def retract_svd(matrices: jax.Array) -> jax.Array:
        return manifold.retract(matrices, method="svd")

    result = benchmark_wrapper(benchmark, retract_svd, x)
    expected_shape = (3, 3) if batch_size == 1 else (batch_size, 3, 3)
    assert result.shape == expected_shape


@pytest.mark.benchmark(group="so3_retract_polar_express")
@pytest.mark.parametrize("batch_size", RETRACTION_BENCH_CASES)
def test_so3_retract_polar_express_benchmark(
    benchmark: BenchmarkFixture,
    batch_size: int,
) -> None:
    """Benchmark polar express retraction across batch sizes."""
    manifold = SO3()
    key = jrandom.PRNGKey(1)

    if batch_size == 1:
        x = jnp.eye(3, dtype=jnp.float32) + jrandom.normal(key, (3, 3), dtype=jnp.float32) * 0.05
    else:
        x = (
            jnp.eye(3, dtype=jnp.float32)[None, ...]
            + jrandom.normal(key, (batch_size, 3, 3), dtype=jnp.float32) * 0.05
        )

    @jax.jit
    def retract_polar(matrices: jax.Array) -> jax.Array:
        return manifold.retract(matrices, method="polar_express")

    result = benchmark_wrapper(benchmark, retract_polar, x)
    expected_shape = (3, 3) if batch_size == 1 else (batch_size, 3, 3)
    assert result.shape == expected_shape


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

    @jax.jit
    def retract_polar(matrices: jax.Array) -> jax.Array:
        return manifold.retract(matrices, method="polar_express")

    result = benchmark_wrapper(benchmark, retract_polar, x)
    assert result.shape == (batch_size, 3, 3)


@pytest.mark.benchmark(group="so3_tangent_projection")
@pytest.mark.parametrize("batch_size", RETRACTION_BENCH_CASES)
def test_so3_project_to_tangent_benchmark(
    benchmark: BenchmarkFixture,
    batch_size: int,
) -> None:
    """Benchmark tangent projection across batch sizes."""
    manifold = SO3()
    key = jrandom.PRNGKey(3)

    if batch_size == 1:
        y = jnp.eye(3, dtype=jnp.float32)
        v = jrandom.normal(key, (3, 3), dtype=jnp.float32)
    else:
        y = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None, ...], (batch_size, 1, 1))
        v = jrandom.normal(key, (batch_size, 3, 3), dtype=jnp.float32)

    @jax.jit
    def project(y_in: jax.Array, v_in: jax.Array) -> jax.Array:
        return manifold.project_to_tangent(y_in, v_in)

    result = benchmark_wrapper(benchmark, project, y, v)
    expected_shape = (3, 3) if batch_size == 1 else (batch_size, 3, 3)
    assert result.shape == expected_shape


@pytest.mark.benchmark(group="so3_combined_operations")
def test_so3_retract_and_project_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark combined retraction and tangent projection (common workflow)."""
    manifold = SO3()
    key = jrandom.PRNGKey(7)

    batch_size = 32
    x = (
        jnp.eye(3, dtype=jnp.float32)[None, ...]
        + jrandom.normal(key, (batch_size, 3, 3), dtype=jnp.float32) * 0.1
    )
    v = jrandom.normal(key, (batch_size, 3, 3), dtype=jnp.float32)

    @jax.jit
    def retract_and_project(matrices: jax.Array, vectors: jax.Array) -> tuple[jax.Array, jax.Array]:
        R = manifold.retract(matrices, method="svd")
        v_tan = manifold.project_to_tangent(R, vectors)
        return R, v_tan

    R, v_tan = benchmark_wrapper(benchmark, retract_and_project, x, v)
    assert R.shape == (batch_size, 3, 3)
    assert v_tan.shape == (batch_size, 3, 3)
