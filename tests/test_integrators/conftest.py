"""Shared fixtures and helpers for integrator tests."""

import jax
import jax.numpy as jnp
import pytest
from pytest_benchmark.fixture import BenchmarkFixture


def _linear_vector_fields(A: jax.Array) -> list:
    """Convert matrix generators to linear vector field callables.

    Args:
        A: Array of shape (dim, n, n) containing matrix generators

    Returns:
        List of callable vector fields y -> A[i] @ y
    """
    return [lambda y, M=A[i]: M @ y for i in range(A.shape[0])]


def _project_to_tangent(y: jax.Array, v: jax.Array) -> jax.Array:
    """Project vector v onto tangent plane at point y on S^2.

    Args:
        y: Point on sphere (unit vector)
        v: Vector to project

    Returns:
        Projected vector orthogonal to y
    """
    return v - jnp.dot(v, y) * y


@pytest.fixture
def rotation_matrix_2d() -> jax.Array:
    """Standard 2D rotation generator (90-degree CCW)."""
    return jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float32)


@pytest.fixture
def euclidean_initial_state() -> jax.Array:
    """Standard initial state for 2D Euclidean tests."""
    return jnp.array([1.0, 0.0], dtype=jnp.float32)


@pytest.fixture
def sphere_initial_state() -> jax.Array:
    """Standard initial state on S^2 (north pole)."""
    return jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)


def benchmark_wrapper(benchmark: BenchmarkFixture, func, *args, **kwargs) -> jax.Array:
    """Wrapper for JAX benchmark that ensures device synchronization.

    Args:
        benchmark: pytest-benchmark fixture
        func: Function to benchmark
        *args: Positional arguments to func
        **kwargs: Keyword arguments to func

    Returns:
        Result of the benchmarked function
    """
    # Warmup
    jax.block_until_ready(func(*args, **kwargs))

    def run() -> jax.Array:
        return jax.block_until_ready(func(*args, **kwargs))

    return benchmark(run)
