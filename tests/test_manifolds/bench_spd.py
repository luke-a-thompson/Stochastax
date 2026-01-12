from pytest_benchmark.fixture import BenchmarkFixture
from stochastax.manifolds.spd import SPDManifold
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from tests.conftest import benchmark_wrapper
import math

@pytest.mark.benchmark(group="spd_vech")
def test_benchmark_naive_vech(benchmark: BenchmarkFixture) -> None:

    def naive_vech(H: jax.Array) -> jax.Array:
        n = H.shape[-1]
        i, j = jnp.tril_indices(n)
        return H[..., i, j]

    H = jrandom.normal(jrandom.PRNGKey(0), (10, 10))
    result = benchmark_wrapper(benchmark, naive_vech, H)
    assert result.shape == (55,)

@pytest.mark.benchmark(group="spd_vech")
def test_benchmark_vech(benchmark: BenchmarkFixture) -> None:
    H = jrandom.normal(jrandom.PRNGKey(0), (10, 10))
    result = benchmark_wrapper(benchmark, SPDManifold.vech, H)
    assert result.shape == (55,)

@pytest.mark.benchmark(group="spd_unvech")
def test_naive_unvech(benchmark: BenchmarkFixture) -> None:

    def naive_unvech(v: jax.Array) -> jax.Array:
        m = v.shape[-1]  # static at trace time

        # m = n(n+1)/2  <=>  8m + 1 = (2n + 1)^2
        disc = 8 * m + 1
        root = math.isqrt(disc)
        if root * root != disc:
            raise ValueError(f"Invalid vech length {m}; expected n(n+1)/2 for integer n.")
        n = (root - 1) // 2

        i, j = jnp.tril_indices(n)
        H = jnp.zeros(v.shape[:-1] + (n, n), dtype=v.dtype)
        H = H.at[..., i, j].set(v)
        H = H.at[..., j, i].set(v)
        return H

    v = jrandom.normal(jrandom.PRNGKey(0), (55,))
    result = benchmark_wrapper(benchmark, naive_unvech, v)
    assert result.shape == (10, 10)

@pytest.mark.benchmark(group="spd_unvech")
def test_unvech(benchmark: BenchmarkFixture) -> None:
    v = jrandom.normal(jrandom.PRNGKey(0), (55,))
    result = benchmark_wrapper(benchmark, SPDManifold.unvech, v)
    assert result.shape == (10, 10)