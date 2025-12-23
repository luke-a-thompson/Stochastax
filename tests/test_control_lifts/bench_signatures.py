import jax
import jax.numpy as jnp
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from stochastax.analytics.signature_sizes import get_log_signature_dim, get_signature_dim
from stochastax.control_lifts.branched_signature_ito import (
    compute_nonplanar_branched_signature,
    compute_planar_branched_signature,
)
from stochastax.control_lifts.log_signature import compute_log_signature, compute_path_signature
from stochastax.hopf_algebras.hopf_algebras import GLHopfAlgebra, MKWHopfAlgebra, ShuffleHopfAlgebra
import signax
from tests.conftest import (
    BENCH_GL_CASES,
    BENCH_MKW_CASES,
    BENCH_SHUFFLE_CASES,
    generate_brownian_path,
    generate_scalar_path,
)
from tests.test_integrators.conftest import benchmark_wrapper


@pytest.mark.benchmark(group="log_signature")
@pytest.mark.parametrize("dim,depth", BENCH_SHUFFLE_CASES)
@pytest.mark.parametrize("num_timesteps", [256])
def test_signature_benchmark_quicksig(
    benchmark: BenchmarkFixture,
    dim: int,
    depth: int,
    num_timesteps: int,
) -> None:
    """Benchmark QuickSig log-signature computation on representative paths."""
    key = jax.random.PRNGKey(7)
    path = generate_scalar_path(key, dim, num_timesteps)
    channels = int(path.shape[1])
    hopf = ShuffleHopfAlgebra.build(ambient_dim=channels, depth=depth)

    def _quicksig_full_signature(x: jax.Array) -> jax.Array:
        log_sig = compute_path_signature(
            x,
            depth,
            hopf,
            mode="full",
        )
        return jnp.concatenate([coeff.flatten() for coeff in log_sig.coeffs])

    flattened = benchmark_wrapper(benchmark, _quicksig_full_signature, path)
    expected_dim = get_signature_dim(depth, channels)
    assert flattened.shape == (expected_dim,)


@pytest.mark.benchmark(group="log_signature")
@pytest.mark.parametrize("dim,depth", BENCH_SHUFFLE_CASES)
@pytest.mark.parametrize("num_timesteps", [256])
def test_log_signature_benchmark_quicksig(
    benchmark: BenchmarkFixture,
    dim: int,
    depth: int,
    num_timesteps: int,
) -> None:
    """Benchmark QuickSig log-signature computation on representative paths."""
    key = jax.random.PRNGKey(7)
    path = generate_scalar_path(key, dim, num_timesteps)
    channels = int(path.shape[1])
    hopf = ShuffleHopfAlgebra.build(ambient_dim=channels, depth=depth)

    def _quicksig_full_logsignature(x: jax.Array) -> jax.Array:
        log_sig = compute_log_signature(
            x,
            depth,
            hopf,
            "Lyndon words",
            mode="full",
        )
        return jnp.concatenate([coeff.flatten() for coeff in log_sig.coeffs])

    flattened = benchmark_wrapper(benchmark, _quicksig_full_logsignature, path)
    expected_dim = get_log_signature_dim(depth, channels)
    assert flattened.shape == (expected_dim,)


@pytest.mark.benchmark(group="log_signature")
@pytest.mark.parametrize("dim,depth", BENCH_SHUFFLE_CASES)
@pytest.mark.parametrize("num_timesteps", [256])
def test_log_signature_benchmark_signax(
    benchmark: BenchmarkFixture,
    dim: int,
    depth: int,
    num_timesteps: int,
) -> None:
    """Benchmark Signax log-signature computation on the same inputs."""
    key = jax.random.PRNGKey(7)
    path = generate_scalar_path(key, dim, num_timesteps)

    def _signax_full_logsignature(x: jax.Array) -> jax.Array:
        return signax.logsignature(x, depth=depth, stream=False)

    log_sig = benchmark_wrapper(benchmark, _signax_full_logsignature, path)
    expected_dim = get_log_signature_dim(depth, int(path.shape[1]))
    assert log_sig.shape == (expected_dim,)


@pytest.mark.benchmark(group="bck_signature")
@pytest.mark.parametrize("dim,depth", BENCH_GL_CASES)
@pytest.mark.parametrize("num_timesteps", [256])
def test_bck_signature_benchmark(
    benchmark: BenchmarkFixture,
    dim: int,
    depth: int,
    num_timesteps: int,
) -> None:
    """Benchmark non-planar BCK branched Itô signatures on Brownian paths."""
    key = jax.random.PRNGKey(5)
    path = generate_brownian_path(key, dim, num_timesteps)
    hopf = GLHopfAlgebra.build(dim, depth)

    def _bck_full_signature(x: jax.Array) -> jax.Array:
        sig = compute_nonplanar_branched_signature(
            path=x,
            depth=depth,
            hopf=hopf,
            mode="full",
        )
        return sig.flatten()

    flattened = benchmark_wrapper(benchmark, _bck_full_signature, path)
    expected_dim = sum(hopf.basis_size(level) for level in range(depth))
    assert flattened.shape == (expected_dim,)


@pytest.mark.benchmark(group="mkw_signature")
@pytest.mark.parametrize("dim,depth", BENCH_MKW_CASES)
@pytest.mark.parametrize("num_timesteps", [256])
def test_mkw_signature_benchmark(
    benchmark: BenchmarkFixture,
    dim: int,
    depth: int,
    num_timesteps: int,
) -> None:
    """Benchmark planar MKW branched Itô signatures on Brownian paths."""
    key = jax.random.PRNGKey(9)
    path = generate_brownian_path(key, dim, num_timesteps)
    hopf = MKWHopfAlgebra.build(dim, depth)

    def _mkw_full_signature(x: jax.Array) -> jax.Array:
        sig = compute_planar_branched_signature(
            path=x,
            depth=depth,
            hopf=hopf,
            mode="full",
        )
        return sig.flatten()

    flattened = benchmark_wrapper(benchmark, _mkw_full_signature, path)
    expected_dim = sum(hopf.basis_size(level) for level in range(depth))
    assert flattened.shape == (expected_dim,)
