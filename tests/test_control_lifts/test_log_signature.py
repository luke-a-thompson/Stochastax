import jax
import jax.numpy as jnp
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from stochastax.control_lifts.log_signature import compute_log_signature
from stochastax.analytics.signature_sizes import (
    get_log_signature_dim,
    get_signature_dim,
)
import signax
from tests.conftest import generate_scalar_path
from tests.test_integrators.conftest import benchmark_wrapper
from typing import Literal

_test_key = jax.random.PRNGKey(42)


def _log_signature_benchmark_inputs(
    num_timesteps: int = 1024, channels: int = 3, depth: int = 3
) -> tuple[jax.Array, int]:
    """Build a deterministic path and depth for benchmarking."""
    key = jax.random.PRNGKey(7)
    path = generate_scalar_path(key, channels, num_timesteps)
    return path, depth


@pytest.mark.parametrize("scalar_path_fixture", [(1, 10), (2, 10)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("log_signature_type", ["Tensor words", "Lyndon words"])
def test_log_signature_shape_full(
    scalar_path_fixture: jax.Array,
    depth: int,
    log_signature_type: Literal["Tensor words", "Lyndon words"],
) -> None:
    """Log signature tensor dimension matches algebraic formula."""
    path = scalar_path_fixture
    channels = path.shape[1]
    log_sig = compute_log_signature(
        path,
        depth=depth,
        log_signature_type=log_signature_type,
        mode="full",
    )
    log_sig_array = log_sig.flatten()

    if log_signature_type == "Tensor words":
        expected_dim = get_signature_dim(depth, channels)
    else:  # Lyndon words
        expected_dim = get_log_signature_dim(depth, channels)

    expected_shape = (expected_dim,)
    assert log_sig_array.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {log_sig_array.shape}"
    )


@pytest.mark.parametrize("scalar_path_fixture", [(1, 10), (2, 10)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("log_signature_type", ["Tensor words", "Lyndon words"])
def test_log_signature_shape_stream(
    scalar_path_fixture: jax.Array,
    depth: int,
    log_signature_type: Literal["Tensor words", "Lyndon words"],
) -> None:
    """Log signature tensor dimension matches algebraic formula."""
    path = scalar_path_fixture
    num_steps, channels = path.shape
    log_sigs = compute_log_signature(
        path,
        depth=depth,
        log_signature_type=log_signature_type,
        mode="stream",
    )

    assert len(log_sigs) == num_steps - 1
    log_sig_array = jnp.stack(
        [jnp.concatenate([coeff.flatten() for coeff in log_sig.coeffs]) for log_sig in log_sigs]
    )

    if log_signature_type == "Tensor words":
        expected_dim = get_signature_dim(depth, channels)
    else:
        expected_dim = get_log_signature_dim(depth, channels)

    expected_shape = (num_steps - 1, expected_dim)
    assert log_sig_array.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {log_sig_array.shape}"
    )


@pytest.mark.parametrize("scalar_path_fixture", [(1, 10), (2, 10)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("log_signature_type", ["Tensor words", "Lyndon words"])
def test_log_signature_shape_incremental(
    scalar_path_fixture: jax.Array,
    depth: int,
    log_signature_type: Literal["Tensor words", "Lyndon words"],
) -> None:
    """Log signature tensor dimension matches algebraic formula."""
    path = scalar_path_fixture
    num_steps, channels = path.shape
    log_sigs = compute_log_signature(
        path,
        depth=depth,
        log_signature_type=log_signature_type,
        mode="incremental",
    )

    assert len(log_sigs) == num_steps - 1
    log_sig_array = jnp.stack(
        [jnp.concatenate([coeff.flatten() for coeff in log_sig.coeffs]) for log_sig in log_sigs]
    )

    if log_signature_type == "Tensor words":
        expected_dim = get_signature_dim(depth, channels)
    else:
        expected_dim = get_log_signature_dim(depth, channels)

    expected_shape = (num_steps - 1, expected_dim)
    assert log_sig_array.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {log_sig_array.shape}"
    )


# Signax does not support 1D paths
@pytest.mark.parametrize("scalar_path_fixture", [(2, 10), (3, 10)], indirect=True)
@pytest.mark.parametrize("depth", [2, 3])
def test_quicksig_signax_equivalence_full(scalar_path_fixture: jax.Array, depth: int) -> None:
    """
    Test that the log signature computed by QuickSig and Signax are equivalent.
    """
    path = scalar_path_fixture
    quicksig_log_sig = compute_log_signature(
        path, depth=depth, log_signature_type="Lyndon words", mode="full"
    )
    quicksig_log_sig = jnp.concatenate([x.flatten() for x in quicksig_log_sig.coeffs])

    signax_log_sig = signax.logsignature(path, depth=depth, stream=False)

    assert jnp.allclose(quicksig_log_sig, signax_log_sig, atol=1e-5, rtol=1e-5)


# Signax does not support 1D paths
@pytest.mark.parametrize("scalar_path_fixture", [(2, 10), (3, 10)], indirect=True)
@pytest.mark.parametrize("depth", [2, 3])
def test_quicksig_signax_equivalence_stream(scalar_path_fixture: jax.Array, depth: int) -> None:
    """
    Test that the log signature computed by QuickSig and Signax are equivalent.
    """
    path = scalar_path_fixture
    quicksig_log_sigs = compute_log_signature(
        path, depth=depth, log_signature_type="Lyndon words", mode="stream"
    )
    quicksig_log_sigs = jnp.stack(
        [
            jnp.concatenate([coeff.flatten() for coeff in log_sig.coeffs])
            for log_sig in quicksig_log_sigs
        ]
    )

    signax_log_sigs = signax.logsignature(path, depth=depth, stream=True)

    assert jnp.allclose(quicksig_log_sigs, signax_log_sigs, atol=1e-5, rtol=1e-5)


@pytest.mark.benchmark(group="log_signature")
def test_log_signature_benchmark_quicksig(benchmark: BenchmarkFixture) -> None:
    """Benchmark QuickSig log-signature computation on a representative path."""
    path, depth = _log_signature_benchmark_inputs()

    def _quicksig_full_logsignature(x: jax.Array) -> jax.Array:
        log_sig = compute_log_signature(
            x,
            depth,
            "Lyndon words",
            mode="full",
        )
        return jnp.concatenate([coeff.flatten() for coeff in log_sig.coeffs])

    flattened = benchmark_wrapper(benchmark, _quicksig_full_logsignature, path)
    expected_dim = get_log_signature_dim(depth, path.shape[1])
    assert flattened.shape == (expected_dim,)


@pytest.mark.benchmark(group="log_signature")
def test_log_signature_benchmark_signax(benchmark: BenchmarkFixture) -> None:
    """Benchmark Signax log-signature computation on the same input."""
    path, depth = _log_signature_benchmark_inputs()

    def _signax_full_logsignature(x: jax.Array) -> jax.Array:
        return signax.logsignature(x, depth=depth, stream=False)

    log_sig = benchmark_wrapper(benchmark, _signax_full_logsignature, path)
    expected_dim = get_log_signature_dim(depth, path.shape[1])
    assert log_sig.shape == (expected_dim,)
