"""Tests for free Lie algebra operations, focusing on Lyndon bracket formation.

These tests verify:
1. Correctness of Lyndon bracket computation with standard factorization
2. Mathematical properties of commutators (antisymmetry, Jacobi identity)
3. Edge cases and gradient computation
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from stochastax.hopf_algebras.free_lie import (
    commutator,
    enumerate_lyndon_basis,
)
from tests.conftest import benchmark_wrapper
from stochastax.integrators.series import form_lie_series


def test_commutator_basic() -> None:
    """Test basic commutator properties: [a,b] = ab - ba."""
    a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    b = jnp.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])

    result = commutator(a, b)
    expected = a @ b - b @ a

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_commutator_antisymmetry() -> None:
    """Test antisymmetry: [a,b] = -[b,a]."""
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

    ab = commutator(a, b)
    ba = commutator(b, a)

    np.testing.assert_allclose(ab, -ba, rtol=1e-10)


def test_commutator_self_zero() -> None:
    """Test that [a,a] = 0."""
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = commutator(a, a)

    np.testing.assert_allclose(result, jnp.zeros_like(result), rtol=1e-10)


def test_jacobi_identity() -> None:
    """Test the Jacobi identity: [a, [b, c]] + [b, [c, a]] + [c, [a, b]] = 0.

    This is a fundamental property of Lie algebras.
    """
    n = 3
    a = jax.random.normal(jax.random.PRNGKey(4), (n, n))
    b = jax.random.normal(jax.random.PRNGKey(5), (n, n))
    c = jax.random.normal(jax.random.PRNGKey(6), (n, n))

    # Compute the three terms
    term1 = commutator(a, commutator(b, c))
    term2 = commutator(b, commutator(c, a))
    term3 = commutator(c, commutator(a, b))

    jacobi_sum = term1 + term2 + term3

    # Jacobi identity holds exactly, but float32 has numerical precision limits
    np.testing.assert_allclose(jacobi_sum, jnp.zeros_like(jacobi_sum), atol=1e-5)


def test_apply_lie_coeffs() -> None:
    """Test that a Lie algebra element can be formed from brackets and coefficients."""
    n = 3
    num_words = 5

    # Random brackets and coefficients
    W = jax.random.normal(jax.random.PRNGKey(12), (num_words, n, n))
    lam = jax.random.normal(jax.random.PRNGKey(13), (num_words,))

    # Wrap into per-level inputs expected by form_series
    lam_by_len = [lam]
    # words_by_len is only used for filtering empties; provide non-empty dummy words
    result = form_lie_series(W, lam_by_len)

    # Should compute sum_i lam[i] * W[i]
    # Note: tensordot and explicit sum may have slightly different floating point behavior
    expected = jnp.sum(lam[:, None, None] * W, axis=0)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_apply_lie_coeffs_shape_error() -> None:
    """Test that mismatched shapes raise appropriate error."""
    n = 2
    W = jax.random.normal(jax.random.PRNGKey(14), (5, n, n))
    lam = jax.random.normal(jax.random.PRNGKey(15), (3,))  # Wrong size

    with pytest.raises(ValueError, match="does not match"):
        lam_by_len = [lam]
        form_lie_series(W, lam_by_len)


@pytest.mark.benchmark(group="free_lie")
def test_enumerate_lyndon_basis_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark enumeration of Lyndon words at depth=6, dim=4."""
    depth = 6
    dim = 4

    def _run_enumeration(_: int) -> list[jax.Array]:
        return enumerate_lyndon_basis(depth, dim)

    _ = benchmark_wrapper(benchmark, _run_enumeration, 0)
