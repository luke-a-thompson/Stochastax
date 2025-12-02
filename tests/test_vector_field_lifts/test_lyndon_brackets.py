import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from stochastax.hopf_algebras.free_lie import enumerate_lyndon_basis
from stochastax.hopf_algebras.free_lie import commutator
from stochastax.hopf_algebras.hopf_algebra_types import ShuffleHopfAlgebra
from stochastax.vector_field_lifts import form_lyndon_brackets_from_words
from stochastax.vector_field_lifts.lie_lift import form_lyndon_lift
from tests.test_integrators.conftest import (
    _linear_vector_fields,
    benchmark_wrapper,
    build_block_rotation_generators,
)


def test_form_lyndon_brackets_single_letter() -> None:
    """Test that single letter Lyndon words [i] return A[i] directly."""
    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(20), (dim, n, n))

    words = enumerate_lyndon_basis(depth=1, dim=A.shape[0])
    result = form_lyndon_brackets_from_words(A, words)

    # For depth=1, dim=2, we get words [0] and [1]
    # result is a list per level, so result[0] is level 1 brackets
    assert len(result) == 1
    assert result[0].shape == (2, n, n)
    np.testing.assert_allclose(result[0][0], A[0], rtol=1e-10)
    np.testing.assert_allclose(result[0][1], A[1], rtol=1e-10)


def test_form_lyndon_brackets_two_letters() -> None:
    """Test that two-letter Lyndon word [0,1] computes [A[0], A[1]]."""
    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(21), (dim, n, n))

    words = enumerate_lyndon_basis(depth=2, dim=A.shape[0])
    result = form_lyndon_brackets_from_words(A, words)

    # For depth=2, dim=2: words are [0], [1] at level 1, [0,1] at level 2
    assert len(result) == 2
    assert result[0].shape == (2, n, n)  # Level 1: [0], [1]
    assert result[1].shape == (1, n, n)  # Level 2: [0,1]

    # Check single letters (level 1)
    np.testing.assert_allclose(result[0][0], A[0], rtol=1e-10)
    np.testing.assert_allclose(result[0][1], A[1], rtol=1e-10)

    # Check two-letter bracket [0,1] = [A[0], A[1]] (level 2)
    bracket_01 = result[1][0]
    expected = commutator(A[0], A[1])
    np.testing.assert_allclose(bracket_01, expected, rtol=1e-10)


def test_form_lyndon_brackets_standard_factorization() -> None:
    """Test that Lyndon brackets use standard factorization correctly.

    For a Lyndon word w = uv where v is the longest proper Lyndon suffix,
    [w] = [[u], [v]].

    Example: [0,0,1] should factorize as [0] and [0,1] (not [0,0] and [1]).
    """
    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(22), (dim, n, n))

    words = enumerate_lyndon_basis(depth=3, dim=A.shape[0])
    result = form_lyndon_brackets_from_words(A, words)

    # For depth=3, dim=2:
    # Level 1: [0], [1]
    # Level 2: [0,1]
    # Level 3: [0,0,1], [0,1,1]
    # Let's verify [0,0,1] = [[0], [0,1]]
    bracket_0 = result[0][0]  # [0] = A[0] (level 1, first bracket)
    bracket_01 = result[1][0]  # [0,1] = [A[0], A[1]] (level 2, first bracket)
    bracket_001 = result[2][0]  # [0,0,1] should be [[0], [0,1]] (level 3, first bracket)

    expected = commutator(bracket_0, bracket_01)
    np.testing.assert_allclose(bracket_001, expected, rtol=1e-10)

    # Also verify [0,1,1] = [[0,1], [1]]
    bracket_1 = result[0][1]  # [1] = A[1] (level 1, second bracket)
    bracket_011 = result[2][1]  # [0,1,1] should be [[0,1], [1]] (level 3, second bracket)

    expected_011 = commutator(bracket_01, bracket_1)
    np.testing.assert_allclose(bracket_011, expected_011, rtol=1e-10)


def test_form_lyndon_brackets_jacobi_identity() -> None:
    """Test that Lyndon brackets satisfy the Jacobi identity.

    For any three elements in the free Lie algebra, the Jacobi identity should hold.
    We'll test with dim=3 using the single-letter brackets directly.
    """
    dim, n = 3, 3
    A = jax.random.normal(jax.random.PRNGKey(23), (dim, n, n))

    # Get single-letter brackets
    bracket_0 = A[0]  # [0] = A[0]
    bracket_1 = A[1]  # [1] = A[1]
    bracket_2 = A[2]  # [2] = A[2]

    # Compute Jacobi identity: [a, [b, c]] + [b, [c, a]] + [c, [a, b]] = 0
    # Using the three single-letter brackets directly
    term1 = commutator(bracket_0, commutator(bracket_1, bracket_2))  # [0, [1, 2]]
    term2 = commutator(bracket_1, commutator(bracket_2, bracket_0))  # [1, [2, 0]]
    term3 = commutator(bracket_2, commutator(bracket_0, bracket_1))  # [2, [0, 1]]

    jacobi_sum = term1 + term2 + term3

    # Jacobi identity should hold (up to numerical precision)
    np.testing.assert_allclose(jacobi_sum, jnp.zeros_like(jacobi_sum), atol=1e-5)


def test_form_lyndon_brackets_three_dimensions() -> None:
    """Test Lyndon brackets for dimension 3 with more complex words."""
    dim, n = 3, 2
    A = jax.random.normal(jax.random.PRNGKey(24), (dim, n, n))

    words = enumerate_lyndon_basis(depth=3, dim=A.shape[0])
    result = form_lyndon_brackets_from_words(A, words)

    # For dim=3, depth=3, we should have:
    # Level 1: [0], [1], [2]
    # Level 2: [0,1], [0,2], [1,2]
    # Level 3: [0,0,1], [0,0,2], [0,1,1], [0,1,2], [0,2,2], [1,1,2], [1,2,2]

    # Verify basic structure
    assert len(result) == 3  # Three levels
    assert result[0].shape[0] >= 3  # At least single letters

    # Verify [0,1] = [A[0], A[1]]
    bracket_0 = result[0][0]  # Level 1: [0]
    bracket_1 = result[0][1]  # Level 1: [1]
    bracket_01 = result[1][0]  # Level 2: [0,1] (first two-letter word)

    expected_01 = commutator(bracket_0, bracket_1)
    np.testing.assert_allclose(bracket_01, expected_01, rtol=1e-10)


def test_form_lyndon_brackets_empty() -> None:
    """Test handling of edge cases (empty/zero depth)."""
    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(25), (dim, n, n))

    # Depth 0 should return empty list
    words = enumerate_lyndon_basis(depth=0, dim=A.shape[0])
    result = form_lyndon_brackets_from_words(A, words)
    assert len(result) == 0

    # Depth 1 with dim=1 should work
    A_single = jax.random.normal(jax.random.PRNGKey(26), (1, n, n))
    words = enumerate_lyndon_basis(depth=2, dim=A_single.shape[0])
    result = form_lyndon_brackets_from_words(A_single, words)
    # For dim=1, we only get [0] at level 1, nothing at level 2+
    assert len(result) == 2
    assert result[0].shape == (1, n, n)  # Level 1: [0]
    assert result[1].shape == (0, n, n)  # Level 2: empty
    np.testing.assert_allclose(result[0][0], A_single[0], rtol=1e-10)


def test_form_lyndon_brackets_reproducibility() -> None:
    """Test that form_lyndon_brackets produces consistent results."""
    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(27), (dim, n, n))

    # Run twice with same input
    words = enumerate_lyndon_basis(depth=2, dim=A.shape[0])
    result1 = form_lyndon_brackets_from_words(A, words)
    words = enumerate_lyndon_basis(depth=2, dim=A.shape[0])
    result2 = form_lyndon_brackets_from_words(A, words)

    # Results should match exactly level-by-level
    assert len(result1) == len(result2)
    for level1, level2 in zip(result1, result2):
        np.testing.assert_allclose(level1, level2, rtol=1e-10)


def test_form_lyndon_brackets_gradients() -> None:
    """Test that gradients can be computed through Lyndon bracket formation."""
    dim, n = 2, 2
    key = jax.random.PRNGKey(28)

    def loss_fn(A: jax.Array) -> jax.Array:
        words = enumerate_lyndon_basis(depth=2, dim=A.shape[0])
        brackets = form_lyndon_brackets_from_words(A, words)
        # Sum of squares as a simple loss - concatenate all levels first
        brackets_flat = (
            jnp.concatenate(brackets, axis=0)
            if brackets
            else jnp.zeros((0, A.shape[-1], A.shape[-1]))
        )
        return jnp.sum(brackets_flat**2)

    A = jax.random.normal(key, (dim, n, n))

    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(A)

    # Gradients should have same shape as A
    assert grads.shape == A.shape
    # Gradients should not be all zeros (for a non-trivial case)
    assert not jnp.allclose(grads, jnp.zeros_like(grads), atol=1e-10)


def test_form_lyndon_brackets_consistency_with_duval() -> None:
    """Test that Lyndon brackets are consistent with duval_generator output."""
    from stochastax.control_lifts.log_signature import enumerate_lyndon_basis

    dim, n = 2, 3
    A = jax.random.normal(jax.random.PRNGKey(29), (dim, n, n))

    # Generate Lyndon words using duval_generator
    words_by_len = enumerate_lyndon_basis(depth=2, dim=dim)

    # Compute brackets using our function
    words = enumerate_lyndon_basis(depth=2, dim=A.shape[0])
    result = form_lyndon_brackets_from_words(A, words)

    # Verify we have the right number of brackets per level
    assert len(result) == len(words_by_len)
    for level_idx, (brackets_level, words_level) in enumerate(zip(result, words_by_len)):
        assert brackets_level.shape[0] == words_level.shape[0], f"Level {level_idx} mismatch"

    # Verify single-letter brackets match (level 1)
    assert words_by_len[0].shape[0] == 2  # [0] and [1]
    assert result[0].shape[0] == 2
    np.testing.assert_allclose(result[0][0], A[0], rtol=1e-10)
    np.testing.assert_allclose(result[0][1], A[1], rtol=1e-10)

    # Verify two-letter bracket (level 2)
    assert words_by_len[1].shape[0] == 1  # [0,1]
    assert result[1].shape[0] == 1
    bracket_01_expected = commutator(A[0], A[1])
    np.testing.assert_allclose(result[1][0], bracket_01_expected, rtol=1e-10)


LIFT_BENCH_CASES: list = [
    pytest.param(2, 2, id="dim-2-depth-2"),
    pytest.param(3, 3, id="dim-3-depth-3"),
    pytest.param(8, 3, id="dim-8-depth-3"),
]


@pytest.mark.benchmark(group="lyndon_lift")
@pytest.mark.parametrize("dim,depth", LIFT_BENCH_CASES)
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
    hopf = ShuffleHopfAlgebra.build(d=dim, max_degree=depth, cache_lyndon_basis=True)

    compiled = jax.jit(lambda y: form_lyndon_lift(vector_fields, y, hopf))
    brackets = benchmark_wrapper(benchmark, compiled, base_point)

    # Basic sanity checks to keep this a proper test.
    assert isinstance(brackets, list)
    assert len(brackets) == depth
