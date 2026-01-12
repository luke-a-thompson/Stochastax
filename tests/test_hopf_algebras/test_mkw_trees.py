import jax
import jax.numpy as jnp
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from stochastax.hopf_algebras.mkw_trees import enumerate_mkw_trees
from tests.conftest import benchmark_wrapper

# OEIS A000108: Catalan numbers; here mapping n -> Catalan(n-1)
# https://oeis.org/A000108
A000108 = {
    1: 1,
    2: 1,
    3: 2,
    4: 5,
    5: 14,
    6: 42,
    7: 132,
    8: 429,
}


def _check_preorder_parent_array(arr: jnp.ndarray) -> None:
    assert arr.dtype == jnp.int32, f"Expected dtype int32, got {arr.dtype}"
    n = int(arr.shape[0])
    assert int(arr[0]) == -1, f"Root parent must be -1, got {int(arr[0])}"
    for i in range(1, n):
        assert int(arr[i]) < i, f"Parent at index {i} must be < {i}, got {int(arr[i])}"


def _assert_parent_batch(batch: jnp.ndarray, n: int) -> None:
    assert batch.dtype == jnp.int32, f"Expected batch dtype int32, got {batch.dtype}"
    assert batch.shape[1] == n, f"Expected second dimension {n}, got {batch.shape[1]}"
    assert jnp.all(batch[:, 0] == -1), "All roots must have parent -1 in column 0"
    if n > 1:
        assert jnp.all(batch[:, 1:] >= 0), "Non-root parents must be >= 0"


@pytest.mark.parametrize("n", sorted(A000108))
def test_ordered_counts_and_conventions_small_n(n: int) -> None:
    batch = enumerate_mkw_trees(n)[n - 1]
    parents = batch.parent
    assert parents.shape[0] == A000108[n], (
        f"For n={n}, expected A000108[n]={A000108[n]} trees, got {parents.shape[0]}"
    )
    for i in range(parents.shape[0]):
        _check_preorder_parent_array(parents[i])


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_ordered_enumerator_counts(n: int) -> None:
    batch = enumerate_mkw_trees(n)[n - 1]
    parents = batch.parent
    expected = A000108[n]
    assert parents.shape == (expected, n), (
        f"Expected parents shape {(expected, n)}, got {parents.shape}"
    )
    _assert_parent_batch(parents, n)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_ordered_is_jittable(n: int) -> None:
    ordered_fn = jax.jit(enumerate_mkw_trees, static_argnums=0)
    ordered_batch = ordered_fn(n)[n - 1]
    _assert_parent_batch(ordered_batch.parent, n)


@pytest.mark.benchmark(group="mkw_trees")
def test_mkw_enumeration_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark enumeration of MKW tree parents at n=8."""
    n = 8

    def _run_enumeration(size: int) -> jnp.ndarray:
        return enumerate_mkw_trees(size)[size - 1].parent

    parents = benchmark_wrapper(benchmark, _run_enumeration, n)
    expected = A000108[n]
    assert parents.shape == (expected, n)
    _assert_parent_batch(parents, n)
