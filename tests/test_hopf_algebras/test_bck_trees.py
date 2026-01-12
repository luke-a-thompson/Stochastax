import jax
import jax.numpy as jnp
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from stochastax.hopf_algebras.bck_trees import enumerate_bck_trees
from stochastax.hopf_algebras.mkw_trees import enumerate_mkw_trees
from tests.conftest import benchmark_wrapper


# OEIS A000081: number of rooted unlabeled trees with n nodes
# https://oeis.org/A000081
A000081 = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 9,
    6: 20,
    7: 48,
    8: 115,
    9: 286,
    10: 719,
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


def _canonical_code(parent: jnp.ndarray) -> tuple:
    values = parent.tolist()
    n = len(values)
    children: list[list[int]] = [[] for _ in range(n)]
    root = 0

    for idx, par in enumerate(values):
        if par == -1:
            root = idx
        else:
            children[par].append(idx)

    def encode(node: int) -> tuple:
        sub = [encode(child) for child in children[node]]
        sub.sort()
        return tuple(sub)

    return encode(root)


@pytest.mark.parametrize("n", sorted(A000081))
def test_unordered_counts_small_n(n: int) -> None:
    batch = enumerate_bck_trees(n)[n - 1]
    parents = batch.parent
    assert parents.shape[0] == A000081[n], (
        f"For n={n}, expected {A000081[n]} unordered trees, got {parents.shape[0]}"
    )
    for i in range(parents.shape[0]):
        _check_preorder_parent_array(parents[i])


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_unordered_enumerator_counts(n: int) -> None:
    batch = enumerate_bck_trees(n)[n - 1]
    parents = batch.parent
    expected = A000081[n]
    assert parents.shape == (expected, n), (
        f"Expected parents shape {(expected, n)}, got {parents.shape}"
    )
    _assert_parent_batch(parents, n)
    as_tuples = {tuple(map(int, row.tolist())) for row in parents}
    assert len(as_tuples) == expected, f"Expected {expected} unique encodings, got {len(as_tuples)}"


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_unordered_matches_plane_canonical_codes(n: int) -> None:
    plane_batch = enumerate_mkw_trees(n)[n - 1]
    unordered_batch = enumerate_bck_trees(n)[n - 1]

    plane_codes = {_canonical_code(row) for row in plane_batch.parent}
    unordered_codes = {_canonical_code(row) for row in unordered_batch.parent}

    assert unordered_codes == plane_codes, (
        f"Canonical code sets differ for n={n}: unordered={len(unordered_codes)}, plane={len(plane_codes)}"
    )


@pytest.mark.parametrize("n", [2, 3, 4])
def test_unordered_is_jittable(n: int) -> None:
    unordered_fn = jax.jit(enumerate_bck_trees, static_argnums=0)
    unordered_batch = unordered_fn(n)[n - 1]
    _assert_parent_batch(unordered_batch.parent, n)


@pytest.mark.benchmark(group="bck_trees")
def test_bck_enumeration_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark enumeration of BCK tree parents at n=8."""
    n = 8

    def _run_enumeration(size: int) -> jnp.ndarray:
        return enumerate_bck_trees(size)[size - 1].parent

    parents = benchmark_wrapper(benchmark, _run_enumeration, n)
    expected = A000081[n]
    assert parents.shape == (expected, n)
    _assert_parent_batch(parents, n)
