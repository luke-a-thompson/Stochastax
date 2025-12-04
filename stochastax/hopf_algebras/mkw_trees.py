"""Enumeration of rooted ordered (plane) trees via Dyck words.

This module provides ``enumerate_mkw_trees`` which returns all plane trees
with a fixed number of nodes by enumerating Dyck words of length ``2*(n-1)``
and mapping each to a parent array in preorder.
"""

import jax.numpy as jnp
from stochastax.hopf_algebras.forest_types import Forest, MKWForest


def _dyck_to_parent_preorder(dyck: list[int]) -> jnp.ndarray:
    """Convert a Dyck word (length 2*(n-1)) to a parent array in preorder.

    Convention:
    - Nodes are indexed in preorder (first time a node is created/entered)
      with root at index 0.
    - The parent array has shape (n,) with dtype jnp.int32.
    - The root has parent -1; for all i > 0, parent[i] < i.

    Dyck encoding used here:
    - We fix the root (index 0) and interpret a Dyck word with m = n-1 pairs.
    - Each opening symbol (1) creates a new child of the current node and
      descends into it; each closing symbol (0) ascends to the parent.
    """
    n_minus_1 = len(dyck) // 2
    n = n_minus_1 + 1
    parent_py: list[int] = [-1] + [0] * (n - 1)
    stack: list[int] = [0]
    next_index = 1
    for bit in dyck:
        if bit == 1:
            child = next_index
            next_index += 1
            parent_py[child] = stack[-1]
            stack.append(child)
        else:
            # close and go up
            stack.pop()
    return jnp.asarray(parent_py)


def _enumerate_mkw_trees_n(n: int) -> MKWForest:
    """Enumerate rooted ordered (plane) trees with exactly ``n`` nodes.

    Args:
        n: Number of nodes per tree. Must satisfy ``n >= 1``.

    Returns:
        A ``Forest`` where ``parent`` has shape ``(C_{n-1}, n)`` and dtype
        ``int32``, with rows ordered by the Dyck-word backtracking order.

    Notes:
    - Nodes are indexed by preorder; ``parent[0] == -1``.
    - Function is JAX-jittable with ``static_argnums=0``.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if n == 1:
        parents = jnp.asarray([[-1]], dtype=jnp.int32)
        return MKWForest(Forest(parent=parents))

    m = n - 1  # number of Dyck pairs

    results_py: list[jnp.ndarray] = []

    def backtrack(open_used: int, close_used: int, dyck: list[int]) -> None:
        if open_used == m and close_used == m:
            results_py.append(_dyck_to_parent_preorder(dyck))
            return
        if open_used < m:
            dyck.append(1)
            backtrack(open_used + 1, close_used, dyck)
            dyck.pop()
        if close_used < open_used:
            dyck.append(0)
            backtrack(open_used, close_used + 1, dyck)
            dyck.pop()

    backtrack(0, 0, [])
    parents = jnp.stack(results_py, axis=0).astype(jnp.int32)
    return MKWForest(Forest(parent=parents))


def enumerate_mkw_trees(max_n: int) -> list[MKWForest]:
    """Enumerate ordered (plane) rooted trees for all degrees 1..``max_n``.

    Returns a list where entry at index n-1 is the ``MKWForest`` for degree n.
    """
    if max_n <= 0:
        raise ValueError("max_n must be >= 1")
    return [_enumerate_mkw_trees_n(n) for n in range(1, max_n + 1)]
