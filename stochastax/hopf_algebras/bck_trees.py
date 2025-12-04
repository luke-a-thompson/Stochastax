"""Enumeration of rooted unordered trees via BH canonical sequences.

This module provides ``enumerate_bck_trees`` which returns all rooted
unordered (non-plane) trees with a fixed number of nodes, using the
Beyer-Hedetniemi (BH) successor on canonical level sequences.
https://combinatorialpress.com/jcmcc-articles/volume-076/an-application-of-level-sequences-to-parallel-generation-of-rootedtrees/
"""

import jax.numpy as jnp
from stochastax.hopf_algebras.forest_types import Forest, BCKForest


def _levelseq_to_parent(levels: list[int]) -> jnp.ndarray:
    """Convert a canonical level sequence to a parent array.

    Args:
        levels: A list of positive integers ``L[0..n-1]`` with ``L[0] == 1``,
            representing depths of nodes in preorder. This must be a valid
            canonical level sequence for an unordered rooted tree.

    Returns:
        A length-``n`` ``jnp.int32`` array ``parent`` with ``parent[0] == -1``
        and ``0 <= parent[i] < i`` for ``i > 0``.
    """
    n = len(levels)
    parent_py: list[int] = [-1] * n
    stack: list[int] = []
    for i in range(n):
        d = levels[i]
        while len(stack) >= d:
            stack.pop()
        if stack:
            parent_py[i] = stack[-1]
        stack.append(i)
    return jnp.asarray(parent_py, dtype=jnp.int32)


def _bh_successor(levels: list[int]) -> list[int]:
    """Beyer-Hedetniemi successor for canonical level sequences (1-based depths)."""
    n = len(levels)

    # p = largest index with level > 2  (1-based depths; minimal sequence has no >2)
    p = -1
    for i in range(n - 1, -1, -1):
        if levels[i] > 2:
            p = i
            break
    if p == -1:
        # Already at the minimal sequence [1, 2, 2, ..., 2]
        return levels[:]

    # q = parent position of node p: last i < p with level == levels[p] - 1
    parent_level = levels[p] - 1
    q = -1
    for i in range(p - 1, -1, -1):
        if levels[i] == parent_level:
            q = i
            break

    # Period length and repeat-copy
    k = p - q
    S = levels[:]
    for i in range(p, n):
        S[i] = S[i - k]  # copy from S (progressively), not from the old array
    return S


def _enumerate_bck_trees_n(n: int) -> BCKForest:
    """Enumerate rooted unordered trees with exactly ``n`` nodes.

    Uses the Beyer-Hedetniemi successor to iterate canonical level sequences.

    Args:
        n: Number of nodes per tree. Must satisfy ``n >= 1``.

    Returns:
        A ``Forest`` where ``parent`` has shape ``(A000081(n), n)`` and dtype
        ``int32``. Rows are ordered by decreasing lexicographic order of the
        canonical level sequences.

    Notes:
    - Function is JAX-jittable with ``static_argnums=0``.
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if n == 1:
        return BCKForest(Forest(parent=jnp.asarray([[-1]], dtype=jnp.int32)))

    # Start and end level sequences
    levels = list(range(1, n + 1))  # [1,2,3,...,n]
    levels_min = [1] + [2] * (n - 1)

    parents_list: list[jnp.ndarray] = []
    while True:
        parents_list.append(_levelseq_to_parent(levels))
        if levels == levels_min:
            break
        levels = _bh_successor(levels)

    parents = jnp.stack(parents_list, axis=0).astype(jnp.int32)
    return BCKForest(Forest(parent=parents))


def enumerate_bck_trees(max_n: int) -> list[BCKForest]:
    """Enumerate unordered rooted trees for all degrees 1..``max_n``.

    Returns a list where entry at index n-1 is the ``BCKForest`` for degree n.
    """
    if max_n <= 0:
        raise ValueError("max_n must be >= 1")
    return [_enumerate_bck_trees_n(n) for n in range(1, max_n + 1)]
