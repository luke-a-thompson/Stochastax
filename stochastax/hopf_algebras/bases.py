"""Tree and word basis enumeration for Hopf algebras.

This module provides:
- Lyndon word enumeration for shuffle/tensor Hopf algebras
- Ordered (plane) tree enumeration via Dyck words for MKW Hopf algebras
- Unordered tree enumeration via BH sequences for GL/BCK Hopf algebras
"""

import jax
import jax.numpy as jnp
import numpy as np
from collections import defaultdict
from stochastax.hopf_algebras.forest_types import Forest, MKWForest, BCKForest


# ============================================================================
# Lyndon word basis for Shuffle Hopf Algebra
# ============================================================================


def commutator(a: jax.Array, b: jax.Array) -> jax.Array:
    return a @ b - b @ a


def enumerate_lyndon_basis(depth: int, dim: int) -> list[jax.Array]:
    """Duval's generator. Generates lists of words (integer sequences) for each level up to a specified depth.
    These words typically correspond to the Lyndon word basis.
    Ref: https://www.lyndex.org/algo.php
    """
    if dim == 1:
        first_level_word = [jnp.array([[0]], dtype=jnp.int32)]
        higher_level_empty_words = [jnp.empty((0, i + 1), dtype=jnp.int32) for i in range(1, depth)]
        return first_level_word + higher_level_empty_words

    list_of_words: dict[int, list[list[int]]] = defaultdict(list)
    word: list[int] = [-1]
    while word:
        word[-1] += 1
        m = len(word)
        list_of_words[m - 1].append(list(word))
        while len(word) < depth:
            word.append(word[-m])
        while word and word[-1] == dim - 1:
            word.pop()

    result: list[jax.Array] = []
    for level in range(depth):
        words_level = list_of_words[level]
        if not words_level:
            result.append(jnp.empty((0, level + 1), dtype=jnp.int32))
        else:
            result.append(jnp.asarray(words_level, dtype=jnp.int32))

    return result


def build_lyndon_dependency_tables(
    words_by_len: list[jax.Array],
) -> tuple[
    tuple[jax.Array, ...],
    tuple[jax.Array, ...],
    tuple[jax.Array, ...],
    tuple[jax.Array, ...],
    tuple[jax.Array, ...],
]:
    """Precompute Lyndon split/prefix/suffix metadata per level.

    For each Lyndon word :math:`w` (length > 1) we cache its **standard
    (Shirshov) factorization** :math:`w = uv`, where :math:`v` is the longest
    proper Lyndon suffix of :math:`w` and :math:`u` is the corresponding proper
    prefix. Downstream routines (e.g. the shuffle/Lyndon vector field lift)
    then evaluate brackets via the recursion :math:`[w] = [[u],[v]]` using
    constant-time table lookups, rather than re-factorizing words at runtime.

    Args:
        words_by_len: output of ``enumerate_lyndon_basis`` grouping Lyndon words
            by length (indexing by level = length - 1).

    Returns:
        Tuple of five tuples (per level):
            - splits[level]: int32 array [N_level] giving the split position of each word
              (0 for length-1 words).
            - prefix_levels[level]: int32 array of Lyndon levels for prefixes
              (-1 for length-1 words).
            - prefix_indices[level]: int32 array of prefix indices within that level
              (-1 for length-1 words).
            - suffix_levels[level]: same as prefix but for suffix.
            - suffix_indices[level]: suffix indices.
    """

    tuple_levels: list[list[tuple[int, ...]]] = []
    index_maps: list[dict[tuple[int, ...], int]] = []
    for words in words_by_len:
        if words.size == 0:
            tuple_levels.append([])
            index_maps.append({})
            continue
        words_host = np.asarray(words, dtype=np.int32)
        tuples_level: list[tuple[int, ...]] = []
        index_map: dict[tuple[int, ...], int] = {}
        for idx, row in enumerate(words_host):
            word_tuple = tuple(int(v) for v in row)
            tuples_level.append(word_tuple)
            index_map[word_tuple] = idx
        tuple_levels.append(tuples_level)
        index_maps.append(index_map)

    splits: list[jax.Array] = []
    prefix_levels: list[jax.Array] = []
    prefix_indices: list[jax.Array] = []
    suffix_levels: list[jax.Array] = []
    suffix_indices: list[jax.Array] = []

    for _, level_words in enumerate(tuple_levels):
        if not level_words:
            splits.append(jnp.zeros((0,), dtype=jnp.int32))
            prefix_levels.append(jnp.zeros((0,), dtype=jnp.int32))
            prefix_indices.append(jnp.zeros((0,), dtype=jnp.int32))
            suffix_levels.append(jnp.zeros((0,), dtype=jnp.int32))
            suffix_indices.append(jnp.zeros((0,), dtype=jnp.int32))
            continue

        level_splits: list[int] = []
        level_prefix_levels: list[int] = []
        level_prefix_indices: list[int] = []
        level_suffix_levels: list[int] = []
        level_suffix_indices: list[int] = []

        for word in level_words:
            word_len = len(word)
            if word_len == 1:
                level_splits.append(0)
                level_prefix_levels.append(-1)
                level_prefix_indices.append(-1)
                level_suffix_levels.append(-1)
                level_suffix_indices.append(-1)
                continue

            split_found = None
            found_prefix_level = -1
            found_prefix_idx = -1
            found_suffix_level = -1
            found_suffix_idx = -1
            for split in range(word_len - 1, 0, -1):
                prefix = word[:split]
                suffix = word[split:]
                prefix_len = len(prefix)
                suffix_len = len(suffix)
                prefix_level = prefix_len - 1
                suffix_level = suffix_len - 1
                prefix_map = index_maps[prefix_level]
                suffix_map = index_maps[suffix_level]
                prefix_ok = prefix in prefix_map
                suffix_ok = suffix in suffix_map
                if prefix_ok and suffix_ok:
                    split_found = split
                    found_prefix_level = prefix_level
                    found_suffix_level = suffix_level
                    found_prefix_idx = prefix_map[prefix]
                    found_suffix_idx = suffix_map[suffix]
                    break
            if split_found is None:
                raise ValueError(f"Unable to split Lyndon word {word}.")

            level_splits.append(split_found)
            level_prefix_levels.append(found_prefix_level)
            level_prefix_indices.append(found_prefix_idx)
            level_suffix_levels.append(found_suffix_level)
            level_suffix_indices.append(found_suffix_idx)

        splits.append(jnp.asarray(level_splits, dtype=jnp.int32))
        prefix_levels.append(jnp.asarray(level_prefix_levels, dtype=jnp.int32))
        prefix_indices.append(jnp.asarray(level_prefix_indices, dtype=jnp.int32))
        suffix_levels.append(jnp.asarray(level_suffix_levels, dtype=jnp.int32))
        suffix_indices.append(jnp.asarray(level_suffix_indices, dtype=jnp.int32))

    return (
        tuple(splits),
        tuple(prefix_levels),
        tuple(prefix_indices),
        tuple(suffix_levels),
        tuple(suffix_indices),
    )


# ============================================================================
# Ordered (plane) trees for MKW Hopf Algebra
# ============================================================================


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


def enumerate_mkw_trees(order: int) -> list[MKWForest]:
    """Enumerate ordered (plane) rooted trees for all degrees 1..``order``.

    Returns a list where entry at index n-1 is the ``MKWForest`` for degree n.
    """
    return [_enumerate_mkw_trees_n(n) for n in range(1, order + 1)]


# ============================================================================
# Unordered trees for GL/BCK Hopf Algebra
# ============================================================================


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


def enumerate_bck_trees(order: int) -> list[BCKForest]:
    """Enumerate unordered rooted trees for all degrees 1..``order``.

    Returns a list where entry at index n-1 is the ``BCKForest`` for degree n.
    """
    if order <= 0:
        raise ValueError("order must be >= 1")
    return [_enumerate_bck_trees_n(n) for n in range(1, order + 1)]
