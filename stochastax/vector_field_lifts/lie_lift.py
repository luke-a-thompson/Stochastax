from typing import Callable

import jax
import jax.numpy as jnp
from stochastax.control_lifts.log_signature import duval_generator
from stochastax.vector_field_lifts.vector_field_lift_types import LyndonBrackets
from stochastax.hopf_algebras.free_lie import (
    find_split_points_vectorized,
    compute_lyndon_level_brackets,
)


def form_lyndon_brackets_from_words(
    A: jax.Array,
    words_by_len: list[jax.Array],
) -> LyndonBrackets:
    """
    Form Lyndon brackets (commutators) for all Lyndon words up to given depth,
    returned per-level to mirror signature inputs.

    Uses the standard factorization: for a Lyndon word w = uv where v is the
    longest proper Lyndon suffix, [w] = [[u], [v]].

    Args:
        A: [dim, n, n] array where A[i] is the i-th Lie algebra basis element.
        words_by_len: Output of ``duval_generator(depth, dim)`` listing Lyndon words.

    Returns:
        A list of length `depth`. Entry k contains the bracket matrices for
        Lyndon words of length k+1 with shape [Nk, n, n]. Empty levels yield
        a [0, n, n] array.
    """
    if not words_by_len:
        n = A.shape[-1]
        depth = 0
        return LyndonBrackets([jnp.zeros((0, n, n), dtype=A.dtype) for _ in range(depth)])

    n = A.shape[-1]
    all_brackets: list[jax.Array] = []

    for word_len_idx, words in enumerate(words_by_len):
        if words.size == 0:
            all_brackets.append(jnp.zeros((0, n, n), dtype=A.dtype))
            continue

        word_length = word_len_idx + 1  # words at index k have length k+1

        # Compute brackets for this level
        if word_length == 1:
            # Level 1: just A[i] for each word
            level_brackets = A[words[:, 0]]  # [N1, n, n]
        else:
            # Level > 1: find splits and compute brackets
            splits = find_split_points_vectorized(words, words_by_len[:word_len_idx])
            level_brackets = compute_lyndon_level_brackets(
                words, splits, words_by_len[:word_len_idx], all_brackets, A
            )

        all_brackets.append(level_brackets)

    return LyndonBrackets(all_brackets)


def form_lyndon_lift(
    V: list[Callable[[jax.Array], jax.Array]],
    x: jax.Array,
    words_by_len: list[jax.Array],
    project_to_tangent: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
) -> LyndonBrackets:
    """
    Build nonlinear (pre-Lie) Lyndon brackets evaluated at a base point.

    Args:
        V: list of vector fields V[i]: R^n -> R^n (or manifold charts).
        x: base point where the brackets' Jacobians are evaluated.
        words_by_len: list produced by duval_generator capturing Lyndon words.
        project_to_tangent: optional projection enforcing tangent dynamics.

    Returns:
        List storing [N_k, n, n] Jacobians per Lyndon level (degree k+1).
    """
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape {x.shape}.")
    projector = project_to_tangent or (lambda _y, v: v)
    n_state = int(x.shape[0])

    words_tuples: list[list[tuple[int, ...]]] = []
    word_maps: list[dict[tuple[int, ...], int]] = []
    for words in words_by_len:
        if words.size == 0:
            words_tuples.append([])
            word_maps.append({})
            continue
        tuples_level: list[tuple[int, ...]] = []
        index_map: dict[tuple[int, ...], int] = {}
        for idx in range(words.shape[0]):
            word_tuple = tuple(int(v) for v in words[idx].tolist())
            tuples_level.append(word_tuple)
            index_map[word_tuple] = idx
        words_tuples.append(tuples_level)
        word_maps.append(index_map)

    splits_by_len: list[list[int]] = [[]]
    for level_idx in range(1, len(words_tuples)):
        level_words = words_tuples[level_idx]
        if not level_words:
            splits_by_len.append([])
            continue
        splits: list[int] = []
        for word in level_words:
            L = len(word)
            split_found = None
            for split in range(L - 1, 0, -1):
                prefix = word[:split]
                suffix = word[split:]
                prefix_len = len(prefix)
                suffix_len = len(suffix)
                prefix_ok = prefix_len == 1 or prefix in word_maps[prefix_len - 1]
                suffix_ok = suffix_len == 1 or suffix in word_maps[suffix_len - 1]
                if prefix_ok and suffix_ok:
                    split_found = split
                    break
            if split_found is None:
                raise ValueError(f"Unable to split Lyndon word {word}.")
            splits.append(split_found)
        splits_by_len.append(splits)

    vector_field_funcs: list[list[Callable[[jax.Array], jax.Array]]] = []
    func_maps: list[dict[tuple[int, ...], Callable[[jax.Array], jax.Array]]] = []

    for level_idx, level_words in enumerate(words_tuples):
        funcs_level: list[Callable[[jax.Array], jax.Array]] = []
        lookup: dict[tuple[int, ...], Callable[[jax.Array], jax.Array]] = {}
        if not level_words:
            vector_field_funcs.append(funcs_level)
            func_maps.append(lookup)
            continue

        if level_idx == 0:
            for word in level_words:
                letter = word[0]

                def make_leaf(index: int) -> Callable[[jax.Array], jax.Array]:
                    def leaf(y: jax.Array) -> jax.Array:
                        return projector(y, V[index](y))

                    return leaf

                fn = make_leaf(letter)
                funcs_level.append(fn)
                lookup[word] = fn
        else:
            splits = splits_by_len[level_idx]
            for word_idx, word in enumerate(level_words):
                split = splits[word_idx]
                prefix = word[:split]
                suffix = word[split:]
                prefix_fn = func_maps[len(prefix) - 1].get(prefix)
                suffix_fn = func_maps[len(suffix) - 1].get(suffix)
                if prefix_fn is None or suffix_fn is None:
                    raise KeyError(f"Missing Lyndon prefix or suffix for word {word}.")

                def make_bracket(
                    f_left: Callable[[jax.Array], jax.Array],
                    f_right: Callable[[jax.Array], jax.Array],
                ) -> Callable[[jax.Array], jax.Array]:
                    def bracket(y: jax.Array) -> jax.Array:
                        left_val = f_left(y)
                        right_val = f_right(y)
                        left_push = jax.jvp(f_left, (y,), (right_val,))[1]
                        right_push = jax.jvp(f_right, (y,), (left_val,))[1]
                        return projector(y, left_push - right_push)

                    return bracket

                fn = make_bracket(prefix_fn, suffix_fn)
                funcs_level.append(fn)
                lookup[word] = fn

        vector_field_funcs.append(funcs_level)
        func_maps.append(lookup)

    brackets_by_len: list[jax.Array] = []
    for funcs_level in vector_field_funcs:
        if not funcs_level:
            brackets_by_len.append(jnp.zeros((0, n_state, n_state), dtype=x.dtype))
            continue
        mats = [jax.jacrev(fn)(x) for fn in funcs_level]
        brackets_by_len.append(jnp.stack(mats, axis=0))

    return LyndonBrackets(brackets_by_len)
