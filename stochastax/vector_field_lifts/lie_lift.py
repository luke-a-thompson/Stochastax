from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from stochastax.vector_field_lifts.vector_field_lift_types import (
    LyndonBrackets,
    LyndonBracketFunctions,
)
from stochastax.hopf_algebras.hopf_algebras import ShuffleHopfAlgebra
from stochastax.hopf_algebras.free_lie import (
    build_lyndon_dependency_tables,
    commutator,
)
from stochastax.manifolds import Manifold, EuclideanSpace


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
    n = A.shape[-1]
    if not words_by_len:
        depth = 0
        return LyndonBrackets([jnp.zeros((0, n, n), dtype=A.dtype) for _ in range(depth)])

    (
        _splits,
        prefix_levels,
        prefix_indices,
        suffix_levels,
        suffix_indices,
    ) = build_lyndon_dependency_tables(words_by_len)

    all_brackets: list[jax.Array] = []

    for level_idx, words in enumerate(words_by_len):
        if words.size == 0:
            all_brackets.append(jnp.zeros((0, n, n), dtype=A.dtype))
            continue

        if level_idx == 0:
            # Level 0: single letters, brackets are just the generators A[i].
            level_brackets = A[words[:, 0]]
        else:
            pl = prefix_levels[level_idx]
            pi = prefix_indices[level_idx]
            sl = suffix_levels[level_idx]
            si = suffix_indices[level_idx]

            num_words = int(words.shape[0])
            level_list: list[jax.Array] = []
            for word_idx in range(num_words):
                prefix_level = int(pl[word_idx])
                prefix_index = int(pi[word_idx])
                suffix_level = int(sl[word_idx])
                suffix_index = int(si[word_idx])
                if prefix_level < 0 or suffix_level < 0:
                    raise ValueError("Invalid Lyndon prefix/suffix metadata encountered.")

                prefix_bracket = all_brackets[prefix_level][prefix_index]
                suffix_bracket = all_brackets[suffix_level][suffix_index]
                level_list.append(commutator(prefix_bracket, suffix_bracket))

            level_brackets = jnp.stack(level_list, axis=0)

        all_brackets.append(level_brackets)

    return LyndonBrackets(all_brackets)


def form_lyndon_bracket_functions(
    vector_fields: list[Callable[[jax.Array], jax.Array]],
    hopf: ShuffleHopfAlgebra,
    manifold: Manifold = EuclideanSpace(),
) -> LyndonBracketFunctions:
    """
    Build callable Lyndon bracket vector fields V_w(y).

    Args:
        vector_fields: driver vector fields f_i: R^n -> R^n
        hopf: ShuffleHopfAlgebra with cached Lyndon metadata (cache_lyndon_basis=True)
        project_to_tangent: optional projector to enforce manifold tangency

    Returns:
        Nested lists: level k contains callables for Lyndon words of length k+1.
    """
    if len(vector_fields) != hopf.ambient_dimension:
        raise ValueError(
            "Number of vector fields must equal hopf.ambient_dimension "
            f"({len(vector_fields)} != {hopf.ambient_dimension})."
        )

    cached_words = hopf.lyndon_basis_by_degree
    prefix_level_by_degree = hopf.lyndon_prefix_level_by_degree
    prefix_index_by_degree = hopf.lyndon_prefix_index_by_degree
    suffix_level_by_degree = hopf.lyndon_suffix_level_by_degree
    suffix_index_by_degree = hopf.lyndon_suffix_index_by_degree
    if not cached_words:
        raise ValueError(
            "ShuffleHopfAlgebra must be constructed via ShuffleHopfAlgebra.build "
            "with cache_lyndon_basis=True to use form_lyndon_bracket_functions."
        )
    if not (
        len(cached_words)
        == len(prefix_level_by_degree)
        == len(prefix_index_by_degree)
        == len(suffix_level_by_degree)
        == len(suffix_index_by_degree)
    ):
        raise ValueError("ShuffleHopfAlgebra Lyndon caches are inconsistent.")

    bracket_fns: list[list[Callable[[jax.Array], jax.Array]]] = []

    for level_idx, words_level in enumerate(cached_words):
        words_np = np.asarray(words_level)
        num_words = int(words_np.shape[0]) if words_np.ndim > 0 else 0
        funcs_level: list[Callable[[jax.Array], jax.Array]] = []

        if num_words == 0:
            bracket_fns.append(funcs_level)
            continue

        if level_idx == 0:
            for word in words_np:
                letter = int(word[0])

                def make_leaf(index: int) -> Callable[[jax.Array], jax.Array]:
                    def leaf(y: jax.Array) -> jax.Array:
                        return manifold.project_to_tangent(y, vector_fields[index](y))

                    return leaf

                fn = jax.jit(make_leaf(letter))
                funcs_level.append(fn)
        else:
            prefix_levels = np.asarray(prefix_level_by_degree[level_idx])
            prefix_indices = np.asarray(prefix_index_by_degree[level_idx])
            suffix_levels = np.asarray(suffix_level_by_degree[level_idx])
            suffix_indices = np.asarray(suffix_index_by_degree[level_idx])

            for word_idx in range(num_words):
                prefix_level = int(prefix_levels[word_idx])
                prefix_index = int(prefix_indices[word_idx])
                suffix_level = int(suffix_levels[word_idx])
                suffix_index = int(suffix_indices[word_idx])
                if prefix_level < 0 or suffix_level < 0:
                    raise ValueError("Invalid Lyndon prefix/suffix metadata encountered.")
                prefix_fn = bracket_fns[prefix_level][prefix_index]
                suffix_fn = bracket_fns[suffix_level][suffix_index]

                def make_bracket(
                    f_left: Callable[[jax.Array], jax.Array],
                    f_right: Callable[[jax.Array], jax.Array],
                ) -> Callable[[jax.Array], jax.Array]:
                    def bracket(y: jax.Array) -> jax.Array:
                        left_val = f_left(y)
                        right_val = f_right(y)
                        left_push = jax.jvp(f_left, (y,), (right_val,))[1]
                        right_push = jax.jvp(f_right, (y,), (left_val,))[1]
                        return manifold.project_to_tangent(y, left_push - right_push)

                    return bracket

                fn = jax.jit(make_bracket(prefix_fn, suffix_fn))
                funcs_level.append(fn)

        bracket_fns.append(funcs_level)

    return LyndonBracketFunctions(bracket_fns)


def form_lyndon_lift(
    vector_fields: list[Callable[[jax.Array], jax.Array]],
    base_point: jax.Array,
    hopf: ShuffleHopfAlgebra,
    manifold: Manifold = EuclideanSpace(),
) -> LyndonBrackets:
    """
    Build nonlinear (pre-Lie) Lyndon brackets evaluated at a base point.

    Args:
        vector_fields: list of driver vector fields vector_fields[i]: R^n -> R^n. One
            vector field per driver dimension.
        base_point: base point where the brackets' Jacobians are evaluated.
        hopf: Shuffle (tensor) Hopf algebra built with ``cache_lyndon_basis=True`` so
            Lyndon combinatorics can be reused by the lift.
        manifold: manifold to project the vector fields onto the tangent space.

    Returns:
        List storing [N_k, n, n] Jacobians per Lyndon level (degree k+1), where N_k is
        the number of Lyndon words of length k+1.
    """
    if base_point.ndim != 1:
        raise ValueError(f"base_point must be 1D, got shape {base_point.shape}.")
    if len(vector_fields) != hopf.ambient_dimension:
        raise ValueError(
            "Number of vector fields must equal hopf.ambient_dimension "
            f"({len(vector_fields)} != {hopf.ambient_dimension})."
        )
    n_state = int(base_point.shape[0])

    vector_field_funcs = form_lyndon_bracket_functions(
        vector_fields=vector_fields,
        hopf=hopf,
        manifold=manifold,
    )

    brackets_by_len: list[jax.Array] = []
    for funcs_level in vector_field_funcs:
        if not funcs_level:
            brackets_by_len.append(jnp.zeros((0, n_state, n_state), dtype=base_point.dtype))
            continue

        level_funcs = tuple(funcs_level)

        def eval_level(
            y: jax.Array,
            funcs: tuple[Callable[[jax.Array], jax.Array], ...] = level_funcs,
        ) -> jax.Array:
            vals = [fn(y) for fn in funcs]
            return jnp.stack(vals, axis=0)

        level_jac = jax.jacfwd(eval_level)(base_point)
        brackets_by_len.append(level_jac)

    return LyndonBrackets(brackets_by_len)
