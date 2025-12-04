"""Form Butcher/Lie-Butcher differentials for BCK and MKW forests.

## Public API

### Butcher Series:
- form_butcher_differentials: Form Butcher differentials for a BCK forest.
- form_lie_butcher_differentials: Form Lie-Butcher differentials for a MKW forest. Suitable for manifolds.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Optional

from stochastax.hopf_algebras.hopf_algebras import MKWForest, BCKForest
from stochastax.vector_field_lifts.vector_field_lift_types import (
    ButcherDifferentials,
    LieButcherDifferentials,
)


def _build_children_from_parent(parent_row: list[int]) -> list[list[int]]:
    """Build child adjacency lists from a single parent array row.
    Nodes are in preorder, with ``parent_row[0] == -1`` and for i>0, 0 <= parent[i] < i.
    Children are listed in ascending node-index order, which matches the planar
    order for MKW and serves as a fixed canonical order for BCK.
    """
    n: int = len(parent_row)
    children: list[list[int]] = [[] for _ in range(n)]
    for node in range(1, n):
        p = parent_row[node]
        if p >= 0:
            children[p].append(node)
    return children


def _build_tree_elementary_differentials_from_fx(
    fx: jax.Array,
    forest: MKWForest | BCKForest,
    apply_dfk: Optional[Callable[[list[jax.Array]], jax.Array]] = None,
) -> jax.Array:
    """Compute elementary differentials using only fx = f(x) and an optional higher-derivative oracle.
    - If apply_dfk is None, only trees with a single node (leaves) are supported.
      For any internal node, a ValueError is raised since higher derivatives are required.
    - If apply_dfk is provided, internal node values are computed as:
        E[i] = apply_dfk([E[child] for child in children[i]])
    Args:
        fx: JAX array with shape [..., d], equal to f(x) at the evaluation point.
        forest: Forest of parent arrays (shape [ntrees, n]).
        apply_dfk: Callable taking a list of direction vectors and returning
            the higher-order directional derivative D^k f(x)[v1, ..., vk].
    Returns:
        JAX array of shape [ntrees_filtered, ..., d].
    """
    parents = jnp.asarray(forest.parent)
    if parents.ndim != 2:
        raise ValueError("forest.parent must have shape [ntrees, n]")
    num_trees: int = int(parents.shape[0])

    results: list[jax.Array] = []
    for t in range(num_trees):
        parent_row: list[int] = list(map(int, parents[t].tolist()))
        if parent_row[0] != -1:
            raise ValueError("Invalid parent encoding: parent[0] must be -1 for the root")
        children: list[list[int]] = _build_children_from_parent(parent_row)

        memo: dict[int, jax.Array] = {}

        def eval_node(node_index: int) -> jax.Array:
            if node_index in memo:
                return memo[node_index]
            child_indices = children[node_index]
            if len(child_indices) == 0:
                value = fx
            else:
                if apply_dfk is None:
                    raise ValueError(
                        "apply_dfk is required to compute internal node differentials (higher derivatives)."
                    )
                child_vectors: list[jax.Array] = [eval_node(ci) for ci in child_indices]
                value = apply_dfk(child_vectors)
            memo[node_index] = value
            return value

        F_t = eval_node(0)
        results.append(F_t)

    if len(results) == 0:
        out_shape = (0,) + tuple(fx.shape)
        return jnp.empty(out_shape, dtype=fx.dtype)
    return jnp.stack(results, axis=0)


def form_butcher_differentials(
    f: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    forest: BCKForest,
) -> ButcherDifferentials:
    """Return elementary differentials F_tau(x) for all BCK trees tau in the forest.
    Coefficients are not applied; contract externally as needed.
    """

    fx = f(x)

    def apply_dfk(child_vecs: list[jax.Array]) -> jax.Array:
        if len(child_vecs) == 0:
            raise ValueError("apply_dfk called with zero children")
        h = f
        for v in child_vecs:

            def h_next(y: jax.Array, h=h, v=v) -> jax.Array:
                _, dg_v = jax.jvp(h, (y,), (v,))
                return dg_v

            h = h_next
        return h(x)

    elementary_differentials = _build_tree_elementary_differentials_from_fx(fx, forest, apply_dfk)
    return ButcherDifferentials(elementary_differentials)


def form_lie_butcher_differentials(
    f: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    forest: MKWForest,
    project_to_tangent: Callable[[jax.Array, jax.Array], jax.Array],
) -> LieButcherDifferentials:
    """Return elementary differentials F_tau(x) for all MKW trees tau (manifold case).
    Coefficients are not applied; arborify or contract externally as needed.
    """

    def f_tan(y: jax.Array) -> jax.Array:
        return project_to_tangent(y, f(y))

    fx_tan = project_to_tangent(x, f(x))

    def apply_dfk(child_vecs: list[jax.Array]) -> jax.Array:
        if len(child_vecs) == 0:
            raise ValueError("apply_dfk called with zero children")
        h = f_tan
        for v in child_vecs:

            def h_next(y: jax.Array, h=h, v=v) -> jax.Array:
                _, dg_v = jax.jvp(h, (y,), (v,))
                return project_to_tangent(y, dg_v)

            h = h_next
        return h(x)

    elementary_differentials = _build_tree_elementary_differentials_from_fx(
        fx_tan, forest, apply_dfk
    )
    return LieButcherDifferentials(elementary_differentials)
