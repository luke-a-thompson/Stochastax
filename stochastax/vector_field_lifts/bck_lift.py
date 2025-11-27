import jax
import jax.numpy as jnp
from typing import Callable

from stochastax.hopf_algebras.hopf_algebra_types import BCKForest
from stochastax.vector_field_lifts.vector_field_lift_types import BCKBrackets
from stochastax.vector_field_lifts.butcher import _build_children_from_parent
from stochastax.vector_field_lifts.combinatorics import unrank_base_d


def form_bck_brackets(
    vector_fields: list[Callable[[jax.Array], jax.Array]],
    base_point: jax.Array,
    forests_by_degree: list[BCKForest],
) -> BCKBrackets:
    """
    Build BCK brackets (unordered rooted forests) evaluated at a base point.

    Args:
        vector_fields: list of driver vector fields vector_fields[i]: R^n -> R^n. One
            vector field per driver dimension.
        base_point: base point where the BCK elementary differentials' Jacobians are
            evaluated.
        forests_by_degree: list where entry k encodes all BCK (unordered) rooted
            forests of degree k+1, as BCKForest objects.

    Returns:
        List storing [num_shapes_k * d^(k+1), n, n] Jacobians per forest degree
        (degree k+1), where num_shapes_k is the number of distinct unordered forest
        shapes with k+1 nodes and d is the number of driver vector fields.
    """
    if base_point.ndim != 1:
        raise ValueError(f"base_point must be a 1D array [n], got shape {base_point.shape}")
    d = len(vector_fields)
    n_state = int(base_point.shape[0])

    results_by_degree: list[jax.Array] = []

    for degree_idx, forest in enumerate(forests_by_degree):
        parents = jnp.asarray(forest.parent)
        if parents.ndim != 2:
            raise ValueError("Each BCKForest.parent must have shape [num_shapes, n_nodes]")
        num_shapes = int(parents.shape[0])
        n_nodes = int(parents.shape[1])
        if n_nodes != degree_idx + 1:
            raise ValueError(
                f"Inconsistent forest at index {degree_idx}: expected {degree_idx + 1} nodes, got {n_nodes}"
            )

        if num_shapes == 0:
            results_by_degree.append(jnp.zeros((0, n_state, n_state), dtype=base_point.dtype))
            continue

        num_colours = d**n_nodes
        level_mats: list[jax.Array] = []

        for shape_id in range(num_shapes):
            parent_row = list(map(int, parents[shape_id].tolist()))
            if parent_row[0] != -1:
                raise ValueError("Invalid parent encoding: parent[0] must be -1 for the root")
            children = _build_children_from_parent(parent_row)

            def build_node_function(
                node_index: int, colours: list[int]
            ) -> Callable[[jax.Array], jax.Array]:
                child_indices = children[node_index]
                colour = colours[node_index]
                if len(child_indices) == 0:
                    return vector_fields[colour]
                child_funcs = [build_node_function(ci, colours) for ci in child_indices]

                def h(y: jax.Array) -> jax.Array:
                    g = vector_fields[colour]
                    for cf in child_funcs:

                        def g_next(z: jax.Array, g=g, cf=cf) -> jax.Array:
                            _, dg_v = jax.jvp(g, (z,), (cf(z),))
                            return dg_v

                        g = g_next
                    return g(y)

                return h

            for colour_index in range(num_colours):
                colours = unrank_base_d(colour_index, n_nodes, d)
                F_root_fn = build_node_function(0, colours)
                J = jax.jacrev(F_root_fn)(base_point)
                level_mats.append(J)

        if len(level_mats) == 0:
            out = jnp.zeros((0, n_state, n_state), dtype=base_point.dtype)
        else:
            out = jnp.stack(level_mats, axis=0)
        results_by_degree.append(out)

    return BCKBrackets(results_by_degree)
