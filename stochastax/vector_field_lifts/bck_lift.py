import jax
import jax.numpy as jnp
from typing import Callable

from stochastax.hopf_algebras.hopf_algebra_types import BCKForest
from stochastax.vector_field_lifts.vector_field_lift_types import BCKBrackets
from stochastax.vector_field_lifts.butcher import _build_children_from_parent
from stochastax.vector_field_lifts.combinatorics import unrank_base_d


def form_bck_brackets(
    V: list[Callable[[jax.Array], jax.Array]],
    x: jax.Array,
    forests_by_degree: list[BCKForest],
) -> BCKBrackets:
    """
    Compute bracket matrices (Jacobian of elementary differentials) for BCK trees, per degree.

    Returns a list per degree k (degree = k+1) with shape [num_shapes_k * d^(k+1), n, n].
    """
    if x.ndim != 1:
        raise ValueError(f"x must be a 1D array [n], got shape {x.shape}")
    d = len(V)
    n_state = int(x.shape[0])

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
            results_by_degree.append(jnp.zeros((0, n_state, n_state), dtype=x.dtype))
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
                    return V[colour]
                child_funcs = [build_node_function(ci, colours) for ci in child_indices]

                def h(y: jax.Array) -> jax.Array:
                    g = V[colour]
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
                J = jax.jacrev(F_root_fn)(x)
                level_mats.append(J)

        if len(level_mats) == 0:
            out = jnp.zeros((0, n_state, n_state), dtype=x.dtype)
        else:
            out = jnp.stack(level_mats, axis=0)
        results_by_degree.append(out)

    return BCKBrackets(results_by_degree)
