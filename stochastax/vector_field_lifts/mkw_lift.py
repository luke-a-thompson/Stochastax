import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional

from stochastax.hopf_algebras.hopf_algebra_types import MKWHopfAlgebra
from stochastax.vector_field_lifts.vector_field_lift_types import MKWBrackets
from stochastax.vector_field_lifts.combinatorics import unrank_base_d


def form_mkw_brackets(
    vector_fields: list[Callable[[jax.Array], jax.Array]],
    base_point: jax.Array,
    hopf: MKWHopfAlgebra,
    project_to_tangent: Callable[[jax.Array, jax.Array], jax.Array],
) -> MKWBrackets:
    """
    Build nonlinear MKW brackets (planar rooted forests) evaluated at a base point.

    Args:
        vector_fields: list of driver vector fields vector_fields[i]: R^n -> R^n. One
            vector field per driver dimension.
        base_point: base point where the MKW elementary differentials' Jacobians are
            evaluated.
        hopf: Hopf algebra containing the planar forests metadata via
            ``MKWHopfAlgebra.build``.
        project_to_tangent: projection map enforcing tangent dynamics on a manifold;
            typically identity in the Euclidean case.

    Returns:
        List storing [num_shapes_k * d^(k+1), n, n] Jacobians per forest degree
        (degree k+1), where num_shapes_k is the number of distinct planar forest
        shapes with k+1 nodes and d is the number of driver vector fields.
    """
    if base_point.ndim != 1:
        raise ValueError(f"base_point must be a 1D array [n], got shape {base_point.shape}")
    forests_by_degree = hopf.forests_by_degree
    if len(vector_fields) != hopf.ambient_dimension:
        raise ValueError(
            "Number of vector fields must equal hopf.ambient_dimension "
            f"({len(vector_fields)} != {hopf.ambient_dimension})."
        )
    if not forests_by_degree:
        raise ValueError(
            "MKWHopfAlgebra instance does not contain any forests. Ensure it "
            "was constructed via MKWHopfAlgebra.build."
        )
    if (
        not hopf.children_by_degree
        or not hopf.child_counts_by_degree
        or not hopf.eval_order_by_degree
    ):
        raise ValueError(
            "MKWHopfAlgebra is missing cached children metadata. Rebuild via MKWHopfAlgebra.build()."
        )

    d = hopf.ambient_dimension
    n_state = int(base_point.shape[0])

    results_by_degree: list[jax.Array] = []

    for degree_idx, forest in enumerate(forests_by_degree):
        parents = np.asarray(forest.parent)
        if parents.ndim != 2:
            raise ValueError("Each MKWForest.parent must have shape [num_shapes, n_nodes]")
        num_shapes = int(parents.shape[0])
        n_nodes = int(parents.shape[1])
        if n_nodes != degree_idx + 1:
            raise ValueError(
                f"Inconsistent forest at index {degree_idx}: expected {degree_idx + 1} nodes, got {n_nodes}"
            )

        expected_count = hopf.basis_size(degree_idx)
        if num_shapes == 0:
            if expected_count != 0:
                raise ValueError(
                    f"Hopf expects {expected_count} basis elements at level {degree_idx}, "
                    "but no forest shapes were provided."
                )
            results_by_degree.append(jnp.zeros((0, n_state, n_state), dtype=base_point.dtype))
            continue

        children_table = hopf.children_by_degree[degree_idx]
        child_counts_table = hopf.child_counts_by_degree[degree_idx]
        eval_order_table = hopf.eval_order_by_degree[degree_idx]
        children_np = np.asarray(children_table)
        child_counts_np = np.asarray(child_counts_table)
        eval_order_np = np.asarray(eval_order_table)

        num_colours = d**n_nodes
        level_mats: list[jax.Array] = []

        for shape_id in range(num_shapes):
            parent_row = list(map(int, parents[shape_id].tolist()))
            if parent_row[0] != -1:
                raise ValueError("Invalid parent encoding: parent[0] must be -1 for the root")
            child_counts = child_counts_np[shape_id]
            children_indices = children_np[shape_id]
            eval_order = eval_order_np[shape_id]

            def build_node_function(
                colours: list[int],
            ) -> Callable[[jax.Array], jax.Array]:
                node_funcs: list[Optional[Callable[[jax.Array], jax.Array]]] = [None] * n_nodes
                for node_idx in eval_order:
                    node_idx = int(node_idx)
                    colour = colours[node_idx]
                    num_children = int(child_counts[node_idx])
                    if num_children == 0:

                        def leaf_field(y: jax.Array, colour_idx: int = colour) -> jax.Array:
                            return project_to_tangent(y, vector_fields[colour_idx](y))

                        node_funcs[node_idx] = leaf_field
                        continue
                    child_ids = [
                        int(children_indices[node_idx, slot])
                        for slot in range(num_children)
                    ]
                    child_funcs: list[Callable[[jax.Array], jax.Array]] = []
                    for c_idx in child_ids:
                        child_fn = node_funcs[c_idx]
                        if child_fn is None:
                            raise ValueError("Encountered unset child function in MKW lift.")
                        child_funcs.append(child_fn)

                    def make_node(
                        colour_idx: int,
                        funcs: list[Callable[[jax.Array], jax.Array]],
                    ) -> Callable[[jax.Array], jax.Array]:
                        def node_fn(y: jax.Array) -> jax.Array:
                            def g(z: jax.Array) -> jax.Array:
                                return project_to_tangent(z, vector_fields[colour_idx](z))

                            current = g
                            for cf in funcs:

                                def g_next(z: jax.Array, g=current, cf=cf) -> jax.Array:
                                    _, dg_v = jax.jvp(g, (z,), (cf(z),))
                                    return project_to_tangent(z, dg_v)

                                current = g_next
                            return current(y)

                        return node_fn

                    node_funcs[node_idx] = make_node(colour, child_funcs)

                root_fn = node_funcs[0]
                assert root_fn is not None
                return root_fn

            for colour_index in range(num_colours):
                colours = unrank_base_d(colour_index, n_nodes, d)
                F_root_fn = build_node_function(colours)
                J = jax.jacrev(F_root_fn)(base_point)
                level_mats.append(J)

        if len(level_mats) != expected_count:
            raise ValueError(
                f"Constructed {len(level_mats)} MKW brackets at level {degree_idx}, "
                f"but hopf.basis_size reports {expected_count}."
            )

        out = (
            jnp.zeros((0, n_state, n_state), dtype=base_point.dtype)
            if len(level_mats) == 0
            else jnp.stack(level_mats, axis=0)
        )
        results_by_degree.append(out)

    return MKWBrackets(results_by_degree)
