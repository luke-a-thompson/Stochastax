import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional

from stochastax.hopf_algebras.hopf_algebras import GLHopfAlgebra
from stochastax.vector_field_lifts.vector_field_lift_types import (
    BCKBrackets,
    BCKBracketFunctions,
)
from stochastax.manifolds.manifolds import Manifold, EuclideanSpace


def form_bck_lift(
    vector_fields: list[Callable[[jax.Array], jax.Array]],
    base_point: jax.Array,
    hopf: GLHopfAlgebra,
    manifold: Manifold = EuclideanSpace(),
) -> BCKBrackets:
    """
    Build BCK brackets (unordered rooted forests) evaluated at a base point.

    Args:
        vector_fields: list of driver vector fields vector_fields[i]: R^n -> R^n. One
            vector field per driver dimension.
        base_point: base point where the BCK elementary differentials' Jacobians are
            evaluated.
        hopf: Hopf algebra containing the unordered forests metadata via
            ``GLHopfAlgebra.build``.

    Returns:
        List storing [num_shapes_k * d^(k+1), n, n] Jacobians per forest degree
        (degree k+1), where num_shapes_k is the number of distinct unordered forest
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
            "GLHopfAlgebra instance does not contain any forests. Ensure it "
            "was constructed via GLHopfAlgebra.build."
        )
    if (
        not hopf.children_by_degree
        or not hopf.child_counts_by_degree
        or not hopf.eval_order_by_degree
    ):
        raise ValueError(
            "GLHopfAlgebra is missing cached children metadata. Rebuild via GLHopfAlgebra.build()."
        )

    if not isinstance(manifold, EuclideanSpace):
        raise ValueError("form_bck_lift currently supports only EuclideanSpace.")

    n_state = int(base_point.shape[0])

    bracket_functions = form_bck_bracket_functions(
        vector_fields=vector_fields,
        hopf=hopf,
        manifold=manifold,
    )

    results_by_degree: list[jax.Array] = []
    for degree_idx, funcs_level in enumerate(bracket_functions):
        expected_count = hopf.basis_size(degree_idx)
        if len(funcs_level) != expected_count:
            raise ValueError(
                f"Constructed {len(funcs_level)} BCK bracket functions at level {degree_idx}, "
                f"but hopf.basis_size reports {expected_count}."
            )
        if expected_count == 0:
            results_by_degree.append(jnp.zeros((0, n_state, n_state), dtype=base_point.dtype))
            continue

        level_funcs = tuple(funcs_level)

        def eval_level(
            y: jax.Array,
            funcs: tuple[Callable[[jax.Array], jax.Array], ...] = level_funcs,
        ) -> jax.Array:
            vals = [fn(y) for fn in funcs]
            return jnp.stack(vals, axis=0)

        level_jac = jax.jacfwd(eval_level)(base_point)
        results_by_degree.append(level_jac)

    return BCKBrackets(results_by_degree)


def form_bck_bracket_functions(
    vector_fields: list[Callable[[jax.Array], jax.Array]],
    hopf: GLHopfAlgebra,
    manifold: Manifold = EuclideanSpace(),
) -> BCKBracketFunctions:
    """
    Build callable BCK bracket vector fields V_w(y) (unordered rooted forests).

    Args:
        vector_fields: list of driver vector fields vector_fields[i]: R^n -> R^n. One
            vector field per driver dimension.
        hopf: Hopf algebra containing the unordered forests metadata via
            ``GLHopfAlgebra.build``.

    Returns:
        Per-degree list of callable bracket functions.
    """
    if not isinstance(manifold, EuclideanSpace):
        raise ValueError("form_bck_bracket_functions currently supports only EuclideanSpace.")

    forests_by_degree = hopf.forests_by_degree
    if len(vector_fields) != hopf.ambient_dimension:
        raise ValueError(
            "Number of vector fields must equal hopf.ambient_dimension "
            f"({len(vector_fields)} != {hopf.ambient_dimension})."
        )
    if not forests_by_degree:
        raise ValueError(
            "GLHopfAlgebra instance does not contain any forests. Ensure it "
            "was constructed via GLHopfAlgebra.build."
        )
    if (
        not hopf.children_by_degree
        or not hopf.child_counts_by_degree
        or not hopf.eval_order_by_degree
    ):
        raise ValueError(
            "GLHopfAlgebra is missing cached children metadata. Rebuild via GLHopfAlgebra.build()."
        )

    d = hopf.ambient_dimension

    bracket_fns_by_degree: list[list[Callable[[jax.Array], jax.Array]]] = []

    for degree_idx, forest in enumerate(forests_by_degree):
        parents = np.asarray(forest.parent)
        if parents.ndim != 2:
            raise ValueError("Each BCKForest.parent must have shape [num_shapes, n_nodes]")
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
            bracket_fns_by_degree.append([])
            continue

        children_table = hopf.children_by_degree[degree_idx]
        child_counts_table = hopf.child_counts_by_degree[degree_idx]
        eval_order_table = hopf.eval_order_by_degree[degree_idx]
        children_np = np.asarray(children_table)
        child_counts_np = np.asarray(child_counts_table)
        eval_order_np = np.asarray(eval_order_table)

        colourings = hopf.colourings_by_degree[degree_idx]
        num_colours = int(colourings.shape[0])
        level_fns: list[Callable[[jax.Array], jax.Array]] = []

        for shape_id in range(num_shapes):
            parent_row = list(map(int, parents[shape_id].tolist()))
            if parent_row[0] != -1:
                raise ValueError("Invalid parent encoding: parent[0] must be -1 for the root")
            child_counts = child_counts_np[shape_id]
            children_indices = children_np[shape_id]
            eval_order = eval_order_np[shape_id]

            def build_node_function(
                colours: np.ndarray,
            ) -> Callable[[jax.Array], jax.Array]:
                node_funcs: list[Optional[Callable[[jax.Array], jax.Array]]] = [None] * n_nodes
                for node_idx in eval_order:
                    node_idx = int(node_idx)
                    colour = int(colours[node_idx])
                    num_children = int(child_counts[node_idx])
                    if num_children == 0:
                        node_funcs[node_idx] = vector_fields[colour]
                        continue
                    child_ids = [
                        int(children_indices[node_idx, slot]) for slot in range(num_children)
                    ]
                    child_funcs: list[Callable[[jax.Array], jax.Array]] = []
                    for c_idx in child_ids:
                        child_fn = node_funcs[c_idx]
                        if child_fn is None:
                            raise ValueError("Encountered unset child function in BCK lift.")
                        child_funcs.append(child_fn)

                    def make_node(
                        colour_idx: int,
                        funcs: list[Callable[[jax.Array], jax.Array]],
                    ) -> Callable[[jax.Array], jax.Array]:
                        def node_fn(y: jax.Array) -> jax.Array:
                            g = vector_fields[colour_idx]
                            for cf in funcs:

                                def g_next(z: jax.Array, g=g, cf=cf) -> jax.Array:
                                    _, dg_v = jax.jvp(g, (z,), (cf(z),))
                                    return manifold.project_to_tangent(z, dg_v)

                                g = g_next
                            return g(y)

                        return node_fn

                    node_funcs[node_idx] = make_node(colour, child_funcs)

                root_fn = node_funcs[0]
                assert root_fn is not None
                return root_fn

            for colour_index in range(num_colours):
                colours = colourings[colour_index]
                F_root_fn = build_node_function(colours)
                level_fns.append(F_root_fn)

        if len(level_fns) != expected_count:
            raise ValueError(
                f"Constructed {len(level_fns)} BCK bracket functions at level {degree_idx}, "
                f"but hopf.basis_size reports {expected_count}."
            )

        bracket_fns_by_degree.append(level_fns)

    return BCKBracketFunctions(bracket_fns_by_degree)
