"""Abstract Hopf algebra interface and shared utilities."""

from __future__ import annotations
from typing import Sequence, TypeVar, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import jax
import jax.numpy as jnp
from stochastax.hopf_algebras.forest_types import Forest, MKWForest, BCKForest


@dataclass(frozen=True)
class GraftingTable:
    shape_out: jax.Array
    shape_a: jax.Array
    shape_b: jax.Array
    colour_index_map: jax.Array
    num_shapes_a: int
    num_shapes_b: int
    num_cols: int


class HopfAlgebra(ABC):
    """Abstract Hopf algebra interface sufficient for signature/log-signature workflows."""

    ambient_dimension: int
    depth: int

    @classmethod
    @abstractmethod
    def build(cls, ambient_dim: int, depth: int) -> HopfAlgebra:
        """Construct a Hopf algebra instance with cached per-degree metadata.

        Implementations must agree on the (ambient_dim, depth) signature so that
        callers can be generic over concrete Hopf algebras.
        """
        raise NotImplementedError

    @abstractmethod
    def basis_size(self, level: int | None = None) -> int:
        """Number of basis elements.

        - If ``level`` is provided: return the number of basis elements at that level
          (degree = level + 1 for signatures).
        - If ``level`` is ``None``: return the total number of basis elements across all
          stored levels (i.e. the sum over ``level=0..depth-1``).

        Implementations must define how many coefficients live at each level.
        """
        raise NotImplementedError

    @abstractmethod
    def product(self, a_levels: list[jax.Array], b_levels: list[jax.Array]) -> list[jax.Array]:
        """Product on truncated coefficients, degree-wise (omits degree 0).

        Args:
            a_levels: list of flattened tensors per degree (omits degree 0)
            b_levels: same shape/depth as a_levels
        Returns:
            list of flattened tensors per degree representing a ⋆ b
        """
        raise NotImplementedError

    @abstractmethod
    def exp(self, x: list[jax.Array]) -> list[jax.Array]:
        """Exponential with respect to the product, truncated to x's depth (omits degree 0)."""
        raise NotImplementedError

    @abstractmethod
    def log(self, g: list[jax.Array]) -> list[jax.Array]:
        """Logarithm with respect to the product, truncated to g's depth (omits degree 0)."""
        raise NotImplementedError

    def zero(self, depth: int, dtype: jnp.dtype) -> list[jax.Array]:
        return [jnp.zeros((self.basis_size(i),), dtype=dtype) for i in range(depth)]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


HopfAlgebraT = TypeVar("HopfAlgebraT", bound=HopfAlgebra, contravariant=True)


# ============================================================================
# Shared utility functions for tree-based Hopf algebras
# ============================================================================


def _build_colourings_by_degree(ambient_dim: int, depth: int) -> tuple[np.ndarray, ...]:
    """Precompute node colourings for each degree.

    For each level k (0-indexed, degree = k+1), returns an integer array of shape
    [ambient_dim**(k+1), k+1] listing all base-ambient_dim digit expansions.
    """
    if depth <= 0:
        return tuple()
    out: list[np.ndarray] = []
    base = int(ambient_dim)
    for level in range(depth):
        n_digits = level + 1
        count = base**n_digits
        idx = np.arange(count, dtype=np.int64)
        digits = np.empty((count, n_digits), dtype=np.int32)
        # Most-significant digit first.
        for j in range(n_digits):
            power = base ** (n_digits - 1 - j)
            digits[:, j] = ((idx // power) % base).astype(np.int32)
        out.append(digits)
    return tuple(out)


def _build_children_from_parent(parent: list[int], ordered: bool) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(len(parent))]
    for i in range(1, len(parent)):
        p = parent[i]
        if p >= 0:
            children[p].append(i)
    if ordered:
        for node_children in children:
            node_children.sort()
    return children


def _traverse(children: list[list[int]], postorder: bool) -> list[int]:
    order: list[int] = []

    def dfs(node: int) -> None:
        if not postorder:
            order.append(node)
        for child in children[node]:
            dfs(child)
        if postorder:
            order.append(node)

    if children:
        dfs(0)
    return order


def _canonical_unordered_sequence_and_order(
    children: list[list[int]], node: int
) -> tuple[list[int], list[int]]:
    if not children:
        return [1], [node]
    child_entries: list[tuple[list[int], list[int]]] = []
    for child in children[node]:
        seq, order = _canonical_unordered_sequence_and_order(children, child)
        child_entries.append((seq, order))
    child_entries.sort(key=lambda item: item[0], reverse=True)
    seq_out: list[int] = [1]
    order_out: list[int] = [node]
    for seq, order in child_entries:
        seq_out.extend([level + 1 for level in seq])
        order_out.extend(order)
    return seq_out, order_out


def _parent_array_from_order(parent_old: list[int], order_old: list[int]) -> list[int]:
    new_index = {old: new for new, old in enumerate(order_old)}
    parent_new: list[int] = [-1] * len(order_old)
    for old in order_old:
        if old == 0:
            continue
        p_old = parent_old[old]
        parent_new[new_index[old]] = new_index[p_old]
    return parent_new


def _levelseq_to_parent(levels: list[int]) -> list[int]:
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
    return parent_py



def _build_shape_index_by_degree(
    forests: Sequence[Forest] | Sequence[MKWForest] | Sequence[BCKForest],
) -> tuple[dict[tuple[int, ...], int], ...]:
    indices: list[dict[tuple[int, ...], int]] = []
    for forest in forests:
        parents = np.asarray(forest.parent)
        if parents.ndim != 2:
            raise ValueError("Forest.parent must have shape [num_shapes, n_nodes].")
        local: dict[tuple[int, ...], int] = {}
        for idx, row in enumerate(parents):
            key = tuple(int(v) for v in row.tolist())
            local[key] = idx
        indices.append(local)
    return tuple(indices)


def _graft_unordered_parent(
    parent_a: list[int],
    parent_b: list[int],
    node_b: int,
) -> tuple[list[int], list[int]]:
    n_a = len(parent_a)
    n_b = len(parent_b)
    n_total = n_a + n_b
    parent_old: list[int] = [-1] * n_total
    for idx in range(n_b):
        parent_old[idx] = parent_b[idx]
    for idx in range(n_a):
        offset_idx = n_b + idx
        if idx == 0:
            parent_old[offset_idx] = node_b
        else:
            parent_old[offset_idx] = n_b + parent_a[idx]
    children = _build_children_from_parent(parent_old, ordered=False)
    seq, order_old = _canonical_unordered_sequence_and_order(children, 0)
    parent_new = _levelseq_to_parent(seq)
    return parent_new, order_old


def _graft_ordered_parent(
    parent_a: list[int],
    parent_b: list[int],
    node_b: int,
    position: int,
) -> tuple[list[int], list[int]]:
    n_a = len(parent_a)
    n_b = len(parent_b)
    n_total = n_a + n_b
    parent_old: list[int] = [-1] * n_total
    for idx in range(n_b):
        parent_old[idx] = parent_b[idx]
    for idx in range(n_a):
        offset_idx = n_b + idx
        if idx == 0:
            parent_old[offset_idx] = node_b
        else:
            parent_old[offset_idx] = n_b + parent_a[idx]
    children = _build_children_from_parent(parent_old, ordered=True)
    root_a = n_b
    children_b = children[node_b]
    if position < 0 or position > len(children_b):
        raise ValueError("Invalid insertion position for ordered graft.")
    if root_a in children_b:
        children_b = [child for child in children_b if child != root_a]
    children[node_b] = children_b[:position] + [root_a] + children_b[position:]
    order_old = _traverse(children, postorder=False)
    parent_new = _parent_array_from_order(parent_old, order_old)
    return parent_new, order_old


def _build_children_eval_tables(
    forests: Sequence[Forest] | Sequence[MKWForest] | Sequence[BCKForest],
) -> tuple[
    tuple[jax.Array, ...],
    tuple[jax.Array, ...],
    tuple[jax.Array, ...],
]:
    children_tables: list[jax.Array] = []
    child_counts_tables: list[jax.Array] = []
    eval_orders_tables: list[jax.Array] = []
    for forest in forests:
        parents = jnp.asarray(forest.parent)
        if parents.ndim != 2:
            raise ValueError("Forest.parent must have shape [num_shapes, n_nodes].")
        num_shapes = int(parents.shape[0])
        n_nodes = int(parents.shape[1]) if parents.shape[1:] else 0
        if num_shapes == 0 or n_nodes == 0:
            children_tables.append(jnp.zeros((num_shapes, n_nodes, 0), dtype=jnp.int32))
            child_counts_tables.append(jnp.zeros((num_shapes, n_nodes), dtype=jnp.int32))
            eval_orders_tables.append(jnp.zeros((num_shapes, n_nodes), dtype=jnp.int32))
            continue

        child_lists_per_shape: list[list[list[int]]] = []
        eval_orders_per_shape: list[list[int]] = []
        max_children = 0
        for shape_idx in range(num_shapes):
            parent_row = list(map(int, parents[shape_idx].tolist()))
            children = _build_children_from_parent(parent_row, ordered=False)
            child_lists_per_shape.append(children)
            local_max = max((len(c) for c in children), default=0)
            max_children = max(max_children, local_max)
            eval_orders_per_shape.append(_traverse(children, postorder=True))

        width = max_children if max_children > 0 else 0
        children_np = np.full(
            (num_shapes, n_nodes, width),
            -1,
            dtype=np.int32,
        )
        child_counts_np = np.zeros((num_shapes, n_nodes), dtype=np.int32)
        eval_order_np = np.zeros((num_shapes, n_nodes), dtype=np.int32)

        for shape_idx, children in enumerate(child_lists_per_shape):
            for node_idx, c_list in enumerate(children):
                child_counts_np[shape_idx, node_idx] = len(c_list)
                for slot_idx, child in enumerate(c_list):
                    if width == 0:
                        break
                    children_np[shape_idx, node_idx, slot_idx] = child
            eval_order_np[shape_idx, :] = np.asarray(
                eval_orders_per_shape[shape_idx], dtype=np.int32
            )

        children_tables.append(jnp.asarray(children_np))
        child_counts_tables.append(jnp.asarray(child_counts_np))
        eval_orders_tables.append(jnp.asarray(eval_order_np))

    return (
        tuple(children_tables),
        tuple(child_counts_tables),
        tuple(eval_orders_tables),
    )


def _apply_grafting_table(
    a_level: jax.Array,
    b_level: jax.Array,
    num_shapes_out: int,
    table: GraftingTable,
) -> jax.Array:
    colour_map = table.colour_index_map
    if colour_map.size == 0:
        return jnp.zeros((num_shapes_out * table.num_cols,), dtype=a_level.dtype)
    a_coeffs = jnp.reshape(a_level, (table.num_shapes_a, -1))
    b_coeffs = jnp.reshape(b_level, (table.num_shapes_b, -1))
    a_pick = a_coeffs[table.shape_a]
    b_pick = b_coeffs[table.shape_b]
    outer = a_pick[:, :, None] * b_pick[:, None, :]
    out_level = jnp.zeros((num_shapes_out, table.num_cols), dtype=outer.dtype)
    out_level = out_level.at[table.shape_out[:, None, None], colour_map].add(outer)
    return out_level.reshape(-1)


def _build_grafting_tables(
    forests_by_degree: Sequence[Forest] | Sequence[MKWForest] | Sequence[BCKForest],
    parent_index_by_degree: tuple[dict[tuple[int, ...], int], ...],
    colourings_by_degree: tuple[np.ndarray, ...],
    ambient_dim: int,
    ordered: bool,
) -> tuple[tuple[GraftingTable | None, ...], ...]:
    base = int(ambient_dim)
    depth = len(forests_by_degree)
    out_tables: list[list[GraftingTable | None]] = []
    for degree_a in range(1, depth + 1):
        row: list[GraftingTable | None] = []
        parents_a = np.asarray(forests_by_degree[degree_a - 1].parent)
        num_shapes_a = int(parents_a.shape[0]) if parents_a.ndim == 2 else 0
        colourings_a = np.asarray(colourings_by_degree[degree_a - 1])
        for degree_b in range(1, depth + 1):
            if degree_a + degree_b > depth:
                row.append(None)
                continue
            parents_b = np.asarray(forests_by_degree[degree_b - 1].parent)
            num_shapes_b = int(parents_b.shape[0]) if parents_b.ndim == 2 else 0
            colourings_b = np.asarray(colourings_by_degree[degree_b - 1])
            out_index = parent_index_by_degree[degree_a + degree_b - 1]
            entries_shape_out: list[int] = []
            entries_shape_a: list[int] = []
            entries_shape_b: list[int] = []
            entries_colour_map: list[np.ndarray] = []
            if num_shapes_a == 0 or num_shapes_b == 0:
                row.append(
                    GraftingTable(
                        shape_out=jnp.zeros((0,), dtype=jnp.int32),
                        shape_a=jnp.zeros((0,), dtype=jnp.int32),
                        shape_b=jnp.zeros((0,), dtype=jnp.int32),
                        colour_index_map=jnp.zeros(
                            (0, colourings_a.shape[0], colourings_b.shape[0]),
                            dtype=jnp.int32,
                        ),
                        num_shapes_a=num_shapes_a,
                        num_shapes_b=num_shapes_b,
                        num_cols=int(colourings_a.shape[0] * colourings_b.shape[0]),
                    )
                )
                continue
            for shape_a in range(num_shapes_a):
                parent_a = list(map(int, parents_a[shape_a].tolist()))
                for shape_b in range(num_shapes_b):
                    parent_b = list(map(int, parents_b[shape_b].tolist()))
                    if ordered:
                        children_b = _build_children_from_parent(parent_b, ordered=True)
                    else:
                        children_b = _build_children_from_parent(parent_b, ordered=False)
                    for node_b in range(degree_b):
                        if ordered:
                            positions = range(len(children_b[node_b]) + 1)
                        else:
                            positions = range(1)
                        for pos in positions:
                            if ordered:
                                parent_new, order_old = _graft_ordered_parent(
                                    parent_a, parent_b, node_b, pos
                                )
                            else:
                                parent_new, order_old = _graft_unordered_parent(
                                    parent_a, parent_b, node_b
                                )
                            shape_out = out_index.get(tuple(parent_new))
                            if shape_out is None:
                                raise ValueError("Grafting produced an unknown tree shape.")
                            mapping: list[tuple[str, int]] = []
                            for old in order_old:
                                if old < degree_b:
                                    mapping.append(("b", old))
                                else:
                                    mapping.append(("a", old - degree_b))
                            n_total = len(mapping)
                            colour_index_map = np.zeros(
                                (colourings_a.shape[0], colourings_b.shape[0]), dtype=np.int32
                            )
                            for pos, (source, node_idx) in enumerate(mapping):
                                weight = base ** (n_total - 1 - pos)
                                if source == "a":
                                    digits = colourings_a[:, node_idx][:, None]
                                else:
                                    digits = colourings_b[:, node_idx][None, :]
                                colour_index_map += (digits * weight).astype(np.int32)
                            entries_shape_out.append(shape_out)
                            entries_shape_a.append(shape_a)
                            entries_shape_b.append(shape_b)
                            entries_colour_map.append(colour_index_map)
            if entries_colour_map:
                colour_map_arr = jnp.asarray(np.stack(entries_colour_map, axis=0), dtype=jnp.int32)
            else:
                colour_map_arr = jnp.zeros(
                    (0, colourings_a.shape[0], colourings_b.shape[0]), dtype=jnp.int32
                )
            table = GraftingTable(
                shape_out=jnp.asarray(entries_shape_out, dtype=jnp.int32),
                shape_a=jnp.asarray(entries_shape_a, dtype=jnp.int32),
                shape_b=jnp.asarray(entries_shape_b, dtype=jnp.int32),
                colour_index_map=colour_map_arr,
                num_shapes_a=num_shapes_a,
                num_shapes_b=num_shapes_b,
                num_cols=int(colourings_a.shape[0] * colourings_b.shape[0]),
            )
            row.append(table)
        out_tables.append(row)
    return tuple(tuple(row) for row in out_tables)
