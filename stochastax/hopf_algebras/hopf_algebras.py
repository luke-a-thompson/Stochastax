from __future__ import annotations
from typing import final, override, Sequence, TypeVar, Literal
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import jax
import jax.numpy as jnp
from stochastax.tensor_ops import tensor_product
from stochastax.hopf_algebras.free_lie import enumerate_lyndon_basis, build_lyndon_dependency_tables
from stochastax.hopf_algebras.bck_trees import enumerate_bck_trees
from stochastax.hopf_algebras.mkw_trees import enumerate_mkw_trees
from stochastax.hopf_algebras.forest_types import Forest, MKWForest, BCKForest


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
    def coproduct(self, levels: list[jax.Array]) -> list[list[jax.Array]]:
        """Coproduct (deconcatenation) listing splits per degree.

        For degree n (index n-1), return a flat list encoding the pairs
        (deg k, deg n-k) for all splits k=1..n-1. Degree-0 parts are omitted.
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

    @override
    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


@dataclass(frozen=True, eq=False)
class ShuffleHopfAlgebra(HopfAlgebra):
    """Shuffle/Tensor Hopf algebra used for path signatures.

    The representation uses per-degree flattened tensors, omitting degree 0.
    Instances built via ``build`` cache per-degree metadata so downstream
    consumers can reuse the same combinatorics instead of recomputing them.

    In particular, we cache the Lyndon word basis up to the chosen depth and
    lookup tables for each Lyndon word's **standard (Shirshov) factorization**
    :math:`w=uv` (longest proper Lyndon suffix split). These tables allow
    vector-field lifts and bracket formation to evaluate the recursion
    :math:`[w]=[[u],[v]]` via constant-time index lookups.
    """

    ambient_dimension: int
    depth: int = 0  # truncation depth (number of stored positive-degree levels)
    shape_count_by_degree: list[int] = field(default_factory=list)
    lyndon_basis_by_degree: tuple[jax.Array, ...] = field(default_factory=tuple)
    lyndon_split_by_degree: tuple[jax.Array, ...] = field(default_factory=tuple)
    lyndon_prefix_level_by_degree: tuple[jax.Array, ...] = field(default_factory=tuple)
    lyndon_prefix_index_by_degree: tuple[jax.Array, ...] = field(default_factory=tuple)
    lyndon_suffix_level_by_degree: tuple[jax.Array, ...] = field(default_factory=tuple)
    lyndon_suffix_index_by_degree: tuple[jax.Array, ...] = field(default_factory=tuple)

    @classmethod
    @override
    def build(
        cls,
        ambient_dim: int,
        depth: int,
    ) -> ShuffleHopfAlgebra:
        """Construct a shuffle Hopf algebra with cached per-degree metadata."""
        if depth <= 0:
            raise ValueError(f"max_degree must be >= 1, got {depth}.")
        shape_counts = [int(ambient_dim ** (level + 1)) for level in range(depth)]
        lyndon_tuple: tuple[jax.Array, ...] = tuple()
        split_tuple: tuple[jax.Array, ...] = tuple()
        prefix_level_tuple: tuple[jax.Array, ...] = tuple()
        prefix_index_tuple: tuple[jax.Array, ...] = tuple()
        suffix_level_tuple: tuple[jax.Array, ...] = tuple()
        suffix_index_tuple: tuple[jax.Array, ...] = tuple()
        lyndon_levels = enumerate_lyndon_basis(depth, ambient_dim)
        (
            split_tuple,
            prefix_level_tuple,
            prefix_index_tuple,
            suffix_level_tuple,
            suffix_index_tuple,
        ) = build_lyndon_dependency_tables(lyndon_levels)
        lyndon_tuple = tuple(lyndon_levels)
        return cls(
            ambient_dimension=ambient_dim,
            depth=depth,
            shape_count_by_degree=shape_counts,
            lyndon_basis_by_degree=lyndon_tuple,
            lyndon_split_by_degree=split_tuple,
            lyndon_prefix_level_by_degree=prefix_level_tuple,
            lyndon_prefix_index_by_degree=prefix_index_tuple,
            lyndon_suffix_level_by_degree=suffix_level_tuple,
            lyndon_suffix_index_by_degree=suffix_index_tuple,
        )

    @override
    def basis_size(
        self, level: int | None = None, basis: Literal["lyndon", "tensor"] = "lyndon"
    ) -> int:
        if not self.shape_count_by_degree:
            raise ValueError(
                "ShuffleHopfAlgebra must be constructed via ShuffleHopfAlgebra.build to "
                "populate per-degree shape counts."
            )
        if level is None:
            if basis == "lyndon":
                return sum(len(self.lyndon_basis_by_degree[i]) for i in range(self.depth))
            return sum(self.shape_count_by_degree)
        if level < 0 or level >= self.depth:
            raise ValueError(
                f"Requested level {level} outside available range [0, {self.depth - 1}]."
            )
        if basis == "lyndon":
            return len(self.lyndon_basis_by_degree[level])
        return self.shape_count_by_degree[level]

    @override
    def zero(self, depth: int, dtype: jnp.dtype) -> list[jax.Array]:
        """Return zero element in tensor basis (internal representation)."""
        return [jnp.zeros((self.basis_size(i, basis="tensor"),), dtype=dtype) for i in range(depth)]

    def _unflatten_levels(self, levels: list[jax.Array]) -> list[jax.Array]:
        dim = self.ambient_dimension
        return [term.reshape((dim,) * (i + 1)) for i, term in enumerate(levels)]

    def _flatten_levels(self, levels: list[jax.Array]) -> list[jax.Array]:
        return [term.reshape(-1) for term in levels]

    def _cauchy_convolution(
        self, a_levels: list[jax.Array], b_levels: list[jax.Array]
    ) -> list[jax.Array]:
        r"""
        Computes the degree-m component of the graded tensor-concatenation product
        or Cauchy convolution product in the truncated free tensor algebra.
        $$
        Z^{(m)} \;=\;\sum_{p+q=m} X^{(p)} \otimes Y^{(q)}
        $$
        This is the degree-m truncation of the full (Cauchy) product
        $$
        X \cdot Y = \sum_{p,q\ge0} X^{(p)} \otimes Y^{(q)}.
        $$
        """

        if len(a_levels) == 0 or len(b_levels) == 0:
            return []

        if a_levels[0].shape != b_levels[0].shape:
            raise ValueError(
                f"a_levels and b_levels must have the same shape, got "
                f"{a_levels[0].shape} and {b_levels[0].shape}"
            )

        depth = len(a_levels)
        base = a_levels[0]
        n_features = base.shape[-1]
        out = [jnp.zeros((n_features,) * (k + 1)) for k in range(depth)]
        # order-1 term is zero as there is no way to split $$1 = (p+1)+(q+1)$$ with $$p,q ≥ 0$$
        for i in range(1, depth):  # i is the index for out, e.g., out[i] is order i+1
            # we want $$Z^{(i+1)} = \sum_{(j+1)+(k+1)=i+1} X^{(j+1)}⊗Y^{(k+1)}$$
            # i.e. we want to sum over all ways to split $$i+1 = (j+1)+(k+1)$$ with $$j,k ≥ 0$$
            acc = jnp.zeros_like(out[i])
            for j in range(i):
                if j < len(a_levels) and (i - 1 - j) < len(b_levels):  # Ensure terms exist
                    k = i - 1 - j
                    term = tensor_product(a_levels[j], b_levels[k])
                    acc = acc + term  # $$X^{(j+1)}⊗Y^{(k+1)}$$
            out[i] = acc
        return out

    @override
    def product(self, a_levels: list[jax.Array], b_levels: list[jax.Array]) -> list[jax.Array]:
        if len(a_levels) != len(b_levels):
            raise ValueError("Truncations must match for product.")
        # Work in unflattened tensor shapes for the convolution; then re-flatten
        a_unflat = self._unflatten_levels(a_levels)
        b_unflat = self._unflatten_levels(b_levels)
        cross_unflat = self._cauchy_convolution(a_unflat, b_unflat)
        out_unflat = [a + b + c for a, b, c in zip(a_unflat, b_unflat, cross_unflat)]
        return self._flatten_levels(out_unflat)

    def _pure_product(
        self, a_levels: list[jax.Array], b_levels: list[jax.Array]
    ) -> list[jax.Array]:
        """(a ⋆ b) with linear parts removed; used by exp/log series."""
        ab = self.product(a_levels, b_levels)
        return [x - y - z for x, y, z in zip(ab, a_levels, b_levels)]

    @override
    def coproduct(self, levels: list[jax.Array]) -> list[list[jax.Array]]:
        """
        The deconcatenation coproduct splits each level into pairs of levels.
        """
        depth = len(levels)
        result: list[list[jax.Array]] = []
        for n in range(1, depth + 1):
            splits: list[jax.Array] = []
            for k in range(1, n):
                splits.append(levels[k - 1])
                splits.append(levels[n - k - 1])
            result.append(splits)
        return result

    @override
    def exp(self, x: list[jax.Array]) -> list[jax.Array]:
        if len(x) == 0:
            return []
        depth = len(x)
        acc = self.zero(depth, dtype=x[0].dtype)
        factorial = 1.0
        current_power = x  # k = 1
        acc = [a + (1.0 / factorial) * cp for a, cp in zip(acc, current_power)]
        for k in range(2, depth + 1):
            factorial *= float(k)
            current_power = self._pure_product(current_power, x)
            acc = [a + (1.0 / factorial) * cp for a, cp in zip(acc, current_power)]
        return acc

    @override
    def log(self, g: list[jax.Array]) -> list[jax.Array]:
        if len(g) == 0:
            return []
        dtype = g[0].dtype
        depth = len(g)
        acc = self.zero(depth, dtype)
        current_power = g  # k = 1
        coeff = 1.0
        acc = [a + coeff * cp for a, cp in zip(acc, current_power)]
        for k in range(2, depth + 1):
            current_power = self._pure_product(current_power, g)
            coeff = ((-1.0) ** (k + 1)) / float(k)
            acc = [a + coeff * cp for a, cp in zip(acc, current_power)]
        return acc

    @override
    def __str__(self) -> str:
        return "Shuffle Hopf Algebra"


def _build_children_from_parent(parent: list[int]) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(len(parent))]
    for i in range(1, len(parent)):
        p = parent[i]
        if p >= 0:
            children[p].append(i)
    return children


def _parents_to_children_ordered(parent: list[int]) -> list[list[int]]:
    children = _build_children_from_parent(parent)
    for node_children in children:
        node_children.sort()
    return children


def _preorder_from_children(children: list[list[int]]) -> list[int]:
    order: list[int] = []

    def dfs(node: int) -> None:
        order.append(node)
        for child in children[node]:
            dfs(child)

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


def _colouring_index_from_digits(digits: list[int], base: int) -> int:
    idx = 0
    for digit in digits:
        idx = idx * base + int(digit)
    return idx


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
    children = _build_children_from_parent(parent_old)
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
    children = _parents_to_children_ordered(parent_old)
    root_a = n_b
    children_b = children[node_b]
    if position < 0 or position > len(children_b):
        raise ValueError("Invalid insertion position for ordered graft.")
    if root_a in children_b:
        children_b = [child for child in children_b if child != root_a]
    children[node_b] = children_b[:position] + [root_a] + children_b[position:]
    order_old = _preorder_from_children(children)
    parent_new = _parent_array_from_order(parent_old, order_old)
    return parent_new, order_old


def _grafting_product_level(
    a_level: jax.Array,
    b_level: jax.Array,
    degree_a: int,
    degree_b: int,
    ambient_dim: int,
    forests_by_degree: Sequence[Forest] | Sequence[MKWForest] | Sequence[BCKForest],
    parent_index_by_degree: tuple[dict[tuple[int, ...], int], ...],
    colourings_by_degree: tuple[np.ndarray, ...],
    ordered: bool,
) -> jax.Array:
    n_a = degree_a
    n_b = degree_b
    n_total = n_a + n_b
    if n_a <= 0 or n_b <= 0:
        raise ValueError("Degrees must be >= 1 for grafting.")
    if n_total - 1 >= len(forests_by_degree):
        raise ValueError("Requested grafting degree exceeds available forest metadata.")

    parents_a = np.asarray(forests_by_degree[n_a - 1].parent)
    parents_b = np.asarray(forests_by_degree[n_b - 1].parent)
    num_shapes_a = int(parents_a.shape[0]) if parents_a.ndim == 2 else 0
    num_shapes_b = int(parents_b.shape[0]) if parents_b.ndim == 2 else 0
    num_shapes_out = int(forests_by_degree[n_total - 1].parent.shape[0])

    dtype = jnp.result_type(a_level, b_level)
    out = jnp.zeros((num_shapes_out, ambient_dim**n_total), dtype=dtype)
    if num_shapes_a == 0 or num_shapes_b == 0:
        return out.reshape(-1)

    a_coeffs = jnp.reshape(a_level, (num_shapes_a, -1))
    b_coeffs = jnp.reshape(b_level, (num_shapes_b, -1))
    colourings_a = np.asarray(colourings_by_degree[n_a - 1])
    colourings_b = np.asarray(colourings_by_degree[n_b - 1])
    out_index = parent_index_by_degree[n_total - 1]

    for shape_a in range(num_shapes_a):
        parent_a = list(map(int, parents_a[shape_a].tolist()))
        for shape_b in range(num_shapes_b):
            parent_b = list(map(int, parents_b[shape_b].tolist()))
            if ordered:
                children_b = _parents_to_children_ordered(parent_b)
            else:
                children_b = _build_children_from_parent(parent_b)
            for node_b in range(n_b):
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
                        if old < n_b:
                            mapping.append(("b", old))
                        else:
                            mapping.append(("a", old - n_b))
                    colour_index_map = np.empty(
                        (colourings_a.shape[0], colourings_b.shape[0]), dtype=np.int32
                    )
                    for idx_a, col_a in enumerate(colourings_a):
                        for idx_b, col_b in enumerate(colourings_b):
                            digits: list[int] = []
                            for source, node_idx in mapping:
                                if source == "b":
                                    digits.append(int(col_b[node_idx]))
                                else:
                                    digits.append(int(col_a[node_idx]))
                            colour_index_map[idx_a, idx_b] = _colouring_index_from_digits(
                                digits, ambient_dim
                            )
                    colour_index_map_jax = jnp.asarray(colour_index_map)
                    outer = jnp.outer(a_coeffs[shape_a], b_coeffs[shape_b])
                    out = out.at[shape_out, colour_index_map_jax].add(outer)

    return out.reshape(-1)


def _postorder(children: list[list[int]]) -> list[int]:
    order: list[int] = []

    def dfs(node: int) -> None:
        for child in children[node]:
            dfs(child)
        order.append(node)

    if not children:
        return order
    dfs(0)
    return order


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
            children = _build_children_from_parent(parent_row)
            child_lists_per_shape.append(children)
            local_max = max((len(c) for c in children), default=0)
            max_children = max(max_children, local_max)
            eval_orders_per_shape.append(_postorder(children))

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


@dataclass(frozen=True, eq=False)
class GLHopfAlgebra(HopfAlgebra):
    """Grossman-Larson / Connes-Kreimer Hopf algebra on unordered rooted forests.

    basis_size(level): number of unordered rooted forests with (level+1) nodes,
    multiplied by ambient_dim^(level+1) if nodes are coloured by driver components.
    """

    ambient_dimension: int
    depth: int = 0
    # Optional precomputed structures
    degree2_chain_indices: jax.Array | None = (
        None  # (ambient_dim, ambient_dim) mapping for degree-2 chains
    )
    shape_count_by_degree: list[int] = field(default_factory=list)
    forests_by_degree: tuple[BCKForest, ...] = field(default_factory=tuple)
    children_by_degree: tuple[jax.Array, ...] = field(default_factory=tuple)
    child_counts_by_degree: tuple[jax.Array, ...] = field(default_factory=tuple)
    eval_order_by_degree: tuple[jax.Array, ...] = field(default_factory=tuple)
    colourings_by_degree: tuple[np.ndarray, ...] = field(default_factory=tuple)
    parent_index_by_degree: tuple[dict[tuple[int, ...], int], ...] = field(
        default_factory=tuple
    )

    @override
    def basis_size(self, level: int | None = None) -> int:
        if not self.shape_count_by_degree:
            raise ValueError(
                "GLHopfAlgebra must be constructed via GLHopfAlgebra.build to "
                "populate per-degree shape counts."
            )
        if level is None:
            return sum(
                num_shapes * (self.ambient_dimension ** (idx + 1))
                for idx, num_shapes in enumerate(self.shape_count_by_degree)
            )
        if level < 0 or level >= self.depth:
            raise ValueError(
                f"Requested level {level} outside available range [0, {self.depth - 1}]."
            )
        n = level + 1
        num_shapes = self.shape_count_by_degree[level]
        return num_shapes * (self.ambient_dimension**n)

    @classmethod
    @override
    def build(
        cls,
        ambient_dim: int,
        depth: int,
    ) -> GLHopfAlgebra:
        # Build degree-2 map if applicable (no dependency on TreeIndex to avoid cycles)
        forests = enumerate_bck_trees(depth)
        deg2_map: jax.Array | None = None
        if len(forests) >= 2:
            parents = forests[1].parent  # degree 2 is index 1
            target = jnp.asarray([-1, 0], dtype=jnp.int32)
            eq = jnp.all(parents == target[None, :], axis=1)
            matches = jnp.where(eq, size=1, fill_value=-1)[0]
            shape_id = int(matches[0].item())
            if shape_id >= 0:
                rows = []
                for i in range(ambient_dim):
                    row = []
                    for j in range(ambient_dim):
                        row.append(shape_id * (ambient_dim**2) + i * ambient_dim + j)
                    rows.append(row)
                deg2_map = jnp.asarray(rows, dtype=jnp.int32)
        shape_counts = [int(f.parent.shape[0]) for f in forests]
        (
            children_tables,
            child_counts_tables,
            eval_orders_tables,
        ) = _build_children_eval_tables(forests)
        colourings_by_degree = _build_colourings_by_degree(ambient_dim, len(forests))
        parent_index_by_degree = _build_shape_index_by_degree(forests)
        return cls(
            ambient_dimension=ambient_dim,
            degree2_chain_indices=deg2_map,
            depth=len(forests),
            shape_count_by_degree=shape_counts,
            forests_by_degree=tuple(forests),
            children_by_degree=children_tables,
            child_counts_by_degree=child_counts_tables,
            eval_order_by_degree=eval_orders_tables,
            colourings_by_degree=colourings_by_degree,
            parent_index_by_degree=parent_index_by_degree,
        )

    @override
    def product(self, a_levels: list[jax.Array], b_levels: list[jax.Array]) -> list[jax.Array]:
        if len(a_levels) != len(b_levels):
            raise ValueError("Truncations must match for product.")
        depth = len(a_levels)
        out = [ai + bi for ai, bi in zip(a_levels, b_levels)]
        for out_idx in range(1, depth):
            acc = jnp.zeros_like(out[out_idx])
            for p in range(out_idx):
                degree_a = p + 1
                degree_b = out_idx - p
                acc = acc + _grafting_product_level(
                    a_level=a_levels[p],
                    b_level=b_levels[out_idx - 1 - p],
                    degree_a=degree_a,
                    degree_b=degree_b,
                    ambient_dim=self.ambient_dimension,
                    forests_by_degree=self.forests_by_degree,
                    parent_index_by_degree=self.parent_index_by_degree,
                    colourings_by_degree=self.colourings_by_degree,
                    ordered=False,
                )
            out[out_idx] = out[out_idx] + acc
        return out

    @override
    def coproduct(self, levels: list[jax.Array]) -> list[list[jax.Array]]:
        raise NotImplementedError("GLHopfAlgebra.coproduct is not implemented yet.")

    @override
    def exp(self, x: list[jax.Array]) -> list[jax.Array]:
        if len(x) == 0:
            return []
        depth = len(x)
        acc = [jnp.zeros_like(t) for t in x]
        current = [t for t in x]
        factorial = 1.0
        for k in range(1, depth + 1):
            factorial *= float(k)
            acc = [a + (1.0 / factorial) * c for a, c in zip(acc, current)]
            if k < depth:
                ab = self.product(current, x)
                # pure product: remove linear parts
                current = [u - v - w for u, v, w in zip(ab, current, x)]
        return acc

    @override
    def log(self, g: list[jax.Array]) -> list[jax.Array]:
        if len(g) == 0:
            return []
        depth = len(g)
        acc = [jnp.zeros_like(t) for t in g]
        current = [t for t in g]
        coeff = 1.0
        acc = [a + coeff * c for a, c in zip(acc, current)]
        for k in range(2, depth + 1):
            ab = self.product(current, g)
            current = [u - v - w for u, v, w in zip(ab, current, g)]
            coeff = ((-1.0) ** (k + 1)) / float(k)
            acc = [a + coeff * c for a, c in zip(acc, current)]
        return acc

    @override
    def __str__(self) -> str:
        return "Grossman-Larson Hopf Algebra"


@final
@dataclass(frozen=True, eq=False)
class MKWHopfAlgebra(HopfAlgebra):
    """Munthe-Kaas-Wright Hopf algebra on ordered (planar) rooted forests.

    basis_size(level): number of plane rooted forests with (level+1) nodes,
    multiplied by ambient_dim^(level+1) if nodes are coloured by driver components.
    """

    ambient_dimension: int
    depth: int = 0
    # Optional precomputed structures
    degree2_chain_indices: jax.Array | None = (
        None  # (ambient_dim, ambient_dim) mapping for degree-2 chains
    )
    shape_count_by_degree: list[int] = field(default_factory=list)
    forests_by_degree: tuple[MKWForest, ...] = field(default_factory=tuple)
    children_by_degree: tuple[jax.Array, ...] = field(default_factory=tuple)
    child_counts_by_degree: tuple[jax.Array, ...] = field(default_factory=tuple)
    eval_order_by_degree: tuple[jax.Array, ...] = field(default_factory=tuple)
    colourings_by_degree: tuple[np.ndarray, ...] = field(default_factory=tuple)
    parent_index_by_degree: tuple[dict[tuple[int, ...], int], ...] = field(
        default_factory=tuple
    )

    @override
    def basis_size(self, level: int | None = None) -> int:
        if not self.shape_count_by_degree:
            raise ValueError(
                "MKWHopfAlgebra must be constructed via MKWHopfAlgebra.build to "
                "populate per-degree shape counts."
            )
        if level is None:
            return sum(
                num_shapes * (self.ambient_dimension ** (idx + 1))
                for idx, num_shapes in enumerate(self.shape_count_by_degree)
            )
        if level < 0 or level >= self.depth:
            raise ValueError(
                f"Requested level {level} outside available range [0, {self.depth - 1}]."
            )
        n = level + 1
        num_shapes = self.shape_count_by_degree[level]
        return num_shapes * (self.ambient_dimension**n)

    @classmethod
    @override
    def build(
        cls,
        ambient_dim: int,
        depth: int,
    ) -> MKWHopfAlgebra:
        forests = enumerate_mkw_trees(depth)
        deg2_map: jax.Array | None = None
        if len(forests) >= 2:
            parents = forests[1].parent  # degree 2 is index 1
            target = jnp.asarray([-1, 0], dtype=jnp.int32)
            eq = jnp.all(parents == target[None, :], axis=1)
            matches = jnp.where(eq, size=1, fill_value=-1)[0]
            shape_id = int(matches[0].item())
            if shape_id >= 0:
                rows = []
                for i in range(ambient_dim):
                    row = []
                    for j in range(ambient_dim):
                        row.append(shape_id * (ambient_dim**2) + i * ambient_dim + j)
                    rows.append(row)
                deg2_map = jnp.asarray(rows, dtype=jnp.int32)
        shape_counts = [int(f.parent.shape[0]) for f in forests]
        (
            children_tables,
            child_counts_tables,
            eval_orders_tables,
        ) = _build_children_eval_tables(forests)
        colourings_by_degree = _build_colourings_by_degree(ambient_dim, len(forests))
        parent_index_by_degree = _build_shape_index_by_degree(forests)
        return cls(
            ambient_dimension=ambient_dim,
            degree2_chain_indices=deg2_map,
            depth=len(forests),
            shape_count_by_degree=shape_counts,
            forests_by_degree=tuple(forests),
            children_by_degree=children_tables,
            child_counts_by_degree=child_counts_tables,
            eval_order_by_degree=eval_orders_tables,
            colourings_by_degree=colourings_by_degree,
            parent_index_by_degree=parent_index_by_degree,
        )

    @override
    def product(self, a_levels: list[jax.Array], b_levels: list[jax.Array]) -> list[jax.Array]:
        if len(a_levels) != len(b_levels):
            raise ValueError("Truncation orders must match for product.")
        depth = len(a_levels)
        out = [ai + bi for ai, bi in zip(a_levels, b_levels)]
        for out_idx in range(1, depth):
            acc = jnp.zeros_like(out[out_idx])
            for p in range(out_idx):
                degree_a = p + 1
                degree_b = out_idx - p
                acc = acc + _grafting_product_level(
                    a_level=a_levels[p],
                    b_level=b_levels[out_idx - 1 - p],
                    degree_a=degree_a,
                    degree_b=degree_b,
                    ambient_dim=self.ambient_dimension,
                    forests_by_degree=self.forests_by_degree,
                    parent_index_by_degree=self.parent_index_by_degree,
                    colourings_by_degree=self.colourings_by_degree,
                    ordered=True,
                )
            out[out_idx] = out[out_idx] + acc
        return out

    @override
    def coproduct(self, levels: list[jax.Array]) -> list[list[jax.Array]]:
        raise NotImplementedError("MKWHopfAlgebra.coproduct is not implemented yet.")

    @override
    def exp(self, x: list[jax.Array]) -> list[jax.Array]:
        if len(x) == 0:
            return []
        depth = len(x)
        acc = [jnp.zeros_like(t) for t in x]
        current = [t for t in x]
        factorial = 1.0
        for k in range(1, depth + 1):
            factorial *= float(k)
            acc = [a + (1.0 / factorial) * c for a, c in zip(acc, current)]
            if k < depth:
                ab = self.product(current, x)
                current = [u - v - w for u, v, w in zip(ab, current, x)]
        return acc

    @override
    def log(self, g: list[jax.Array]) -> list[jax.Array]:
        if len(g) == 0:
            return []
        depth = len(g)
        acc = [jnp.zeros_like(t) for t in g]
        current = [t for t in g]
        coeff = 1.0
        acc = [a + coeff * c for a, c in zip(acc, current)]
        for k in range(2, depth + 1):
            ab = self.product(current, g)
            current = [u - v - w for u, v, w in zip(ab, current, g)]
            coeff = ((-1.0) ** (k + 1)) / float(k)
            acc = [a + coeff * c for a, c in zip(acc, current)]
        return acc

    @override
    def __str__(self) -> str:
        return "Munthe-Kaas-Wright Hopf Algebra"


HopfAlgebraT = TypeVar("HopfAlgebraT", bound=HopfAlgebra, contravariant=True)
