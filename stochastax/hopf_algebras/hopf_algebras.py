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
        )

    @override
    def product(self, a_levels: list[jax.Array], b_levels: list[jax.Array]) -> list[jax.Array]:
        if len(a_levels) != len(b_levels):
            raise ValueError("Truncations must match for product.")
        m = len(a_levels)
        out = [ai + bi for ai, bi in zip(a_levels, b_levels)]
        if m >= 2:
            if self.degree2_chain_indices is None:
                raise ValueError("GLHopfAlgebra not built with degree-2 tables.")
            d = self.ambient_dimension
            a1 = a_levels[0].reshape(-1)
            b1 = b_levels[0].reshape(-1)
            if a1.shape[0] != d or b1.shape[0] != d:
                raise NotImplementedError("Degree-1 basis must be single-node with d colours.")
            outer = jnp.outer(a1, b1)  # (d, d)
            idx = self.degree2_chain_indices
            updates = jnp.zeros_like(out[1])
            updates = updates.at[idx].add(outer)
            out[1] = out[1] + updates
        if m >= 3:
            raise NotImplementedError("GLHopfAlgebra product for degree>=3 not implemented yet.")
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
        )

    @override
    def product(self, a_levels: list[jax.Array], b_levels: list[jax.Array]) -> list[jax.Array]:
        if len(a_levels) != len(b_levels):
            raise ValueError("Truncation orders must match for product.")
        order = len(a_levels)
        out = [ai + bi for ai, bi in zip(a_levels, b_levels)]
        if order >= 2:
            if self.degree2_chain_indices is None:
                raise ValueError("MKWHopfAlgebra not built with degree-2 tables.")
            a1 = a_levels[0].reshape(-1)
            b1 = b_levels[0].reshape(-1)
            if a1.shape[0] != self.ambient_dimension or b1.shape[0] != self.ambient_dimension:
                raise NotImplementedError(
                    "Degree-1 basis must be single-node with ambient dimension colours."
                )
            outer = jnp.outer(a1, b1)  # (ambient_dimension, ambient_dimension)
            idx = self.degree2_chain_indices
            updates = jnp.zeros_like(out[1])
            updates = updates.at[idx].add(outer)
            out[1] = out[1] + updates
        if order >= 3:
            raise NotImplementedError("MKWHopfAlgebra product for order>=3 not implemented yet.")
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
