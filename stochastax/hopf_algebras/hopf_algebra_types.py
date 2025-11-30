from __future__ import annotations
from typing import NamedTuple, NewType, final, override, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from stochastax.tensor_ops import cauchy_convolution


class HopfAlgebra(ABC):
    """Abstract Hopf algebra interface sufficient for signature/log-signature workflows."""

    ambient_dimension: int

    @abstractmethod
    def basis_size(self, level: int) -> int:
        """Number of basis elements at given level (degree = level + 1 for signatures).

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


@dataclass(frozen=True)
class ShuffleHopfAlgebra(HopfAlgebra):
    """Shuffle/Tensor Hopf algebra used for path signatures.

    The representation uses per-degree flattened tensors, omitting degree 0.
    Instances built via ``build`` cache per-degree metadata so downstream
    consumers can reuse the same combinatorics instead of recomputing them.
    """

    ambient_dimension: int
    max_degree: int = 0
    shape_count_by_degree: list[int] = field(default_factory=list)
    lyndon_basis_by_degree: tuple[jax.Array, ...] = field(default_factory=tuple)

    @classmethod
    def build(
        cls,
        d: int,
        max_degree: int,
        cache_lyndon_basis: bool = False,
    ) -> "ShuffleHopfAlgebra":
        """Construct a shuffle Hopf algebra with cached per-degree metadata."""
        if max_degree <= 0:
            raise ValueError(f"max_degree must be >= 1, got {max_degree}.")
        shape_counts = [int(d ** (level + 1)) for level in range(max_degree)]
        if cache_lyndon_basis:
            from stochastax.hopf_algebras.free_lie import enumerate_lyndon_basis

            lyndon_levels = enumerate_lyndon_basis(max_degree, d)
            lyndon_tuple = tuple(lyndon_levels)
        else:
            lyndon_tuple: tuple[jax.Array, ...] = tuple()
        return cls(
            ambient_dimension=d,
            max_degree=max_degree,
            shape_count_by_degree=shape_counts,
            lyndon_basis_by_degree=lyndon_tuple,
        )

    @override
    def basis_size(self, level: int) -> int:
        if self.shape_count_by_degree:
            if level < 0 or level >= len(self.shape_count_by_degree):
                raise ValueError(
                    f"Requested level {level} outside available range "
                    f"[0, {len(self.shape_count_by_degree) - 1}]."
                )
            return self.shape_count_by_degree[level]
        # Stateless fallback (legacy instances not created via build()).
        return int(self.ambient_dimension ** (level + 1))

    def _unflatten_levels(self, levels: list[jax.Array]) -> list[jax.Array]:
        dim = self.ambient_dimension
        return [term.reshape((dim,) * (i + 1)) for i, term in enumerate(levels)]

    def _flatten_levels(self, levels: list[jax.Array]) -> list[jax.Array]:
        return [term.reshape(-1) for term in levels]

    @override
    def product(self, a_levels: list[jax.Array], b_levels: list[jax.Array]) -> list[jax.Array]:
        if len(a_levels) != len(b_levels):
            raise ValueError("Truncations must match for product.")
        depth = len(a_levels)
        # Work in unflattened tensor shapes for the convolution; then re-flatten
        a_unflat = self._unflatten_levels(a_levels)
        b_unflat = self._unflatten_levels(b_levels)
        cross_unflat = cauchy_convolution(a_unflat, b_unflat, depth)
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


class Forest(NamedTuple):
    """A batch container for a forest of rooted trees.

    Parameters
    - parent: 2D array of shape ``(num_trees, n)`` with dtype ``int32``.
      Each row encodes one rooted tree via its parent array in preorder:
      ``parent[0] == -1`` and for ``i > 0`` we have ``0 <= parent[i] < i``.

    Notes
    - This container is compatible with JAX; the array can be a ``jax.Array``.
    - The number of nodes ``n`` is the same for all trees in the forest.

    Example
    >>> import jax.numpy as jnp
    >>> forest = Forest(parent=jnp.array([[-1, 0, 0]], dtype=jnp.int32))
    >>> forest.parent.shape
    (1, 3)
    """

    parent: jnp.ndarray

    @property
    def order(self) -> int:
        """Order of the forest.

        The order of a forest is the number of nodes in each tree of the forest.
        """
        return len(self.parent[0])

    @property
    def size(self) -> int:
        """Size of the forest.

        The size of a forest is the number of trees in the forest.
        """
        return self.parent.shape[0]


MKWForest = NewType("MKWForest", Forest)
BCKForest = NewType("BCKForest", Forest)


@dataclass(frozen=True, eq=False)
class GLHopfAlgebra(HopfAlgebra):
    """Grossman-Larson / Connes-Kreimer Hopf algebra on unordered rooted forests.

    basis_size(level): number of unordered rooted forests with (level+1) nodes,
    multiplied by ambient_dim^(level+1) if nodes are coloured by driver components.
    """

    ambient_dimension: int
    max_order: int = 0
    # Optional precomputed structures
    degree2_chain_indices: Optional[jax.Array] = None  # (d, d) mapping for degree-2 chains
    shape_count_by_degree: list[int] = field(default_factory=list)
    forests_by_degree: tuple[BCKForest, ...] = field(default_factory=tuple)

    @override
    def basis_size(self, level: int) -> int:
        if not self.shape_count_by_degree:
            raise ValueError(
                "GLHopfAlgebra must be constructed via GLHopfAlgebra.build to "
                "populate per-degree shape counts."
            )
        if level < 0 or level >= len(self.shape_count_by_degree):
            raise ValueError(
                f"Requested level {level} outside available range "
                f"[0, {len(self.shape_count_by_degree) - 1}]."
            )
        n = level + 1
        num_shapes = self.shape_count_by_degree[level]
        return num_shapes * (self.ambient_dimension**n)

    @classmethod
    def build(
        cls,
        d: int,
        forests: list[BCKForest],
    ) -> GLHopfAlgebra:
        # Build degree-2 map if applicable (no dependency on TreeIndex to avoid cycles)
        deg2_map: Optional[jax.Array] = None
        if len(forests) >= 2:
            parents = forests[1].parent  # degree 2 is index 1
            target = jnp.asarray([-1, 0], dtype=jnp.int32)
            eq = jnp.all(parents == target[None, :], axis=1)
            matches = jnp.where(eq, size=1, fill_value=-1)[0]
            shape_id = int(matches[0].item())
            if shape_id >= 0:
                rows = []
                for i in range(d):
                    row = []
                    for j in range(d):
                        row.append(shape_id * (d**2) + i * d + j)
                    rows.append(row)
                deg2_map = jnp.asarray(rows, dtype=jnp.int32)
        shape_counts = [int(f.parent.shape[0]) for f in forests]
        return cls(
            ambient_dimension=d,
            degree2_chain_indices=deg2_map,
            max_order=len(forests),
            shape_count_by_degree=shape_counts,
            forests_by_degree=tuple(forests),
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
    max_order: int = 0
    # Optional precomputed structures
    degree2_chain_indices: Optional[jax.Array] = None  # (d, d) mapping for degree-2 chains
    shape_count_by_degree: list[int] = field(default_factory=list)
    forests_by_degree: tuple[MKWForest, ...] = field(default_factory=tuple)

    @override
    def basis_size(self, level: int) -> int:
        if not self.shape_count_by_degree:
            raise ValueError(
                "MKWHopfAlgebra must be constructed via MKWHopfAlgebra.build to "
                "populate per-degree shape counts."
            )
        if level < 0 or level >= len(self.shape_count_by_degree):
            raise ValueError(
                f"Requested level {level} outside available range "
                f"[0, {len(self.shape_count_by_degree) - 1}]."
            )
        n = level + 1
        num_shapes = self.shape_count_by_degree[level]
        return num_shapes * (self.ambient_dimension**n)

    @classmethod
    def build(
        cls,
        ambient_dim: int,
        forests: list[MKWForest],
    ) -> MKWHopfAlgebra:
        deg2_map: Optional[jax.Array] = None
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
        return cls(
            ambient_dimension=ambient_dim,
            degree2_chain_indices=deg2_map,
            max_order=len(forests),
            shape_count_by_degree=shape_counts,
            forests_by_degree=tuple(forests),
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
