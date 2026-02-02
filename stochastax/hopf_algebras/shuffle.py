"""Shuffle/Tensor Hopf algebra for path signatures."""

from __future__ import annotations
from typing import Literal, override
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from stochastax.tensor_ops import tensor_product
from stochastax.hopf_algebras.hopf_algebra_types import HopfAlgebra
from stochastax.hopf_algebras.bases import enumerate_lyndon_basis, build_lyndon_dependency_tables


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
