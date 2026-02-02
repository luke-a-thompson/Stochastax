"""Grossman-Larson / Connes-Kreimer Hopf algebra on unordered rooted forests."""

from __future__ import annotations
from typing import override
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from stochastax.hopf_algebras.hopf_algebra_types import (
    HopfAlgebra,
    _build_colourings_by_degree,
    _build_shape_index_by_degree,
    _build_grafting_tables,
    _build_children_eval_tables,
    _apply_grafting_table,
)
from stochastax.hopf_algebras.bases import enumerate_bck_trees
from stochastax.hopf_algebras.forest_types import BCKForest


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
    colourings_by_degree: tuple[jax.Array, ...] = field(default_factory=tuple)
    parent_index_by_degree: tuple[dict[tuple[int, ...], int], ...] = field(default_factory=tuple)
    grafting_tables_by_degree: tuple[tuple[jax.Array | None, ...], ...] = field(
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
        grafting_tables_by_degree = _build_grafting_tables(
            forests_by_degree=forests,
            parent_index_by_degree=parent_index_by_degree,
            colourings_by_degree=colourings_by_degree,
            ambient_dim=ambient_dim,
            ordered=False,
        )
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
            grafting_tables_by_degree=grafting_tables_by_degree,
        )

    @override
    def product(self, a_levels: list[jax.Array], b_levels: list[jax.Array]) -> list[jax.Array]:
        if len(a_levels) != len(b_levels):
            raise ValueError("Truncations must match for product.")
        depth = len(a_levels)
        out = [ai + bi for ai, bi in zip(a_levels, b_levels)]
        for out_idx in range(1, depth):
            acc = jnp.zeros_like(out[out_idx])
            num_shapes_out = self.shape_count_by_degree[out_idx]
            for p in range(out_idx):
                degree_a = p + 1
                degree_b = out_idx - p
                table = self.grafting_tables_by_degree[degree_a - 1][degree_b - 1]
                if table is None:
                    continue
                acc = acc + _apply_grafting_table(
                    a_level=a_levels[p],
                    b_level=b_levels[out_idx - 1 - p],
                    num_shapes_out=num_shapes_out,
                    table=table,
                )
            out[out_idx] = out[out_idx] + acc
        return out

    @override
    def exp(self, x: list[jax.Array]) -> list[jax.Array]:
        if len(x) == 0:
            return []
        depth = len(x)
        if depth > self.depth:
            raise ValueError("depth exceeds available Hopf algebra depth.")
        x1 = x[0]
        x2 = x[1] if depth >= 2 else jnp.zeros((0,), dtype=x1.dtype)
        acc = [jnp.zeros_like(t) for t in x]
        current = [jnp.zeros_like(t) for t in x]
        current[0] = x1
        if depth >= 2:
            current[1] = x2
        factorial = 1.0
        for k in range(1, depth + 1):
            factorial *= float(k)
            acc = [a + (1.0 / factorial) * c for a, c in zip(acc, current)]
            if k < depth:
                current = self._bilinear_deg12(current, x1, x2, depth)
        return acc

    def _bilinear_deg12(
        self,
        a_levels: list[jax.Array],
        x1: jax.Array,
        x2: jax.Array,
        depth: int,
    ) -> list[jax.Array]:
        out = [jnp.zeros_like(level) for level in a_levels]
        for out_idx in range(1, depth):
            num_shapes_out = self.shape_count_by_degree[out_idx]
            acc = jnp.zeros_like(out[out_idx])
            table = self.grafting_tables_by_degree[out_idx - 1][0]
            if table is not None:
                acc = acc + _apply_grafting_table(
                    a_level=a_levels[out_idx - 1],
                    b_level=x1,
                    num_shapes_out=num_shapes_out,
                    table=table,
                )
            if out_idx >= 2:
                table = self.grafting_tables_by_degree[out_idx - 2][1]
                if table is not None:
                    acc = acc + _apply_grafting_table(
                        a_level=a_levels[out_idx - 2],
                        b_level=x2,
                        num_shapes_out=num_shapes_out,
                        table=table,
                    )
            out[out_idx] = acc
        return out


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
