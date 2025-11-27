"""Utilities for working with rooted trees and forests.

This module defines the ``Forest`` container type and helpers to render
collections of rooted trees as Unicode art. A forest is represented by a
"parent array" for each tree, stacked along the first axis.

Conventions
- Nodes are indexed in preorder with the root at index 0.
- For each tree, ``parent[0] == -1`` and for all ``i > 0``, ``0 <= parent[i] < i``.
"""

import jax.numpy as jnp
from stochastax.hopf_algebras.hopf_algebra_types import Forest


def _build_children(parent: list[int]) -> list[list[int]]:
    """Compute adjacency lists (children) from a parent array.

    Args:
        parent: A single-tree parent array in preorder, length ``n``.

    Returns:
        A list of length ``n`` where entry ``i`` contains the child indices of ``i``.

    Complexity:
        O(n) time and O(n) additional space.
    """
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    for i in range(1, n):
        p = parent[i]
        if p >= 0:
            children[p].append(i)
    return children


def _render_tree_centered(parent: list[int], show_ids: bool) -> list[str]:
    """Render a single rooted tree with the root centered and diagonal branches."""
    children = _build_children(parent)

    def node_label(i: int) -> str:
        return f"•{i}" if show_ids else "•"

    n = len(parent)

    # Compute depth of each node
    depth: list[int] = [0] * n
    for i in range(1, n):
        depth[i] = depth[parent[i]] + 1
    max_depth = max(depth) if n > 0 else 0

    # Compute subtree layout widths in "units" where each leaf has width 1.
    # Positions are x-coordinates measured in leaf units (floats), later scaled.
    def layout(node: int, offset_units: float) -> tuple[int, dict[int, float]]:
        if len(children[node]) == 0:
            x = offset_units + 0.5
            return 1, {node: x}
        total_width = 0
        x_positions: dict[int, float] = {}
        child_centers: list[float] = []
        running_offset = offset_units
        for child in children[node]:
            child_width, child_pos = layout(child, running_offset)
            x_positions.update(child_pos)
            # Child center is the average of its occupied unit interval
            child_center = running_offset + child_width / 2.0
            child_centers.append(child_center)
            running_offset += child_width
            total_width += child_width
        # Center the current node above the span of its children
        x_node = sum(child_centers) / len(child_centers)
        x_positions[node] = x_node
        return total_width, x_positions

    if n == 0:
        return []

    _, x_pos_units = layout(0, 0.0)

    # Scale positions to columns for ASCII grid
    scale = 4  # columns per unit; larger spreads subtrees further apart
    col_pos: dict[int, int] = {node: int(round(x * scale)) for node, x in x_pos_units.items()}

    # Determine canvas size
    rows = max_depth * 3 + 1
    max_col = 0
    # Account for labels
    for node, col in col_pos.items():
        label = node_label(node)
        start = col - (len(label) // 2)
        end = start + len(label) - 1
        max_col = max(max_col, end)
    # Account for connectors (span to furthest child)
    for node, col_parent in col_pos.items():
        for child in children[node]:
            col_child = col_pos[child]
            max_col = max(max_col, col_parent, col_child)

    width = max(1, max_col + 1)
    canvas: list[list[str]] = [[" "] * width for _ in range(rows)]

    def put(r: int, c: int, ch: str) -> None:
        if 0 <= r < rows and 0 <= c < width:
            canvas[r][c] = ch

    def hline(r: int, c0: int, c1: int, ch: str) -> None:
        start = min(c0, c1)
        end = max(c0, c1)
        for x in range(start, end + 1):
            put(r, x, ch)

    # Draw connectors (3-row per depth: label, bar, stems)
    for node, col_parent in col_pos.items():
        if len(children[node]) == 0:
            continue
        r_label = depth[node] * 3
        r_bar = r_label + 1
        r_stem = r_label + 2
        child_cols = sorted(col_pos[ch] for ch in children[node])
        if len(child_cols) == 1:
            # Single child: simple verticals
            cc = child_cols[0]
            put(r_bar, col_parent, "│")
            put(r_stem, cc, "│")
            continue
        left = child_cols[0]
        right = child_cols[-1]
        # Horizontal rail with tees
        hline(r_bar, left, right, "─")
        put(r_bar, left, "┌")
        put(r_bar, right, "┐")
        for cc in child_cols[1:-1]:
            put(r_bar, cc, "┬")
        # Parent connector into the rail
        if col_parent in child_cols:
            put(r_bar, col_parent, "┼")
        else:
            put(r_bar, col_parent, "┴")
        # Child stems down to next label row
        for cc in child_cols:
            put(r_stem, cc, "│")

    # Draw node labels
    for node, col in col_pos.items():
        r = depth[node] * 3
        label = node_label(node)
        start = col - (len(label) // 2)
        for i, ch in enumerate(label):
            put(r, start + i, ch)

    return ["".join(row).rstrip() for row in canvas]


def print_forest(batch: Forest, show_node_ids: bool = True) -> str:
    """Render a ``Forest`` as a fenced Markdown code block using a centered layout.

    Args:
        batch: A ``Forest`` with ``parent`` of shape ``(num_trees, n)``.
        show_node_ids: If ``True``, include node indices next to bullets.

    Returns:
        A single string containing a fenced code block with one Unicode tree
        per forest row, separated by a blank line.
    """
    parents = jnp.asarray(batch.parent)
    drawings: list[str] = []
    for row in range(parents.shape[0]):
        parent_row: list[int] = list(map(int, parents[row].tolist()))
        tree_lines = _render_tree_centered(parent_row, show_node_ids)
        drawings.append("\n".join(tree_lines))
    body = "\n\n".join(drawings)
    return f"```\n{body}\n```"
