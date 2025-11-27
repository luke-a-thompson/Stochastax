import jax.numpy as jnp

from stochastax.hopf_algebras.print_rooted_trees import print_forest
from stochastax.hopf_algebras.hopf_algebra_types import Forest


def test_render_forest_markdown_minimal_centered() -> None:
    parent = jnp.array([[-1, 0, 0]], dtype=jnp.int32)
    batch = Forest(parent=parent)
    md = print_forest(batch, show_node_ids=False)
    expected = (
        """
```
    •
  ┌─┴─┐
  │   │
  •   •
```
"""
    ).strip()
    assert md == expected, (
        f"Rendered markdown (centered) does not match expected. Got:\n{md}\nExpected:\n{expected}"
    )
