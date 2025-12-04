from __future__ import annotations
from typing import NamedTuple, NewType
import jax.numpy as jnp


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
