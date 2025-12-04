import jax
from typing import Callable


def split_multi_vector_field(
    multi_vf: Callable[[jax.Array], jax.Array],
    dim: int,
    state_dim: int,
) -> list[Callable[[jax.Array], jax.Array]]:
    """
    Wrap a batched vector field ``multi_vf(y) -> [dim, state_dim]`` into
    a list of ``dim`` individual vector fields ``V[i](y) -> [state_dim]``.

    Useful when a single neural network produces all driver channels at once
    (e.g. shape ``[data_dim * hidden_dim]`` reshaped to ``[data_dim, hidden_dim]``),
    but downstream lifts (``form_lyndon_lift``, ``form_bck_lift``,
    ``form_mkw_lift``) expect a list of per-channel callables.
    """

    def make_V(i: int) -> Callable[[jax.Array], jax.Array]:
        def V_i(y: jax.Array) -> jax.Array:
            out = multi_vf(y).reshape(dim, state_dim)
            return out[i]

        return V_i

    return [make_V(i) for i in range(dim)]
