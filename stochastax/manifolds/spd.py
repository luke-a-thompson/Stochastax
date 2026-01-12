from stochastax.manifolds import Manifold
import jax
import jax.numpy as jnp
import numpy as np
import math

class SPDManifold(Manifold):
    @classmethod
    def vech(cls, X: jax.Array) -> jax.Array:
        # X: (..., n, n) symmetric (or treated as such)
        n = X.shape[-1]  # static at trace/compile time for a given call signature

        # Precomputed at trace/compile time (Python runs once per (n, dtype, device) specialisation)
        i, j = np.tril_indices(n)
        lin = np.ravel_multi_index((i, j), (n, n))
        lin = jnp.asarray(lin)

        flat = X.reshape(X.shape[:-2] + (n * n,))
        return jnp.take(flat, lin, axis=-1)

    @classmethod
    def unvech(cls, v: jax.Array) -> jax.Array:
        # v: (..., m), where m = n(n+1)/2
        m = v.shape[-1]  # static at trace/compile time for a given call signature

        # Solve m = n(n+1)/2  <=>  8m + 1 = (2n + 1)^2   (all trace-time Python integers)
        disc = 8 * m + 1
        root = math.isqrt(disc)
        if root * root != disc:
            raise ValueError(f"Invalid vech length {m}; expected n(n+1)/2 for integer n.")
        n = (root - 1) // 2

        # Precomputed at trace/compile time
        i, j = np.tril_indices(n)
        lin = np.ravel_multi_index((i, j), (n, n))
        lin = jnp.asarray(lin)

        flat = jnp.zeros(v.shape[:-1] + (n * n,), dtype=v.dtype)
        flat = flat.at[..., lin].set(v)
        H = flat.reshape(v.shape[:-1] + (n, n))

        # Symmetrise: H + H^T - diag(H) (batched-safe)
        diag = jnp.diagonal(H, axis1=-2, axis2=-1)                         # (..., n)
        D = jnp.eye(n, dtype=v.dtype) * diag[..., None, :]                 # (..., n, n)
        return H + jnp.swapaxes(H, -1, -2) - D
