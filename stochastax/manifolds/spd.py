from stochastax.manifolds import Manifold
import jax
import jax.numpy as jnp
import numpy as np
import math


class SPDManifold(Manifold):
    eps: float = 1e-6

    @classmethod
    def retract(cls, x: jax.Array, *, eps: float | None = None) -> jax.Array:
        """
        Project an ambient matrix to the SPD manifold via eigenvalue clipping.

        Under the affine-invariant geometry, the manifold is the cone of SPD
        matrices. We symmetrise and clamp eigenvalues to ensure positive-definite
        output.
        """
        if x.shape[-2:] != (x.shape[-1], x.shape[-1]):
            raise ValueError(f"SPD expects (..., n, n); got x={x.shape}.")

        sym = cls._symmetrize(x)
        evals, evecs = jnp.linalg.eigh(sym)
        min_eval = jnp.asarray(cls.eps if eps is None else eps, dtype=sym.dtype)
        evals = jnp.maximum(evals, min_eval)
        return (evecs * evals[..., None, :]) @ jnp.swapaxes(evecs, -2, -1)

    @classmethod
    def project_to_tangent(cls, y: jax.Array, v: jax.Array) -> jax.Array:
        """
        Project an ambient matrix onto the tangent space at y under the
        affine-invariant metric.

        Tangent space at y consists of symmetric matrices. The orthogonal
        projection in affine-invariant coordinates is:
            P_y(V) = y^{1/2} * sym(y^{-1/2} V y^{-1/2}) * y^{1/2}.
        """
        if y.shape[-2:] != (y.shape[-1], y.shape[-1]) or v.shape[-2:] != (
            v.shape[-1],
            v.shape[-1],
        ):
            raise ValueError(f"SPD expects (..., n, n); got y={y.shape}, v={v.shape}.")
        if y.shape != v.shape:
            raise ValueError(f"SPD expects matching shapes; got y={y.shape}, v={v.shape}.")

        y_sqrt, y_inv_sqrt = cls._matrix_sqrt_and_inv_sqrt(y)
        mid = y_inv_sqrt @ v @ y_inv_sqrt
        mid_sym = cls._symmetrize(mid)
        return y_sqrt @ mid_sym @ y_sqrt

    @staticmethod
    def _symmetrize(x: jax.Array) -> jax.Array:
        return 0.5 * (x + jnp.swapaxes(x, -2, -1))

    @classmethod
    def _matrix_sqrt_and_inv_sqrt(cls, y: jax.Array) -> tuple[jax.Array, jax.Array]:
        y_sym = cls._symmetrize(y)
        evals, evecs = jnp.linalg.eigh(y_sym)
        min_eval = jnp.asarray(cls.eps, dtype=y.dtype)
        evals = jnp.maximum(evals, min_eval)
        sqrt_evals = jnp.sqrt(evals)
        inv_sqrt_evals = 1.0 / sqrt_evals
        y_sqrt = (evecs * sqrt_evals[..., None, :]) @ jnp.swapaxes(evecs, -2, -1)
        y_inv_sqrt = (evecs * inv_sqrt_evals[..., None, :]) @ jnp.swapaxes(evecs, -2, -1)
        return y_sqrt, y_inv_sqrt

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
        diag = jnp.diagonal(H, axis1=-2, axis2=-1)  # (..., n)
        D = jnp.eye(n, dtype=v.dtype) * diag[..., None, :]  # (..., n, n)
        return H + jnp.swapaxes(H, -1, -2) - D
