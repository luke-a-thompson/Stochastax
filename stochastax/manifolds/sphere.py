from stochastax.manifolds import Manifold
import jax
import jax.numpy as jnp


class Sphere(Manifold):
    """Unit sphere S^(n-1) embedded in R^n.

    Points satisfy ||y|| = 1. The tangent space at y consists of
    all vectors orthogonal to y.
    """

    @classmethod
    def retract(cls, x: jax.Array) -> jax.Array:
        """Normalize vector to unit length.

        Args:
            x: Point in ambient space, shape (..., n).

        Returns:
            Normalized point on unit sphere.
        """
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        norm = jnp.maximum(norm, 1e-12)
        return x / norm

    @classmethod
    def project_to_tangent(cls, y: jax.Array, v: jax.Array) -> jax.Array:
        """Project vector onto tangent space orthogonal to y.

        For the sphere, the tangent space at y is the orthogonal complement
        of y. We remove the component of v along y: v_tangent = v - <v,y>y.

        Args:
            y: Point on the sphere, shape (..., n).
            v: Vector in ambient space, shape (..., n).

        Returns:
            Tangent vector orthogonal to y.
        """
        assert y.shape == v.shape, f"Shape mismatch: y.shape={y.shape}, v.shape={v.shape}"
        # Compute <v, y> with proper broadcasting
        dot_product = jnp.sum(v * y, axis=-1, keepdims=True)
        return v - dot_product * y
