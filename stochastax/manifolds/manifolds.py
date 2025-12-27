from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp


class Manifold(ABC):
    """Base class for Riemannian manifold operations.

    Used for constrained dynamics on manifolds, including Neural CDEs,
    log-ODE integration, and geometric deep learning applications.
    """

    @abstractmethod
    def retract(self, x: jax.Array) -> jax.Array:
        """Map a point from ambient space onto the manifold.

        This is a retraction operation that takes an arbitrary point in the
        ambient space and returns the nearest (or a nearby) point on the manifold.

        Args:
            x: Point in ambient space, shape (..., n) where n is the ambient dimension.

        Returns:
            Point on the manifold with the same shape as input.
        """
        pass

    @abstractmethod
    def project_to_tangent(self, y: jax.Array, v: jax.Array) -> jax.Array:
        """Project an ambient vector onto the tangent space at a manifold point.

        The tangent space T_y(M) at point y consists of all vectors tangent to
        the manifold at that point. This operation removes any normal components.

        Args:
            y: Point on the manifold, shape (..., n).
            v: Vector in ambient space, shape (..., n).

        Returns:
            Tangent vector at y with the same shape as input.
        """
        pass


class EuclideanSpace(Manifold):
    """Euclidean space R^n as a manifold.

    The tangent space at any point is the entire ambient space,
    so retraction and projection are identity operations.
    """

    def retract(self, x: jax.Array) -> jax.Array:
        """Identity map for Euclidean space."""
        return x

    def project_to_tangent(self, y: jax.Array, v: jax.Array) -> jax.Array:
        """Identity projection for Euclidean space."""
        assert y.shape == v.shape, f"Shape mismatch: y.shape={y.shape}, v.shape={v.shape}"
        return v


class Sphere(Manifold):
    """Unit sphere S^(n-1) embedded in R^n.

    Points satisfy ||y|| = 1. The tangent space at y consists of
    all vectors orthogonal to y.
    """

    def retract(self, x: jax.Array) -> jax.Array:
        """Normalize vector to unit length.

        Args:
            x: Point in ambient space, shape (..., n).

        Returns:
            Normalized point on unit sphere.
        """
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        assert jnp.all(norm > 0), "Cannot normalize zero vector to sphere"
        return x / norm

    def project_to_tangent(self, y: jax.Array, v: jax.Array) -> jax.Array:
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
