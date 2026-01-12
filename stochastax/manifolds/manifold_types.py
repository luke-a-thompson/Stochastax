from abc import ABC, abstractmethod
import jax


class Manifold(ABC):
    """Base class for Riemannian manifold operations.

    Used for constrained dynamics on manifolds, including Neural CDEs,
    log-ODE integration, and geometric deep learning applications.
    """

    @classmethod
    @abstractmethod
    def retract(cls, x: jax.Array) -> jax.Array:
        """Map a point from ambient space onto the manifold.

        This is a retraction operation that takes an arbitrary point in the
        ambient space and returns the nearest (or a nearby) point on the manifold.

        Args:
            x: Point in ambient space, shape (..., n) where n is the ambient dimension.

        Returns:
            Point on the manifold with the same shape as input.
        """
        pass

    @classmethod
    @abstractmethod
    def project_to_tangent(cls, y: jax.Array, v: jax.Array) -> jax.Array:
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

    @classmethod
    def retract(cls, x: jax.Array) -> jax.Array:
        """Identity map for Euclidean space."""
        return x

    @classmethod
    def project_to_tangent(cls, y: jax.Array, v: jax.Array) -> jax.Array:
        """Identity projection for Euclidean space."""
        assert y.shape == v.shape, f"Shape mismatch: y.shape={y.shape}, v.shape={v.shape}"
        return v
