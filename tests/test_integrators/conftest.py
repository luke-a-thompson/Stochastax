"""Shared fixtures and helpers for integrator tests."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from stochastax.control_lifts.branched_signature_ito import (
    compute_nonplanar_branched_signature,
    compute_planar_branched_signature,
)
from stochastax.control_lifts.log_signature import compute_log_signature
from stochastax.hopf_algebras.hopf_algebras import ShuffleHopfAlgebra
from typing import Callable


def _linear_vector_fields(A: jax.Array) -> list:
    """Convert matrix generators to linear vector field callables.

    Args:
        A: Array of shape (dim, n, n) containing matrix generators

    Returns:
        List of callable vector fields y -> A[i] @ y
    """
    return [lambda y, M=A[i]: M @ y for i in range(A.shape[0])]


def _project_to_tangent(y: jax.Array, v: jax.Array) -> jax.Array:
    """Project vector v onto tangent plane at point y on S^2."""
    return v - jnp.dot(v, y) * y


@pytest.fixture
def rotation_matrix_2d() -> jax.Array:
    """Standard 2D rotation generator (90-degree CCW)."""
    return jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float32)


@pytest.fixture
def euclidean_initial_state() -> jax.Array:
    """Standard initial state for 2D Euclidean tests."""
    return jnp.array([1.0, 0.0], dtype=jnp.float32)


@pytest.fixture
def sphere_initial_state() -> jax.Array:
    """Standard initial state on S^2 (north pole)."""
    return jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)


def build_block_rotation_generators(num_generators: int) -> jax.Array:
    """Create commuting 2x2 rotation blocks for a Euclidean system."""
    R2 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float32)
    state_dim = 2 * num_generators
    matrices: list[jax.Array] = []
    for i in range(num_generators):
        mat = jnp.zeros((state_dim, state_dim), dtype=jnp.float32)
        start = 2 * i
        mat = mat.at[start : start + 2, start : start + 2].set(R2)
        matrices.append(mat)
    return jnp.stack(matrices, axis=0)


def build_block_initial_state(num_generators: int) -> jax.Array:
    """Create a normalized initial state aligned with the first axis of each block."""
    base = jnp.array([1.0, 0.0], dtype=jnp.float32)
    tiled = jnp.tile(base, reps=(num_generators,))
    return tiled / jnp.linalg.norm(tiled)


def build_two_point_path(delta: float, dim: int) -> jax.Array:
    """Construct a two-point control path with identical increments per driver."""
    delta_vec = jnp.full((1, dim), delta, dtype=jnp.float32)
    return jnp.vstack([jnp.zeros_like(delta_vec), delta_vec])


def build_deterministic_increments(
    dim: int, steps: int, seed: int = 0, scale: float = 0.05
) -> jax.Array:
    """Generate a deterministic-yet-smooth set of increments for benchmarking."""
    key = jrandom.PRNGKey(seed)
    noise = jrandom.normal(key, shape=(steps, dim), dtype=jnp.float32)
    axis = jnp.linspace(0.0, jnp.pi, steps, dtype=jnp.float32)
    waves = jnp.stack(
        [jnp.sin(axis + 0.3 * float(i)) for i in range(dim)],
        axis=1,
    )
    return scale * (noise + waves)
