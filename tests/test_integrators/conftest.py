"""Shared fixtures and helpers for integrator tests."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from stochastax.control_lifts.branched_signature_ito import (
    compute_nonplanar_branched_signature,
    compute_planar_branched_signature,
)
from stochastax.control_lifts.log_signature import compute_log_signature
from stochastax.hopf_algebras.hopf_algebras import ShuffleHopfAlgebra
from stochastax.control_lifts.signature_types import LogSignature
from stochastax.hopf_algebras.free_lie import enumerate_lyndon_basis
from stochastax.hopf_algebras.hopf_algebras import GLHopfAlgebra, MKWHopfAlgebra
from stochastax.vector_field_lifts.bck_lift import form_bck_lift
from stochastax.vector_field_lifts.lie_lift import form_lyndon_brackets_from_words
from stochastax.vector_field_lifts.mkw_lift import form_mkw_lift
from stochastax.vector_field_lifts.vector_field_lift_types import LyndonBrackets

from tests.conftest import _so3_generators


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


def benchmark_wrapper(benchmark: BenchmarkFixture, func, *args, **kwargs) -> jax.Array:
    """Wrapper for JAX benchmark that ensures device synchronization."""
    warmed = jax.block_until_ready(func(*args, **kwargs))

    def run() -> None:
        jax.block_until_ready(func(*args, **kwargs))
        return None

    benchmark(run)
    return warmed


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


def build_standard_log_ode_inputs(
    depth: int, dim: int, delta: float = 0.3
) -> tuple[LyndonBrackets, LogSignature, jax.Array]:
    """Assemble brackets, log-signature, and initial state for Euclidean log-ODE."""
    generators = build_block_rotation_generators(dim)
    words = enumerate_lyndon_basis(depth, dim)
    brackets = form_lyndon_brackets_from_words(generators, words)
    path = build_two_point_path(delta, dim)
    hopf = ShuffleHopfAlgebra.build(ambient_dim=dim, depth=depth)
    primitive = compute_log_signature(path, depth, hopf, "Lyndon words", mode="full")
    y0 = build_block_initial_state(dim)
    return brackets, primitive, y0


def build_standard_manifold_case(
    depth: int, steps: int = 32, seed: int = 0, dt: float = 0.01
) -> tuple[LyndonBrackets, jax.Array, jax.Array]:
    """Create deterministic increments for manifold log-ODE benchmarking."""
    A = _so3_generators()
    dim = A.shape[0]
    words = enumerate_lyndon_basis(depth, dim)
    brackets = form_lyndon_brackets_from_words(A, words)
    increments = build_deterministic_increments(dim, steps, seed, scale=float(dt) ** 0.5)
    y0 = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    return brackets, increments, y0


def build_bck_log_ode_inputs(
    depth: int, dim: int, delta: float = 0.35, cov_scale: float = 0.0
) -> tuple[list, object, jax.Array]:
    """Assemble BCK brackets and non-planar branched log-signature."""
    hopf = GLHopfAlgebra.build(dim, depth)
    generators = build_block_rotation_generators(dim)
    vector_fields = _linear_vector_fields(generators)
    y0 = build_block_initial_state(dim)
    path = build_two_point_path(delta, dim)
    steps = path.shape[0] - 1
    identity = jnp.eye(dim, dtype=jnp.float32)
    cov = jnp.tile((cov_scale * identity)[None, :, :], reps=(steps, 1, 1))
    signature = compute_nonplanar_branched_signature(
        path=path,
        order_m=depth,
        hopf=hopf,
        mode="full",
        cov_increments=cov,
    )
    logsig = signature.log()
    brackets = form_bck_lift(vector_fields, y0, hopf)
    return brackets, logsig, y0


def build_mkw_log_ode_inputs(
    depth: int, steps: int = 12, seed: int = 0, cov_scale: float = 0.0
) -> tuple[list, object, jax.Array]:
    """Assemble MKW brackets and planar branched log-signature on S^2."""
    A = _so3_generators()
    dim = A.shape[0]
    hopf = MKWHopfAlgebra.build(dim, depth)
    vector_fields = _linear_vector_fields(A)
    y0 = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    brackets = form_mkw_lift(vector_fields, y0, hopf, _project_to_tangent)

    increments = build_deterministic_increments(dim, steps, seed, scale=0.03)
    origin = jnp.zeros((1, dim), dtype=jnp.float32)
    path = jnp.concatenate([origin, origin + jnp.cumsum(increments, axis=0)], axis=0)
    identity = jnp.eye(dim, dtype=jnp.float32)
    cov = jnp.tile((cov_scale * identity)[None, :, :], reps=(steps, 1, 1))
    signature = compute_planar_branched_signature(
        path=path,
        order_m=depth,
        hopf=hopf,
        mode="full",
        cov_increments=cov,
    )
    logsig = signature.log()
    return brackets, logsig, y0
