import jax
import jax.numpy as jnp
import pytest
from typing import Callable
from jax.scipy.linalg import expm as jexpm

from stochastax.integrators.log_ode import log_ode
from stochastax.hopf_algebras.free_lie import enumerate_lyndon_basis
from stochastax.hopf_algebras.hopf_algebras import ShuffleHopfAlgebra
from stochastax.control_lifts.log_signature import compute_log_signature
from stochastax.vector_field_lifts.lie_lift import (
    form_lyndon_bracket_functions,
)
import jax.random as jrandom

from tests.conftest import _so3_generators
from tests.test_integrators.conftest import (
    _linear_vector_fields,
    build_block_rotation_generators,
)
from stochastax.manifolds import Sphere


def _simple_vector_fields(dim: int) -> list:
    mats = jnp.stack(
        [jnp.eye(dim, dtype=jnp.float32) * float(i + 1) for i in range(dim)],
        axis=0,
    )

    def make_field(matrix: jax.Array) -> Callable[[jax.Array], jax.Array]:
        def vf(y: jax.Array) -> jax.Array:
            return matrix @ y

        return vf

    return [make_field(mats[i]) for i in range(dim)]


def test_lyndon_log_ode_zero_control_identity() -> None:
    """With zero coefficients, state-dependent log-ODE should return the same state."""
    depth = 2
    dim = 3
    hopf = ShuffleHopfAlgebra.build(ambient_dim=dim, depth=depth)

    A = _so3_generators()
    V = _linear_vector_fields(A)
    bracket_functions = form_lyndon_bracket_functions(V, hopf, Sphere())

    y0 = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    zero_path = jnp.zeros((2, dim), dtype=jnp.float32)
    primitive = compute_log_signature(zero_path, depth, hopf, "Lyndon words", mode="full")
    y_next = log_ode(bracket_functions, primitive, y0, Sphere())

    assert jnp.allclose(y_next, y0, rtol=1e-7, atol=0.0)


@pytest.mark.parametrize("dim", [1, 3, 5])
def test_lyndon_log_ode_euclidean_depth1_linear_matches_matrix_exponential(dim: int) -> None:
    """Depth-1 state-dependent log-ODE matches exp(Δ * sum_i A_i) @ y0 for linear fields."""
    depth = 1
    delta: float = 0.3

    hopf = ShuffleHopfAlgebra.build(ambient_dim=dim, depth=depth)
    generators = build_block_rotation_generators(dim)
    vector_fields = _linear_vector_fields(generators)
    bracket_functions = form_lyndon_bracket_functions(vector_fields, hopf)

    path = jnp.stack(
        [jnp.zeros((dim,), dtype=jnp.float32), jnp.full((dim,), delta, dtype=jnp.float32)]
    )
    primitive = compute_log_signature(path, depth, hopf, "Lyndon words", mode="full")

    y0 = jnp.array([1.0, 0.0] * dim, dtype=jnp.float32)
    y0 = y0 / jnp.linalg.norm(y0)

    y_logode = log_ode(bracket_functions, primitive, y0)

    combined_generator = jnp.sum(generators, axis=0)
    expected = jexpm(delta * combined_generator) @ y0
    assert jnp.allclose(y_logode, expected, rtol=1e-6, atol=1e-6)


def test_lyndon_log_ode_manifold_brownian_statistics() -> None:
    """Combined check for spherical Brownian properties with skew-symmetric generators.
    - Norm preservation along trajectory
    - Small-time tangent-plane mean ~ 0 and covariance ~ t * I
    - Long-time empirical mean ~ 0 (approximate uniformity)
    """
    # Construct two skew-symmetric generators in R^3 that span the tangent plane at e3
    # A1 rotates in (e1, e3)-plane, A2 rotates in (e2, e3)-plane
    A1 = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=jnp.float32)
    A2 = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=jnp.float32)
    A: jax.Array = jnp.stack([A1, A2], axis=0)  # [2, 3, 3]
    depth: int = 1
    dim: int = A.shape[0]
    hopf = ShuffleHopfAlgebra.build(ambient_dim=dim, depth=depth)
    bracket_functions = form_lyndon_bracket_functions(_linear_vector_fields(A), hopf, Sphere())
    y0: jax.Array = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    # JIT step integrator across a path of increments at depth=1
    @jax.jit
    def integrate_path(increments: jax.Array, y_init: jax.Array) -> tuple[jax.Array, jax.Array]:
        def step(carry: jax.Array, inc: jax.Array) -> tuple[jax.Array, jax.Array]:
            y_curr = carry
            # depth=1 => coefficients list is [inc]
            seg_W = jnp.vstack([jnp.zeros((1, dim), dtype=inc.dtype), inc.reshape(1, -1)])
            primitive = compute_log_signature(seg_W, depth, hopf, "Lyndon words", mode="full")
            y_next = log_ode(bracket_functions, primitive, y_curr, Sphere())
            return y_next, y_next

        y_T, ys = jax.lax.scan(step, y_init, increments)
        return y_T, ys

    key = jrandom.PRNGKey(0)

    # 1) Norm preservation along a single long trajectory
    T_long = 2.0
    N_long = 200
    dt_long = T_long / N_long
    key, sub = jrandom.split(key)
    dW_long: jax.Array = jrandom.normal(sub, shape=(N_long, dim), dtype=jnp.float32) * jnp.sqrt(
        dt_long
    )
    _, ys_long = integrate_path(dW_long, y0)
    norms = jnp.linalg.norm(ys_long, axis=1)
    assert jnp.allclose(norms, 1.0, rtol=1e-6, atol=1e-6)

    # 2) Small-time tangent-plane statistics (mean ~ 0, Cov ~ t * I_2)
    M_small = 512
    T_small = 0.05
    N_small = 50
    dt_small = T_small / N_small
    key, sub = jrandom.split(key)
    dW_small: jax.Array = jrandom.normal(
        sub, shape=(M_small, N_small, dim), dtype=jnp.float32
    ) * jnp.sqrt(dt_small)

    @jax.jit
    def integrate_batch(increments_batch: jax.Array) -> jax.Array:
        # vmap over trials to get final states
        def one_traj(incs: jax.Array) -> jax.Array:
            yT, _ = integrate_path(incs, y0)
            return yT

        return jax.vmap(one_traj, in_axes=0)(increments_batch)

    yT_small: jax.Array = integrate_batch(dW_small)  # [M_small, 3]
    disp_small: jax.Array = yT_small - y0  # [M_small, 3]

    # Project onto tangent plane at y0 = e3
    P_tan = jnp.eye(3, dtype=jnp.float32) - jnp.outer(y0, y0)  # diag(1,1,0)
    disp_tan = disp_small @ P_tan.T  # [M_small, 3], last component near 0
    # Use first two components for covariance check
    disp_xy = disp_tan[:, :2]  # [M_small, 2]

    mean_xy = jnp.mean(disp_xy, axis=0)
    # Tolerance scales like sqrt(Var/M) ~ sqrt(T_small/M_small)
    assert jnp.linalg.norm(mean_xy) < 0.07

    # Empirical covariance
    centered = disp_xy - mean_xy
    cov_xy = (centered.T @ centered) / float(M_small)
    target_cov = T_small * jnp.eye(2, dtype=jnp.float32)
    assert jnp.allclose(cov_xy, target_cov, rtol=0.15, atol=0.02)

    # 3) Long-time approximate uniformity: empirical mean near zero
    M_large = 512
    key, sub = jrandom.split(key)
    dW_large: jax.Array = jrandom.normal(
        sub, shape=(M_large, N_long, dim), dtype=jnp.float32
    ) * jnp.sqrt(dt_long)
    yT_large: jax.Array = integrate_batch(dW_large)
    mean_large = jnp.mean(yT_large, axis=0)
    # Expect exponential decay of the mean from the pole at finite time
    expected_decay = jnp.exp(-T_long)
    assert jnp.allclose(jnp.linalg.norm(mean_large), expected_decay, rtol=0.3, atol=0.03)
