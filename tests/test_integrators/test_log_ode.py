from typing import Callable

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.linalg import expm as jexpm

from stochastax.integrators.log_ode import log_ode
from stochastax.control_lifts.log_signature import duval_generator
from stochastax.control_lifts.log_signature import compute_log_signature
from stochastax.vector_field_lifts.lie_lift import (
    form_lyndon_brackets_from_words,
    form_lyndon_lift,
)
from stochastax.vector_field_lifts.vector_field_lift_types import LyndonBrackets
import jax.random as jrandom


def _so3_generators() -> jax.Array:
    A1 = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    A2 = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    A3 = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return jnp.stack([A1, A2, A3], axis=0)  # [3, 3, 3]


def test_lyndon_log_ode_manifold_zero_control_identity() -> None:
    """With zero coefficients, log-ODE should return the same normalized state."""
    A: jax.Array = _so3_generators()
    depth: int = 2
    dim: int = A.shape[0]
    words = duval_generator(depth, dim)
    bracket_basis: LyndonBrackets = form_lyndon_brackets_from_words(A, words)

    y0: jax.Array = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    # Build a constant path (zero increments) and compute its log-signature
    zero_path = jnp.zeros((2, dim), dtype=jnp.float32)  # shape (N+1, dim)
    primitive = compute_log_signature(zero_path, depth, "Lyndon words", mode="full")
    y_next: jax.Array = log_ode(bracket_basis, primitive, y0)

    # y0 already unit norm; expect exact equality within tolerance
    assert jnp.allclose(y_next, y0, rtol=1e-7)


def test_lyndon_log_ode_euclidean_linear_matches_matrix_exponential() -> None:
    """In 1D with depth=1, log-ODE equals exp(Δ A) @ y0."""
    # Single generator in R^2: 90-degree rotation generator
    A0 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float32)
    A: jax.Array = A0[jnp.newaxis, ...]  # [1, 2, 2]
    depth: int = 1
    words = duval_generator(depth, 1)
    bracket_basis = form_lyndon_brackets_from_words(A, words)  # [1, 2, 2] == A0

    delta: float = 0.3
    # Build a 2-point path with increment delta
    seg_W = jnp.array([[0.0], [delta]], dtype=jnp.float32)

    y0: jax.Array = jnp.array([1.0, 0.0], dtype=jnp.float32)
    primitive = compute_log_signature(seg_W, depth, "Lyndon words", mode="full")
    y_logode: jax.Array = log_ode(bracket_basis, primitive, y0)

    expected: jax.Array = jexpm(delta * A0) @ y0
    expected = expected / jnp.linalg.norm(expected)

    assert jnp.allclose(y_logode, expected, rtol=1e-6)


def test_log_ode_works_with_precomputed_words() -> None:
    """log_ode with brackets built from precomputed Lyndon words reproduces linear result."""
    A0 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float32)
    A = A0[jnp.newaxis, ...]
    depth = 1
    words = duval_generator(depth, 1)
    brackets = form_lyndon_brackets_from_words(A, words)

    delta = 0.2
    seg_W = jnp.array([[0.0], [delta]], dtype=jnp.float32)
    y0 = jnp.array([1.0, 0.0], dtype=jnp.float32)
    primitive = compute_log_signature(seg_W, depth, "Lyndon words", mode="full")
    y_next = log_ode(brackets, primitive, y0)

    expected = jexpm(delta * A0) @ y0
    expected = expected / jnp.linalg.norm(expected)
    assert jnp.allclose(y_next, expected, rtol=1e-6, atol=1e-6)


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
    words = duval_generator(depth, dim)
    bracket_basis: LyndonBrackets = form_lyndon_brackets_from_words(A, words)
    y0: jax.Array = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    # JIT step integrator across a path of increments at depth=1
    @jax.jit
    def integrate_path(increments: jax.Array, y_init: jax.Array) -> tuple[jax.Array, jax.Array]:
        def step(carry: jax.Array, inc: jax.Array) -> tuple[jax.Array, jax.Array]:
            y_curr = carry
            # depth=1 => coefficients list is [inc]
            seg_W = jnp.vstack([jnp.zeros((1, dim), dtype=inc.dtype), inc.reshape(1, -1)])
            primitive = compute_log_signature(seg_W, depth, "Lyndon words", mode="full")
            y_next = log_ode(bracket_basis, primitive, y_curr)
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


def test_lyndon_lift_matches_linear_brackets() -> None:
    """Nonlinear free-Lie lift reproduces matrix brackets for linear vector fields."""
    A = _so3_generators()
    depth = 2
    dim = A.shape[0]

    def _linear_field(matrix: jax.Array) -> Callable[[jax.Array], jax.Array]:
        def field(y: jax.Array) -> jax.Array:
            return matrix @ y

        return field

    V = [_linear_field(A[i]) for i in range(dim)]
    x0 = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32)
    words = duval_generator(depth, dim)
    nonlinear = form_lyndon_lift(V, x0, words)
    words_direct = duval_generator(depth, dim)
    linear = form_lyndon_brackets_from_words(A, words_direct)

    for non_level, lin_level in zip(nonlinear, linear):
        assert non_level.shape == lin_level.shape
        assert jnp.allclose(non_level, lin_level, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("brownian_path_fixture", [(1, 200)], indirect=True)
def test_lyndon_log_ode_euclidean_segmentation_invariance(
    brownian_path_fixture: jax.Array,
) -> None:
    """Sequential windowed application equals whole-interval application for commuting case (dim=1)."""
    W: jax.Array = brownian_path_fixture  # shape (N+1, 1)
    depth: int = 1  # depth=1 and dim=1 => commuting flows so product of exponentials equals single exponential
    # Single 2x2 skew-symmetric generator
    A0 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float32)
    A: jax.Array = A0[jnp.newaxis, ...]  # [1, 2, 2]
    words = duval_generator(depth, A.shape[0])
    bracket_basis: LyndonBrackets = form_lyndon_brackets_from_words(A, words)

    y0: jax.Array = jnp.array([1.0, 0.0], dtype=jnp.float32)

    # Whole interval
    log_sig_full = compute_log_signature(W, depth, "Lyndon words", mode="full")
    y_full: jax.Array = log_ode(bracket_basis, log_sig_full, y0)

    # Windowed
    window: int = 10
    y_win: jax.Array = y0
    N: int = W.shape[0] - 1
    for s in range(0, N, window):
        e = min(s + window, N)
        seg: jax.Array = W[s : e + 1, :]
        log_sig_seg = compute_log_signature(seg, depth, "Lyndon words", mode="full")
        y_win = log_ode(bracket_basis, log_sig_seg, y_win)

    assert jnp.allclose(y_full, y_win, rtol=1e-5)


@pytest.mark.parametrize("brownian_path_fixture", [(3, 300)], indirect=True)
def test_lyndon_log_ode_euclidean_segmentation_invariance_commuting_high_depth(
    brownian_path_fixture: jax.Array,
) -> None:
    """Higher depth and multi-dim path on Euclidean (commuting diagonal generators).

    Since generators commute, product of exponentials along windows equals
    the exponential of the sum, so whole-interval equals windowed trajectory.
    """
    W: jax.Array = brownian_path_fixture  # shape (N+1, 3)
    depth: int = 3

    # Build commuting skew-symmetric generators acting on disjoint 2D planes in R^6
    # Each is an independent 2x2 rotation block; they commute and are norm-preserving.
    R2 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float32)
    zeros2 = jnp.zeros((2, 2), dtype=jnp.float32)
    # A1 acts on coordinates (0,1), A2 on (2,3), A3 on (4,5)
    A1 = jnp.block([[R2, zeros2, zeros2], [zeros2, zeros2, zeros2], [zeros2, zeros2, zeros2]])
    A2 = jnp.block([[zeros2, zeros2, zeros2], [zeros2, R2, zeros2], [zeros2, zeros2, zeros2]])
    A3 = jnp.block([[zeros2, zeros2, zeros2], [zeros2, zeros2, zeros2], [zeros2, zeros2, R2]])
    A: jax.Array = jnp.stack([A1, A2, A3], axis=0)  # [3, 6, 6]

    words = duval_generator(depth, A.shape[0])
    bracket_basis: LyndonBrackets = form_lyndon_brackets_from_words(A, words)

    y0: jax.Array = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    y0 = y0 / jnp.linalg.norm(y0)

    # Whole interval
    log_sig_full = compute_log_signature(W, depth, "Lyndon words", mode="full")
    y_full: jax.Array = log_ode(bracket_basis, log_sig_full, y0)

    # Windowed
    window: int = 25
    y_win: jax.Array = y0
    N: int = W.shape[0] - 1
    for s in range(0, N, window):
        e = min(s + window, N)
        seg: jax.Array = W[s : e + 1, :]
        log_sig_seg = compute_log_signature(seg, depth, "Lyndon words", mode="full")
        y_win = log_ode(bracket_basis, log_sig_seg, y_win)

    assert jnp.allclose(y_full, y_win, rtol=1e-5)
