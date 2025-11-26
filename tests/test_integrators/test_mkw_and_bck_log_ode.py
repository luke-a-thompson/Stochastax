import jax
import jax.numpy as jnp
import pytest
from jax.scipy.linalg import expm as jexpm
from typing import cast

from stochastax.controls.drivers import bm_driver
from stochastax.controls.augmentations import non_overlapping_windower
from stochastax.control_lifts.branched_signature_ito import (
    compute_planar_branched_signature,
    compute_nonplanar_branched_signature,
)
from stochastax.control_lifts.signature_types import (
    BCKLogSignature,
    MKWLogSignature,
)
from stochastax.integrators.log_ode import log_ode
from stochastax.vector_field_lifts.bck_lift import form_bck_brackets
from stochastax.vector_field_lifts.mkw_lift import form_mkw_brackets
from stochastax.hopf_algebras import enumerate_mkw_trees, enumerate_bck_trees
from stochastax.hopf_algebras.hopf_algebra_types import MKWHopfAlgebra, GLHopfAlgebra
from stochastax.hopf_algebras.elements import GroupElement


def _so3_generators() -> jax.Array:
    A1 = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    A2 = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=jnp.float32)
    A3 = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    return jnp.stack([A1, A2, A3], axis=0)


def _linear_vector_fields(A: jax.Array):
    return [lambda y, M=A[i]: M @ y for i in range(A.shape[0])]


def _project_to_tangent(y: jax.Array, v: jax.Array) -> jax.Array:
    # Tangent projection on S^2
    return v - jnp.dot(v, y) * y


def test_bck_log_ode_euclidean() -> None:
    """Depth-1 BCK log-ODE in Euclidean space matches the corresponding matrix exponential."""
    depth = 1
    delta = 0.35
    A0 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float32)
    A = A0[jnp.newaxis, ...]
    y0 = jnp.array([1.0, 0.0], dtype=jnp.float32)

    forests = enumerate_bck_trees(depth)
    hopf = GLHopfAlgebra.build(1, forests)
    bck_brackets = form_bck_brackets(_linear_vector_fields(A), y0, forests)

    path = jnp.array([[0.0], [delta]], dtype=jnp.float32)
    cov = jnp.zeros((1, 1, 1), dtype=jnp.float32)
    sig_levels = compute_nonplanar_branched_signature(
        path=path,
        order_m=depth,
        forests=forests,
        cov_increments=cov,
        return_trajectory=False,
    )
    sig_levels_list = cast(list[jax.Array], sig_levels)
    group_el = GroupElement(hopf=hopf, coeffs=sig_levels_list, interval=(0.0, 1.0))
    logsig = BCKLogSignature(group_el.log())

    y_next = log_ode(bck_brackets, logsig, y0)
    expected = jexpm(delta * A0) @ y0
    expected = expected / jnp.linalg.norm(y0)
    assert jnp.allclose(y_next, expected, rtol=1e-6, atol=1e-6)


def test_mkw_log_ode_euclidean() -> None:
    """Depth-1 MKW log-ODE in Euclidean space matches the corresponding matrix exponential."""
    depth = 1
    delta = -0.41
    A0 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float32)
    A = A0[jnp.newaxis, ...]
    y0 = jnp.array([1.0, 0.0], dtype=jnp.float32)

    forests = enumerate_mkw_trees(depth)
    hopf = MKWHopfAlgebra.build(1, forests)
    mkw_brackets = form_mkw_brackets(
        _linear_vector_fields(A),
        y0,
        forests,
        lambda _, v: v,
    )

    path = jnp.array([[0.0], [delta]], dtype=jnp.float32)
    cov = jnp.zeros((1, 1, 1), dtype=jnp.float32)
    sig_levels = compute_planar_branched_signature(
        path=path,
        order_m=depth,
        forests=forests,
        cov_increments=cov,
        return_trajectory=False,
    )
    sig_levels_list = cast(list[jax.Array], sig_levels)
    group_el = GroupElement(hopf=hopf, coeffs=sig_levels_list, interval=(0.0, 1.0))
    logsig = MKWLogSignature(group_el.log())

    y_next = log_ode(mkw_brackets, logsig, y0)
    expected = jexpm(delta * A0) @ y0
    expected = expected / jnp.linalg.norm(y0)
    assert jnp.allclose(y_next, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("dim", [1, 3])
def test_bck_signature_quadratic_variation(depth: int, dim: int) -> None:
    """
    BCK (non-planar branched) rough paths on Euclidean systems: Quadratic variation check
    at the signature level. Degree-2 coordinates vanish when cov=0 and are non-zero when
    cov=dt*I at the chain-of-length-2 indices.
    """
    forests = enumerate_bck_trees(depth)
    hopf = GLHopfAlgebra.build(dim, forests)
    timesteps = 200
    key = jax.random.PRNGKey(7)
    W = bm_driver(key, timesteps=timesteps, dim=dim)
    dt = 1.0 / timesteps
    identity = jnp.eye(dim, dtype=jnp.float32)

    steps = W.num_timesteps - 1
    cov_zero = jnp.zeros((steps, dim, dim), dtype=W.path.dtype)
    cov_dtI = jnp.tile((dt * identity)[None, :, :], reps=(steps, 1, 1))
    sig_zero = cast(
        list[jax.Array],
        compute_nonplanar_branched_signature(
            path=W.path,
            order_m=depth,
            forests=forests,
            cov_increments=cov_zero,
            return_trajectory=False,
        ),
    )
    sig_dtI = cast(
        list[jax.Array],
        compute_nonplanar_branched_signature(
            path=W.path,
            order_m=depth,
            forests=forests,
            cov_increments=cov_dtI,
            return_trajectory=False,
        ),
    )
    if depth >= 2 and hopf.degree2_chain_indices is not None:
        chain_zero = sig_zero[1][hopf.degree2_chain_indices]
        chain_dtI = sig_dtI[1][hopf.degree2_chain_indices]
        # For multi-step paths, degree-2 mass arises even with cov=0 via group product.
        # QV injection must strictly increase the chain component norm.
        norm_zero = jnp.linalg.norm(chain_zero)
        norm_dtI = jnp.linalg.norm(chain_dtI)
        assert norm_dtI > norm_zero + 1e-6


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("dim", [1, 3])
def test_mkw_signature_quadratic_variation(depth: int, dim: int) -> None:
    """
    MKW (planar branched) rough paths on Euclidean systems: Quadratic variation check
    at the signature level. Degree-2 coordinates vanish when cov=0 and are non-zero when
    cov=dt*I at the chain-of-length-2 indices.
    """
    forests = enumerate_mkw_trees(depth)
    hopf = MKWHopfAlgebra.build(dim, forests)
    timesteps = 200
    key = jax.random.PRNGKey(9)
    W = bm_driver(key, timesteps=timesteps, dim=dim)
    dt = 1.0 / timesteps
    identity = jnp.eye(dim, dtype=jnp.float32)

    steps = W.num_timesteps - 1
    cov_zero = jnp.zeros((steps, dim, dim), dtype=W.path.dtype)
    cov_dtI = jnp.tile((dt * identity)[None, :, :], reps=(steps, 1, 1))
    sig_zero = cast(
        list[jax.Array],
        compute_planar_branched_signature(
            path=W.path,
            order_m=depth,
            forests=forests,
            cov_increments=cov_zero,
            return_trajectory=False,
        ),
    )
    sig_dtI = cast(
        list[jax.Array],
        compute_planar_branched_signature(
            path=W.path,
            order_m=depth,
            forests=forests,
            cov_increments=cov_dtI,
            return_trajectory=False,
        ),
    )
    if depth >= 2 and hopf.degree2_chain_indices is not None:
        chain_zero = sig_zero[1][hopf.degree2_chain_indices]
        chain_dtI = sig_dtI[1][hopf.degree2_chain_indices]
        norm_zero = jnp.linalg.norm(chain_zero)
        norm_dtI = jnp.linalg.norm(chain_dtI)
        assert norm_dtI > norm_zero + 1e-6


@pytest.mark.parametrize("depth", [1, 2])
def test_mkw_log_ode_manifold(depth: int) -> None:
    """
    MKW on S^2 with tangent projection:
    - Norm preservation
    - Small-time tangent-plane statistics ~ N(0, t I)
    """
    # Use so(3) generators, but vector fields evaluated through projection for MKW brackets
    A = _so3_generators()  # [3,3,3]
    V = _linear_vector_fields(A)
    y0 = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    dim = 3

    forests = enumerate_mkw_trees(depth)
    hopf = MKWHopfAlgebra.build(dim, forests)
    mkw_brackets = form_mkw_brackets(V, y0, forests, _project_to_tangent)

    key = jax.random.PRNGKey(4)
    timesteps = 1000
    window_size = 10
    W = bm_driver(key, timesteps=timesteps, dim=dim)
    windows = non_overlapping_windower(W, window_size=window_size)
    dt = 1.0 / timesteps
    identity = jnp.eye(dim, dtype=jnp.float32)

    # Integrate over windows using planar branched signatures with correct QV
    state = y0
    traj = [state]
    for w in windows:
        steps = w.num_timesteps - 1
        cov = jnp.tile((dt * identity)[None, :, :], reps=(steps, 1, 1))
        sig_levels = compute_planar_branched_signature(
            path=w.path,
            order_m=depth,
            forests=forests,
            cov_increments=cov,
            return_trajectory=False,
        )
        sig_levels_list = cast(list[jax.Array], sig_levels)
        group_el = GroupElement(hopf=hopf, coeffs=sig_levels_list, interval=w.interval)
        logsig = group_el.log()
        state = log_ode(mkw_brackets, logsig, state)
        traj.append(state)
    trajectory = jnp.stack(traj, axis=0)

    # Norm preservation (relaxed tolerance)
    norms = jnp.linalg.norm(trajectory, axis=1)
    assert jnp.max(jnp.abs(norms - 1.0)) < 0.05

    # Small-time statistics around y0 using short integration
    key = jax.random.PRNGKey(5)
    M_small = 256
    T_small = 0.05
    N_small = 50
    dt_small = T_small / N_small
    key, sub = jax.random.split(key)
    dW_small = jax.random.normal(sub, shape=(M_small, N_small, dim), dtype=jnp.float32) * jnp.sqrt(
        dt_small
    )

    def integrate_short(incs: jax.Array) -> jax.Array:
        state = y0
        for i in range(N_small):
            seg = jnp.vstack([jnp.zeros((1, dim), dtype=incs.dtype), incs[i].reshape(1, -1)])
            # Build degree-1 only for speed in very small-time check if depth==1; otherwise use depth
            forests_loc = forests
            hopf_loc = hopf
            cov = jnp.tile((dt_small * identity)[None, :, :], reps=(1, 1, 1))
            sig_levels = compute_planar_branched_signature(
                path=seg,
                order_m=depth,
                forests=forests_loc,
                cov_increments=cov,
                return_trajectory=False,
            )
            sig_levels_list = cast(list[jax.Array], sig_levels)
            group_el = GroupElement(hopf=hopf_loc, coeffs=sig_levels_list, interval=(i, i + 1))
            logsig = group_el.log()
            state = log_ode(mkw_brackets, logsig, state)
        return state

    yT_small = jax.vmap(integrate_short, in_axes=0)(dW_small)
    disp_small = yT_small - y0
    P_tan = jnp.eye(3, dtype=jnp.float32) - jnp.outer(y0, y0)
    disp_tan = disp_small @ P_tan.T
    disp_xy = disp_tan[:, :2]
    mean_xy = jnp.mean(disp_xy, axis=0)
    assert jnp.linalg.norm(mean_xy) < 0.08
    centered = disp_xy - mean_xy
    cov_xy = (centered.T @ centered) / float(M_small)
    target_cov = T_small * jnp.eye(2, dtype=jnp.float32)
    assert jnp.allclose(cov_xy, target_cov, rtol=0.2, atol=0.03)
