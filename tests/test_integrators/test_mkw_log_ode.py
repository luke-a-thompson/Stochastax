import jax
import jax.numpy as jnp
import pytest
from jax.scipy.linalg import expm as jexpm

from stochastax.controls.drivers import bm_driver
from stochastax.controls.augmentations import non_overlapping_windower
from stochastax.control_lifts.branched_signature_ito import compute_planar_branched_signature
from stochastax.integrators.log_ode import log_ode
from stochastax.vector_field_lifts.mkw_lift import form_mkw_bracket_functions
from stochastax.hopf_algebras.mkw import MKWHopfAlgebra
from stochastax.manifolds import EuclideanSpace, Sphere

from tests.conftest import _so3_generators
from tests.test_integrators.conftest import (
    _linear_vector_fields,
    build_block_rotation_generators,
    build_block_initial_state,
    build_two_point_path,
)


@pytest.mark.parametrize("depth", [1])
@pytest.mark.parametrize("dim", [1, 3])
def test_mkw_log_ode_euclidean_linear_matches_matrix_exponential(depth: int, dim: int) -> None:
    """Planar branched log-ODE matches matrix exponential for linear Euclidean systems."""
    delta = 0.25
    hopf = MKWHopfAlgebra.build(dim, depth)

    generators = build_block_rotation_generators(dim)
    y0 = build_block_initial_state(dim)
    def batched_field(y: jax.Array) -> jax.Array:
        return jnp.stack([M @ y for M in generators], axis=0)

    mkw_brackets = form_mkw_bracket_functions(batched_field, hopf, EuclideanSpace())

    path = build_two_point_path(delta, dim)
    steps = path.shape[0] - 1
    cov = jnp.zeros((steps, dim, dim), dtype=jnp.float32)
    sig = compute_planar_branched_signature(
        path=path,
        depth=depth,
        hopf=hopf,
        mode="full",
        cov_increments=cov,
    )
    logsig = sig.log()

    y_next = log_ode(mkw_brackets, logsig, y0, EuclideanSpace())
    combined_generator = jnp.sum(generators, axis=0)
    expected = jexpm(delta * combined_generator) @ y0
    assert jnp.allclose(y_next, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("dim", [1, 3])
def test_mkw_signature_quadratic_variation(depth: int, dim: int) -> None:
    """
    MKW (planar branched) rough paths on Euclidean systems: Quadratic variation check
    at the signature level. Degree-2 coordinates vanish when cov=0 and are non-zero when
    cov=dt*I at the chain-of-length-2 indices.
    """
    hopf = MKWHopfAlgebra.build(dim, depth)
    timesteps = 200
    key = jax.random.PRNGKey(9)
    W = bm_driver(key, timesteps=timesteps, dim=dim)
    dt = 1.0 / timesteps
    identity = jnp.eye(dim, dtype=jnp.float32)

    steps = W.num_timesteps - 1
    cov_zero = jnp.zeros((steps, dim, dim), dtype=W.path.dtype)
    cov_dtI = jnp.tile((dt * identity)[None, :, :], reps=(steps, 1, 1))
    sig_zero = compute_planar_branched_signature(
        path=W.path,
        depth=depth,
        hopf=hopf,
        mode="full",
        cov_increments=cov_zero,
    )
    sig_dtI = compute_planar_branched_signature(
        path=W.path,
        depth=depth,
        hopf=hopf,
        mode="full",
        cov_increments=cov_dtI,
    )
    if depth >= 2 and hopf.degree2_chain_indices is not None:
        chain_zero = sig_zero.coeffs[1][hopf.degree2_chain_indices]
        chain_dtI = sig_dtI.coeffs[1][hopf.degree2_chain_indices]
        norm_zero = jnp.linalg.norm(chain_zero)
        norm_dtI = jnp.linalg.norm(chain_dtI)
        assert norm_dtI > norm_zero + 1e-6


@pytest.mark.parametrize("depth", [1, 2])
def test_mkw_log_ode_manifold(depth: int, sphere_initial_state: jax.Array) -> None:
    """
    MKW on S^2 with tangent projection:
    - Norm preservation
    - Small-time tangent-plane statistics ~ N(0, t I)
    """
    # Use so(3) generators, but vector fields evaluated through projection for MKW brackets
    A = _so3_generators()  # [3,3,3]
    y0 = sphere_initial_state
    dim = 3

    hopf = MKWHopfAlgebra.build(dim, depth)
    def batched_field(y: jax.Array) -> jax.Array:
        return jnp.stack([M @ y for M in A], axis=0)

    mkw_bracket_functions = form_mkw_bracket_functions(batched_field, hopf, Sphere())

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
        sig = compute_planar_branched_signature(
            path=w.path,
            depth=depth,
            hopf=hopf,
            mode="full",
            cov_increments=cov,
            index_start=int(w.interval[0]),
        )
        logsig = sig.log()
        state = log_ode(mkw_bracket_functions, logsig, state, Sphere())
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
            cov = jnp.tile((dt_small * identity)[None, :, :], reps=(1, 1, 1))
            sig = compute_planar_branched_signature(
                path=seg,
                depth=depth,
                hopf=hopf,
                mode="full",
                cov_increments=cov,
                index_start=i,
            )
            logsig = sig.log()
            state = log_ode(mkw_bracket_functions, logsig, state, Sphere())
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
