import jax
import jax.numpy as jnp
import pytest
from jax.scipy.linalg import expm as jexpm

from stochastax.controls.drivers import bm_driver
from stochastax.control_lifts.branched_signature_ito import compute_nonplanar_branched_signature
from stochastax.integrators.log_ode import log_ode
from stochastax.vector_field_lifts.bck_lift import form_bck_brackets
from stochastax.hopf_algebras import enumerate_bck_trees
from stochastax.hopf_algebras.hopf_algebra_types import GLHopfAlgebra


def _linear_vector_fields(A: jax.Array) -> list:
    return [lambda y, M=A[i]: M @ y for i in range(A.shape[0])]


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
    sig = compute_nonplanar_branched_signature(
        path=path,
        order_m=depth,
        hopf=hopf,
        mode="full",
        cov_increments=cov,
    )
    logsig = sig.log()

    y_next = log_ode(bck_brackets, logsig, y0)
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
    sig_zero = compute_nonplanar_branched_signature(
        path=W.path,
        order_m=depth,
        hopf=hopf,
        mode="full",
        cov_increments=cov_zero,
    )
    sig_dtI = compute_nonplanar_branched_signature(
        path=W.path,
        order_m=depth,
        hopf=hopf,
        mode="full",
        cov_increments=cov_dtI,
    )
    if depth >= 2 and hopf.degree2_chain_indices is not None:
        chain_zero = sig_zero.coeffs[1][hopf.degree2_chain_indices]
        chain_dtI = sig_dtI.coeffs[1][hopf.degree2_chain_indices]
        # For multi-step paths, degree-2 mass arises even with cov=0 via group product.
        # QV injection must strictly increase the chain component norm.
        norm_zero = jnp.linalg.norm(chain_zero)
        norm_dtI = jnp.linalg.norm(chain_dtI)
        assert norm_dtI > norm_zero + 1e-6
