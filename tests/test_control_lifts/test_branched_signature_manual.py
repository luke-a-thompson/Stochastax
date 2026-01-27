import numpy as np
import jax.numpy as jnp

from stochastax.control_lifts.branched_signature_ito import (
    compute_nonplanar_branched_signature,
    compute_planar_branched_signature,
)
from stochastax.hopf_algebras.hopf_algebras import GLHopfAlgebra, MKWHopfAlgebra
from stochastax.controls.drivers import bm_driver


def _find_shape_id(hopf: GLHopfAlgebra | MKWHopfAlgebra, degree: int, parent: list[int]) -> int:
    parents = np.asarray(hopf.forests_by_degree[degree - 1].parent)
    target = np.asarray(parent, dtype=np.int32)
    matches = np.all(parents == target[None, :], axis=1)
    where = np.where(matches)[0]
    if where.size != 1:
        raise AssertionError(f"Expected exactly one match for parent={parent}, got {where}.")
    return int(where[0])


def _digits_all_j_index(base: int, degree: int, j: int) -> int:
    # digits are most-significant first: [j, j, ..., j]
    idx = 0
    for _ in range(degree):
        idx = idx * base + j
    return int(idx)


def test_bck_signature_manual_coordinates_depth3_dim1() -> None:
    """
    For a single-step 1D path with increment v and cov=0, the implemented GL/BCK exp/product
    yields only the chain coordinates:
    - level1: v
    - level2 (2-node chain): v^2 / 2!
    - level3 (3-node chain): v^3 / 3!
    and the 3-node bush [-1,0,0] coordinate is 0.
    """
    dim = 1
    depth = 3
    v = jnp.array(0.25, dtype=jnp.float32)
    path = jnp.array([[0.0], [v]], dtype=jnp.float32)
    cov = jnp.zeros((1, dim, dim), dtype=jnp.float32)

    hopf = GLHopfAlgebra.build(dim, depth)
    sig = compute_nonplanar_branched_signature(
        path=path,
        depth=depth,
        hopf=hopf,
        mode="full",
        cov_increments=cov,
    )

    # Shape ids at degree 3.
    chain3 = _find_shape_id(hopf, degree=3, parent=[-1, 0, 1])
    bush3 = _find_shape_id(hopf, degree=3, parent=[-1, 0, 0])

    # Level 1: only one colour.
    assert jnp.allclose(sig.coeffs[0][0], v)
    # Level 2: only one shape (2-node chain), only one colouring.
    assert jnp.allclose(sig.coeffs[1][0], v * v / 2.0)
    # Level 3: chain is v^3/6, bush is 0.
    assert jnp.allclose(sig.coeffs[2][chain3], v * v * v / 6.0)
    assert jnp.allclose(sig.coeffs[2][bush3], 0.0)


def test_mkw_signature_manual_coordinates_depth3_dim1() -> None:
    """
    Same manual coordinate check as GL/BCK, but for planar MKW.
    """
    dim = 1
    depth = 3
    v = jnp.array(0.25, dtype=jnp.float32)
    path = jnp.array([[0.0], [v]], dtype=jnp.float32)
    cov = jnp.zeros((1, dim, dim), dtype=jnp.float32)

    hopf = MKWHopfAlgebra.build(dim, depth)
    sig = compute_planar_branched_signature(
        path=path,
        depth=depth,
        hopf=hopf,
        mode="full",
        cov_increments=cov,
    )

    chain3 = _find_shape_id(hopf, degree=3, parent=[-1, 0, 1])
    bush3 = _find_shape_id(hopf, degree=3, parent=[-1, 0, 0])

    assert jnp.allclose(sig.coeffs[0][0], v)
    assert jnp.allclose(sig.coeffs[1][0], v * v / 2.0)
    assert jnp.allclose(sig.coeffs[2][chain3], v * v * v / 6.0)
    assert jnp.allclose(sig.coeffs[2][bush3], 0.0)


def test_bck_mkw_repeated_colour_coordinate_depth3_dim3() -> None:
    """
    For a single-step path where only component j moves, the "all-j" chain coordinate
    at levels 1..3 should match v^k/k! (cov=0), in both GL/BCK and MKW.
    """
    dim = 3
    depth = 3
    j = 2
    v = jnp.array(0.5, dtype=jnp.float32)
    inc = jnp.zeros((dim,), dtype=jnp.float32).at[j].set(v)
    path = jnp.stack([jnp.zeros((dim,), dtype=jnp.float32), inc], axis=0)
    cov = jnp.zeros((1, dim, dim), dtype=jnp.float32)

    hopf_bck = GLHopfAlgebra.build(dim, depth)
    sig_bck = compute_nonplanar_branched_signature(
        path=path,
        depth=depth,
        hopf=hopf_bck,
        mode="full",
        cov_increments=cov,
    )
    hopf_mkw = MKWHopfAlgebra.build(dim, depth)
    sig_mkw = compute_planar_branched_signature(
        path=path,
        depth=depth,
        hopf=hopf_mkw,
        mode="full",
        cov_increments=cov,
    )

    # Degree-1 indices: shape-major then colour-lexicographic; shape 0 occupies 0..dim-1.
    assert jnp.allclose(sig_bck.coeffs[0][j], v)
    assert jnp.allclose(sig_mkw.coeffs[0][j], v)

    # Degree-2 has only one shape at depth 2 (the chain). Colour index for (j,j) is j*dim + j.
    jj2 = _digits_all_j_index(base=dim, degree=2, j=j)
    assert jnp.allclose(sig_bck.coeffs[1][jj2], v * v / 2.0)
    assert jnp.allclose(sig_mkw.coeffs[1][jj2], v * v / 2.0)

    # Degree-3: pick the chain shape id and then offset by dim^3 for colourings.
    chain3_bck = _find_shape_id(hopf_bck, degree=3, parent=[-1, 0, 1])
    chain3_mkw = _find_shape_id(hopf_mkw, degree=3, parent=[-1, 0, 1])
    jj3 = _digits_all_j_index(base=dim, degree=3, j=j)
    idx_chain_jjj_bck = chain3_bck * (dim**3) + jj3
    idx_chain_jjj_mkw = chain3_mkw * (dim**3) + jj3

    assert jnp.allclose(sig_bck.coeffs[2][idx_chain_jjj_bck], v * v * v / 6.0)
    assert jnp.allclose(sig_mkw.coeffs[2][idx_chain_jjj_mkw], v * v * v / 6.0)


def test_bck_mkw_brownian_covariance_injection_degree2_chain() -> None:
    """
    Brownian quadratic variation check with an actual Brownian sample path.

    We run the same Brownian path twice:
    - cov=0
    - cov=dt*I (the correct quadratic variation for standard Brownian)

    At degree 2, the difference on the chain coordinates should be exactly
    sum_k 0.5 * cov_k = 0.5 * I, because:
    - local_ito_character injects 0.5*cov per step into the chain basis
    - the degree-2 component of the product does not mix with higher degrees
    """
    dim = 3
    depth = 3
    timesteps = 200
    key = jnp.array([7, 0], dtype=jnp.uint32)  # deterministic PRNGKey-like
    W = bm_driver(key, timesteps=timesteps, dim=dim)

    steps = W.num_timesteps - 1
    dt = 1.0 / float(timesteps)
    identity = jnp.eye(dim, dtype=jnp.float32)
    cov_zero = jnp.zeros((steps, dim, dim), dtype=W.path.dtype)
    cov_dtI = jnp.tile((dt * identity)[None, :, :], reps=(steps, 1, 1))
    expected = identity  # since sum(dt) = 1 and we inject Sym(cov)

    hopf_bck = GLHopfAlgebra.build(dim, depth)
    sig_bck_zero = compute_nonplanar_branched_signature(
        path=W.path,
        depth=depth,
        hopf=hopf_bck,
        mode="full",
        cov_increments=cov_zero,
    )
    sig_bck_dtI = compute_nonplanar_branched_signature(
        path=W.path,
        depth=depth,
        hopf=hopf_bck,
        mode="full",
        cov_increments=cov_dtI,
    )
    assert hopf_bck.degree2_chain_indices is not None
    diff_bck = (
        sig_bck_dtI.coeffs[1][hopf_bck.degree2_chain_indices]
        - sig_bck_zero.coeffs[1][hopf_bck.degree2_chain_indices]
    )
    assert jnp.allclose(diff_bck, expected, atol=1e-6, rtol=1e-6)

    hopf_mkw = MKWHopfAlgebra.build(dim, depth)
    sig_mkw_zero = compute_planar_branched_signature(
        path=W.path,
        depth=depth,
        hopf=hopf_mkw,
        mode="full",
        cov_increments=cov_zero,
    )
    sig_mkw_dtI = compute_planar_branched_signature(
        path=W.path,
        depth=depth,
        hopf=hopf_mkw,
        mode="full",
        cov_increments=cov_dtI,
    )
    assert hopf_mkw.degree2_chain_indices is not None
    diff_mkw = (
        sig_mkw_dtI.coeffs[1][hopf_mkw.degree2_chain_indices]
        - sig_mkw_zero.coeffs[1][hopf_mkw.degree2_chain_indices]
    )
    assert jnp.allclose(diff_mkw, expected, atol=1e-6, rtol=1e-6)

