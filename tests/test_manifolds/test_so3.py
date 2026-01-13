import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from stochastax.manifolds import SO3


STRICT_RTOL: float = 1e-5
STRICT_ATOL: float = 1e-6
POLAR_RTOL: float = 1e-5
POLAR_ATOL: float = 1e-6


def _as_bool(x: jax.Array) -> bool:
    """Convert a scalar JAX boolean array to a Python bool."""
    return bool(x.item())


def _is_orthogonal(R: jax.Array, rtol: float = STRICT_RTOL, atol: float = STRICT_ATOL) -> bool:
    """Check if R^T R = I."""
    n = R.shape[-1]
    RtR = jnp.swapaxes(R, -2, -1) @ R
    identity = jnp.eye(n, dtype=R.dtype)
    return _as_bool(jnp.allclose(RtR, identity, rtol=rtol, atol=atol))


def _is_special_orthogonal(
    R: jax.Array, rtol: float = STRICT_RTOL, atol: float = STRICT_ATOL
) -> bool:
    """Check if R is in SO(3): orthogonal with det(R) = 1."""
    det = jnp.linalg.det(R)
    orth_ok = _is_orthogonal(R, rtol, atol)
    det_ok = _as_bool(jnp.allclose(det, 1.0, rtol=rtol, atol=atol))
    return orth_ok and det_ok


def _so3_diagnostics(R: jax.Array) -> tuple[bool, float, float, float]:
    """
    Return (is_finite, det, max_abs(R^T R - I), fro_norm(R^T R - I)) for a single 3x3 matrix.
    """
    RtR = jnp.swapaxes(R, -2, -1) @ R
    eye = jnp.eye(3, dtype=R.dtype)
    err = RtR - eye
    is_finite = _as_bool(jnp.all(jnp.isfinite(R)))
    det = float(jnp.linalg.det(R).item())
    max_abs_err = float(jnp.max(jnp.abs(err)).item())
    fro_err = float(jnp.linalg.norm(err).item())
    return is_finite, det, max_abs_err, fro_err


def _assert_special_orthogonal(
    R: jax.Array,
    *,
    rtol: float = STRICT_RTOL,
    atol: float = STRICT_ATOL,
    context: str = "R",
) -> None:
    """
    Assert R ∈ SO(3) with a descriptive error message.
    """
    is_finite, det, max_abs_err, fro_err = _so3_diagnostics(R)
    assert is_finite, f"{context}: non-finite entries"

    n = R.shape[-1]
    RtR = jnp.swapaxes(R, -2, -1) @ R
    identity = jnp.eye(n, dtype=R.dtype)
    orth_close = _as_bool(jnp.allclose(RtR, identity, rtol=rtol, atol=atol))
    det_close = _as_bool(jnp.allclose(jnp.asarray(det, dtype=R.dtype), 1.0, rtol=rtol, atol=atol))

    assert orth_close, (
        f"{context}: not orthogonal within rtol={rtol} atol={atol}. "
        f"max|R^T R - I|={max_abs_err:.3e}, ||R^T R - I||_F={fro_err:.3e}, det={det:.8f}"
    )
    assert det_close, (
        f"{context}: det not close to 1 within rtol={rtol} atol={atol}. "
        f"det={det:.8f}, max|R^T R - I|={max_abs_err:.3e}, ||R^T R - I||_F={fro_err:.3e}"
    )


def _is_skew_symmetric(A: jax.Array, rtol: float = STRICT_RTOL, atol: float = STRICT_ATOL) -> bool:
    """Check if A^T = -A."""
    At = jnp.swapaxes(A, -2, -1)
    return _as_bool(jnp.allclose(At, -A, rtol=rtol, atol=atol))


def test_retract_svd_identity() -> None:
    """SVD retraction on identity matrix returns identity."""
    manifold = SO3()
    x = jnp.eye(3, dtype=jnp.float32)
    R = manifold.retract(x, method="svd")

    _assert_special_orthogonal(R, context="svd(identity)")
    assert _as_bool(jnp.allclose(R, x, rtol=1e-6, atol=1e-7))


def test_retract_polar_express_identity() -> None:
    """Polar express retraction on identity matrix returns identity."""
    manifold = SO3()
    x = jnp.eye(3, dtype=jnp.float32)
    R = manifold.retract(x, method="polar_express")

    _assert_special_orthogonal(
        R, rtol=POLAR_RTOL, atol=POLAR_ATOL, context="polar_express(identity)"
    )
    assert _as_bool(jnp.allclose(R, x, rtol=POLAR_RTOL, atol=POLAR_ATOL))


def test_retract_default_method() -> None:
    """Default method is SVD."""
    manifold = SO3()
    x = jnp.eye(3, dtype=jnp.float32)
    R = manifold.retract(x)

    _assert_special_orthogonal(R, context="default_retract(identity)")
    assert _as_bool(jnp.allclose(R, x, rtol=1e-6, atol=1e-7))


def test_retract_svd_near_rotation() -> None:
    """SVD retraction on near-rotation matrix."""
    manifold = SO3()
    key = jrandom.PRNGKey(42)

    # Create a rotation matrix and perturb it slightly
    theta = 0.5
    R_true = jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta), 0.0],
            [jnp.sin(theta), jnp.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    noise = jrandom.normal(key, (3, 3), dtype=jnp.float32) * 0.01
    x = R_true + noise

    R = manifold.retract(x, method="svd")

    _assert_special_orthogonal(R, context="svd(near_rotation)")
    # Should be close to the original rotation
    assert jnp.linalg.norm(R - R_true) < 0.1


def test_retract_polar_express_near_rotation() -> None:
    """Polar express retraction on near-rotation matrix."""
    manifold = SO3()
    key = jrandom.PRNGKey(43)

    # Create a rotation matrix (small angle for better convergence)
    theta = 0.1
    R_true = jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta), 0.0],
            [jnp.sin(theta), jnp.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    # Small perturbation
    noise = jrandom.normal(key, (3, 3), dtype=jnp.float32) * 0.001
    x = R_true + noise

    R = manifold.retract(x, method="polar_express")

    # Check result is approximately orthogonal and close to input
    _assert_special_orthogonal(
        R, rtol=POLAR_RTOL, atol=POLAR_ATOL, context="polar_express(near_rotation)"
    )
    assert jnp.linalg.norm(R - R_true) < 0.05


def test_retract_svd_reflection() -> None:
    """SVD retraction correctly handles reflection (det < 0)."""
    manifold = SO3()

    # Create a reflection matrix: flip x-axis
    reflection = jnp.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32)

    R = manifold.retract(reflection, method="svd")

    _assert_special_orthogonal(R, context="svd(reflection)")
    # Should not return the reflection itself
    det = jnp.linalg.det(R)
    assert _as_bool(jnp.allclose(det, 1.0, rtol=STRICT_RTOL, atol=STRICT_ATOL))


def test_retract_polar_express_reflection() -> None:
    """Polar express correctly falls back to SVD for reflections."""
    manifold = SO3()

    # Create a reflection matrix
    reflection = jnp.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32)

    R = manifold.retract(reflection, method="polar_express")

    _assert_special_orthogonal(R, context="polar_express(reflection)")
    det = jnp.linalg.det(R)
    assert _as_bool(jnp.allclose(det, 1.0, rtol=STRICT_RTOL, atol=STRICT_ATOL))


def test_retract_batched_svd() -> None:
    """SVD retraction works on batched inputs."""
    manifold = SO3()
    key = jrandom.PRNGKey(44)

    batch_size = 5
    # Create batch of near-identity matrices
    x = (
        jnp.eye(3, dtype=jnp.float32)[None, ...]
        + jrandom.normal(key, (batch_size, 3, 3), dtype=jnp.float32) * 0.05
    )

    R = manifold.retract(x, method="svd")

    assert R.shape == (batch_size, 3, 3)
    # Check each matrix in batch
    for i in range(batch_size):
        _assert_special_orthogonal(R[i], context=f"svd(batch)[{i}]")


def test_retract_batched_polar_express() -> None:
    """Polar express retraction works on batched inputs."""
    manifold = SO3()
    key = jrandom.PRNGKey(45)

    batch_size = 5
    x = (
        jnp.eye(3, dtype=jnp.float32)[None, ...]
        + jrandom.normal(key, (batch_size, 3, 3), dtype=jnp.float32) * 0.05
    )

    R = manifold.retract(x, method="polar_express")

    assert R.shape == (batch_size, 3, 3)
    for i in range(batch_size):
        _assert_special_orthogonal(
            R[i], rtol=POLAR_RTOL, atol=POLAR_ATOL, context=f"polar_express(batch)[{i}]"
        )


def test_retract_invalid_method() -> None:
    """Invalid method raises ValueError."""
    manifold = SO3()
    x = jnp.eye(3, dtype=jnp.float32)

    with pytest.raises(ValueError, match="Unknown method"):
        manifold.retract(x, method="invalid")  # type: ignore


def test_retract_invalid_shape() -> None:
    """Invalid shape raises ValueError."""
    manifold = SO3()
    x = jnp.eye(2, dtype=jnp.float32)

    with pytest.raises(ValueError, match="SO3 expects"):
        manifold.retract(x, method="svd")


def test_project_to_tangent_identity() -> None:
    """Tangent projection at identity."""
    manifold = SO3()
    y = jnp.eye(3, dtype=jnp.float32)
    v = jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=jnp.float32)

    v_tan = manifold.project_to_tangent(y, v)

    # Result should be skew-symmetric
    assert _is_skew_symmetric(v_tan)


def test_project_to_tangent_general_point() -> None:
    """Tangent projection at a general rotation matrix."""
    manifold = SO3()

    # Create a rotation matrix
    theta = 0.3
    y = jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta), 0.0],
            [jnp.sin(theta), jnp.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    v = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float32)

    v_tan = manifold.project_to_tangent(y, v)

    # v_tan should be in the tangent space at y
    # For SO(3), tangent space at R is {R * A : A^T = -A}
    # So R^T v_tan should be skew-symmetric
    Rt_v_tan = jnp.swapaxes(y, -2, -1) @ v_tan
    assert _is_skew_symmetric(Rt_v_tan)


def test_project_to_tangent_already_tangent() -> None:
    """Projection of a tangent vector returns itself."""
    manifold = SO3()
    y = jnp.eye(3, dtype=jnp.float32)

    # Create a skew-symmetric matrix
    A = jnp.array([[0.0, -1.0, 2.0], [1.0, 0.0, -3.0], [-2.0, 3.0, 0.0]], dtype=jnp.float32)
    v = y @ A  # v is already in tangent space

    v_tan = manifold.project_to_tangent(y, v)

    # Should be unchanged (up to numerical error)
    assert _as_bool(jnp.allclose(v_tan, v, rtol=1e-6, atol=1e-7))


def test_project_to_tangent_batched() -> None:
    """Tangent projection works on batched inputs."""
    manifold = SO3()
    key = jrandom.PRNGKey(46)

    batch_size = 3
    y = jnp.tile(jnp.eye(3, dtype=jnp.float32)[None, ...], (batch_size, 1, 1))
    v = jrandom.normal(key, (batch_size, 3, 3), dtype=jnp.float32)

    v_tan = manifold.project_to_tangent(y, v)

    assert v_tan.shape == (batch_size, 3, 3)
    for i in range(batch_size):
        assert _is_skew_symmetric(v_tan[i])


def test_project_to_tangent_invalid_shape() -> None:
    """Invalid shape raises ValueError."""
    manifold = SO3()
    y = jnp.eye(2, dtype=jnp.float32)
    v = jnp.eye(2, dtype=jnp.float32)

    with pytest.raises(ValueError, match="SO3 expects"):
        manifold.project_to_tangent(y, v)


def test_from_6d_produces_so3_batched() -> None:
    key = jrandom.PRNGKey(124)
    x6 = jrandom.normal(key, (32, 6), dtype=jnp.float32)
    R = SO3.retract(x6, method="gram_schmidt")
    assert R.shape == (32, 3, 3)
    for i in range(32):
        _assert_special_orthogonal(R[i], context=f"from_6d(batch)[{i}]")


def test_polar_steps_parameter() -> None:
    """Polar steps parameter affects convergence."""
    # Create a matrix far from SO(3)
    key = jrandom.PRNGKey(47)
    x = jrandom.normal(key, (3, 3), dtype=jnp.float32) * 2.0

    manifold_few_steps = SO3(polar_steps=1)
    manifold_many_steps = SO3(polar_steps=10)

    R_few = manifold_few_steps.retract(x, method="polar_express")
    R_many = manifold_many_steps.retract(x, method="polar_express")

    # Both should be in SO(3)
    _assert_special_orthogonal(R_few, rtol=1e-4, atol=1e-5, context="polar_express(steps=1)")
    _assert_special_orthogonal(R_many, rtol=1e-5, atol=1e-6, context="polar_express(steps=10)")

    # More steps should give better accuracy
    error_few = jnp.linalg.norm(jnp.swapaxes(R_few, -2, -1) @ R_few - jnp.eye(3))
    error_many = jnp.linalg.norm(jnp.swapaxes(R_many, -2, -1) @ R_many - jnp.eye(3))
    assert error_many <= error_few


def test_project_to_tangent_jit_compatible() -> None:
    """Tangent projection is JIT-compatible."""
    manifold = SO3()

    @jax.jit
    def project_jitted(y: jax.Array, v: jax.Array) -> jax.Array:
        return manifold.project_to_tangent(y, v)

    y = jnp.eye(3, dtype=jnp.float32)
    v = jnp.ones((3, 3), dtype=jnp.float32)
    v_tan = project_jitted(y, v)

    assert _is_skew_symmetric(v_tan)


@pytest.mark.parametrize("method", ["svd", "polar_express"])
def test_retract_methods_agree_near_identity(method: str) -> None:
    """Different retraction methods give similar results near identity."""
    manifold = SO3()
    key = jrandom.PRNGKey(48)

    # Small perturbation from identity
    x = jnp.eye(3, dtype=jnp.float32) + jrandom.normal(key, (3, 3), dtype=jnp.float32) * 0.01

    R = manifold.retract(x, method=method)  # type: ignore

    # Use looser tolerance for polar express
    if method == "polar_express":
        _assert_special_orthogonal(
            R, rtol=POLAR_RTOL, atol=POLAR_ATOL, context="agree_near_identity(polar_express)"
        )
    else:
        _assert_special_orthogonal(R, context="agree_near_identity(svd)")
    # Should be close to identity
    assert jnp.linalg.norm(R - jnp.eye(3)) < 0.1


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("scale", [0.01, 0.05, 0.1])
def test_retract_polar_express_random_inputs_in_so3(
    seed: int,
    batch_size: int,
    scale: float,
) -> None:
    """Stress test (near-identity): polar_express should produce finite SO(3) matrices."""
    manifold = SO3()
    key = jrandom.PRNGKey(seed)

    shape = (3, 3) if batch_size == 1 else (batch_size, 3, 3)
    noise = jrandom.normal(key, shape, dtype=jnp.float32) * jnp.asarray(scale, dtype=jnp.float32)
    x = (
        jnp.eye(3, dtype=jnp.float32)
        if batch_size == 1
        else jnp.eye(3, dtype=jnp.float32)[None, ...]
    )
    x = x + noise
    R = manifold.retract(x, method="polar_express")

    assert _as_bool(jnp.all(jnp.isfinite(R))), "polar_express(random): non-finite entries"
    if batch_size == 1:
        _assert_special_orthogonal(
            R,
            rtol=POLAR_RTOL,
            atol=POLAR_ATOL,
            context=f"polar_express(random, seed={seed}, scale={scale}, batch=1)",
        )
    else:
        # Check each item to avoid accidental broadcasting mistakes.
        for i in range(batch_size):
            _assert_special_orthogonal(
                R[i],
                rtol=POLAR_RTOL,
                atol=POLAR_ATOL,
                context=f"polar_express(random, seed={seed}, scale={scale}, batch={batch_size})[{i}]",
            )
