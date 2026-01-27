import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from stochastax.manifolds.spd import SPDManifold


STRICT_RTOL: float = 1e-5
STRICT_ATOL: float = 1e-6


def _as_bool(x: jax.Array) -> bool:
    return bool(x.item())


def _is_symmetric(a: jax.Array, rtol: float = STRICT_RTOL, atol: float = STRICT_ATOL) -> bool:
    return _as_bool(jnp.allclose(a, jnp.swapaxes(a, -2, -1), rtol=rtol, atol=atol))


def _min_eigval(a: jax.Array) -> float:
    evals = jnp.linalg.eigvalsh(a)
    return float(jnp.min(evals).item())


def _make_spd(key: jax.Array, n: int, eps: float = 0.5) -> jax.Array:
    a = jrandom.normal(key, (n, n), dtype=jnp.float32)
    return a @ jnp.swapaxes(a, -2, -1) + eps * jnp.eye(n, dtype=jnp.float32)


def test_retract_identity() -> None:
    manifold = SPDManifold()
    x = jnp.eye(3, dtype=jnp.float32)
    y = manifold.retract(x)

    assert _is_symmetric(y)
    assert _as_bool(jnp.allclose(y, x, rtol=1e-6, atol=1e-7))
    assert _min_eigval(y) >= manifold.eps


def test_retract_spd_input_is_stable() -> None:
    manifold = SPDManifold()
    key = jrandom.PRNGKey(0)
    x = _make_spd(key, 4, eps=0.25)
    y = manifold.retract(x)

    assert _is_symmetric(y)
    assert _as_bool(jnp.allclose(y, x, rtol=1e-4, atol=1e-5))
    assert _min_eigval(y) >= manifold.eps


def test_retract_clamps_negative_eigenvalues() -> None:
    manifold = SPDManifold()
    x = jnp.diag(jnp.array([-1.0, 0.1, 2.0], dtype=jnp.float32))
    y = manifold.retract(x, eps=1e-4)

    assert _is_symmetric(y)
    assert _min_eigval(y) >= 0.999e-4


def test_retract_batched() -> None:
    manifold = SPDManifold()
    key = jrandom.PRNGKey(1)
    a = jrandom.normal(key, (5, 3, 3), dtype=jnp.float32)
    x = a @ jnp.swapaxes(a, -2, -1) + 0.1 * jnp.eye(3, dtype=jnp.float32)
    y = manifold.retract(x)

    assert y.shape == (5, 3, 3)
    assert _is_symmetric(y)
    assert _min_eigval(y) >= manifold.eps


def test_project_to_tangent_identity() -> None:
    manifold = SPDManifold()
    y = jnp.eye(3, dtype=jnp.float32)
    v = jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=jnp.float32)

    v_tan = manifold.project_to_tangent(y, v)
    v_sym = 0.5 * (v + jnp.swapaxes(v, -2, -1))

    assert _is_symmetric(v_tan)
    assert _as_bool(jnp.allclose(v_tan, v_sym, rtol=1e-6, atol=1e-7))


def test_project_to_tangent_symmetric_is_fixed() -> None:
    manifold = SPDManifold()
    key = jrandom.PRNGKey(2)
    y = _make_spd(key, 4, eps=0.5)
    b = jrandom.normal(key, (4, 4), dtype=jnp.float32)
    v = 0.5 * (b + jnp.swapaxes(b, -2, -1))

    v_tan = manifold.project_to_tangent(y, v)

    assert _is_symmetric(v_tan)
    assert _as_bool(jnp.allclose(v_tan, v, rtol=1e-5, atol=1e-6))


def test_project_to_tangent_batched() -> None:
    manifold = SPDManifold()
    key = jrandom.PRNGKey(3)
    a = jrandom.normal(key, (4, 3, 3), dtype=jnp.float32)
    y = a @ jnp.swapaxes(a, -2, -1) + 0.2 * jnp.eye(3, dtype=jnp.float32)
    v = jrandom.normal(key, (4, 3, 3), dtype=jnp.float32)

    v_tan = manifold.project_to_tangent(y, v)

    assert v_tan.shape == (4, 3, 3)
    assert _is_symmetric(v_tan)


def test_project_to_tangent_invalid_shape() -> None:
    manifold = SPDManifold()
    y = jnp.eye(2, dtype=jnp.float32)
    v = jnp.eye(3, dtype=jnp.float32)

    with pytest.raises(ValueError, match="SPD expects"):
        manifold.project_to_tangent(y, v)
