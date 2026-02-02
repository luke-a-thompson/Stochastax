import jax
import jax.numpy as jnp

from stochastax.hopf_algebras.gl import GLHopfAlgebra
from stochastax.hopf_algebras.mkw import MKWHopfAlgebra


def _build_deg12_inputs(
    hopf: GLHopfAlgebra | MKWHopfAlgebra, depth: int, key: jax.Array
) -> tuple[jax.Array, jax.Array, list[jax.Array]]:
    key1, key2 = jax.random.split(key)
    x1 = jax.random.normal(key1, (hopf.basis_size(0),), dtype=jnp.float32)
    x2 = (
        jax.random.normal(key2, (hopf.basis_size(1),), dtype=jnp.float32)
        if depth >= 2
        else jnp.zeros((0,), dtype=jnp.float32)
    )
    x = [jnp.zeros((hopf.basis_size(i),), dtype=jnp.float32) for i in range(depth)]
    x[0] = x1
    if depth >= 2:
        x[1] = x2
    return x1, x2, x


def _reference_exp(
    hopf: GLHopfAlgebra | MKWHopfAlgebra, x: list[jax.Array]
) -> list[jax.Array]:
    if len(x) == 0:
        return []
    depth = len(x)
    acc = [jnp.zeros_like(t) for t in x]
    current = [t for t in x]
    factorial = 1.0
    for k in range(1, depth + 1):
        factorial *= float(k)
        acc = [a + (1.0 / factorial) * c for a, c in zip(acc, current)]
        if k < depth:
            ab = hopf.product(current, x)
            current = [u - v - w for u, v, w in zip(ab, current, x)]
    return acc


def _assert_exp_matches_reference(
    hopf: GLHopfAlgebra | MKWHopfAlgebra, depth: int
) -> None:
    key = jax.random.PRNGKey(depth)
    _, _, x = _build_deg12_inputs(hopf, depth, key)
    exp_ref = _reference_exp(hopf, x)
    exp_actual = hopf.exp(x)
    for level_ref, level_actual in zip(exp_ref, exp_actual):
        assert jnp.allclose(level_ref, level_actual, rtol=1e-6, atol=1e-6)


def test_exp_matches_reference_gl() -> None:
    hopf = GLHopfAlgebra.build(ambient_dim=2, depth=5)
    for depth in (2, 3, 4, 5):
        _assert_exp_matches_reference(hopf, depth)


def test_exp_matches_reference_mkw() -> None:
    hopf = MKWHopfAlgebra.build(ambient_dim=2, depth=5)
    for depth in (2, 3, 4, 5):
        _assert_exp_matches_reference(hopf, depth)


def test_exp_jit_and_grad_gl() -> None:
    hopf = GLHopfAlgebra.build(ambient_dim=2, depth=4)
    depth = 4
    key = jax.random.PRNGKey(0)
    x1, x2, _ = _build_deg12_inputs(hopf, depth, key)

    def f(a: jax.Array, b: jax.Array) -> jax.Array:
        x = [jnp.zeros((hopf.basis_size(i),), dtype=a.dtype) for i in range(depth)]
        x[0] = a
        if depth >= 2:
            x[1] = b
        out = hopf.exp(x)
        return sum(jnp.sum(level) for level in out)

    jit_f = jax.jit(f)
    _ = jit_f(x1, x2)
    grad_x1 = jax.grad(f, argnums=0)(x1, x2)
    assert grad_x1.shape == x1.shape
