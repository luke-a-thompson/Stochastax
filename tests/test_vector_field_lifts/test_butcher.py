import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stochastax.vector_field_lifts.butcher import (
    form_butcher_differentials,
    form_lie_butcher_differentials,
)
from stochastax.hopf_algebras.hopf_algebras import MKWForest, BCKForest
from tests.conftest import (
    forest_from_parents,
    chain_parent,
    binary_root_parent,
    power_matrix,
    linear_field,
    elementwise_square_field,
    identity_projection,
    sphere_tangent_projection,
)


@pytest.mark.parametrize("dim", [1, 2, 4])
def test_butcher_single_node_various_dims(dim: int) -> None:
    # Forest with a single-node tree should return f(x) directly
    key = jax.random.PRNGKey(0 + dim)
    x = jax.random.normal(key, (dim,))
    A = jax.random.normal(jax.random.PRNGKey(10 + dim), (dim, dim)) * 0.1
    f = linear_field(A)

    forest = BCKForest(forest_from_parents([[-1]]))
    out = form_butcher_differentials(f, x, forest)
    np.testing.assert_allclose(np.asarray(out), np.asarray(f(x))[None, :], rtol=1e-10, atol=1e-12)


def test_butcher_linear_chain_depths() -> None:
    # For linear f(x)=A x, a chain with n nodes yields A^n x
    x = jnp.array([0.3, -0.2, 0.5], dtype=jnp.float32)
    A = jnp.array([[0.2, 0.0, -0.1], [0.0, 0.1, 0.0], [0.05, 0.0, 0.15]], dtype=jnp.float32)
    f = linear_field(A)

    for n_nodes in [1, 2, 3, 4]:
        forest = BCKForest(forest_from_parents([chain_parent(n_nodes)]))
        out = form_butcher_differentials(f, x, forest)
        expected = (power_matrix(A, n_nodes) @ x)[None, :]
        np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-7, atol=1e-8)


def test_butcher_linear_binary_root_is_zero() -> None:
    # For linear f, any node with ≥2 children uses D^2 f which is zero → result zero
    dim = 2
    x = jnp.array([0.7, -0.4], dtype=jnp.float32)
    A = jnp.array([[0.3, -0.1], [0.2, 0.05]], dtype=jnp.float32)
    f = linear_field(A)

    forest = BCKForest(forest_from_parents([binary_root_parent()]))
    out = form_butcher_differentials(f, x, forest)
    np.testing.assert_allclose(
        np.asarray(out), np.zeros((1, dim), dtype=np.asarray(out).dtype), rtol=1e-10, atol=1e-12
    )


def test_butcher_nonlinear_scalar_cases() -> None:
    # Scalar field f(x)=x^2:
    # - single node: x^2
    # - chain of length 2: D f(x)[x^2] = 2 x * x^2 = 2 x^3
    # - binary root: D^2 f(x)[x^2, x^2] = 2 x^4
    f = elementwise_square_field()

    x = jnp.array([1.2], dtype=jnp.float32)
    # single node
    forest1 = BCKForest(forest_from_parents([[-1]]))
    out1 = form_butcher_differentials(f, x, forest1)
    np.testing.assert_allclose(
        np.asarray(out1), np.asarray(jnp.square(x))[None, :], rtol=1e-7, atol=1e-8
    )

    # chain length 2
    forest2 = BCKForest(forest_from_parents([chain_parent(2)]))
    out2 = form_butcher_differentials(f, x, forest2)
    expected2 = (2.0 * x**3)[None, :]
    np.testing.assert_allclose(np.asarray(out2), np.asarray(expected2), rtol=1e-7, atol=1e-8)

    # binary root
    forest3 = BCKForest(forest_from_parents([binary_root_parent()]))
    out3 = form_butcher_differentials(f, x, forest3)
    expected3 = (2.0 * x**4)[None, :]
    np.testing.assert_allclose(np.asarray(out3), np.asarray(expected3), rtol=1e-7, atol=1e-8)


def test_butcher_nonlinear_vector_elementwise() -> None:
    # Vector field f(x)=x^2 applied elementwise; check multi-dim binary root and chain
    f = elementwise_square_field()
    x = jnp.array([0.8, -0.6, 0.3], dtype=jnp.float32)

    # chain length 3 → apply D f three times: 4 x^4 elementwise
    forest_chain = BCKForest(forest_from_parents([chain_parent(3)]))
    out_chain = form_butcher_differentials(f, x, forest_chain)
    expected_chain = (4.0 * x**4)[None, :]
    np.testing.assert_allclose(
        np.asarray(out_chain), np.asarray(expected_chain), rtol=1e-6, atol=1e-7
    )

    # binary root → elementwise 2 * (x^2)*(x^2) = 2 x^4
    forest_bin = BCKForest(forest_from_parents([binary_root_parent()]))
    out_bin = form_butcher_differentials(f, x, forest_bin)
    expected_bin = (2.0 * x**4)[None, :]
    np.testing.assert_allclose(np.asarray(out_bin), np.asarray(expected_bin), rtol=1e-6, atol=1e-7)


def test_lie_butcher_identity_projection_matches_butcher() -> None:
    # With identity projection, Lie-Butcher should match Butcher
    x = jnp.array([0.2, 0.5, -0.1], dtype=jnp.float32)
    A = jnp.array([[0.1, 0.0, 0.0], [0.2, -0.05, 0.0], [0.0, 0.0, 0.15]], dtype=jnp.float32)
    f = linear_field(A)

    parents = [chain_parent(3), binary_root_parent(), [-1]]

    project = identity_projection  # type: ignore[assignment]

    # Construct forests per shape (parent length) to respect the Forest contract,
    # then concatenate outputs for comparison.
    outs_lb = []
    outs_b = []
    for parent in parents:
        forest_mkw = MKWForest(forest_from_parents([parent]))
        forest_bck = BCKForest(forest_from_parents([parent]))
        outs_lb.append(form_lie_butcher_differentials(f, x, forest_mkw, project))
        outs_b.append(form_butcher_differentials(f, x, forest_bck))

    out_lb = jnp.concatenate(outs_lb, axis=0)
    out_b = jnp.concatenate(outs_b, axis=0)
    np.testing.assert_allclose(np.asarray(out_lb), np.asarray(out_b), rtol=1e-7, atol=1e-8)


def test_lie_butcher_tangent_projection_property() -> None:
    # Use sphere-like projection: P_y(v) = v - (v·y) y; outputs should be tangent: <out, x> = 0
    x = jnp.array([1.0, 0.0], dtype=jnp.float32)  # unit vector along e1
    f = elementwise_square_field()  # simple nonlinear field

    def project_to_tangent(y: jax.Array, v: jax.Array) -> jax.Array:
        return sphere_tangent_projection(y, v)

    parents = [chain_parent(2), binary_root_parent(), [-1]]

    # Build forests per shape and concatenate outputs to respect Forest's
    # requirement that all trees in a batch share the same number of nodes.
    outs = []
    for parent in parents:
        forest = MKWForest(forest_from_parents([parent]))
        outs.append(form_lie_butcher_differentials(f, x, forest, project_to_tangent))

    out = jnp.concatenate(outs, axis=0)
    out_arr = np.asarray(out)
    # Check tangency for each tree result
    dots = out_arr @ np.asarray(x)
    np.testing.assert_allclose(
        dots, np.zeros((out_arr.shape[0],), dtype=out_arr.dtype), rtol=1e-6, atol=1e-7
    )
