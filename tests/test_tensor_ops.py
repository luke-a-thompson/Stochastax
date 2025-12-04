import jax
import jax.numpy as jnp
import pytest
from stochastax.tensor_ops import (
    restricted_tensor_exp,
    tensor_product,
    seq_tensor_product,
)


@pytest.mark.parametrize(
    "x_shape_suffix, y_shape_suffix, expected_out_suffix",
    [
        ((2,), (3,), (2, 3)),
        ((2, 4), (3,), (2, 4, 3)),  # x is (B, M, K), y is (B, N)
        ((2,), (3, 5), (2, 3, 5)),  # x is (B, M), y is (B, N, L)
        ((1,), (3,), (1, 3)),  # Edge case: M=1
        ((2,), (1,), (2, 1)),  # Edge case: N=1
        ((1,), (1,), (1, 1)),  # Edge case: M=1, N=1
    ],
)
def test_batch_tensor_product_shapes(
    x_shape_suffix: tuple[int, ...],
    y_shape_suffix: tuple[int, ...],
    expected_out_suffix: tuple[int, ...],
) -> None:
    B: int = 4  # Batch size
    key = jax.random.PRNGKey(0)

    x_shape = (B,) + x_shape_suffix
    y_shape = (B,) + y_shape_suffix
    expected_shape = (B,) + expected_out_suffix

    x = jax.random.normal(key, x_shape)
    y = jax.random.normal(key, y_shape)

    result = jax.vmap(tensor_product)(x, y)
    assert result.shape == expected_shape, (
        f"Input x:{x.shape}, y:{y.shape}. Expected {expected_shape}, got {result.shape}"
    )


# Minimal value test for batch_tensor_product
def test_batch_tensor_product_values() -> None:
    B: int = 1
    # These are already batched with B=1
    x_b = jnp.array([[1.0, 2.0]])  # Shape (1, 2)
    y_b = jnp.array([[3.0, 4.0, 5.0]])  # Shape (1, 3)

    # Expected: (1, 2, 3)
    # x[b, i] * y[b, j]
    # result[0,0,0] = x[0,0]*y[0,0] = 1*3 = 3
    # result[0,0,1] = x[0,0]*y[0,1] = 1*4 = 4
    # result[0,0,2] = x[0,0]*y[0,2] = 1*5 = 5
    # result[0,1,0] = x[0,1]*y[0,0] = 2*3 = 6
    # result[0,1,1] = x[0,1]*y[0,1] = 2*4 = 8
    # result[0,1,2] = x[0,1]*y[0,2] = 2*5 = 10
    expected_result = jnp.array([[[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]])

    result = jax.vmap(tensor_product)(x_b, y_b)
    assert result.shape == (B, 2, 3)
    assert jnp.allclose(result, expected_result)

    # Test with more feature dimensions
    # These are already batched with B=1
    x2_b = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    y2_b = jnp.array([[5.0, 6.0]])  # (1, 2)
    # Expected: (1, 2, 2, 2)
    # result[0,i,k,j] = x2[0,i,k] * y2[0,j]
    # result[0,0,0,0] = x2[0,0,0]*y2[0,0] = 1*5=5
    # result[0,0,0,1] = x2[0,0,0]*y2[0,1] = 1*6=6
    expected_result2 = jnp.array([[[[5.0, 6.0], [10.0, 12.0]], [[15.0, 18.0], [20.0, 24.0]]]])
    result2 = jax.vmap(tensor_product)(x2_b, y2_b)
    assert result2.shape == (B, 2, 2, 2)
    assert jnp.allclose(result2, expected_result2)


@pytest.mark.parametrize(
    "x_shape_suffix, y_shape_suffix, expected_out_suffix",
    [
        ((2,), (3,), (2, 3)),  # x:(B,S,M), y:(B,S,N) -> (B,S,M,N)
        ((2, 4), (3,), (2, 4, 3)),  # x:(B,S,M,K), y:(B,S,N) -> (B,S,M,K,N)
        ((2,), (3, 5), (2, 3, 5)),  # x:(B,S,M), y:(B,S,N,L) -> (B,S,M,N,L)
        ((1,), (3,), (1, 3)),  # Edge: M=1
        ((2,), (1,), (2, 1)),  # Edge: N=1
        ((1,), (1,), (1, 1)),  # Edge: M=1, N=1
    ],
)
def test_batch_seq_tensor_product_shapes(
    x_shape_suffix: tuple[int, ...],
    y_shape_suffix: tuple[int, ...],
    expected_out_suffix: tuple[int, ...],
) -> None:
    B: int = 3  # Batch size
    S: int = 5  # Sequence length
    key = jax.random.PRNGKey(1)

    x_shape = (B, S) + x_shape_suffix
    y_shape = (B, S) + y_shape_suffix
    expected_shape = (B, S) + expected_out_suffix

    x = jax.random.normal(key, x_shape)
    y = jax.random.normal(key, y_shape)

    result = jax.vmap(seq_tensor_product)(x, y)
    assert result.shape == expected_shape, (
        f"Input x:{x.shape}, y:{y.shape}. Expected {expected_shape}, got {result.shape}"
    )


def test_batch_seq_tensor_product_values() -> None:
    B: int = 1
    S: int = 1
    # x_b_s: (B, S, M) = (1,1,2)
    x_b_s = jnp.array([[[1.0, 2.0]]])
    # y_b_s: (B, S, N) = (1,1,3)
    y_b_s = jnp.array([[[3.0, 4.0, 5.0]]])

    # Expected: (B, S, M, N) = (1,1,2,3)
    expected_result = jnp.array([[[[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]]])

    result = jax.vmap(seq_tensor_product)(x_b_s, y_b_s)
    assert result.shape == (B, S, 2, 3)
    assert jnp.allclose(result, expected_result)

    # Test with more feature dimensions and S > 1
    B2: int = 1
    S2: int = 2
    # x2_b_s: (B2, S2, M, K) = (1, 2, 2, 2)
    x2_b_s = jnp.array([[[[1.0, 2.0], [3.0, 4.0]], [[1.1, 2.1], [3.1, 4.1]]]])
    # y2_b_s: (B2, S2, N) = (1, 2, 2)
    y2_b_s = jnp.array([[[5.0, 6.0], [5.1, 6.1]]])

    # Manually construct expected output by applying vmap(tensor_product) per sequence element
    # and then stacking. This also indirectly tests tensor_product logic.
    # x2_b_s[:, 0, :, :] has shape (B2, M, K)
    # y2_b_s[:, 0, :] has shape (B2, N)
    expected_s0 = jax.vmap(tensor_product)(
        x2_b_s[:, 0, :, :], y2_b_s[:, 0, :]
    )  # Shape (B2, M, K, N)
    expected_s1 = jax.vmap(tensor_product)(
        x2_b_s[:, 1, :, :], y2_b_s[:, 1, :]
    )  # Shape (B2, M, K, N)
    expected_result2 = jnp.stack([expected_s0, expected_s1], axis=1)  # Shape (B2, S2, M, K, N)

    result2 = jax.vmap(seq_tensor_product)(x2_b_s, y2_b_s)
    assert result2.shape == (B2, S2, 2, 2, 2), f"Expected {(B2, S2, 2, 2, 2)}, got {result2.shape}"
    assert jnp.allclose(result2, expected_result2)


@pytest.mark.parametrize(
    "depth, n_features",
    [
        (1, 3),
        (2, 2),
        (3, 2),  # A common case
        (5, 1),  # Edge case: n_features = 1
    ],
)
def test_batch_restricted_tensor_exp_output_structure(depth: int, n_features: int) -> None:
    B: int = 2
    key = jax.random.PRNGKey(depth + n_features)
    # x_b is batched input for vmap
    x_b: jax.Array = jax.random.normal(key, shape=(B, n_features))

    # restricted_tensor_exp takes unbatched x, depth is static
    # Output is a list of batched tensors
    result_list = jax.vmap(restricted_tensor_exp, in_axes=(0, None))(x_b, depth)

    assert len(result_list) == depth, f"Expected tuple of length {depth}, got {len(result_list)}"

    for k_idx, term in enumerate(result_list):
        order = k_idx + 1
        expected_term_shape = (B,) + (n_features,) * order
        assert term.shape == expected_term_shape, (
            f"Term {k_idx} (order {order}) has shape {term.shape}, expected {expected_term_shape}"
        )


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_batch_restricted_tensor_exp_values(depth: int) -> None:
    # Batched input (B=1, n_features=2)
    x_val_b = jnp.array([[1.0, 2.0]])

    # Compute result terms up to the given depth
    result_terms = jax.vmap(restricted_tensor_exp, in_axes=(0, None))(x_val_b, depth)
    assert len(result_terms) == depth

    # Build expected terms generically: T1 = x, Tk = Tk-1 ⊗ (x / k)
    expected_terms: list[jax.Array] = []
    for k in range(1, depth + 1):
        if k == 1:
            expected_terms.append(x_val_b)
        else:
            expected_terms.append(jax.vmap(tensor_product)(expected_terms[-1], x_val_b / float(k)))

    for k in range(depth):
        assert jnp.allclose(result_terms[k], expected_terms[k])


# @pytest.mark.parametrize(
#     "x_terms_defs, y_terms_defs, depth, n_features",
#     [
#         # Case 1: Basic - X=(X1), Y=(Y1), depth=2. Expect Z=(0, X1⊗Y1)
#         (
#             [(1,)],
#             [(1,)],
#             2,
#             2,
#         ),  # x_terms: one term of order 1  # y_terms: one term of order 1  # depth, n_features
#         # Case 2: X=(X1,X2), Y=(Y1), depth=3. Expect Z=(0, X1⊗Y1, X2⊗Y1)
#         ([(1,), (2,)], [(1,)], 3, 2),  # x_terms: order 1, order 2  # y_terms: order 1
#         # Case 3: X=(X1), Y=(Y1,Y2), depth=3. Expect Z=(0, X1⊗Y1, X1⊗Y2)
#         ([(1,)], [(1,), (2,)], 3, 2),
#         # Case 4: X=(X1,X2), Y=(Y1,Y2), depth=3. Expect Z=(0, X1⊗Y1, X1⊗Y2 + X2⊗Y1)
#         ([(1,), (2,)], [(1,), (2,)], 3, 2),
#         # Case 5: Deeper - X=(X1,X2), Y=(Y1,Y2), depth=4
#         # Z1=0, Z2=X1Y1, Z3=X1Y2+X2Y1, Z4=X2Y2
#         ([(1,), (2,)], [(1,), (2,)], 4, 2),
#         # Case 6: Truncation by depth - X=(X1,X2,X3), Y=(Y1), depth=2. Expect Z=(0, X1Y1) (X2Y1, X3Y1 ignored)
#         ([(1,), (2,), (3,)], [(1,)], 2, 2),
#         # Case 7: n_features = 1
#         ([(1,)], [(1,)], 2, 1),
#     ],
# )
# def test_batch_cauchy_prod_logic(
#     x_terms_defs: list[tuple[int, ...]],
#     y_terms_defs: list[tuple[int, ...]],
#     depth: int,
#     n_features: int,
# ) -> None:
#     B: int = 2
#     key = jax.random.PRNGKey(
#         sum(d[0] for d in x_terms_defs) + sum(d[0] for d in y_terms_defs) + depth + n_features
#     )

#     def _create_terms(
#         term_defs: list[tuple[int, ...]], current_key: jax.Array, batch_size: int
#     ) -> list[jax.Array]:
#         terms = []
#         for i, order_def in enumerate(term_defs):
#             order = order_def[0]  # Assuming the first element in tuple is the order indicator
#             current_key, subkey = jax.random.split(current_key)
#             # Create batched terms directly
#             term_shape = (batch_size,) + (n_features,) * order
#             # Small integer values for easier debugging if needed, scaled by order
#             terms.append(jax.random.uniform(subkey, term_shape, minval=1, maxval=3) * order)
#         return terms

#     key, x_key, y_key, s_key = jax.random.split(key, 4)
#     # x_terms and y_terms are lists of batched tensors
#     x_terms: list[jax.Array] = _create_terms(x_terms_defs, x_key, B)
#     y_terms: list[jax.Array] = _create_terms(y_terms_defs, y_key, B)

#     # Expected shapes derived directly from unflattened levels
#     def _expected_shape_for_level(k_idx: int) -> tuple[int, ...]:
#         # k_idx is 0-based index (order = k_idx + 1)
#         return (B,) + (n_features,) * (k_idx + 1)

#     # Calculate expected output manually (already batched)
#     expected_out_terms: list[jax.Array] = [
#         jnp.zeros(_expected_shape_for_level(k)) for k in range(depth)
#     ]

#     # Loop over output orders (k_out for Z^{(k_out+1)})
#     # Z order k_out+1. This means k_out index in expected_out_terms.
#     for k_out in range(depth):  # k_out from 0 to depth-1
#         target_order_Z = k_out + 1  # order of Z term we are computing
#         current_sum_for_order_Zk = jnp.zeros(_expected_shape_for_level(k_out))

#         # Sum over X_i Y_j where order(X_i) + order(Y_j) = target_order_Z
#         for i_x_term_idx in range(len(x_terms)):
#             x_term_order = x_terms_defs[i_x_term_idx][0]  # Actual order of X term, e.g. 1, 2...

#             for j_y_term_idx in range(len(y_terms)):
#                 y_term_order = y_terms_defs[j_y_term_idx][0]  # Actual order of Y term

#                 if x_term_order + y_term_order == target_order_Z:
#                     # x_terms[i_x_term_idx] is (B, feat_x)
#                     # y_terms[j_y_term_idx] is (B, feat_y)
#                     # vmap tensor_product over these batched tensors
#                     prod = jax.vmap(tensor_product)(x_terms[i_x_term_idx], y_terms[j_y_term_idx])
#                     current_sum_for_order_Zk += prod
#         if target_order_Z > 0:  # Z_0 is always zero in this context (not computed by cauchy_prod)
#             expected_out_terms[k_out] = current_sum_for_order_Zk

#     # cauchy_prod expects lists of unbatched tensors.
#     # vmap handles the batching. in_axes=(0,0,None) means:
#     #   - x_terms: each tensor in list is unstacked at axis 0
#     #   - y_terms: each tensor in list is unstacked at axis 0
#     #   - depth: static
#     result_terms = jax.vmap(cauchy_convolution, in_axes=(0, 0, None))(x_terms, y_terms, depth)

#     assert len(result_terms) == depth
#     for i in range(depth):
#         expected_shape_i = _expected_shape_for_level(i)
#         assert result_terms[i].shape == expected_shape_i, (
#             f"Term {i} shape mismatch. Expected {expected_shape_i}, got {result_terms[i].shape}"
#         )
#         # Check if expected_out_terms[i] is non-zero or if result_terms[i] is also zero
#         # This handles cases where an order might not be produced (e.g. Z1 for X=(X2), Y=(Y2))
#         if jnp.any(expected_out_terms[i] != 0) or jnp.any(result_terms[i] != 0):
#             assert jnp.allclose(result_terms[i], expected_out_terms[i], atol=1e-5), (
#                 f"Term {i} (order {i + 1}) mismatch. Got:\n{result_terms[i]}\nExpected:\n{expected_out_terms[i]}"
#             )
#         else:  # Both are zero, which is fine
#             pass
