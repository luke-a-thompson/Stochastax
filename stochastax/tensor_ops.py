import jax
import jax.numpy as jnp


def tensor_product(x: jax.Array, y: jax.Array) -> jax.Array:
    """Computes the outer product of two tensors.

    This is a pure feature-space operation. It does not align batch dimensions.
    For inputs `x` with shape `A` and `y` with shape `B`, the result has shape `A + B`.

    Args:
        x (jax.Array): The first tensor.
        y (jax.Array): The second tensor.

    Returns:
        jax.Array: The tensor product of x and y.
    """
    x_bcast = jnp.reshape(x, x.shape + (1,) * y.ndim)  # Make col vector
    y_bcast = jnp.reshape(y, (1,) * x.ndim + y.shape)  # Make row vector
    return x_bcast * y_bcast


def seq_tensor_product(x: jax.Array, y: jax.Array) -> jax.Array:
    """
    Outer product of the trailing dimensions while preserving the
    leading sequence axis.

    Shapes
    -------
    x : (S, *A)          # *A is any tuple of ≥1 dims
    y : (S, *B_)         # *B_ is any tuple of ≥1 dims

    Returns
    --------
    (S, *A, *B_)
    """
    # trailing ranks
    a_rank = x.ndim - 1
    b_rank = y.ndim - 1

    # add singleton axes **once** instead of in a Python loop
    x_bcast = jnp.reshape(x, x.shape + (1,) * b_rank)
    y_bcast = jnp.reshape(
        y, y.shape[:1] + (1,) * a_rank + y.shape[1:]
    )  # Only take first dim (sequence)

    return x_bcast * y_bcast


def restricted_tensor_exp(x: jax.Array, depth: int) -> list[jax.Array]:
    r"""
    Return the truncated tensor-exponential terms

    $$\exp(x) = \sum_{k=0}^{\infty} \frac{x^{\otimes k}}{k!}$$

    Args:
        x: ArrayLike shape (..., n)
        depth: int. The truncation order of the restricted tensor exponential, usually denoted k in literature.

    Returns:
        A tuple of length max_order, where the k-th entry is $$x^{⊗(k+1)}/(k+1)!$$.

        terms[k-1] is the k-th order term $$\frac{x^{\otimes k}}{k!}$$ so has shape `(..., n, n, …, n)` with `k` copies of the last dimension.
    """
    terms = [x]
    for k in range(1, depth):
        divisor = k + 1
        next_factor = x / divisor
        next_power = tensor_product(terms[-1], next_factor)  # $$x^{\otimes (k+1)} / (k+1)!$$
        terms.append(next_power)
    return terms
