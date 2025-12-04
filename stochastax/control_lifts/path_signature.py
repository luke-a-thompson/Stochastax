import jax
import jax.numpy as jnp
from stochastax.tensor_ops import restricted_tensor_exp, seq_tensor_product
from typing import Literal, overload
from functools import partial
from stochastax.hopf_algebras.hopf_algebras import ShuffleHopfAlgebra
from stochastax.control_lifts.signature_types import PathSignature
from stochastax.hopf_algebras.elements import GroupElement


def _compute_incremental_levels(path_increments: jax.Array, depth: int) -> list[jax.Array]:
    """The core of signature computation from KerasSig."""
    depth_1_stream = jnp.cumsum(path_increments, axis=0)
    incremental_signatures: list[jax.Array] = [depth_1_stream]

    first_inc_tensor_exp_terms = restricted_tensor_exp(path_increments[0, :], depth=depth)

    divisors = jnp.arange(2, depth + 1, dtype=path_increments.dtype).reshape(depth - 1, 1, 1)
    path_increment_divided = jnp.expand_dims(path_increments, axis=0) / divisors

    for k in range(1, depth):
        sig_accm = incremental_signatures[0][:-1, :] + path_increment_divided[k - 1, 1:, :]

        for j in range(k - 1):
            prev_signature_level_term = incremental_signatures[j + 1][:-1, :]
            scaled_increment = path_increment_divided[k - j - 2, 1:, :]
            sig_accm = prev_signature_level_term + seq_tensor_product(sig_accm, scaled_increment)

        sig_accm = seq_tensor_product(sig_accm, path_increments[1:, :])

        first_inc_expanded = jnp.expand_dims(first_inc_tensor_exp_terms[k], axis=0)
        sig_accm = jnp.concatenate([first_inc_expanded, sig_accm], axis=0)

        incremental_signatures.append(jnp.cumsum(sig_accm, axis=0))

    return incremental_signatures


@overload
def compute_path_signature(
    path: jax.Array,
    depth: int,
    hopf: ShuffleHopfAlgebra,
    mode: Literal["full"],
    index_start: int = 0,
) -> PathSignature: ...


@overload
def compute_path_signature(
    path: jax.Array,
    depth: int,
    hopf: ShuffleHopfAlgebra,
    mode: Literal["stream", "incremental"],
    index_start: int = 0,
) -> list[PathSignature]: ...


@partial(jax.jit, static_argnames=["hopf", "depth", "mode", "index_start"])
def compute_path_signature(
    path: jax.Array,
    depth: int,
    hopf: ShuffleHopfAlgebra,
    mode: Literal["full", "stream", "incremental"],
    index_start: int = 0,
) -> PathSignature | list[PathSignature]:
    r"""Computes the truncated path signature
    $$\operatorname{Sig}_{0,T}(X)=\bigl(S^{(1)}_{0,T},\,S^{(2)}_{0,T},\ldots,S^{(m)}_{0,T}\bigr),\qquad m=\text{depth}.$$
    The constant term $$S^{(0)}_{0,T}=1$$ is omitted.
    """
    assert depth > 0 and isinstance(depth, int), f"Depth must be a positive integer, got {depth}."
    if path.ndim == 1:
        raise ValueError(
            f"QuickSig requires 2D arrays of shape [seq_len, n_features]. Got shape: {path.shape}. \n Consider using path.reshape(-1, 1)."
        )
    # Extract shape info before heavy computation
    seq_len = int(path.shape[0])
    n_features = int(path.shape[1])
    if hopf.ambient_dimension != n_features:
        raise ValueError(
            f"Mismatch between hopf ambient dimension ({hopf.ambient_dimension}) and path features ({n_features})."
        )
    if hopf.max_degree != depth:
        raise ValueError(
            f"Mismatch between hopf.max_degree ({hopf.max_degree}) and depth argument ({depth})."
        )

    if seq_len <= 1:
        if mode == "full":
            zero_terms = [
                jnp.zeros((n_features ** (i + 1),), dtype=path.dtype) for i in range(depth)
            ]
            return PathSignature(
                GroupElement(
                    hopf=hopf,
                    coeffs=zero_terms,
                    interval=(index_start, index_start + max(seq_len - 1, 0)),
                )
            )
        elif mode in ("stream", "incremental"):
            return []
        else:
            raise ValueError(f"Invalid mode: {mode}")

    path_increments = path[1:, :] - path[:-1, :]

    match mode:
        case "incremental":
            return [
                PathSignature(
                    GroupElement(
                        hopf=hopf,
                        coeffs=[
                            term.reshape(-1)
                            for term in restricted_tensor_exp(path_increments[i, :], depth=depth)
                        ],
                        interval=(index_start + i, index_start + i + 1),
                    )
                )
                for i in range(path_increments.shape[0])
            ]
        case "full":
            incremental_signatures = _compute_incremental_levels(path_increments, depth)
            final_levels = [jnp.ravel(c[-1]) for c in incremental_signatures]
            return PathSignature(
                GroupElement(
                    hopf=hopf,
                    coeffs=final_levels,
                    interval=(index_start, index_start + path.shape[0] - 1),
                )
            )
        case "stream":
            incremental_signatures = _compute_incremental_levels(path_increments, depth)
            return [
                PathSignature(
                    GroupElement(
                        hopf=hopf,
                        coeffs=[jnp.ravel(term[i, :]) for term in incremental_signatures],
                        interval=(index_start, index_start + i + 1),
                    )
                )
                for i in range(path_increments.shape[0])
            ]
        case _:
            raise ValueError(f"Invalid mode: {mode}")
