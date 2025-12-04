from functools import partial
import jax
from stochastax.control_lifts.path_signature import compute_path_signature
from typing import Literal, overload
import jax.numpy as jnp
from stochastax.hopf_algebras.free_lie import enumerate_lyndon_basis
from stochastax.hopf_algebras.elements import LieElement
from stochastax.control_lifts.signature_types import PathSignature, LogSignature
from stochastax.hopf_algebras.hopf_algebras import ShuffleHopfAlgebra


@overload
def compute_log_signature(
    path: jax.Array,
    depth: int,
    hopf: ShuffleHopfAlgebra,
    log_signature_type: Literal["Tensor words", "Lyndon words"],
    mode: Literal["full"],
) -> LogSignature: ...


@overload
def compute_log_signature(
    path: jax.Array,
    depth: int,
    hopf: ShuffleHopfAlgebra,
    log_signature_type: Literal["Tensor words", "Lyndon words"],
    mode: Literal["stream", "incremental"],
) -> list[LogSignature]: ...


@partial(jax.jit, static_argnames=["hopf", "depth", "log_signature_type", "mode"])
def compute_log_signature(
    path: jax.Array,
    depth: int,
    hopf: ShuffleHopfAlgebra,
    log_signature_type: Literal["Tensor words", "Lyndon words"],
    mode: Literal["full", "stream", "incremental"],
) -> LogSignature | list[LogSignature]:
    n_features = path.shape[-1]
    if hopf.ambient_dimension != n_features:
        raise ValueError(
            f"Mismatch between hopf ambient dimension ({hopf.ambient_dimension}) and path features ({n_features})."
        )
    if hopf.max_degree != depth:
        raise ValueError(
            f"Mismatch between hopf.max_degree ({hopf.max_degree}) and depth argument ({depth})."
        )
    signature_result = compute_path_signature(path, depth, hopf, mode=mode)

    def _group_to_lie(group_el: PathSignature) -> LogSignature:
        lie_el = group_el.log()
        if log_signature_type == "Tensor words":
            return lie_el
        if log_signature_type == "Lyndon words":
            basis = enumerate_lyndon_basis(depth, n_features)
            # reshape each level to expanded tensor shape, then compress
            expanded = [
                coeff.reshape((n_features,) * (i + 1)) for i, coeff in enumerate(lie_el.coeffs)
            ]
            compressed = compress(expanded, basis)
            return LogSignature(
                LieElement(hopf=lie_el.hopf, coeffs=compressed, interval=lie_el.interval)
            )
        raise ValueError(f"Invalid log signature type: {log_signature_type}")

    if isinstance(signature_result, list):
        return [LogSignature(_group_to_lie(sig)) for sig in signature_result]
    return LogSignature(_group_to_lie(signature_result))


def index_select(input: jax.Array, indices: jax.Array) -> jax.Array:
    """
    Select entries in m-level tensor based on given indices
    This function will help compressing log-signatures
    """
    if input.ndim == 0:
        return jnp.zeros(indices.shape[0], dtype=input.dtype)

    dim_first_axis = input.shape[0]
    ndim_input_tensor = input.ndim
    n_components_in_indices = indices.shape[1]

    if n_components_in_indices > ndim_input_tensor:
        return jnp.zeros(indices.shape[0], dtype=input.dtype)

    # Coefficients elsewhere in the code are flattened in C order; match that here
    flattened = input.ravel()

    # In C-order flattening, the last axis varies fastest.
    # For a tensor with shape (d, d, ..., d) and k indices (i0, ..., i_{k-1}),
    # the linear index is: i0*d^{k-1} + i1*d^{k-2} + ... + i_{k-1}*d^{0}
    powers = jnp.arange(n_components_in_indices - 1, -1, -1, dtype=jnp.int32)
    strides_array = dim_first_axis**powers

    def _select(one_index_row: jax.Array) -> jax.Array:
        position = jnp.sum(one_index_row * strides_array)
        return flattened[position]

    return jax.vmap(_select)(indices)


def compress(expanded_terms: list[jax.Array], lyndon_basis: list[jax.Array]) -> list[jax.Array]:
    result_compressed = []
    for term, term_lyndon_basis in zip(expanded_terms, lyndon_basis):
        compressed_term = index_select(term, term_lyndon_basis)
        result_compressed.append(compressed_term)
    return result_compressed
