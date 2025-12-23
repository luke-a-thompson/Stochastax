from __future__ import annotations

from typing import Literal, Optional, overload, cast

import jax
import jax.numpy as jnp

from stochastax.hopf_algebras.hopf_algebras import (
    HopfAlgebra,
    GLHopfAlgebra,
    MKWHopfAlgebra,
)
from stochastax.hopf_algebras.elements import GroupElement
from stochastax.control_lifts.signature_types import BCKSignature, MKWSignature


def _zero_coeffs_from_hopf(hopf: HopfAlgebra, depth: int, dtype: jnp.dtype) -> list[jax.Array]:
    return [jnp.zeros((hopf.basis_size(i),), dtype=dtype) for i in range(depth)]


def local_ito_character(
    delta_x: jax.Array,
    cov: Optional[jax.Array],
    order_m: int,
    hopf: GLHopfAlgebra | MKWHopfAlgebra,
    extra: Optional[dict[int, float]] = None,
) -> list[jax.Array]:
    """Build per-step infinitesimal character for Itô branched signature."""
    if order_m != hopf.depth:
        raise ValueError("order_m must equal hopf.depth")
    d = hopf.ambient_dimension
    dtype = delta_x.dtype
    out = _zero_coeffs_from_hopf(hopf, order_m, dtype)

    # Degree 1: single-node tree with colour i gets delta_x[i]
    # We assume shape-major, then colour-lexicographic layout; colours for shape 0 occupy indices 0..d-1.
    out[0] = out[0].at[jnp.arange(d)].set(delta_x)

    # Degree 2: Itô correction on chain-of-length-2
    if order_m >= 2 and cov is not None:
        if hopf.degree2_chain_indices is None:
            raise ValueError("Degree-2 chain indices not available in Hopf algebra.")
        idx = hopf.degree2_chain_indices  # shape (d, d) of flattened indices
        updates = jnp.zeros_like(out[1])
        updates = updates.at[idx].set(cov)
        out[1] = updates

    # Degree >= 3: optional overrides (sparse)
    if extra:
        # The interpretation of keys is left to the caller; no offsets are stored here.
        # For safety, we ignore extras unless future extensions provide per-degree maps.
        pass

    return out


def _branched_signature_ito_impl(
    path: jax.Array,
    order_m: int,
    hopf: GLHopfAlgebra | MKWHopfAlgebra,
    mode: Literal["full", "stream"],
    cov_increments: Optional[jax.Array] = None,
    higher_local_moments: Optional[list[dict[int, float]]] = None,
) -> list[jax.Array] | list[list[jax.Array]]:
    """Compute the Itô branched signature along a sampled path (shared implementation).

    - ``mode="full"``: return the terminal signature levels as a single list of arrays.
    - ``mode="stream"``: return a list over time of signature levels, one entry per increment,
      in the style of ``compute_path_signature(..., mode="stream")``.
    """
    if path.ndim != 2:
        raise ValueError(f"Expected path of shape [T, d], got {path.shape}")
    T, d = path.shape
    if order_m <= 0:
        raise ValueError("order_m must be >= 1")
    if T <= 1:
        dtype = path.dtype
        S0 = _zero_coeffs_from_hopf(hopf, order_m, dtype)
        if mode == "full":
            return S0
        # For an empty or length-1 path, there are no non-trivial prefixes.
        return []

    if hopf.depth != order_m:
        raise ValueError("forests must cover degrees 1..order_m (exact).")

    dtype = path.dtype
    sig = _zero_coeffs_from_hopf(hopf, order_m, dtype)  # unit (tail)

    stream_sig: Optional[list[list[jax.Array]]] = [] if mode == "stream" else None

    for k in range(T - 1):
        delta_x = path[k + 1] - path[k]
        cov = cov_increments[k] if cov_increments is not None else None
        extra = higher_local_moments[k] if higher_local_moments is not None else None
        a_k = local_ito_character(delta_x, cov, order_m, hopf, extra)
        E_k = hopf.exp(a_k)
        sig = hopf.product(sig, E_k)
        if stream_sig is not None:
            stream_sig.append(sig)

    if mode == "full":
        return sig
    assert stream_sig is not None
    return stream_sig


@overload
def compute_planar_branched_signature(
    path: jax.Array,
    order_m: int,
    hopf: MKWHopfAlgebra,
    mode: Literal["full"],
    cov_increments: Optional[jax.Array] = ...,
    higher_local_moments: Optional[list[dict[int, float]]] = ...,
    index_start: int = ...,
) -> MKWSignature: ...


@overload
def compute_planar_branched_signature(
    path: jax.Array,
    order_m: int,
    hopf: MKWHopfAlgebra,
    mode: Literal["stream"],
    cov_increments: Optional[jax.Array] = ...,
    higher_local_moments: Optional[list[dict[int, float]]] = ...,
    index_start: int = ...,
) -> list[MKWSignature]: ...


def compute_planar_branched_signature(
    path: jax.Array,
    order_m: int,
    hopf: MKWHopfAlgebra,
    mode: Literal["full", "stream"],
    cov_increments: Optional[jax.Array] = None,
    higher_local_moments: Optional[list[dict[int, float]]] = None,
    index_start: int = 0,
) -> MKWSignature | list[MKWSignature]:
    """Planar (MKW) Itô branched signature wrapper.

    Returns:
        - ``mode="full"``: a single ``MKWSignature`` for the terminal signature.
        - ``mode="stream"``: a list of ``MKWSignature``, one per prefix.
    """
    if path.ndim != 2:
        raise ValueError(f"Expected path of shape [T, d], got {path.shape}")
    T = path.shape[0]
    result = _branched_signature_ito_impl(
        path=path,
        order_m=order_m,
        hopf=hopf,
        mode=mode,
        cov_increments=cov_increments,
        higher_local_moments=higher_local_moments,
    )
    match mode:
        case "full":
            coeffs = cast(list[jax.Array], result)
            return MKWSignature(
                GroupElement(
                    hopf=hopf, coeffs=coeffs, interval=(index_start, index_start + max(T - 1, 0))
                )
            )
        case "stream":
            stream_coeffs = cast(list[list[jax.Array]], result)
            return [
                MKWSignature(
                    GroupElement(
                        hopf=hopf, coeffs=coeffs, interval=(index_start, index_start + i + 1)
                    )
                )
                for i, coeffs in enumerate(stream_coeffs)
            ]


@overload
def compute_nonplanar_branched_signature(
    path: jax.Array,
    order_m: int,
    hopf: GLHopfAlgebra,
    mode: Literal["full"],
    cov_increments: Optional[jax.Array] = ...,
    higher_local_moments: Optional[list[dict[int, float]]] = ...,
    index_start: int = ...,
) -> BCKSignature: ...


@overload
def compute_nonplanar_branched_signature(
    path: jax.Array,
    order_m: int,
    hopf: GLHopfAlgebra,
    mode: Literal["stream"],
    cov_increments: Optional[jax.Array] = ...,
    higher_local_moments: Optional[list[dict[int, float]]] = ...,
    index_start: int = ...,
) -> list[BCKSignature]: ...


def compute_nonplanar_branched_signature(
    path: jax.Array,
    order_m: int,
    hopf: GLHopfAlgebra,
    mode: Literal["full", "stream"],
    cov_increments: Optional[jax.Array] = None,
    higher_local_moments: Optional[list[dict[int, float]]] = None,
    index_start: int = 0,
) -> BCKSignature | list[BCKSignature]:
    """Nonplanar (BCK/GL) Itô branched signature wrapper.

    Returns:
        - ``mode="full"``: a single ``BCKSignature`` for the terminal signature.
        - ``mode="stream"``: a list of ``BCKSignature``, one per prefix.
    """
    if path.ndim != 2:
        raise ValueError(f"Expected path of shape [T, d], got {path.shape}")
    T = path.shape[0]
    result = _branched_signature_ito_impl(
        path=path,
        order_m=order_m,
        hopf=hopf,
        mode=mode,
        cov_increments=cov_increments,
        higher_local_moments=higher_local_moments,
    )
    match mode:
        case "full":
            coeffs = cast(list[jax.Array], result)
            return BCKSignature(
                GroupElement(
                    hopf=hopf, coeffs=coeffs, interval=(index_start, index_start + max(T - 1, 0))
                )
            )
        case "stream":
            stream_coeffs = cast(list[list[jax.Array]], result)
            return [
                BCKSignature(
                    GroupElement(
                        hopf=hopf, coeffs=coeffs, interval=(index_start, index_start + i + 1)
                    )
                )
                for i, coeffs in enumerate(stream_coeffs)
            ]


if __name__ == "__main__":
    # Minimal example: compare standard tensor signature with branched Itô signature (m=2).
    import jax.numpy as jnp
    from stochastax.control_lifts.path_signature import compute_path_signature
    from stochastax.hopf_algebras.hopf_algebras import GLHopfAlgebra, ShuffleHopfAlgebra

    # Build a simple 2D path
    path = jnp.array(
        [
            [0.0, 0.0],
            [1.0, -0.5],
            [1.7, 0.2],
            [2.0, 0.8],
        ],
        dtype=jnp.float32,
    )
    depth = 2
    d = int(path.shape[1])

    # Standard (shuffle/tensor) signature
    shuffle_hopf = ShuffleHopfAlgebra.build(ambient_dim=d, depth=depth)
    std = compute_path_signature(path, depth=depth, hopf=shuffle_hopf, mode="full")
    std_levels = std.coeffs  # list[jax.Array], levels 1..depth

    # Branched Itô signature on BCK and MKW trees up to degree 2
    bck_hopf = GLHopfAlgebra.build(d, depth)
    mkw_hopf = MKWHopfAlgebra.build(d, depth)
    increments = path[1:, :] - path[:-1, :]
    cov_zero = jnp.zeros((increments.shape[0], d, d), dtype=path.dtype)
    cov_dx_dx = jnp.einsum("td,te->tde", increments, increments)

    ito_zero_sig = compute_nonplanar_branched_signature(
        path=path,
        order_m=depth,
        hopf=bck_hopf,
        mode="full",
        cov_increments=cov_zero,
    )
    ito_dx_dx_sig = compute_planar_branched_signature(
        path=path,
        order_m=depth,
        hopf=mkw_hopf,
        mode="full",
        cov_increments=cov_dx_dx,
    )

    # Compare level-1 (should match total increment)
    lvl1_diff = jnp.linalg.norm(std_levels[0] - ito_zero_sig.coeffs[0])
    print(f"Level-1 difference norm (standard vs branched Itô, cov=0): {float(lvl1_diff):.6f}")

    # Inspect degree-2
    print(f"Standard signature level-2 norm: {float(jnp.linalg.norm(std_levels[1])):.6f}")
    print(
        f"Branched Itô (cov=0) level-2 norm: {float(jnp.linalg.norm(ito_zero_sig.coeffs[1])):.6f}"
    )
    print(
        f"Branched Itô (cov=Δx⊗Δx) level-2 norm: {float(jnp.linalg.norm(ito_dx_dx_sig.coeffs[1])):.6f}"
    )

    # Show chain-of-length-2 matrix entries for the BCK hopf
    bck_hopf = GLHopfAlgebra.build(d, depth)
    if bck_hopf.degree2_chain_indices is not None:
        chain_zero = ito_zero_sig.coeffs[1][bck_hopf.degree2_chain_indices]
        chain_dx_dx = ito_dx_dx_sig.coeffs[1][bck_hopf.degree2_chain_indices]
        print("Branched Itô chain (cov=0):")
        print(chain_zero)
        print("Branched Itô chain (cov=Δx⊗Δx):")
        print(chain_dx_dx)
