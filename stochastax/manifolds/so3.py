from __future__ import annotations

from stochastax.manifolds import Manifold
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp


def so3_from_6d(x: jax.Array, *, eps: float = 1e-7) -> jax.Array:
    """
    Convert a 6D rotation representation into an SO(3) rotation matrix.

    This is the common "6D rotation representation" used in neural networks:
    interpret x = [a, b] with a,b ∈ R^3 and orthonormalize via Gram–Schmidt:
      r1 = normalize(a)
      r2 = normalize(b - <r1,b> r1)
      r3 = r1 × r2
    Return R = [r1 r2 r3] (columns), which is in SO(3) (up to numerical error).

    Args:
        x: Array with shape (..., 6).
        eps: Small constant for numerical stability in normalization.

    Returns:
        Rotation matrix with shape (..., 3, 3).
    """
    if x.shape[-1] != 6:
        raise ValueError(f"so3_from_6d expects (..., 6), got {x.shape}.")

    a = x[..., 0:3]
    b = x[..., 3:6]

    def _normalize(v: jax.Array) -> jax.Array:
        return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + jnp.asarray(eps, v.dtype))

    r1 = _normalize(a)
    b_orth = b - jnp.sum(r1 * b, axis=-1, keepdims=True) * r1
    r2 = _normalize(b_orth)
    r3 = jnp.cross(r1, r2, axis=-1)

    r1 = r1[..., :, None]
    r2 = r2[..., :, None]
    r3 = r3[..., :, None]
    return jnp.concatenate([r1, r2, r3], axis=-1)


@dataclass(frozen=True)
class SO3(Manifold):
    """
    SO(3) represented as rotation matrices R ∈ R^{3×3} with R^T R = I and det(R) = 1.

    Retract options:
      - "polar_express": fast matmul-only polar approximation (often good on GPU)
      - "svd": exact nearest-rotation projection via SVD (slower, default)

    Notes:
      - The Polar Express step here is the degree-5 polynomial iteration with the published
        coefficient schedule + stabilisation tweaks (safety factor + cushioning).
    """

    polar_steps: int = 8
    eps: float = 1e-7

    def from_6d(self, x: jax.Array) -> jax.Array:
        """
        Convenience wrapper around :func:`so3_from_6d` using this manifold's eps.

        Args:
            x: Array with shape (..., 6).

        Returns:
            Rotation matrix with shape (..., 3, 3).
        """
        return so3_from_6d(x, eps=self.eps)

    def retract(
        self,
        x: jax.Array,
        method: Literal["svd", "polar_express"] = "polar_express",
    ) -> jax.Array:
        if x.shape[-2:] != (3, 3):
            raise ValueError(f"SO3 expects (..., 3, 3), got {x.shape}.")

        match method:
            case "svd":
                return self._retract_svd(x)

            case "polar_express":
                # returns an orthogonal matrix; may be in O(3) not SO(3)
                q = self._retract_polar_express(x, steps=self.polar_steps, eps=self.eps)
                # user asked for SO(3): make it principled by falling back to SVD only if needed
                det = jnp.linalg.det(q)
                return jax.lax.cond(
                    jnp.any(det < 0.0),
                    lambda _: self._retract_svd(x),
                    lambda _: q,
                    operand=None,
                )

            case _:
                raise ValueError(f"Unknown method '{method}'. Must be 'svd' or 'polar_express'.")

    def project_to_tangent(self, y: jax.Array, v: jax.Array) -> jax.Array:
        """
        Tangent space at R is { R A : A^T = -A }.
        Orthogonal projection: P_R(V) = R * skew(R^T V),
        where skew(M) = 0.5 (M - M^T).
        """
        if y.shape[-2:] != (3, 3) or v.shape[-2:] != (3, 3):
            raise ValueError(f"SO3 expects (..., 3, 3); got y={y.shape}, v={v.shape}.")

        rt_v = jnp.swapaxes(y, -2, -1) @ v
        skew = 0.5 * (rt_v - jnp.swapaxes(rt_v, -2, -1))
        return y @ skew

    def _retract_svd(self, x: jax.Array) -> jax.Array:
        """
        Nearest rotation (Kabsch/Procrustes): R = U diag(1,1,det(UV^T)) V^T.
        """
        u, _, vt = jnp.linalg.svd(x, full_matrices=False)
        r = u @ vt
        det = jnp.linalg.det(r)

        # If det < 0, flip the last column of U (equivalently multiply by diag(1,1,-1))
        u_fixed = u.at[..., :, 2].set(u[..., :, 2] * jnp.where(det < 0.0, -1.0, 1.0)[..., None])
        r_fixed = u_fixed @ vt
        return r_fixed

    def _retract_polar_express(self, g: jax.Array, steps: int, eps: float) -> jax.Array:
        """
        Degree-5 Polar Express iteration (matmul-only), following the published coefficient
        schedule + stabilisation tweaks (safety factor + cushioning). :contentReference[oaicite:1]{index=1}
        """
        # coeffs_list from the paper's Algorithm 1 python code :contentReference[oaicite:2]{index=2}
        coeffs = jnp.array(
            [
                [8.28721201814563, -23.595886519098837, 17.300387312530933],
                [4.107059111542203, -2.9478499167379106, 0.5448431082926601],
                [3.9486908534822946, -2.908902115962949, 0.5518191394370137],
                [3.3184196573706015, -2.488488024314874, 0.51004894012372],
                [2.300652019954817, -1.6689039845747493, 0.4188073119525673],
                [1.891301407787398, -1.2679958271945868, 0.37680408948524835],
                [1.8750014808534479, -1.2500016453999487, 0.3750001645474248],
                [1.875, -1.25, 0.375],  # repeat thereafter
            ],
            dtype=g.dtype,
        )

        # safety factor scaling (exclude last polynomial) :contentReference[oaicite:3]{index=3}
        safety = jnp.asarray(1.01, dtype=g.dtype)
        coeffs_scaled = coeffs.at[:-1, 0].set(coeffs[:-1, 0] / safety)
        coeffs_scaled = coeffs_scaled.at[:-1, 1].set(coeffs[:-1, 1] / (safety**3))
        coeffs_scaled = coeffs_scaled.at[:-1, 2].set(coeffs[:-1, 2] / (safety**5))

        # cushioning / normalisation :contentReference[oaicite:4]{index=4}
        x = g
        fro = jnp.linalg.norm(x, axis=(-2, -1), keepdims=True)
        x = x / (fro * safety + eps)

        def coeff_at(i: jax.Array) -> jax.Array:
            return jax.lax.cond(
                i < coeffs_scaled.shape[0],
                lambda _: coeffs_scaled[i],
                lambda _: coeffs_scaled[-1],
                operand=None,
            )

        def body(i: jax.Array, x: jax.Array) -> jax.Array:
            a, b, c = coeff_at(i)
            a = jnp.asarray(a, dtype=x.dtype)
            b = jnp.asarray(b, dtype=x.dtype)
            c = jnp.asarray(c, dtype=x.dtype)

            # Right-sided update matches the Polar Express polynomial):
            # X <- X (aI + b (X^T X) + c (X^T X)^2)
            a_mat = jnp.swapaxes(x, -2, -1) @ x  # A = X^T X
            b_mat = b * a_mat + c * (a_mat @ a_mat)  # B = bA + cA^2
            x = a * x + (x @ b_mat)  # X <- aX + bX^3 + cX^5
            return x

        x = jax.lax.fori_loop(0, steps, body, x)
        return x
