import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Any
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from stochastax.controls.drivers import bm_driver
from stochastax.controls.augmentations import non_overlapping_windower
from stochastax.controls.paths_types import Path
from stochastax.control_lifts.log_signature import compute_log_signature
from stochastax.hopf_algebras.hopf_algebras import ShuffleHopfAlgebra
from stochastax.vector_field_lifts.lie_lift import form_lyndon_bracket_functions
from stochastax.integrators.log_ode import log_ode
from stochastax.manifolds import Sphere


def _so3_generators() -> jax.Array:
    """
    Return a [3, 3, 3] stack of so(3) generators acting on R^3 via matrix multiplication.
    Basis corresponds to cross-product with e1, e2, e3 respectively.
    """
    A1 = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    A2 = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    A3 = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return jnp.stack([A1, A2, A3], axis=0)


def _sphere_mesh(nu: int = 50, nv: int = 50) -> tuple[jax.Array, jax.Array, jax.Array]:
    u = jnp.linspace(0.0, 2.0 * jnp.pi, nu)
    v = jnp.linspace(0.0, jnp.pi, nv)
    uu, vv = jnp.meshgrid(u, v)
    x = jnp.cos(uu) * jnp.sin(vv)
    y = jnp.sin(uu) * jnp.sin(vv)
    z = jnp.cos(vv)
    return x, y, z


def _plot_on_sphere(trajectory: jax.Array) -> None:
    """
    Plot trajectory [T,3] as a line on the unit sphere using Matplotlib (interactive).
    """
    X, Y, Z = _sphere_mesh()
    fig = plt.figure()
    ax: Any = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, color="lightblue", alpha=0.5, linewidth=0, rstride=1, cstride=1)
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        color="red",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_title("LogODE on the Sphere (S^2)")
    plt.show(block=True)


def main() -> None:
    # Configuration
    key = jax.random.PRNGKey(0)
    timesteps = 2000
    depth = 3  # Lie series depth
    dim = 3
    window_size = 10  # signatures over 10 time-steps

    # Drivers (Brownian in R^3)
    bm_path: Path = bm_driver(key, timesteps=timesteps, dim=dim)
    windows: list[Path] = non_overlapping_windower(bm_path, window_size=window_size)

    # Lie brackets (Lyndon basis) for so(3) action on S^2 ⊂ R^3
    hopf = ShuffleHopfAlgebra.build(ambient_dim=dim, depth=depth)
    A = _so3_generators()  # [3,3,3]

    def batched_field(y: jax.Array) -> jax.Array:
        return jnp.stack([M @ y for M in A], axis=0)

    lyndon_bracket_functions = form_lyndon_bracket_functions(batched_field, hopf, Sphere())

    # Integrate the Log-ODE window-by-window on the sphere
    state = jnp.array([0.0, 0.0, 1.0])
    traj: list[jax.Array] = [state]
    for w in windows:
        logsig = compute_log_signature(
            w.path, depth=depth, hopf=hopf, log_signature_type="Lyndon words", mode="full"
        )
        state = log_ode(lyndon_bracket_functions, logsig, state, Sphere())
        traj.append(state)
    trajectory = jnp.stack(traj, axis=0)

    _plot_on_sphere(trajectory)


if __name__ == "__main__":
    main()
