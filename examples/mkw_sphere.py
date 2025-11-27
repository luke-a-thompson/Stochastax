import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Any
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Callable

from stochastax.controls.drivers import bm_driver
from stochastax.controls.augmentations import non_overlapping_windower
from stochastax.controls.paths_types import Path
from stochastax.hopf_algebras import enumerate_mkw_trees
from stochastax.hopf_algebras.hopf_algebra_types import MKWHopfAlgebra
from stochastax.hopf_algebras.elements import GroupElement
from stochastax.control_lifts.branched_signature_ito import compute_planar_branched_signature
from stochastax.integrators.log_ode import log_ode
from stochastax.vector_field_lifts.mkw_lift import form_mkw_brackets


def _so3_generators() -> jax.Array:
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
    X, Y, Z = _sphere_mesh()
    fig = plt.figure()
    ax: Any = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, color="lightblue", alpha=0.5, linewidth=0, rstride=1, cstride=1)
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        color="green",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_title("MKW LogODE on the Sphere (S^2) with QV")
    plt.show(block=True)


def _project_to_tangent(y: jax.Array, v: jax.Array) -> jax.Array:
    return v - jnp.dot(y, v) * y


def _linear_vector_fields(A: jax.Array) -> list[Callable[[jax.Array], jax.Array]]:
    # A shape [d, n, n] -> list of callables V_i(y) = A[i] @ y
    vecs: list[Callable[[jax.Array], jax.Array]] = [
        lambda y, M=A[i]: M @ y for i in range(A.shape[0])
    ]
    return vecs


def main() -> None:
    # Configuration
    key = jax.random.PRNGKey(4)
    timesteps = 2000
    depth = 2
    dim = 3
    window_size = 10

    # Drivers and windowing
    bm_path: Path = bm_driver(key, timesteps=timesteps, dim=dim)
    windows: list[Path] = non_overlapping_windower(bm_path, window_size=window_size)
    dt = 1.0 / timesteps

    # MKW forests and Hopf algebra for branched signatures
    forests = enumerate_mkw_trees(depth)
    hopf = MKWHopfAlgebra.build(dim, forests)

    # Vector-field brackets (MKW) using tangent projection
    A = _so3_generators()
    V = _linear_vector_fields(A)
    x0 = jnp.array([0.0, 0.0, 1.0])
    mkw_brackets = form_mkw_brackets(V, x0, forests, _project_to_tangent)

    # words_by_len placeholder: use forests' parent arrays for filtering nonempty levels
    _ = [f.parent for f in forests]  # unused after API change

    # Integrate Log-ODE window-by-window using branched Itô log signatures with known QV
    state = x0
    traj: list[jax.Array] = [state]
    identity = jnp.eye(dim)
    for w in windows:
        steps = w.num_timesteps - 1
        cov_increments = jnp.tile((dt * identity)[None, :, :], reps=(steps, 1, 1))
        # Planar branched signature (group element tail per degree)
        sig_levels = compute_planar_branched_signature(
            path=w.path,
            order_m=depth,
            hopf=hopf,
            mode="full",
            cov_increments=cov_increments,
        )
        # Convert to MKW log-signature
        from typing import cast

        sig_levels_list = cast(list[jax.Array], sig_levels)
        group_el = GroupElement(hopf=hopf, coeffs=sig_levels_list, interval=w.interval)
        logsig = group_el.log()  # MKWLogSignature
        # One Log-ODE update on sphere (normalization keeps state on S^2)
        state = log_ode(mkw_brackets, logsig, state)
        traj.append(state)
    trajectory = jnp.stack(traj, axis=0)

    _plot_on_sphere(trajectory)


if __name__ == "__main__":
    main()
