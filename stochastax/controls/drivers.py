import jax
import jax.numpy as jnp
from stochastax.controls.paths_types import Path


def bm_driver(key: jax.Array, timesteps: int, dim: int) -> Path:
    """
    Generates a Brownian motion path.
    """
    dt = 1.0 / timesteps
    increments = jax.random.normal(key, (timesteps, dim)) * jnp.sqrt(dt)
    path = jnp.concatenate([jnp.zeros((1, dim)), jnp.cumsum(increments, axis=0)], axis=0)
    return Path(path, (0, timesteps + 1))


def correlate_bm_driver_against_reference(
    reference_path: Path, indep_path: Path, rho: float
) -> Path:
    """
    Correlates a Brownian motion path against a reference path by their increments.
    """
    if reference_path.path.shape != indep_path.path.shape:
        raise ValueError(
            f"Reference path and indep path must have the same shape. Got shapes {reference_path.path.shape} and {indep_path.path.shape}"
        )
    if rho < -1 or rho > 1:
        raise ValueError(f"rho must be between -1 and 1. Got {rho}")

    reference_increments = jnp.diff(reference_path.path, axis=0)
    indep_increments = jnp.diff(indep_path.path, axis=0)

    correlated_increments = rho * reference_increments + jnp.sqrt(1 - rho**2) * indep_increments

    initial_cond = indep_path.path[0, :]
    correlated_path = initial_cond + jnp.cumsum(correlated_increments, axis=0)
    correlated_path = jnp.concatenate([initial_cond[None, :], correlated_path], axis=0)
    return Path(correlated_path, indep_path.interval)


def correlated_bm_drivers(indep_bm_paths: Path, corr_matrix: jax.Array) -> Path:
    """
    Generates a new Brownian motion path that is correlated with a reference path.
    """
    num_paths = indep_bm_paths.path.shape[0]
    if corr_matrix.shape != (num_paths, num_paths):
        raise ValueError(
            f"Received {num_paths} paths, but got a correlation matrix with shape {corr_matrix.shape}. Corr matrix must be shape (num_paths, num_paths)."
        )
    if not jnp.allclose(jnp.diag(corr_matrix), 1.0):
        raise ValueError(
            f"The diagonal of the correlation matrix must be 1. Got {jnp.diag(corr_matrix)}"
        )
    if not jnp.allclose(corr_matrix, corr_matrix.T):
        raise ValueError("The correlation matrix must be symmetric.")

    chol_matrix = jnp.linalg.cholesky(corr_matrix)

    indep_increments = jnp.diff(indep_bm_paths.path, axis=1)

    correlated_increments = jnp.einsum("qtd,pq->ptd", indep_increments, chol_matrix)

    new_path = jnp.cumsum(correlated_increments, axis=1)

    zeros_shape = (new_path.shape[0], 1, new_path.shape[2])
    new_path = jnp.concatenate([jnp.zeros(zeros_shape), new_path], axis=1)

    return Path(new_path, indep_bm_paths.interval)


def fractional_bm_driver(key: jax.Array, timesteps: int, dim: int, hurst: float) -> Path:
    """
    Generates sample paths of fractional Brownian Motion using the Davies Harte method with JAX.
    """

    def get_path(key: jax.Array, timesteps: int, hurst: float) -> jax.Array:
        def gamma(k: jax.Array, H: float) -> jax.Array:
            return 0.5 * (
                jnp.abs(k - 1) ** (2 * H) - 2 * jnp.abs(k) ** (2 * H) + jnp.abs(k + 1) ** (2 * H)
            )

        k_vals = jnp.arange(0, timesteps)
        g = gamma(k_vals, hurst)
        r = jnp.concatenate([g, jnp.array([0.0]), jnp.flip(g)[:-1]])

        lk = jnp.fft.fft(r).real

        key1, key2, key3 = jax.random.split(key, 3)

        rvs = jax.random.normal(key1, shape=(timesteps - 1, 2))
        v_0_0 = jax.random.normal(key2)
        v_n_0 = jax.random.normal(key3)

        Vj = jnp.zeros((2 * timesteps, 2))
        Vj = Vj.at[0, 0].set(v_0_0)
        Vj = Vj.at[timesteps, 0].set(v_n_0)

        indices1 = jnp.arange(1, timesteps)
        indices2 = jnp.arange(2 * timesteps - 1, timesteps, -1)

        Vj = Vj.at[indices1, :].set(rvs)
        Vj = Vj.at[indices2, :].set(rvs)

        N = 2 * timesteps
        lk = jnp.maximum(lk, 0.0)

        wk = jnp.zeros(N, dtype=jnp.complex64)
        wk = wk.at[0].set(jnp.sqrt(lk[0]) * Vj[0, 0])
        wk = wk.at[1:timesteps].set(
            jnp.sqrt(lk[1:timesteps] / 2.0) * (Vj[1:timesteps, 0] + 1j * Vj[1:timesteps, 1])
        )
        wk = wk.at[timesteps].set(jnp.sqrt(lk[timesteps]) * Vj[timesteps, 0])
        wk = wk.at[timesteps + 1 : N].set(jnp.conj(wk[timesteps - 1 : 0 : -1]))

        wk = jnp.sqrt(jnp.asarray(N, dtype=wk.dtype)) * wk
        Z = jnp.fft.ifft(wk)
        fGn = Z[0:timesteps].real
        fBm = jnp.cumsum(fGn) * (timesteps ** (-hurst))
        path = jnp.concatenate([jnp.array([0.0]), fBm])
        return path

    keys = jax.random.split(key, dim)
    paths = jax.vmap(get_path, in_axes=(0, None, None))(keys, timesteps, hurst)
    return Path(paths.T, (0, timesteps))


def riemann_liouville_driver(
    key: jax.Array,
    timesteps: int,
    hurst: float,
    bm_path: Path,
) -> Path:
    """
    Hybrid scheme (kappa = 1) for the RL/type-II fBM driver used in rBergomi.
    """
    assert bm_path.num_timesteps == timesteps + 1, "bm_path must have shape (timesteps+1, dim)."

    dim = bm_path.ambient_dimension
    Δ = 1.0 / timesteps
    α = hurst - 0.5
    sqrt2H = jnp.sqrt(2.0 * hurst)

    dW = jnp.diff(bm_path.path, axis=0)

    a = (Δ**α) / (α + 1.0)
    var_I = (Δ ** (2.0 * α + 1.0)) / (2.0 * α + 1.0)
    b = jnp.sqrt(jnp.maximum(var_I - (a * a) * Δ, 0.0))

    Z = jax.random.normal(key, shape=dW.shape)
    I = a * dW + b * Z

    i = jnp.arange(2, timesteps + 1, dtype=dW.dtype)
    w = (Δ**α) * (i ** (α + 1.0) - (i - 1.0) ** (α + 1.0)) / (α + 1.0)

    def conv_full_1d(w: jax.Array, x: jax.Array) -> jax.Array:
        L = int(2 ** jnp.ceil(jnp.log2(w.shape[0] + x.shape[0] - 1)))
        wf = jnp.fft.rfft(jnp.pad(w, (0, L - w.shape[0])))
        xf = jnp.fft.rfft(jnp.pad(x, (0, L - x.shape[0])))
        y = jnp.fft.irfft(wf * xf, n=L)[: w.shape[0] + x.shape[0] - 1]
        return y

    def per_dim(x: jax.Array) -> jax.Array:
        y = conv_full_1d(w, x[:-1])
        return jnp.concatenate([jnp.zeros((1,), x.dtype), y[: timesteps - 1]])

    Y2 = jnp.stack([per_dim(dW[:, d]) for d in range(dim)], axis=1)

    Y_tail = sqrt2H * (I + Y2)
    Y_path = jnp.concatenate([jnp.zeros((1, dim), Y_tail.dtype), Y_tail], axis=0)
    return Path(Y_path, bm_path.interval)
