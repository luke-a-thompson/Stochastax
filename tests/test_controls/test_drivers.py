import jax
import jax.numpy as jnp
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from stochastax.controls.drivers import (
    bm_driver,
    correlate_bm_driver_against_reference,
    fractional_bm_driver,
    riemann_liouville_driver,
)
from stochastax.controls.paths_types import Path
from tests.test_integrators.conftest import benchmark_wrapper
from typing import cast


@pytest.fixture(scope="module")
def bm_samples() -> Path:
    """Generate and cache multiple BM paths for reuse across tests."""
    seed = 42
    timesteps = 1000
    dim = 1
    num_paths = 500

    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)

    vmap_bm = jax.vmap(lambda k: bm_driver(k, timesteps=timesteps, dim=dim))
    paths = vmap_bm(keys)

    return paths


@pytest.mark.benchmark(group="controls_drivers")
def test_bm_driver_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark Brownian motion path generation at 500 steps and 3 dimensions."""
    timesteps = 500
    dim = 3

    def _generate(seed: int) -> Path:
        key = jax.random.key(seed)
        return bm_driver(key, timesteps=timesteps, dim=dim)

    path = cast(Path, benchmark_wrapper(benchmark, _generate, 0))
    assert path.path.shape == (timesteps + 1, dim)


@pytest.mark.benchmark(group="controls_drivers")
def test_fractional_bm_driver_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark Davies-Harte fractional Brownian motion generation."""
    timesteps = 500
    dim = 3
    hurst = 0.4

    def _generate(seed: int) -> Path:
        key = jax.random.key(seed)
        return fractional_bm_driver(key, timesteps=timesteps, dim=dim, hurst=hurst)

    path = cast(Path, benchmark_wrapper(benchmark, _generate, 0))
    assert path.path.shape == (timesteps + 1, dim)


@pytest.mark.benchmark(group="controls_drivers")
def test_riemann_liouville_driver_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark Riemann-Liouville fractional process generation."""
    timesteps = 500
    dim = 3
    hurst = 0.3

    def _generate(seed: int) -> Path:
        key = jax.random.key(seed)
        bm_key, rl_key = jax.random.split(key, 2)
        bm_path = bm_driver(bm_key, timesteps=timesteps, dim=dim)
        return riemann_liouville_driver(
            rl_key,
            timesteps=timesteps,
            hurst=hurst,
            bm_path=bm_path,
        )

    path = cast(Path, benchmark_wrapper(benchmark, _generate, 0))
    assert path.path.shape == (timesteps + 1, dim)


def test_bm_variance_scales_linearly(bm_samples: Path) -> None:
    r"""
    Brownian motion \(B_t\) satisfies \(\Var(B_t) = t\). On the uniform grid
    \(t_k = k/N\), the sample variance across \(M\) paths is linear in time.
    This test regresses empirical variances \(v_k\) against \(t_k\) and expects
    slope \(\hat s \approx 1\):
    \[
      \Var(B_{t_k}) = t_k,\quad \hat s \approx 1.
    \]
    """
    path = bm_samples.path
    timesteps = path.shape[1] - 1
    times = jnp.linspace(0.0, 1.0, timesteps + 1)

    empirical_variances = jnp.var(path, axis=0, ddof=1)
    t = times[1:]
    v = empirical_variances[1:].flatten()

    slope, _ = jnp.polyfit(t, v, 1)
    assert jnp.isclose(slope, 1.0, atol=0.1)


def test_bm_increment_mean_and_variance(bm_samples: Path) -> None:
    r"""
    For a standard Brownian motion \(B_t\) on a uniform grid with \(\Delta t = 1/N\),
    the increments \(\Delta B_k = B_{t_{k+1}} - B_{t_k}\) are i.i.d. \(\mathcal N(0, \Delta t)\).
    Pooling increments across paths, we test
    \[
      \E[\Delta B_k] = 0,\qquad \Var(\Delta B_k) = \Delta t = 1/N.
    \]
    """
    path = bm_samples.path
    timesteps = path.shape[1] - 1

    increments = jnp.diff(path, axis=1).flatten()
    mean_increments = jnp.mean(increments)
    assert jnp.isclose(mean_increments, 0.0, atol=0.1)

    expected_increment_var = 1.0 / timesteps
    empirical_increment_var = jnp.var(increments, ddof=1)
    assert jnp.isclose(
        empirical_increment_var,
        expected_increment_var,
        atol=0.1 * expected_increment_var,
    )


def test_bm_zero_mean_at_all_times(bm_samples: Path) -> None:
    r"""
    For each fixed time \(t_k\), Brownian motion has zero mean: \(\E[B_{t_k}] = 0\).
    We average across simulated paths and check \(\bar B_{t_k} \approx 0\) for all \(k\).
    """
    path = bm_samples.path
    means_at_times = jnp.mean(path, axis=0)
    assert jnp.allclose(means_at_times, 0.0, atol=0.1)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("rho", [-0.9, 0.0, 0.9])
@pytest.mark.parametrize("timesteps", [500, 1000])
@pytest.mark.parametrize("dim", [1, 5])
def test_correlated_bm_driver_correlation(seed: int, rho: float, timesteps: int, dim: int):
    """
    Test that the correlated_bm_driver produces a path with the correct
    correlation.
    """
    key = jax.random.key(seed)
    key1, key2 = jax.random.split(key)

    # Generate two independent Brownian paths
    path1 = bm_driver(key1, timesteps=timesteps, dim=dim)
    path2 = bm_driver(key2, timesteps=timesteps, dim=dim)

    # Generate the correlated path
    correlated_path = correlate_bm_driver_against_reference(path1, path2, rho)

    # The correlation is defined for the increments
    increments1 = jnp.diff(path1.path, axis=0)
    correlated_increments = jnp.diff(correlated_path.path, axis=0)

    # Compute empirical correlation
    # We flatten in case dim > 1
    empirical_corr = jnp.corrcoef(increments1.flatten(), correlated_increments.flatten())[0, 1]

    # Check if the empirical correlation is close to the target rho
    assert jnp.isclose(empirical_corr, rho, atol=1e-1)

    # Also test correlation with the second path
    increments2 = jnp.diff(path2.path, axis=0)
    empirical_corr_vs_path2 = jnp.corrcoef(increments2.flatten(), correlated_increments.flatten())[
        0, 1
    ]
    expected_corr_vs_path2 = jnp.sqrt(1 - rho**2)
    assert jnp.isclose(empirical_corr_vs_path2, expected_corr_vs_path2, atol=1e-1)


@pytest.fixture(scope="module")
def rl_samples() -> Path:
    """Generate and cache multiple RL fBM paths for reuse across tests."""
    seed = 42
    timesteps = 1000
    hurst = 0.3
    num_paths = 2000

    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)

    # Generate BM paths first, then RL paths
    bm_keys = jax.random.split(key, num_paths)
    vmap_bm = jax.vmap(lambda k: bm_driver(k, timesteps=timesteps, dim=1))
    bm_paths = vmap_bm(bm_keys)

    vmap_rl = jax.vmap(
        lambda k, bm: riemann_liouville_driver(k, timesteps=timesteps, hurst=hurst, bm_path=bm)
    )
    rl_paths = vmap_rl(keys, bm_paths)

    return rl_paths


def test_rl_zero_mean_at_all_times(rl_samples: Path) -> None:
    r"""
    For each fixed time \(t_k\), the RL process has zero mean: \(\E[V_{t_k}] = 0\).
    We average across simulated paths and check \(\bar V_{t_k} \approx 0\) for all \(k\).
    """
    paths = rl_samples.path  # Shape: (num_paths, timesteps+1, dim)
    means_at_times = jnp.mean(paths, axis=0)
    assert jnp.allclose(means_at_times, 0.0, atol=0.1)


def test_rl_marginal_variance_scaling(rl_samples: Path) -> None:
    r"""
    Marginal variance of RL process \(V_t\): for Hurst \(H\),
    \[
      \Var(V_{t_k}) \propto t_k^{2H}.
    \]
    We compute the ratio \(R_k = \widehat{\Var}(V_{\cdot,k}) / t_k^{2H}\) across selected times
    and expect \(R_k \approx 1\).
    """
    paths = rl_samples.path  # Shape: (num_paths, timesteps+1, dim)
    num_paths = paths.shape[0]
    timesteps = paths.shape[1] - 1
    hurst = 0.3

    # Test at several time points (skip t=0)
    test_indices = [100, 250, 500, 750, 1000]
    times = jnp.array(test_indices) / timesteps  # Normalize to [0,1]

    for k, t in zip(test_indices, times):
        if k <= timesteps:
            # Compute sample variance at time k
            var_k = jnp.var(paths[:, k, :], axis=0, ddof=1)
            theoretical_var = t ** (2 * hurst)

            # Ratio should be approximately 1
            ratio = jnp.squeeze(var_k) / theoretical_var  # make scalar
            tolerance = 3.0 / (num_paths**0.5)

            assert jnp.isclose(ratio, 1.0, atol=tolerance), (
                f"Variance ratio at t={float(t):.3f} is {float(ratio):.3f}, expected ~1.0 ± {tolerance:.3f}"
            )


def test_rl_gaussianity(rl_samples: Path) -> None:
    r"""
    Gaussianity: for fixed \(t_k\), \(V_{t_k}\) is Gaussian with variance scaling \(t_k^{2H}\).
    Standardising \(Z_k = V_{t_k} / t_k^{H}\) yields \(Z_k \sim \mathcal N(0,1)\).
    We check mean \(\approx 0\), variance \(\approx 1\), and normality via skew/kurtosis.
    """
    paths = rl_samples.path
    timesteps = paths.shape[1] - 1
    hurst = 0.3

    # Test at a few time points
    test_indices = [250, 500, 750]
    times = jnp.array(test_indices) / timesteps

    for k, t in zip(test_indices, times):
        if k <= timesteps:
            # Extract values at time k
            V_k = paths[:, k, :].flatten()

            # Standardize: Z_k = V_k / t_k^H
            Z_k = V_k / (t**hurst)

            # Test mean ≈ 0 and variance ≈ 1
            mean_Z = jnp.mean(Z_k)
            var_Z = jnp.var(Z_k, ddof=1)

            assert jnp.isclose(mean_Z, 0.0, atol=0.1), (
                f"Standardized mean at t={float(t):.3f} is {float(mean_Z):.3f}, expected ~0.0"
            )
            assert jnp.isclose(var_Z, 1.0, atol=0.2), (
                f"Standardized variance at t={float(t):.3f} is {float(var_Z):.3f}, expected ~1.0"
            )

            # Jarque-Bera test (simplified - just check skewness and kurtosis)
            skew = jnp.mean(((Z_k - mean_Z) / jnp.sqrt(var_Z)) ** 3)
            kurt = jnp.mean(((Z_k - mean_Z) / jnp.sqrt(var_Z)) ** 4)

            # For normality: skew ≈ 0, kurt ≈ 3
            assert jnp.abs(skew) < 0.5, (
                f"Skewness at t={float(t):.3f} is {float(skew):.3f}, expected ~0.0"
            )
            assert jnp.abs(kurt - 3.0) < 1.0, (
                f"Kurtosis at t={float(t):.3f} is {float(kurt):.3f}, expected ~3.0"
            )


def test_rl_correlation_with_bm(rl_samples: Path) -> None:
    r"""
    Correlation with BM: if RL is built from Brownian motion \(W^{(1)}\), and an asset Brownian
    \(W^{(2)}\) is constructed with target correlation \(\rho\) to \(W^{(1)}\), then increments
    of the resulting RL path inherit a non-zero correlation with the driving Brownian increments.
    We validate by correlating pooled increments after passing correlated BMs through the RL driver.
    """
    # Generate correlated BM paths
    seed = 42
    timesteps = 1000
    num_paths = 500
    rho = 0.7  # Target correlation

    key = jax.random.key(seed)
    key1, key2 = jax.random.split(key)
    rl_keys = jax.random.split(key, num_paths)

    # Generate two sets of BM paths
    bm_keys1 = jax.random.split(key1, num_paths)
    bm_keys2 = jax.random.split(key2, num_paths)  # Different seed

    vmap_bm = jax.vmap(lambda k: bm_driver(k, timesteps=timesteps, dim=1))
    bm_paths1 = vmap_bm(bm_keys1)
    bm_paths2 = vmap_bm(bm_keys2)

    # Correlate the second set against the first
    vmap_corr = jax.vmap(lambda p1, p2: correlate_bm_driver_against_reference(p1, p2, rho))
    correlated_bm = vmap_corr(bm_paths1, bm_paths2)

    # Generate RL paths using the correlated BM
    hurst = 0.3
    vmap_rl = jax.vmap(
        lambda k, bm: riemann_liouville_driver(k, timesteps=timesteps, hurst=hurst, bm_path=bm)
    )
    rl_paths = vmap_rl(rl_keys, correlated_bm)

    # Compute increments of both BM and RL
    bm_increments = jnp.diff(bm_paths1.path, axis=1)  # Shape: (num_paths, timesteps, dim)
    rl_increments = jnp.diff(rl_paths.path, axis=1)

    # Flatten for correlation computation
    bm_flat = bm_increments.flatten()
    rl_flat = rl_increments.flatten()

    # Compute correlation
    empirical_corr = jnp.corrcoef(bm_flat, rl_flat)[0, 1]

    # The RL process should have some correlation with the underlying BM
    # (though not necessarily equal to rho due to the RL transformation)
    assert jnp.abs(empirical_corr) > 0.1, (
        f"Correlation between BM and RL increments is {float(empirical_corr):.3f}, expected > 0.1"
    )


@pytest.mark.parametrize("hurst", [0.3, 0.5, 0.7])
def test_rl_multivariate_correlation_preservation(hurst: float) -> None:
    """
    Verify that cross-sectional correlations across Brownian channels are preserved
    (up to sampling error) by the RL driver when applied channelwise.
    """
    seed = 7
    N = 1000
    M = 800
    dim = 3

    # Target SPD correlation matrix
    corr = jnp.array(
        [
            [1.0, 0.6, -0.3],
            [0.6, 1.0, 0.2],
            [-0.3, 0.2, 1.0],
        ]
    )
    chol = jnp.linalg.cholesky(corr)

    key = jax.random.key(seed)
    key_bm, key_rl = jax.random.split(key)
    bm_keys = jax.random.split(key_bm, M)
    rl_keys = jax.random.split(key_rl, M)

    # Build correlated-dimension BM paths by post-multiplying increments with chol
    def correlated_bm_path(k: jax.Array) -> Path:
        base = bm_driver(k, timesteps=N, dim=dim)
        inc = jnp.diff(base.path, axis=0)  # (N, dim)
        corr_inc = inc @ chol.T  # (N, dim)
        new_path = jnp.concatenate(
            [jnp.zeros((1, dim), dtype=corr_inc.dtype), jnp.cumsum(corr_inc, axis=0)],
            axis=0,
        )
        return Path(new_path, base.interval)

    bm_paths = jax.vmap(correlated_bm_path)(bm_keys)

    # Apply RL driver channelwise
    rl_paths = jax.vmap(
        lambda rk, bp: riemann_liouville_driver(rk, timesteps=N, hurst=hurst, bm_path=bp)
    )(rl_keys, bm_paths)

    # Pool RL increments across time and batch, compute empirical correlation matrix
    rl_increments = jnp.diff(rl_paths.path, axis=1)  # (M, N, dim)
    X = rl_increments.reshape(-1, dim)  # (M*N, dim)
    # jnp.corrcoef expects variables in columns when rowvar=False
    emp_corr = jnp.corrcoef(X, rowvar=False)

    max_err = jnp.max(jnp.abs(emp_corr - corr))
    tol = 0.08 if hurst < 0.5 else 0.05
    assert max_err <= tol, (
        f"Max entrywise correlation error {float(max_err):.3f} exceeds {tol:.2f}\n"
        f"Empirical:\n{emp_corr}\nTarget:\n{corr}"
    )


@pytest.mark.parametrize("hurst", [0.3, 0.5, 0.7])
def test_davies_harte_fbm_terminal_variance(hurst: float) -> None:
    r"""
    Davies-Harte fBM driver: B^H_1 should have variance 1 for any H in (0,1)
    when using the canonical normalization.

    We simulate many paths and assert Var[B^H_1] ≈ 1.
    """
    seed = 123
    timesteps = 2048
    dim = 1
    num_paths = 2000

    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)

    vmap_fbm = jax.vmap(
        lambda k: fractional_bm_driver(k, timesteps=timesteps, dim=dim, hurst=hurst)
    )
    paths = vmap_fbm(keys)

    terminal = paths.path[:, -1, 0]
    var_emp = jnp.var(terminal, ddof=1)

    # Tolerance scales with 1/sqrt(M). Use a modest cushion.
    tol = 5.0 / jnp.sqrt(num_paths)
    assert jnp.isclose(var_emp, 1.0, atol=float(tol)), (
        f"Empirical Var[B^H_1]={float(var_emp):.3f} not close to 1.0 for H={hurst}"
    )


@pytest.mark.parametrize("hurst", [0.3, 0.5, 0.7])
def test_davies_harte_fbm_zero_mean(hurst: float) -> None:
    r"""
    Davies-Harte fBM paths have zero mean at all times.
    We average across simulated paths and check \bar B^H_{t_k} \approx 0.
    """
    seed = 7
    timesteps = 2000
    dim = 1
    num_paths = 1500

    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)

    vmap_fbm = jax.vmap(
        lambda k: fractional_bm_driver(k, timesteps=timesteps, dim=dim, hurst=hurst)
    )
    paths = vmap_fbm(keys)

    means_at_times = jnp.mean(paths.path, axis=0)
    assert jnp.allclose(means_at_times, 0.0, atol=0.1)


@pytest.mark.parametrize("hurst", [0.3, 0.7])
def test_davies_harte_fbm_variance_scaling(hurst: float) -> None:
    r"""
    For fBM, Var[B^H_t] = t^{2H}. We regress log Var against log t
    and expect slope \approx 2H.
    """
    seed = 11
    timesteps = 3000
    dim = 1
    num_paths = 1200

    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)

    vmap_fbm = jax.vmap(
        lambda k: fractional_bm_driver(k, timesteps=timesteps, dim=dim, hurst=hurst)
    )
    paths = vmap_fbm(keys)

    times = jnp.linspace(0.0, 1.0, timesteps + 1)
    variances = jnp.var(paths.path, axis=0, ddof=1)

    t = times[1:]
    v = variances[1:]
    logt = jnp.log(t)
    logv = jnp.log(v)
    slope, _ = jnp.polyfit(logt, logv, 1)

    expected = 2.0 * hurst
    assert jnp.isclose(slope, expected, atol=0.15)


@pytest.mark.parametrize("hurst", [0.3, 0.7])
def test_davies_harte_fbm_gaussianity(hurst: float) -> None:
    r"""
    For fixed t_k, B^H_{t_k} is Gaussian with Var t_k^{2H}.
    Standardising Z_k = B^H_{t_k} / t_k^H yields ~ N(0,1): mean ~0, var ~1,
    and reasonable skew/kurtosis.
    """
    seed = 17
    timesteps = 2500
    dim = 1
    num_paths = 1500

    key = jax.random.key(seed)
    keys = jax.random.split(key, num_paths)

    vmap_fbm = jax.vmap(
        lambda k: fractional_bm_driver(k, timesteps=timesteps, dim=dim, hurst=hurst)
    )
    paths = vmap_fbm(keys)

    test_indices = [500, 1000, 2000]
    times = jnp.array(test_indices) / timesteps

    for idx, t in zip(test_indices, times):
        B_k = paths.path[:, idx, :].flatten()
        Z_k = B_k / (t**hurst)

        mean_Z = jnp.mean(Z_k)
        var_Z = jnp.var(Z_k, ddof=1)
        assert jnp.isclose(mean_Z, 0.0, atol=0.1)
        assert jnp.isclose(var_Z, 1.0, atol=0.2)

        # Simple skew / kurtosis checks
        std_Z = jnp.sqrt(var_Z)
        z = (Z_k - mean_Z) / std_Z
        skew = jnp.mean(z**3)
        kurt = jnp.mean(z**4)
        assert jnp.abs(skew) < 0.5
        assert jnp.abs(kurt - 3.0) < 1.0


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("timesteps", [1000, 3000])
@pytest.mark.parametrize("hurst", [0.25, 0.5, 0.75])
def test_riemann_liouville_variance_scaling(seed: int, timesteps: int, hurst: float):
    """
    Test the variance scaling of the Riemann-Liouville fBM implementation.
    The variance of fBM scales as t^(2H), so a log-log plot of variance vs. time
    should have a slope of 2H.
    """
    key = jax.random.key(seed)
    N = timesteps
    T = 1.0
    M = 500  # Number of paths for the test

    # Generate M paths for the variance scaling test
    keys = jax.random.split(key, M)
    vmap_driver = jax.vmap(
        lambda k: riemann_liouville_driver(
            k,
            timesteps=N,
            hurst=hurst,
            bm_path=bm_driver(k, timesteps=N, dim=1),
        )
    )
    paths = vmap_driver(keys)

    # The time grid must have N+1 points to match the path length
    times = jnp.linspace(0, T, N + 1)

    # Compute empirical variance at each time point
    variances = jnp.var(paths.path, axis=0, ddof=1)  # unbiased estimate

    # Exclude t=0 (variance zero)
    t = times[1:]
    v = variances[1:]

    # Fit log-log regression
    logt = jnp.log(t)
    logv = jnp.log(v)
    slope, _ = jnp.polyfit(logt, logv, 1)

    # The theoretical slope is 2*H
    expected_slope = 2 * hurst

    # Check if the estimated slope is close to the theoretical value
    assert jnp.isclose(slope, expected_slope, atol=1e-1)
