import jax
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from stochastax.controls.drivers import bm_driver, fractional_bm_driver, riemann_liouville_driver
from stochastax.controls.paths_types import Path
from tests.test_integrators.conftest import benchmark_wrapper
from typing import cast


@pytest.mark.benchmark(group="controls_drivers")
def test_bm_driver_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark Brownian motion path generation at 500 steps and 3 dimensions."""
    timesteps = 1024
    dim = 3

    def _generate(seed: int) -> Path:
        key = jax.random.key(seed)
        return bm_driver(key, timesteps=timesteps, dim=dim)

    path = cast(Path, benchmark_wrapper(benchmark, _generate, 0))
    assert path.path.shape == (timesteps + 1, dim)


@pytest.mark.benchmark(group="controls_drivers")
def test_fractional_bm_driver_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark Davies-Harte fractional Brownian motion generation."""
    timesteps = 1024
    dim = 3
    hurst = 0.33

    def _generate(seed: int) -> Path:
        key = jax.random.key(seed)
        return fractional_bm_driver(key, timesteps=timesteps, dim=dim, hurst=hurst)

    path = cast(Path, benchmark_wrapper(benchmark, _generate, 0))
    assert path.path.shape == (timesteps + 1, dim)


@pytest.mark.benchmark(group="controls_drivers")
def test_riemann_liouville_driver_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark Riemann-Liouville fractional process generation."""
    timesteps = 1024
    dim = 3
    hurst = 0.33

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
