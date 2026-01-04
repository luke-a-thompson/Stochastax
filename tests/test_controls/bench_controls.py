import jax
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from stochastax.controls.drivers import bm_driver, fractional_bm_driver, riemann_liouville_driver
from stochastax.controls.paths_types import Path
from tests.test_integrators.conftest import benchmark_wrapper
from typing import cast

TIMESTEPS = 1024
DIM = 3
HURST = 0.33

@pytest.mark.benchmark(group="controls_drivers")
def test_bm_driver_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark Brownian motion path generation at 500 steps and 3 dimensions."""
    def _generate(seed: int) -> Path:
        key = jax.random.key(seed)
        return bm_driver(key, timesteps=TIMESTEPS, dim=DIM)

    path = cast(Path, benchmark_wrapper(benchmark, _generate, 0))
    assert path.path.shape == (TIMESTEPS + 1, DIM)


@pytest.mark.benchmark(group="controls_drivers")
def test_fractional_bm_driver_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark Davies-Harte fractional Brownian motion generation."""
    def _generate(seed: int) -> Path:
        key = jax.random.key(seed)
        return fractional_bm_driver(key, timesteps=TIMESTEPS, dim=DIM, hurst=HURST)

    path = cast(Path, benchmark_wrapper(benchmark, _generate, 0))
    assert path.path.shape == (TIMESTEPS + 1, DIM)


@pytest.mark.benchmark(group="controls_drivers")
def test_riemann_liouville_driver_benchmark(benchmark: BenchmarkFixture) -> None:
    """Benchmark Riemann-Liouville fractional process generation."""
    def _generate(seed: int) -> Path:
        key = jax.random.key(seed)
        bm_key, rl_key = jax.random.split(key, 2)
        bm_path = bm_driver(bm_key, timesteps=TIMESTEPS, dim=DIM)
        return riemann_liouville_driver(
            rl_key,
            timesteps=TIMESTEPS,
            hurst=HURST,
            bm_path=bm_path,
        )

    path = cast(Path, benchmark_wrapper(benchmark, _generate, 0))
    assert path.path.shape == (TIMESTEPS + 1, DIM)
