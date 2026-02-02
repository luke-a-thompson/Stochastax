import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from stochastax.hopf_algebras.gl import GLHopfAlgebra
from stochastax.hopf_algebras.mkw import MKWHopfAlgebra
from stochastax.hopf_algebras.shuffle import ShuffleHopfAlgebra
from tests.conftest import BENCH_GL_CASES, BENCH_MKW_CASES, BENCH_SHUFFLE_CASES


@pytest.mark.benchmark(group="hopf_build")
@pytest.mark.parametrize("dim,depth", BENCH_SHUFFLE_CASES)
def test_shuffle_hopf_build_benchmark(
    benchmark: BenchmarkFixture,
    dim: int,
    depth: int,
) -> None:
    """Benchmark ShuffleHopfAlgebra.build with moderate depth/dimension."""

    def _builder() -> ShuffleHopfAlgebra:
        return ShuffleHopfAlgebra.build(ambient_dim=dim, depth=depth)

    hopf = benchmark(_builder)
    assert isinstance(hopf, ShuffleHopfAlgebra)
    assert hopf.ambient_dimension == dim
    assert hopf.depth == depth
    assert len(hopf.lyndon_basis_by_degree) == depth


@pytest.mark.benchmark(group="hopf_build")
@pytest.mark.parametrize("dim,depth", BENCH_GL_CASES)
def test_gl_hopf_build_benchmark(
    benchmark: BenchmarkFixture,
    dim: int,
    depth: int,
) -> None:
    """Benchmark GLHopfAlgebra.build which precomputes unordered forests."""

    def _builder() -> GLHopfAlgebra:
        return GLHopfAlgebra.build(dim, depth)

    hopf = benchmark(_builder)
    assert isinstance(hopf, GLHopfAlgebra)
    assert hopf.ambient_dimension == dim
    assert hopf.depth == depth
    assert len(hopf.forests_by_degree) == depth
    assert hopf.degree2_chain_indices is not None


@pytest.mark.benchmark(group="hopf_build")
@pytest.mark.parametrize("dim,depth", BENCH_MKW_CASES)
def test_mkw_hopf_build_benchmark(
    benchmark: BenchmarkFixture,
    dim: int,
    depth: int,
) -> None:
    """Benchmark MKWHopfAlgebra.build which precomputes planar forests."""

    def _builder() -> MKWHopfAlgebra:
        return MKWHopfAlgebra.build(dim, depth)

    hopf = benchmark(_builder)
    assert isinstance(hopf, MKWHopfAlgebra)
    assert hopf.ambient_dimension == dim
    assert hopf.depth == depth
    assert len(hopf.forests_by_degree) == depth
    assert hopf.degree2_chain_indices is not None
