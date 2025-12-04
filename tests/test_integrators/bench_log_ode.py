import jax
import jax.numpy as jnp
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from stochastax.hopf_algebras.hopf_algebras import GLHopfAlgebra, MKWHopfAlgebra
from stochastax.vector_field_lifts.mkw_lift import form_mkw_lift
from tests.conftest import _so3_generators
from tests.test_integrators.conftest import (
    _linear_vector_fields,
    _project_to_tangent,
    benchmark_wrapper,
    build_deterministic_increments,
    build_block_rotation_generators,
    build_block_initial_state,
    form_bck_lift,
    compute_nonplanar_branched_signature,
    build_standard_log_ode_inputs,
    build_standard_manifold_case,
)

from stochastax.control_lifts.branched_signature_ito import compute_planar_branched_signature
from stochastax.integrators.log_ode import log_ode
from stochastax.hopf_algebras.hopf_algebras import ShuffleHopfAlgebra
from stochastax.control_lifts.log_signature import compute_log_signature

LOG_ODE_STANDARD_BENCH_CASES: list = [
    pytest.param(1, 1, 12, id="depth-1-dim-1-step-12"),
    pytest.param(1, 8, 12, id="depth-1-dim-8-step-12"),
    pytest.param(2, 1, 12, id="depth-2-dim-1-step-12"),
    pytest.param(2, 8, 12, id="depth-2-dim-8-step-12"),
    pytest.param(3, 1, 12, id="depth-3-dim-1-step-12"),
    pytest.param(3, 8, 12, id="depth-3-dim-8-step-12"),
]


@pytest.mark.benchmark(group="log_ode_standard_stepwise")
@pytest.mark.parametrize(
    "depth,dim,steps",
    LOG_ODE_STANDARD_BENCH_CASES,
)
def test_log_ode_benchmark_standard_stepwise(
    benchmark: BenchmarkFixture,
    depth: int,
    dim: int,
    steps: int,
) -> None:
    """Benchmark Euclidean log-ODE by stepping through deterministic increments."""
    # Reuse the standard Euclidean brackets/initial state; replace the control with a
    # multi-step deterministic path.
    brackets, _, y0 = build_standard_log_ode_inputs(depth, dim, delta=0.3)
    increments = build_deterministic_increments(dim, steps, seed=0, scale=0.05)
    local_hopf = ShuffleHopfAlgebra.build(ambient_dim=dim, depth=depth)

    @jax.jit
    def integrate_path(path_increments: jax.Array, y_init: jax.Array) -> jax.Array:
        def step(carry: jax.Array, inc: jax.Array) -> tuple[jax.Array, None]:
            seg = jnp.vstack([jnp.zeros((1, dim), dtype=inc.dtype), inc.reshape(1, -1)])
            primitive = compute_log_signature(seg, depth, local_hopf, "Lyndon words", mode="full")
            y_next = log_ode(brackets, primitive, carry)
            return y_next, None

        y_final, _ = jax.lax.scan(step, y_init, path_increments)
        return y_final

    result = benchmark_wrapper(benchmark, integrate_path, increments, y0)
    assert result.shape == y0.shape


LOG_ODE_MANIFOLD_BENCH_CASES: list = [
    pytest.param(1, 12, id="depth-1-dim-3-step-12"),
    pytest.param(2, 12, id="depth-2-dim-3-step-12"),
    pytest.param(3, 12, id="depth-3-dim-3-step-12"),
]


@pytest.mark.benchmark(group="log_ode_manifold_stepwise")
@pytest.mark.parametrize(
    "depth,steps",
    LOG_ODE_MANIFOLD_BENCH_CASES,
)
def test_log_ode_benchmark_manifold_stepwise(
    benchmark: BenchmarkFixture, depth: int, steps: int
) -> None:
    """Benchmark manifold integration by stepping through increments."""
    bracket_basis, increments, y0 = build_standard_manifold_case(depth, steps)
    dim: int = increments.shape[1]
    local_hopf = ShuffleHopfAlgebra.build(ambient_dim=dim, depth=depth)

    @jax.jit
    def integrate_path(path_increments: jax.Array, y_init: jax.Array) -> jax.Array:
        def step(carry: jax.Array, inc: jax.Array) -> tuple[jax.Array, None]:
            seg_W = jnp.vstack([jnp.zeros((1, dim), dtype=inc.dtype), inc.reshape(1, -1)])
            primitive = compute_log_signature(seg_W, depth, local_hopf, "Lyndon words", mode="full")
            y_next = log_ode(bracket_basis, primitive, carry)
            return y_next, None

        y_final, _ = jax.lax.scan(step, y_init, path_increments)
        return y_final

    result = benchmark_wrapper(benchmark, integrate_path, increments, y0)
    assert result.shape == y0.shape


BCK_BENCH_CASES: list = [
    pytest.param(1, 1, 12, id="depth-1-dim-1-step-12"),
    pytest.param(1, 8, 12, id="depth-1-dim-8-step-12"),
    pytest.param(2, 1, 12, id="depth-2-dim-1-step-12"),
    pytest.param(2, 8, 12, id="depth-2-dim-8-step-12"),
]


@pytest.mark.benchmark(group="log_ode_bck_stepwise")
@pytest.mark.parametrize(
    "depth,dim,steps",
    BCK_BENCH_CASES,
)
def test_bck_log_ode_benchmark_stepwise(
    benchmark: BenchmarkFixture,
    depth: int,
    dim: int,
    steps: int,
) -> None:
    """Benchmark BCK integration by stepping through deterministic increments."""
    hopf = GLHopfAlgebra.build(dim, depth)
    generators = build_block_rotation_generators(dim)
    vector_fields = _linear_vector_fields(generators)
    y0 = build_block_initial_state(dim)
    bck_brackets = form_bck_lift(vector_fields, y0, hopf)
    increments = build_deterministic_increments(dim, steps, seed=depth + dim, scale=0.04)

    @jax.jit
    def integrate_path(path_increments: jax.Array, y_init: jax.Array) -> jax.Array:
        def step(carry: jax.Array, inc: jax.Array) -> tuple[jax.Array, None]:
            seg_path = jnp.vstack([jnp.zeros((1, dim), dtype=inc.dtype), inc.reshape(1, -1)])
            cov = jnp.zeros((1, dim, dim), dtype=inc.dtype)
            signature = compute_nonplanar_branched_signature(
                path=seg_path,
                order_m=depth,
                hopf=hopf,
                mode="full",
                cov_increments=cov,
            )
            logsig = signature.log()
            y_next = log_ode(bck_brackets, logsig, carry)
            return y_next, None

        y_final, _ = jax.lax.scan(step, y_init, path_increments)
        return y_final

    result = benchmark_wrapper(benchmark, integrate_path, increments, y0)
    assert result.shape == y0.shape


MKW_BENCH_CASES: list = [
    pytest.param(1, 12, id="depth-1-dim-3-step-12"),
    pytest.param(2, 12, id="depth-2-dim-3-step-12"),
]


@pytest.mark.benchmark(group="log_ode_mkw_stepwise")
@pytest.mark.parametrize("depth,steps", MKW_BENCH_CASES)
def test_mkw_log_ode_benchmark_manifold_stepwise(
    benchmark: BenchmarkFixture, depth: int, steps: int
) -> None:
    """Benchmark MKW manifold integration by stepping through increments."""
    A = _so3_generators()
    dim = A.shape[0]
    hopf = MKWHopfAlgebra.build(dim, depth)
    vector_fields = _linear_vector_fields(A)
    y0 = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    mkw_brackets = form_mkw_lift(vector_fields, y0, hopf, _project_to_tangent)
    increments = build_deterministic_increments(dim, steps, seed=depth + steps, scale=0.03)

    @jax.jit
    def integrate_path(path_increments: jax.Array, y_init: jax.Array) -> jax.Array:
        def step(carry: jax.Array, inc: jax.Array) -> tuple[jax.Array, None]:
            seg_path = jnp.vstack([jnp.zeros((1, dim), dtype=inc.dtype), inc.reshape(1, -1)])
            cov = jnp.zeros((1, dim, dim), dtype=inc.dtype)
            signature = compute_planar_branched_signature(
                path=seg_path,
                order_m=depth,
                hopf=hopf,
                mode="full",
                cov_increments=cov,
            )
            logsig = signature.log()
            y_next = log_ode(mkw_brackets, logsig, carry)
            return y_next, None

        y_final, _ = jax.lax.scan(step, y_init, path_increments)
        return y_final

    result = benchmark_wrapper(benchmark, integrate_path, increments, y0)
    assert result.shape == y0.shape
