"""Benchmarks for vector field bracket-function lifts."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from stochastax.hopf_algebras.hopf_algebras import GLHopfAlgebra, MKWHopfAlgebra, ShuffleHopfAlgebra
from stochastax.manifolds import EuclideanSpace
from stochastax.vector_field_lifts.bck_lift import form_bck_bracket_functions
from stochastax.vector_field_lifts.lie_lift import form_lyndon_bracket_functions
from stochastax.vector_field_lifts.mkw_lift import form_mkw_bracket_functions
from tests.conftest import benchmark_wrapper


def _init_mlp_params(
    key: jax.Array,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    depth: int,
) -> dict[str, list[jax.Array]]:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    sizes = [in_dim] + [hidden_dim] * (depth - 1) + [out_dim]
    keys = jax.random.split(key, num=len(sizes) - 1)
    weights: list[jax.Array] = []
    biases: list[jax.Array] = []
    for k, fan_in, fan_out in zip(keys, sizes[:-1], sizes[1:]):
        scale = 1.0 / jnp.sqrt(float(fan_in))
        W = scale * jax.random.normal(k, shape=(fan_out, fan_in))
        b = jnp.zeros((fan_out,), dtype=W.dtype)
        weights.append(W)
        biases.append(b)
    return {"weights": weights, "biases": biases}


def _mlp_apply(params: dict[str, list[jax.Array]], x: jax.Array) -> jax.Array:
    weights = params["weights"]
    biases = params["biases"]
    h = x
    for i in range(len(weights)):
        h = weights[i] @ h + biases[i]
        if i < len(weights) - 1:
            h = jax.nn.tanh(h)
    return h


def _make_mlp(
    key: jax.Array,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    depth: int,
) -> Callable[[jax.Array], jax.Array]:
    params = _init_mlp_params(key, in_dim, hidden_dim, out_dim, depth)

    def mlp(x: jax.Array) -> jax.Array:
        return _mlp_apply(params, x)

    return mlp


def _make_batched_vector_field(
    key: jax.Array,
    dim: int,
    state_dim: int,
    hidden_dim: int,
    depth: int,
) -> Callable[[jax.Array], jax.Array]:
    mlp = _make_mlp(key, state_dim, hidden_dim, dim * state_dim, depth)

    def batched_field(y: jax.Array) -> jax.Array:
        return mlp(y).reshape(dim, state_dim)

    return batched_field


def _make_eval_fn(
    bracket_functions: list[list[Callable[[jax.Array], jax.Array]]],
    max_terms: int = 4,
) -> Callable[[jax.Array], jax.Array]:
    level0 = bracket_functions[0]
    if not level0:
        raise ValueError("Expected non-empty level-0 bracket functions.")
    num_terms = min(max_terms, len(level0))
    funcs = tuple(level0[:num_terms])

    def eval_fn(y: jax.Array) -> jax.Array:
        out = jnp.zeros_like(y)
        for fn in funcs:
            out = out + fn(y)
        return out

    return eval_fn


BENCH_CASES = [
    pytest.param(2, 3, 8, 32, 3, id="depth2-dim3-state8-h16"),
    pytest.param(2, 4, 16, 64, 3, id="depth2-dim4-state16-h32"),
]


@pytest.mark.benchmark(group="vf_lift_build")
@pytest.mark.parametrize("depth,dim,state_dim,hidden_dim,mlp_depth", BENCH_CASES)
def test_bench_lyndon_bracket_functions_build(
    benchmark: BenchmarkFixture,
    depth: int,
    dim: int,
    state_dim: int,
    hidden_dim: int,
    mlp_depth: int,
) -> None:
    key = jax.random.PRNGKey(0)
    batched_field = _make_batched_vector_field(key, dim, state_dim, hidden_dim, mlp_depth)
    hopf = ShuffleHopfAlgebra.build(ambient_dim=dim, depth=depth)

    def build() -> list[list[Callable[[jax.Array], jax.Array]]]:
        return form_lyndon_bracket_functions(batched_field, hopf)

    bracket_functions = benchmark(build)
    assert len(bracket_functions) == depth


@pytest.mark.benchmark(group="vf_lift_eval")
@pytest.mark.parametrize("depth,dim,state_dim,hidden_dim,mlp_depth", BENCH_CASES)
def test_bench_lyndon_bracket_functions_eval(
    benchmark: BenchmarkFixture,
    depth: int,
    dim: int,
    state_dim: int,
    hidden_dim: int,
    mlp_depth: int,
) -> None:
    key = jax.random.PRNGKey(0)
    batched_field = _make_batched_vector_field(key, dim, state_dim, hidden_dim, mlp_depth)
    hopf = ShuffleHopfAlgebra.build(ambient_dim=dim, depth=depth)
    bracket_functions = form_lyndon_bracket_functions(batched_field, hopf)
    eval_fn = _make_eval_fn(bracket_functions)
    y0 = jnp.linspace(0.1, 0.2, num=state_dim, dtype=jnp.float32)

    _ = benchmark_wrapper(benchmark, eval_fn, y0)


@pytest.mark.benchmark(group="vf_lift_build")
@pytest.mark.parametrize("depth,dim,state_dim,hidden_dim,mlp_depth", BENCH_CASES)
def test_bench_bck_bracket_functions_build(
    benchmark: BenchmarkFixture,
    depth: int,
    dim: int,
    state_dim: int,
    hidden_dim: int,
    mlp_depth: int,
) -> None:
    key = jax.random.PRNGKey(1)
    batched_field = _make_batched_vector_field(key, dim, state_dim, hidden_dim, mlp_depth)
    hopf = GLHopfAlgebra.build(dim, depth)

    def build() -> list[list[Callable[[jax.Array], jax.Array]]]:
        return form_bck_bracket_functions(batched_field, hopf, EuclideanSpace())

    bracket_functions = benchmark(build)
    assert len(bracket_functions) == depth


@pytest.mark.benchmark(group="vf_lift_eval")
@pytest.mark.parametrize("depth,dim,state_dim,hidden_dim,mlp_depth", BENCH_CASES)
def test_bench_bck_bracket_functions_eval(
    benchmark: BenchmarkFixture,
    depth: int,
    dim: int,
    state_dim: int,
    hidden_dim: int,
    mlp_depth: int,
) -> None:
    key = jax.random.PRNGKey(1)
    batched_field = _make_batched_vector_field(key, dim, state_dim, hidden_dim, mlp_depth)
    hopf = GLHopfAlgebra.build(dim, depth)
    bracket_functions = form_bck_bracket_functions(batched_field, hopf, EuclideanSpace())
    eval_fn = _make_eval_fn(bracket_functions)
    y0 = jnp.linspace(0.1, 0.2, num=state_dim, dtype=jnp.float32)

    _ = benchmark_wrapper(benchmark, eval_fn, y0)


@pytest.mark.benchmark(group="vf_lift_build")
@pytest.mark.parametrize("depth,dim,state_dim,hidden_dim,mlp_depth", BENCH_CASES)
def test_bench_mkw_bracket_functions_build(
    benchmark: BenchmarkFixture,
    depth: int,
    dim: int,
    state_dim: int,
    hidden_dim: int,
    mlp_depth: int,
) -> None:
    key = jax.random.PRNGKey(2)
    batched_field = _make_batched_vector_field(key, dim, state_dim, hidden_dim, mlp_depth)
    hopf = MKWHopfAlgebra.build(dim, depth)

    def build() -> list[list[Callable[[jax.Array], jax.Array]]]:
        return form_mkw_bracket_functions(batched_field, hopf, EuclideanSpace())

    bracket_functions = benchmark(build)
    assert len(bracket_functions) == depth


@pytest.mark.benchmark(group="vf_lift_eval")
@pytest.mark.parametrize("depth,dim,state_dim,hidden_dim,mlp_depth", BENCH_CASES)
def test_bench_mkw_bracket_functions_eval(
    benchmark: BenchmarkFixture,
    depth: int,
    dim: int,
    state_dim: int,
    hidden_dim: int,
    mlp_depth: int,
) -> None:
    key = jax.random.PRNGKey(2)
    batched_field = _make_batched_vector_field(key, dim, state_dim, hidden_dim, mlp_depth)
    hopf = MKWHopfAlgebra.build(dim, depth)
    bracket_functions = form_mkw_bracket_functions(batched_field, hopf, EuclideanSpace())
    eval_fn = _make_eval_fn(bracket_functions)
    y0 = jnp.linspace(0.1, 0.2, num=state_dim, dtype=jnp.float32)

    _ = benchmark_wrapper(benchmark, eval_fn, y0)
