import jax
import jax.numpy as jnp
from jax import lax
import pytest
from typing import Callable
from pytest_benchmark.fixture import BenchmarkFixture
from stochastax.hopf_algebras.forest_types import Forest
import numpy as np

_test_key = jax.random.PRNGKey(42)

BENCH_SHUFFLE_CASES: list = [
    pytest.param(2, 2, id="dim-2-depth-2"),
    pytest.param(8, 2, id="dim-8-depth-2"),
    pytest.param(2, 3, id="dim-2-depth-3"),
    pytest.param(8, 3, id="dim-8-depth-3"),
]


BENCH_GL_CASES: list = [
    pytest.param(2, 2, id="dim-2-depth-2"),
    pytest.param(8, 2, id="dim-8-depth-2"),
    pytest.param(2, 3, id="dim-2-depth-3"),
    pytest.param(8, 3, id="dim-8-depth-3"),
    pytest.param(2, 5, id="dim-8-depth-5"),
]

BENCH_MKW_CASES: list = [
    pytest.param(2, 2, id="dim-2-depth-2"),
    pytest.param(8, 2, id="dim-8-depth-2"),
    pytest.param(2, 3, id="dim-2-depth-3"),
    pytest.param(8, 3, id="dim-8-depth-3"),
    pytest.param(2, 5, id="dim-8-depth-5"),
]


def _block(x):
    return jax.tree_util.tree_map(lambda y: y.block_until_ready(), x)


def _maybe_device_put(x):
    # Put only array-like values on device; leave Python scalars/objects alone.
    if isinstance(x, (jax.Array, np.ndarray)):
        return jax.device_put(x)
    return x


def benchmark_wrapper(
    benchmark: BenchmarkFixture,
    func: Callable,
    *args,
    jit: bool = True,
    **kwargs,
):
    args = jax.tree_util.tree_map(_maybe_device_put, args)
    kwargs = jax.tree_util.tree_map(_maybe_device_put, kwargs)
    f = jax.jit(func) if jit else func

    # Warm-up: includes tracing/compile (if jit=True) and one execution
    warmed = _block(f(*args, **kwargs))

    def run():
        _block(f(*args, **kwargs))

    benchmark(run)
    return warmed


@pytest.fixture
def scalar_path_fixture(request: pytest.FixtureRequest) -> jax.Array:
    """Generates a path with a given number of channels and timesteps."""
    n_features, num_timesteps = request.param
    key, subkey = jax.random.split(_test_key)
    return generate_scalar_path(subkey, n_features, num_timesteps)


@pytest.fixture
def linear_path_fixture(request: pytest.FixtureRequest) -> jax.Array:
    """Generates a linear path with a given number of channels and timesteps."""
    n_features, num_timesteps = request.param
    return generate_linear_path(n_features, num_timesteps)


@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("num_timesteps", [10, 100])
def test_gbm_shape_and_positive(num_timesteps: int, n_features: int) -> None:
    """GBM path has expected shape and stays strictly positive."""
    global _test_key
    _test_key, subkey = jax.random.split(_test_key)

    path = generate_scalar_path(subkey, n_features, num_timesteps)
    assert path.shape == (num_timesteps, n_features)
    assert jnp.all(path > 0.0), "GBM should remain positive"


@pytest.fixture
def brownian_path_fixture(request: pytest.FixtureRequest) -> jax.Array:
    """Generates a Brownian path W with W[0]=0, shape (num_steps+1, dim)."""
    dim, num_steps = request.param
    global _test_key
    _test_key, subkey = jax.random.split(_test_key)
    return generate_brownian_path(subkey, dim, num_steps)


def generate_scalar_path(
    key: jax.Array,
    n_features: int,
    num_timesteps: int,
    mu: float = 0.5,
    sigma: float = 0.3,
) -> jax.Array:
    """
    Generate a multi-dimensional path following geometric Brownian motion (GBM).
    Uses JAX for random number generation and path computation.

    Args:
        key: JAX PRNGKey for random number generation.
        num_timesteps: Number of timesteps in the path.
        mu: Drift coefficient (default: 0.5).
        sigma: Volatility coefficient (default: 0.3).
        n_features: Number of dimensions/features in the path (default: 1).

    Returns:
        Tuple containing:
        - timestamps: JAX Array of timestamps, shape (num_timesteps,)
        - values: JAX Array of path values following GBM, shape (num_timesteps, n_features)
    """
    dtype = jnp.float32
    dt = jnp.array(1.0 / (num_timesteps - 1), dtype=dtype)
    std_normal_increments = jax.random.normal(key, (num_timesteps - 1, n_features), dtype=dtype)

    # Scale increments to be N(0, sqrt(dt)) for the GBM formula
    dW_increments = std_normal_increments * jnp.sqrt(dt)  # Shape: (num_timesteps - 1, n_features)

    initial_path_value = jnp.ones(n_features, dtype=dtype)  # S_0, shape (n_features,)

    def gbm_euler_step(s_prev: jax.Array, dW_i: jax.Array) -> tuple[jax.Array, jax.Array]:
        # s_prev: path value at t-1, shape (n_features,)
        # dW_i: Brownian increment N(0,sqrt(dt)) for the step, shape (n_features,)
        s_next = s_prev * (1 + mu * dt + sigma * dW_i)
        return s_next, s_next  # (new_carry_state, value_to_scan_out)

    # Perform the scan over the time steps using dW_increments
    # initial_carry is S_0
    _, path_values_from_t1 = lax.scan(gbm_euler_step, initial_path_value, dW_increments)
    # path_values_from_t1 contains S_1, S_2, ..., S_{T-1} if num_timesteps-1 increments
    # Shape: (num_timesteps - 1, n_features)

    # Prepend S_0 to the path
    s0_reshaped = initial_path_value.reshape(1, n_features)  # Shape: (1, n_features)
    values = jnp.concatenate([s0_reshaped, path_values_from_t1], axis=0)
    # values shape: (num_timesteps, n_features)

    return values


def generate_linear_path(
    n_features: int, start: float = 0.0, stop: float = 1.0, num_timesteps: int = 100
) -> jax.Array:
    """Deterministic straight-line path for ground-truth tests."""
    t = jnp.linspace(start, stop, num_timesteps).reshape(-1, 1)  # (steps, 1)
    vals = jnp.repeat(t, n_features, axis=1)  # (steps, n_features)  # (steps, n_features)
    return vals


def generate_brownian_path(
    key: jax.Array,
    dim: int,
    num_steps: int,
    dt: float | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Generate a (num_steps+1, dim) Brownian path starting at zero."""
    if dt is None:
        dt = 1.0 / float(num_steps)
    dt_arr = jnp.array(dt, dtype=dtype)
    dW = jax.random.normal(key, (num_steps, dim), dtype=dtype) * jnp.sqrt(dt_arr)
    W0 = jnp.zeros((1, dim), dtype=dtype)
    W = jnp.vstack([W0, jnp.cumsum(dW, axis=0)])
    return W


def forest_from_parents(parents: list[list[int]]) -> Forest:
    """Create a Forest from a list of parent arrays (one per tree)."""
    arr = jnp.asarray(parents, dtype=jnp.int32)
    return Forest(parent=arr)


def chain_parent(n_nodes: int) -> list[int]:
    """Return parent array for a preorder linear chain of length n_nodes."""
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")
    return [-1] + list(range(0, n_nodes - 1))


def binary_root_parent() -> list[int]:
    """Return parent array for a 3-node tree with two children at the root."""
    return [-1, 0, 0]


def power_matrix(A: jax.Array, k: int) -> jax.Array:
    """Compute A^k for square matrix A and integer k >= 0."""
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return jnp.eye(A.shape[0], dtype=A.dtype)
    acc = A
    for _ in range(1, k):
        acc = acc @ A
    return acc


def linear_field(A: jax.Array) -> Callable[[jax.Array], jax.Array]:
    """Return f(x) = A @ x."""

    def f(x: jax.Array) -> jax.Array:
        return A @ x

    return f


def elementwise_square_field() -> Callable[[jax.Array], jax.Array]:
    """Return f(x) = x^2 elementwise."""

    def f(x: jax.Array) -> jax.Array:
        return jnp.square(x)

    return f


def identity_projection(y: jax.Array, v: jax.Array) -> jax.Array:
    """Projection that returns the input vector unchanged."""
    return v


def sphere_tangent_projection(y: jax.Array, v: jax.Array) -> jax.Array:
    """Project v onto the tangent space at y on the unit sphere: v - (v·y) y."""
    return v - jnp.dot(v, y) * y


def _so3_generators() -> jax.Array:
    """Return the three generators of so(3) Lie algebra as a [3, 3, 3] array."""
    A1 = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    A2 = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=jnp.float32)
    A3 = jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    return jnp.stack([A1, A2, A3], axis=0)  # [3, 3, 3]
