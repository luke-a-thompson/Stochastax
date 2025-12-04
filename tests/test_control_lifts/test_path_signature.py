import jax
import jax.numpy as jnp
import pytest
from stochastax.control_lifts.path_signature import compute_path_signature
from stochastax.analytics.signature_sizes import get_signature_dim
from stochastax.hopf_algebras.hopf_algebras import ShuffleHopfAlgebra
import signax
import math

_test_key = jax.random.PRNGKey(42)


@pytest.mark.parametrize("scalar_path_fixture", [(1, 10), (2, 10)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_signature_shape_full(scalar_path_fixture: jax.Array, depth: int) -> None:
    """Signature tensor dimension matches algebraic formula."""
    channels = int(scalar_path_fixture.shape[1])
    hopf = ShuffleHopfAlgebra.build(ambient_dim=channels, depth=depth)
    sig = compute_path_signature(scalar_path_fixture, depth=depth, hopf=hopf, mode="full").flatten()
    expected_dim = get_signature_dim(depth, channels)
    expected_shape = (expected_dim,)
    assert sig.shape == expected_shape, f"Expected shape {expected_shape}, got {sig.shape}"


@pytest.mark.parametrize("scalar_path_fixture", [(1, 10), (2, 20)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_signature_shape_stream(scalar_path_fixture: jax.Array, depth: int) -> None:
    """Signature tensor dimension matches algebraic formula."""
    num_steps, channels = scalar_path_fixture.shape
    channels = int(channels)
    hopf = ShuffleHopfAlgebra.build(ambient_dim=channels, depth=depth)
    sigs = compute_path_signature(scalar_path_fixture, depth=depth, hopf=hopf, mode="stream")

    assert len(sigs) == num_steps - 1
    sig_array = jnp.stack([s.flatten() for s in sigs])

    expected_dim = get_signature_dim(depth, channels)
    expected_shape = (num_steps - 1, expected_dim)
    assert sig_array.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {sig_array.shape}"
    )


@pytest.mark.parametrize("scalar_path_fixture", [(1, 10), (2, 10)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_signature_shape_incremental(scalar_path_fixture: jax.Array, depth: int) -> None:
    """Signature tensor dimension matches algebraic formula."""
    num_steps, channels = scalar_path_fixture.shape
    channels = int(channels)
    hopf = ShuffleHopfAlgebra.build(ambient_dim=channels, depth=depth)
    sigs = compute_path_signature(scalar_path_fixture, depth=depth, hopf=hopf, mode="incremental")
    assert len(sigs) == num_steps - 1
    sig_array = jnp.stack([s.flatten() for s in sigs])

    expected_dim = get_signature_dim(depth, channels)
    expected_shape = (num_steps - 1, expected_dim)
    assert sig_array.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {sig_array.shape}"
    )


@pytest.mark.parametrize("linear_path_fixture", [(1, 20)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_linear_path_exactness(linear_path_fixture: jax.Array, depth: int) -> None:
    r"""
    For a straight-line 1-D path $$x(t) = \alpha t$$ with total increment $$\Delta x$$,
    the level-k iterated integral equals $$(\Delta x)^k / k!$$.
    """
    path = linear_path_fixture
    delta_x = path[-1, 0] - path[0, 0]

    channels = int(path.shape[1])
    hopf = ShuffleHopfAlgebra.build(ambient_dim=channels, depth=depth)
    sig = compute_path_signature(path, depth=depth, hopf=hopf, mode="full")
    signature = sig.flatten()

    expected = jnp.array([delta_x**k / math.factorial(k) for k in range(1, depth + 1)])

    assert jnp.allclose(signature, expected, atol=1e-4, rtol=1e-4), (
        f"Signature {signature} does not match expected {expected}"
    )


def test_zero_path_vanishes() -> None:
    """
    A constant path has zero increment, hence zero signature beyond level 0.
    """
    const_path = jnp.zeros((50, 2), dtype=jnp.float32)
    hopf = ShuffleHopfAlgebra.build(ambient_dim=2, depth=4)
    sig = compute_path_signature(const_path, depth=4, hopf=hopf, mode="full")
    signature = sig.flatten()
    assert jnp.allclose(signature, 0.0)


@pytest.mark.parametrize(
    "a, b",
    [
        (1.0, 1.0),
        (2.0, 1.0),
        (1.0, 3.0),
        (-1.0, 2.0),
    ],
)
def test_quadratic_path_signature(a: float, b: float) -> None:
    """
    Tests the signature of a 2D path x(t) = (a*t, b*t^2/2).
    The analytical signature is known and can be compared against.
    """
    T = 1.0
    num_steps = 1000
    depth = 2

    # Create the path x(t) = (a*t, b*t^2/2)
    t = jnp.linspace(0, T, num_steps)
    path_x = a * t
    path_y = b * t**2 / 2.0
    path = jnp.stack([path_x, path_y], axis=-1)
    hopf = ShuffleHopfAlgebra.build(ambient_dim=2, depth=depth)

    # Compute the signature using the function
    sig = compute_path_signature(path, depth=depth, hopf=hopf, mode="full")
    signature = sig.flatten()

    # Analytical signature
    # S_1 = (a*T, b*T^2/2)
    # S_2 = (a^2*T^2/2, a*b*T^3/3, a*b*T^3/6, b^2*T^4/8)
    expected = jnp.array(
        [
            a * T,
            b * T**2 / 2.0,
            a**2 * T**2 / 2.0,
            a * b * T**3 / 3.0,
            a * b * T**3 / 6.0,
            b**2 * T**4 / 8.0,
        ]
    )

    assert jnp.allclose(signature, expected, atol=1e-5, rtol=1e-5), (
        f"Signature {signature} does not match expected {expected}"
    )


@pytest.mark.parametrize("scalar_path_fixture", [(1, 10), (2, 10)], indirect=True)
@pytest.mark.parametrize("depth", [2, 3])
def test_quicksig_signax_equivalence_full(scalar_path_fixture: jax.Array, depth: int) -> None:
    """
    Test that the signature computed by QuickSig and Signax are equivalent.
    """
    path = scalar_path_fixture
    channels = int(path.shape[1])
    hopf = ShuffleHopfAlgebra.build(ambient_dim=channels, depth=depth)
    quicksig_sig = compute_path_signature(path, depth=depth, hopf=hopf, mode="full")
    quicksig_sig = quicksig_sig.flatten()
    signax_sig = signax.signature(path, depth=depth, stream=False)
    assert jnp.allclose(quicksig_sig, signax_sig, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("scalar_path_fixture", [(1, 10), (2, 10)], indirect=True)
@pytest.mark.parametrize("depth", [2, 3])
def test_quicksig_signax_equivalence_stream(scalar_path_fixture: jax.Array, depth: int) -> None:
    """
    Test that the signature computed by QuickSig and Signax are equivalent.
    """
    path = scalar_path_fixture
    channels = int(path.shape[1])
    hopf = ShuffleHopfAlgebra.build(ambient_dim=channels, depth=depth)
    quicksig_sig = compute_path_signature(path, depth=depth, hopf=hopf, mode="stream")
    quicksig_sig = jnp.stack([s.flatten() for s in quicksig_sig])
    signax_sig = signax.signature(path, depth=depth, stream=True)
    assert jnp.allclose(quicksig_sig, signax_sig, atol=1e-5, rtol=1e-5)
