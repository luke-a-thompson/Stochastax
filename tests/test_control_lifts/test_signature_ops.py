import jax
import jax.numpy as jnp
import pytest
from stochastax.control_lifts.path_signature import compute_path_signature
from stochastax.hopf_algebras.hopf_algebras import ShuffleHopfAlgebra


@pytest.mark.parametrize("scalar_path_fixture", [(1, 20), (2, 30)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_chen_identity(scalar_path_fixture: jax.Array, depth: int):
    path = scalar_path_fixture
    n_features = int(path.shape[1])
    hopf = ShuffleHopfAlgebra.build(ambient_dim=n_features, depth=depth)
    midpoint_idx = len(path) // 2

    # Split the path into two overlapping segments
    path_1 = path[: midpoint_idx + 1]
    path_2 = path[midpoint_idx:]

    # Compute signatures for each sub-path.
    sig_1 = compute_path_signature(path_1, depth=depth, hopf=hopf, mode="full", index_start=0)
    sig_2 = compute_path_signature(
        path_2, depth=depth, hopf=hopf, mode="full", index_start=midpoint_idx
    )

    # Combine the signatures using Chen's identity
    combined_sig = sig_1 @ sig_2

    # Compute the signature over the whole path for comparison
    whole_sig = compute_path_signature(path, depth=depth, hopf=hopf, mode="full")

    # The flattened signature values should be the same
    assert jnp.allclose(combined_sig.flatten(), whole_sig.flatten(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("scalar_path_fixture", [(1, 30), (2, 30)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_chen_identity_three_signatures(scalar_path_fixture: jax.Array, depth: int):
    path = scalar_path_fixture
    n_features = int(path.shape[1])
    hopf = ShuffleHopfAlgebra.build(ambient_dim=n_features, depth=depth)
    third_point_idx = len(path) // 3
    two_thirds_point_idx = 2 * len(path) // 3

    # Split the path into three overlapping segments
    path_1 = path[: third_point_idx + 1]
    path_2 = path[third_point_idx : two_thirds_point_idx + 1]
    path_3 = path[two_thirds_point_idx:]

    # Compute signatures for each sub-path
    sig_1 = compute_path_signature(path_1, depth=depth, hopf=hopf, mode="full", index_start=0)
    sig_2 = compute_path_signature(
        path_2, depth=depth, hopf=hopf, mode="full", index_start=third_point_idx
    )
    sig_3 = compute_path_signature(
        path_3, depth=depth, hopf=hopf, mode="full", index_start=two_thirds_point_idx
    )

    # The @ operator is the Chen identity for signatures.
    combined_sig = sig_1 @ sig_2 @ sig_3

    # Compute the signature over the whole path for comparison
    whole_sig = compute_path_signature(path, depth=depth, hopf=hopf, mode="full")

    # The flattened signature values should be the same
    assert jnp.allclose(combined_sig.flatten(), whole_sig.flatten(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("scalar_path_fixture", [(1, 20)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_chen_identity_non_consecutive_intervals(scalar_path_fixture: jax.Array, depth: int):
    path = scalar_path_fixture
    n_features = int(path.shape[1])
    hopf = ShuffleHopfAlgebra.build(ambient_dim=n_features, depth=depth)
    midpoint_idx = len(path) // 2

    # Split the path into two overlapping segments
    path_1 = path[: midpoint_idx + 1]
    path_2 = path[midpoint_idx:]

    # Compute signatures for each sub-path.
    sig_1 = compute_path_signature(path_1, depth=depth, hopf=hopf, mode="full", index_start=0)
    sig_2_non_consecutive = compute_path_signature(
        path_2, depth=depth, hopf=hopf, mode="full", index_start=0
    )

    # Check that combining signatures with non-consecutive intervals raises a ValueError
    with pytest.raises(ValueError):
        _ = sig_1 @ sig_2_non_consecutive


@pytest.mark.parametrize("scalar_path_fixture", [(1, 20)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_chen_identity_mismatched_ambient_dimension(scalar_path_fixture: jax.Array, depth: int):
    path = scalar_path_fixture
    n_features = int(path.shape[1])
    hopf = ShuffleHopfAlgebra.build(ambient_dim=n_features, depth=depth)
    midpoint_idx = len(path) // 2

    # Split the path into two overlapping segments
    path_1 = path[: midpoint_idx + 1]
    path_2 = path[midpoint_idx:]

    # Compute signatures for each sub-path.
    sig_1 = compute_path_signature(path_1, depth=depth, hopf=hopf, mode="full", index_start=0)

    # Pad path_2 with one zero-feature to change ambient dimension
    path_2_padded = jnp.pad(path_2, ((0, 0), (0, 1)))
    n_features_padded = int(path_2_padded.shape[1])
    hopf_padded = ShuffleHopfAlgebra.build(ambient_dim=n_features_padded, depth=depth)
    sig_2_padded = compute_path_signature(
        path_2_padded,
        depth=depth,
        hopf=hopf_padded,
        mode="full",
        index_start=midpoint_idx,
    )

    # Check that combining signatures with different ambient dimensions raises a ValueError
    with pytest.raises(ValueError):
        _ = sig_1 @ sig_2_padded


@pytest.mark.parametrize("scalar_path_fixture", [(1, 20)], indirect=True)
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_chen_identity_mismatched_depth(scalar_path_fixture: jax.Array, depth: int):
    path = scalar_path_fixture
    n_features = int(path.shape[1])
    hopf = ShuffleHopfAlgebra.build(ambient_dim=n_features, depth=depth)
    midpoint_idx = len(path) // 2

    # Split the path into two overlapping segments
    path_1 = path[: midpoint_idx + 1]
    path_2 = path[midpoint_idx:]

    # Compute signatures for each sub-path.
    sig_1 = compute_path_signature(path_1, depth=depth, hopf=hopf, mode="full", index_start=0)
    hopf_deeper = ShuffleHopfAlgebra.build(ambient_dim=n_features, depth=depth + 1)
    sig_2_deeper = compute_path_signature(
        path_2, depth=depth + 1, hopf=hopf_deeper, mode="full", index_start=midpoint_idx
    )

    # Check that combining signatures with different depths raises a ValueError
    with pytest.raises(ValueError):
        _ = sig_1 @ sig_2_deeper
