"""Tests for internal functions of t-SNE PSO."""

import sys
import warnings
from io import StringIO

import numpy as np
import pytest

# Import internal functions directly from tsne_pso
from tsne_pso._tsne_pso import (
    MACHINE_EPSILON,
    _gradient_descent_step,
    _joint_probabilities,
    _kl_divergence,
)

try:
    from tsne_pso._tsne_pso import SimpleProgress
except ImportError:
    # Create mock for SimpleProgress if not directly importable
    class SimpleProgress:
        def __init__(self, *args, **kwargs):
            pass


class TestSimpleProgress:
    """Tests for the SimpleProgress class."""

    def test_simple_progress_functionality(self):
        """Test SimpleProgress iterator functionality."""
        # Skip if not importable
        if not hasattr(SimpleProgress, "__iter__"):
            pytest.skip("SimpleProgress not directly importable")

        # Capture stdout to test logging
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        try:
            # Create progress object
            total = 10
            progress = SimpleProgress(total, desc="TestProgress")

            # Test iteration
            for _ in progress:
                pass

            # Ensure final message was logged
            output = mystdout.getvalue()
            assert "TestProgress: 100%" in output

        finally:
            sys.stdout = old_stdout


def test_joint_probabilities_edge_cases():
    """Test edge cases for joint probability calculation."""
    # Test with tiny distances
    n_samples = 5
    distances = np.zeros((n_samples, n_samples))
    # Add small values to diagonal to avoid division by zero
    np.fill_diagonal(distances, 1e-12)

    # Should handle this edge case without errors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P = _joint_probabilities(distances, perplexity=2.0, verbose=False)

    # Output should have correct shape
    expected_size = (n_samples * (n_samples - 1)) // 2
    assert P.shape == (expected_size,)
    assert np.all(np.isfinite(P))

    # Test with large distances
    distances = np.ones((n_samples, n_samples)) * 1e6
    np.fill_diagonal(distances, 0.0)

    # Should handle large values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P = _joint_probabilities(distances, perplexity=2.0, verbose=False)

    assert P.shape == (expected_size,)
    assert np.all(np.isfinite(P))


def test_kl_divergence_optimization():
    """Test KL divergence calculation in optimization context."""
    n_samples = 5
    n_components = 2

    # Create embedding and P matrix
    rng = np.random.RandomState(42)
    embedding = rng.randn(n_samples, n_components)
    params = embedding.ravel()

    # Create P matrix (joint probabilities)
    P_shape = (n_samples * (n_samples - 1)) // 2
    P = np.ones(P_shape) / P_shape  # Uniform distribution

    # Test standard calculation
    kl1, grad1 = _kl_divergence(
        params,
        P,
        degrees_of_freedom=1.0,
        n_samples=n_samples,
        n_components=n_components,
    )

    # Test with extreme degrees of freedom
    kl2, grad2 = _kl_divergence(
        params,
        P,
        degrees_of_freedom=0.1,
        n_samples=n_samples,
        n_components=n_components,
    )

    # Results should differ
    assert kl1 != kl2
    assert not np.allclose(grad1, grad2)

    # Test with skip_num_points
    kl3, grad3 = _kl_divergence(
        params,
        P,
        degrees_of_freedom=1.0,
        n_samples=n_samples,
        n_components=n_components,
        skip_num_points=2,
    )

    # Verify the gradient shape is correct
    assert grad3.shape == params.shape
    assert np.all(np.isfinite(grad3))

    # Test without error computation
    kl4, grad4 = _kl_divergence(
        params,
        P,
        degrees_of_freedom=1.0,
        n_samples=n_samples,
        n_components=n_components,
        compute_error=False,
    )

    # KL should be NaN, grad should still be computed
    assert np.isnan(kl4)
    assert not np.all(
        grad4 == 0
    )  # Changed from allclose to ensure grad4 has non-zero values


def test_gradient_descent_optimizers():
    """Test gradient descent optimization steps."""
    n_samples = 5
    n_components = 2

    # Create random embedding and P matrix
    rng = np.random.RandomState(42)
    params = rng.randn(n_samples * n_components)
    P_shape = (n_samples * (n_samples - 1)) // 2
    P = np.ones(P_shape) / P_shape  # Uniform distribution

    # Test basic gradient descent step
    params1, error1, update1, gains1 = _gradient_descent_step(
        params.copy(),
        P,
        degrees_of_freedom=1.0,
        n_samples=n_samples,
        n_components=n_components,
    )

    # Params should be updated
    assert not np.array_equal(params, params1)

    # Test with modified hyperparameters
    params2, error2, update2, gains2 = _gradient_descent_step(
        params.copy(),
        P,
        degrees_of_freedom=1.0,
        n_samples=n_samples,
        n_components=n_components,
        momentum=0.5,  # Lower momentum
        learning_rate=500.0,  # Higher learning rate
    )

    # Updates should be different - use a more specific comparison
    # that tolerates small numerical differences but catches actual differences
    assert np.sum(np.abs(params1 - params2)) > 1e-10

    # Test with pre-initialized gains and update
    initial_update = np.zeros_like(params)
    initial_gains = np.ones_like(params)

    params3, error3, update3, gains3 = _gradient_descent_step(
        params.copy(),
        P,
        degrees_of_freedom=1.0,
        n_samples=n_samples,
        n_components=n_components,
        update=initial_update,
        gains=initial_gains,
    )

    # Updates and gains should now be modified - test more directly
    assert np.any(np.abs(update3) > 1e-10)  # Should have some non-zero values
    assert np.any(np.abs(gains3 - 1.0) > 1e-10)  # Should have changed from all ones

    # Test multiple steps
    current_params = params.copy()
    current_update = initial_update.copy()
    current_gains = initial_gains.copy()

    for _ in range(3):
        current_params, error, current_update, current_gains = _gradient_descent_step(
            current_params,
            P,
            degrees_of_freedom=1.0,
            n_samples=n_samples,
            n_components=n_components,
            update=current_update,
            gains=current_gains,
        )

    # After multiple steps, should have moved significantly
    assert np.sum((current_params - params) ** 2) > np.sum((params1 - params) ** 2)


def test_machine_epsilon_usage():
    """Test handling of very small values with machine epsilon."""
    # Create a small test case
    n_samples = 3
    n_components = 2

    # Create embedding with some values very close to zero
    params = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10, 6e-10])

    # Create P matrix with some very small values
    P = np.array([MACHINE_EPSILON, MACHINE_EPSILON, MACHINE_EPSILON])

    # This should not produce NaN or inf values
    kl, grad = _kl_divergence(
        params,
        P,
        degrees_of_freedom=1.0,
        n_samples=n_samples,
        n_components=n_components,
    )

    # Results should be finite
    assert np.isfinite(kl)
    assert np.all(np.isfinite(grad))
