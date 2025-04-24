"""Tests for TSNE-PSO implementation."""

# Author: Otmane Fatteh <fattehotmane@hotmail.com>
# License: BSD 3 clause

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from tsne_pso import TSNEPSO


def test_tsnepso_basic():
    """Test basic functionality of TSNEPSO."""
    # Create a small synthetic dataset
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    # Create and fit the model with minimal iterations for testing
    model = TSNEPSO(
        n_components=2,
        perplexity=2,  # Small perplexity for small dataset
        n_iter=5,  # Minimal iterations for testing
        n_particles=2,  # Minimal particles for testing
        random_state=42,
    )

    # Test fit method
    model.fit(X)
    assert hasattr(model, "embedding_")
    assert model.embedding_.shape == (4, 2)

    # Test fit_transform method
    X_embedded = model.fit_transform(X)
    assert X_embedded.shape == (4, 2)

    # Test transform method raises NotImplementedError
    with pytest.raises(NotImplementedError):
        model.transform(X)


def test_tsnepso_iris():
    """Test TSNEPSO on the Iris dataset."""
    # Load the iris dataset
    iris = load_iris()
    X = iris.data

    # Standardize the data
    X = StandardScaler().fit_transform(X)

    # Create and fit the model with minimal iterations for testing
    model = TSNEPSO(
        n_components=2,
        perplexity=10,
        n_iter=5,  # Minimal iterations for testing
        n_particles=2,  # Minimal particles for testing
        random_state=42,
    )

    # Test fit_transform method
    X_embedded = model.fit_transform(X)
    assert X_embedded.shape == (150, 2)

    # Check that the embedding has finite values
    assert np.all(np.isfinite(X_embedded))


def test_parameter_validation():
    """Test parameter validation in TSNEPSO."""
    # Test invalid perplexity
    with pytest.raises(ValueError):
        TSNEPSO(perplexity=-1)

    # Test invalid method
    with pytest.raises(ValueError):
        TSNEPSO(method="not_pso")

    # Test invalid n_components
    with pytest.raises(ValueError):
        TSNEPSO(n_components=0)


def test_init_options():
    """Test different initialization options."""
    # Create a small dataset
    X = np.random.rand(10, 5)

    # Test PCA initialization
    model = TSNEPSO(init="pca", n_iter=2, n_particles=2, random_state=42)
    X_pca = model.fit_transform(X)
    assert X_pca.shape == (10, 2)

    # Test random initialization
    model = TSNEPSO(init="random", n_iter=2, n_particles=2, random_state=42)
    X_random = model.fit_transform(X)
    assert X_random.shape == (10, 2)

    # Test custom initialization
    init_embedding = np.random.rand(10, 2) * 0.0001
    model = TSNEPSO(init=init_embedding, n_iter=2, n_particles=2, random_state=42)
    X_custom = model.fit_transform(X)
    assert X_custom.shape == (10, 2)
