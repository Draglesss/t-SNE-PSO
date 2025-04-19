"""Tests for TSNE-PSO."""
# Author: Otmane Fatteh <fattehotmane@hotmail.com>
# License: BSD 3 clause

import numpy as np
import pytest
from sklearn.datasets import load_iris

from tsne_pso import TSNEPSO


def test_tsnepso_init_params():
    """Test TSNEPSO initialization parameters."""
    # Default parameters
    tsne_pso = TSNEPSO()
    assert tsne_pso.n_components == 2
    assert tsne_pso.perplexity == 30.0

    # Custom parameters
    tsne_pso = TSNEPSO(
        n_components=3,
        perplexity=10.0,
        early_exaggeration=6.0,
        learning_rate=50.0,
        n_iter=500,
        n_particles=5,
        inertia_weight=0.7,
        cognitive_weight=1.5,
        social_weight=1.5,
        use_hybrid=False,
        degrees_of_freedom=2.0,
        init="random",
        verbose=1,
        random_state=42,
        method="pso",
        angle=0.3,
        n_jobs=2,
        metric="euclidean",
        metric_params={"p": 2},
    )
    assert tsne_pso.n_components == 3
    assert tsne_pso.perplexity == 10.0
    assert tsne_pso.early_exaggeration == 6.0
    assert tsne_pso.learning_rate == 50.0
    assert tsne_pso.n_iter == 500
    assert tsne_pso.use_hybrid is False
    assert tsne_pso.init == "random"


def test_tsnepso_validation():
    """Test parameter validation in TSNEPSO."""
    # Invalid n_components
    tsne_pso = TSNEPSO(n_components=0)
    with pytest.raises(ValueError, match="n_components.*range.*Got 0"):
        tsne_pso.fit(np.random.randn(10, 5))

    # Invalid perplexity
    tsne_pso = TSNEPSO(perplexity=0)
    with pytest.raises(ValueError, match="perplexity must be greater than 0"):
        tsne_pso.fit(np.random.randn(10, 5))

    # Invalid n_iter
    tsne_pso = TSNEPSO(n_iter=0)
    with pytest.raises(ValueError, match="n_iter.*range"):
        tsne_pso.fit(np.random.randn(10, 5))

    # Invalid method
    tsne_pso = TSNEPSO(method="invalid")
    with pytest.raises(ValueError, match="The method must be 'pso'"):
        tsne_pso.fit(np.random.randn(10, 5))

    # Invalid early_exaggeration
    tsne_pso = TSNEPSO(early_exaggeration=0)
    with pytest.raises(ValueError, match="early_exaggeration.*range"):
        tsne_pso.fit(np.random.randn(10, 5))

    # Invalid n_particles
    tsne_pso = TSNEPSO(n_particles=0)
    with pytest.raises(ValueError, match="n_particles.*range"):
        tsne_pso.fit(np.random.randn(10, 5))

    # Invalid inertia_weight
    tsne_pso = TSNEPSO(inertia_weight=1.5)
    with pytest.raises(ValueError, match="inertia_weight.*range"):
        tsne_pso.fit(np.random.randn(10, 5))


def test_tsnepso_iris():
    """Test TSNEPSO on the Iris dataset."""
    # Load data
    iris = load_iris()
    X = iris.data
    n_samples, n_features = X.shape

    # Fit TSNEPSO with reduced iterations for testing
    tsne_pso = TSNEPSO(
        n_components=2,
        perplexity=10.0,
        n_iter=50,  # Low number for faster tests
        n_particles=3,  # Low number for faster tests
        random_state=0,
    )
    embedding = tsne_pso.fit_transform(X)

    # Check shape of results
    assert embedding.shape == (n_samples, 2)

    # Check attributes
    assert hasattr(tsne_pso, "embedding_")
    assert hasattr(tsne_pso, "kl_divergence_")
    assert hasattr(tsne_pso, "n_iter_")

    # Check feature names out
    feature_names = tsne_pso.get_feature_names_out()
    assert feature_names.shape == (2,)
    assert feature_names[0] == "tsnepso0"
    assert feature_names[1] == "tsnepso1"


def test_tsnepso_iris_init_array():
    """Test TSNEPSO with array initialization."""
    # Load data
    iris = load_iris()
    X = iris.data
    n_samples = X.shape[0]

    # Create a custom initialization
    init_embedding = np.random.RandomState(0).normal(0, 0.0001, (n_samples, 2))

    # Fit TSNEPSO with the initial embedding
    tsne_pso = TSNEPSO(
        n_components=2,
        perplexity=10.0,
        n_iter=50,  # Low number for faster tests
        n_particles=3,  # Low number for faster tests
        init=init_embedding,
        random_state=0,
    )
    embedding = tsne_pso.fit_transform(X)

    # Check shape of results
    assert embedding.shape == (n_samples, 2)

    # Test for wrong shape init
    wrong_shape_init = np.random.RandomState(0).normal(0, 0.0001, (n_samples, 3))
    tsne_pso = TSNEPSO(init=wrong_shape_init, n_components=2)
    with pytest.raises(ValueError, match="init.shape=.*but should be"):
        tsne_pso.fit(X)


def test_tsnepso_transform_raises():
    """Test that transform raises NotImplementedError."""
    # First fit the model
    tsne_pso = TSNEPSO(
        n_components=2,
        perplexity=5.0,
        n_iter=10,
        n_particles=2,
        random_state=42,
    )
    
    X = np.random.RandomState(42).normal(0, 1, (20, 5))
    tsne_pso.fit(X)
    
    # Now try to transform new data
    with pytest.raises(NotImplementedError, match="t-SNE does not support the transform"):
        tsne_pso.transform(np.random.randn(10, 5))


def test_tsnepso_precomputed():
    """Test TSNEPSO with precomputed distances."""
    # Load data
    iris = load_iris()
    X = iris.data
    n_samples = X.shape[0]

    # Compute pairwise distances
    from sklearn.metrics import pairwise_distances
    distances = pairwise_distances(X, metric="euclidean", squared=True)

    # Fit TSNEPSO with precomputed distances
    tsne_pso = TSNEPSO(
        n_components=2,
        perplexity=10.0,
        n_iter=50,  # Low number for faster tests
        n_particles=3,  # Low number for faster tests
        metric="precomputed",
        random_state=0,
    )
    embedding = tsne_pso.fit_transform(distances)

    # Check shape of results
    assert embedding.shape == (n_samples, 2)

    # Test for non-square distance matrix
    non_square = np.random.rand(10, 5)
    tsne_pso = TSNEPSO(metric="precomputed")
    with pytest.raises(ValueError, match="X should be a square distance matrix"):
        tsne_pso.fit(non_square)

    # Test for negative distances
    negative_dist = -np.ones((10, 10))
    tsne_pso = TSNEPSO(metric="precomputed")
    with pytest.raises(ValueError, match="Precomputed distance contains negative values"):
        tsne_pso.fit(negative_dist)


def test_tsnepso_perplexity_warning():
    """Test warning for perplexity too large for the dataset."""
    # Create a small dataset with fixed size
    n_samples = 20
    X = np.random.RandomState(42).randn(n_samples, 5)

    # Perplexity equal to n_samples should trigger a warning
    tsne_pso = TSNEPSO(perplexity=n_samples)
    with pytest.warns(UserWarning, match="Perplexity.*should be less than n_samples"):
        tsne_pso.fit(X)


def test_tsnepso_random_state():
    """Test that random_state controls reproducibility."""
    # Load data
    iris = load_iris()
    X = iris.data[:20]  # Use a subset for faster testing

    # First run with fixed random state
    tsne_pso1 = TSNEPSO(
        n_components=2,
        perplexity=5.0,
        n_iter=50,
        n_particles=3,
        random_state=42,
    )
    embedding1 = tsne_pso1.fit_transform(X)

    # Second run with the same random state
    tsne_pso2 = TSNEPSO(
        n_components=2,
        perplexity=5.0,
        n_iter=50,
        n_particles=3,
        random_state=42,
    )
    embedding2 = tsne_pso2.fit_transform(X)

    # The embeddings should be identical
    assert np.allclose(embedding1, embedding2)
