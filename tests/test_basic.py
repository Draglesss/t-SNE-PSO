"""Basic tests for TSNE-PSO package."""

# Author: Otmane Fatteh <fattehotmane@hotmail.com>
# License: BSD 3 clause


def test_import():
    """Test that the package can be imported."""
    from tsne_pso import TSNEPSO

    assert TSNEPSO is not None

    # Check version
    import tsne_pso

    assert hasattr(tsne_pso, "__version__")
