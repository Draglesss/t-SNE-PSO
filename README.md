# TSNE-PSO

[![PyPI version](https://badge.fury.io/py/tsne-pso.svg?icon=si%3Apython)](https://badge.fury.io/py/tsne-pso)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

t-Distributed Stochastic Neighbor Embedding with Particle Swarm Optimization (TSNE-PSO) is an enhanced version of t-SNE that uses Particle Swarm Optimization instead of gradient descent for the optimization step. This implementation is based on the research paper by Allaoui et al. (2025).

## Features

- **Improved Optimization**: Uses Particle Swarm Optimization for better optimization with less susceptibility to local minima
- **Multiple Initialization Options**: Supports initialization using PCA, UMAP, t-SNE, or custom embeddings
- **Hybrid Approach**: Optional hybrid optimization combining PSO with gradient descent steps
- **Highly Customizable**: Fine-tune parameters for particles, inertia, cognitive/social weights, and more
- **scikit-learn Compatible**: Follows scikit-learn's API conventions for easy integration

## New in 1.1.6:

*   **Lightning Fast:** Up to 62% faster embedding times!
*   **Sharper Results:** 15% lower KL divergence scores for clearer cluster separation
*   **Super Efficient:** Reaches optimal embeddings in 73% fewer steps

## Installation

Install the latest stable version from PyPI:

```bash
pip install tsne-pso
```

### Dependencies

- numpy
- scipy
- scikit-learn
- umap-learn (optional, for UMAP initialization)
- tqdm (optional, for progress bars)

## Quick Start

```python
from tsne_pso import TSNEPSO
import numpy as np
from sklearn.datasets import load_iris

# Load example data
iris = load_iris()
X = iris.data

# Create and fit the TSNE-PSO model
tsne_pso = TSNEPSO(
    n_components=2,
    perplexity=30.0,
    n_particles=10,
    n_iter=500,
    random_state=42
)
X_embedded = tsne_pso.fit_transform(X)

# Visualize the results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=iris.target)
plt.legend(handles=scatter.legend_elements()[0], labels=iris.target_names)
plt.title('TSNE-PSO visualization of Iris dataset')
plt.show()
```

## Advanced Usage

### Different Initialization Methods

```python
# Using UMAP for initialization
model = TSNEPSO(init='umap', perplexity=30)

# Using t-SNE for initialization
model = TSNEPSO(init='tsne', perplexity=30)

# Using custom initialization
initial_embedding = np.random.normal(0, 0.0001, (n_samples, 2))
model = TSNEPSO(init=initial_embedding)
```

### Tuning PSO Parameters

```python
model = TSNEPSO(
    n_particles=20,           # Number of particles
    inertia_weight=0.7,       # Inertia weight
    h=1e-20,                  # Parameter for dynamic cognitive weight
    f=1e-21,                  # Parameter for dynamic social weight
    use_hybrid=True,          # Use hybrid PSO + gradient descent
    n_iter=1000               # Number of iterations
)
```

## How It Works

TSNE-PSO enhances the original t-SNE algorithm by replacing gradient descent with Particle Swarm Optimization. The algorithm:

1. **Initialization**: Creates a swarm of particles with positions initialized via PCA, UMAP, t-SNE, or randomly
2. **Optimization**: Updates particles using:
   - Cognitive component (attraction to personal best position)
   - Social component (attraction to global best position)
   - Inertia (tendency to continue current trajectory)
3. **Dynamic Parameters**: Adapts cognitive and social weights over iterations
4. **Hybrid Approach**: Optionally applies gradient descent steps to accelerate convergence

## Citation

If you use this package in your research, please cite the following paper:

```bibtex
@article{allaoui2025t,
  title={t-SNE-PSO: Optimizing t-SNE using particle swarm optimization},
  author={Allaoui, Mebarka and Belhaouari, Samir Brahim and Hedjam, Rachid and Bouanane, Khadra and Kherfi, Mohammed Lamine},
  journal={Expert Systems with Applications},
  volume={269},
  pages={126398},
  year={2025},
  publisher={Elsevier}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

BSD-3-Clause License

## Development Setup

For development, follow these steps:

```bash
# Clone the repository
git clone https://github.com/draglesss/t-SNE-PSO.git
cd t-SNE-PSO

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in development mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=tsne_pso

# Run tests without the slow ones
pytest -k "not slow"
```

### Code Formatting

```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Run type checking with mypy
mypy tsne_pso
```