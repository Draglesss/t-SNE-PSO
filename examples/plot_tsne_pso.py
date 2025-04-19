"""
====================================================
t-SNE with Particle Swarm Optimization (PSO)
====================================================

An example of t-SNE with Particle Swarm Optimization (PSO) 
applied to the MNIST digits dataset.

This visualization demonstrates how TSNE-PSO can create well-separated clusters
in the embedding space.
"""

# Author: Your Name <ofatteh25@ubishops.ca>
# License: BSD 3 clause

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

from tsne_pso import TSNEPSO


# Load data
digits = load_digits()
X = digits.data
y = digits.target
labels = digits.target_names

# Reduce dimensionality with standard t-SNE
print("Computing standard t-SNE embedding...")
tsne = TSNE(n_components=2, init='pca', random_state=42)
X_tsne = tsne.fit_transform(X)

# Reduce dimensionality with t-SNE PSO
print("Computing t-SNE PSO embedding...")
tsne_pso = TSNEPSO(
    n_components=2,
    perplexity=30.0,
    n_particles=10,
    n_iter=500,
    random_state=42,
    verbose=1,
)
X_tsne_pso = tsne_pso.fit_transform(X)

# Visualize results
plt.figure(figsize=(15, 6))

# Standard t-SNE
plt.subplot(121)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
plt.title('Standard t-SNE')
plt.colorbar(scatter, ticks=range(10), label='Digit')
plt.grid(True)

# t-SNE with PSO
plt.subplot(122)
scatter = plt.scatter(X_tsne_pso[:, 0], X_tsne_pso[:, 1], c=y, cmap='tab10')
plt.title('t-SNE with Particle Swarm Optimization')
plt.colorbar(scatter, ticks=range(10), label='Digit')
plt.grid(True)

plt.suptitle('Comparison of t-SNE and t-SNE PSO on Digits Dataset')
plt.tight_layout()
plt.savefig('tsne_vs_tsne_pso.png', dpi=300)
plt.show()
