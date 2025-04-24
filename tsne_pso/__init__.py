"""t-Distributed Stochastic Neighbor Embedding with Particle Swarm Optimization.

This implementation is based on the paper:
Allaoui, M., Belhaouari, S. B., Hedjam, R., Bouanane, K., & Kherfi, M. L. (2025).
t-SNE-PSO: Optimizing t-SNE using particle swarm optimization.
Expert Systems with Applications, 269, 126398.
"""

from ._tsne_pso import TSNEPSO
from ._version import __version__

__all__ = ["TSNEPSO", "__version__"]
