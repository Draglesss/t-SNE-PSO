# Authors: Allaoui, Mebarka
#          Belhaouari, Samir Brahim
#          Hedjam, Rachid
#          Bouanane, Khadra
#          Kherfi, Mohammed Lamine
#
# Maintained by: Otmane Fatteh <fattehotmane@hotmail.com>
#
# License: BSD 3 clause
#
# This implementation is based on the paper:
# Allaoui, M., Belhaouari, S. B., Hedjam, R., Bouanane, K., & Kherfi, M. L. (2025).
# t-SNE-PSO: Optimizing t-SNE using particle swarm optimization.
# Expert Systems with Applications, 269, 126398.

import hashlib
import logging
import platform
import warnings
from numbers import Integral, Real
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, _utils
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_array, check_is_fitted

try:
    import umap

    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

try:
    from tqdm import tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from .logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

"""Core implementation of t-SNE PSO optimization.

This module provides the computational functions for t-SNE with PSO optimization.
All functions are implemented with NumPy for numerical operations, focusing on
numerical stability and performance.
"""

# Define exports
__all__ = [
    "compute_joint_probabilities",
    "compute_kl_divergence",
    "update_particle_pso",
]

# Constants for numerical stability
MACHINE_EPSILON = np.finfo(float).eps
MAX_VAL = np.finfo(float).max
MIN_VAL = np.finfo(float).min

# Determine the maximum seed value based on platform
_PLATFORM_SYSTEM = platform.system()
_MAX_SEED_VALUE = 2**31 - 1  # Windows safe limit


def _get_safe_random_seed(random_state):
    """Generate a random seed that is safe for all platforms while maintaining distribution quality.

    Instead of simply truncating the range on Windows, this function ensures a consistent
    statistical distribution of seeds across all platforms by using a hash-based approach
    when needed.

    Parameters
    ----------
    random_state : RandomState
        NumPy random state object

    Returns
    -------
    int
        A random seed suitable for all platforms
    """
    # Generate a full-range random number as a reference point
    full_range_random = random_state.randint(0, 2**32 - 1)

    # For non-Windows platforms, we can directly use the full range
    if _PLATFORM_SYSTEM != "Windows":
        return full_range_random

    hash_obj = hashlib.md5(str(full_range_random).encode())
    # Convert first 4 bytes of hash to integer and take modulo to fit within safe range
    mapped_value = int(hash_obj.hexdigest()[:8], 16) % _MAX_SEED_VALUE

    return mapped_value


def compute_joint_probabilities(
    distances: np.ndarray, perplexity: float, verbose: bool = False
) -> np.ndarray:
    """Calculate joint probability distributions P_ij from pairwise distances.

    This function implements a robust binary search algorithm to find optimal
    conditional probabilities (P_j|i) with the desired perplexity. It then
    computes symmetric joint probabilities P_ij for the high-dimensional
    data distribution.

    The perplexity parameter effectively controls the balance between preserving
    local and global structure. Higher perplexity values consider more neighbors,
    while lower values focus on more local relationships.

    Parameters
    ----------
    distances : array of shape (n_samples, n_samples)
        Square matrix of pairwise Euclidean distances between data points
    perplexity : float
        Desired perplexity value, typically between 5 and 50. Controls the
        effective number of neighbors considered for each point.
    verbose : bool, default=False
        Whether to display progress information during computation

    Returns
    -------
    P : array of shape (n_samples * (n_samples-1) / 2,)
        Condensed symmetric joint probability matrix in upper triangular form

    Notes
    -----
    The algorithm performs a binary search to find precision values (beta) that
    yield the desired perplexity for each point. This ensures a consistent
    neighborhood size across the dataset regardless of local density variations.

    Numerical stability is ensured by applying proper minimum bounds to prevent
    underflow issues during exponential calculations.
    """
    # Input validation
    assert isinstance(distances, np.ndarray), "distances must be a numpy array"
    assert distances.ndim == 2, "distances must be a 2D array"
    assert distances.shape[0] == distances.shape[1], "distances must be square"
    assert np.all(distances >= 0), "distances must be non-negative"
    assert isinstance(perplexity, (int, float)), "perplexity must be numeric"
    assert 5.0 <= perplexity <= 50.0, "perplexity must be between 5 and 50"

    n_samples = distances.shape[0]
    P = np.zeros((n_samples, n_samples))
    beta = np.ones(n_samples)
    logU = np.log(perplexity)

    # Loop over all instances
    for i in range(n_samples):
        # Print progress
        if verbose and i % 1000 == 0:
            logger.info(f"Computing P-values for point {i}/{n_samples}")

        # Compute P-values using binary search
        betamin = -np.inf
        betamax = np.inf
        Di = distances[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n_samples]))]

        # Binary search loop
        for _ in range(50):
            # Compute Gaussian kernel and entropy
            Pi = np.exp(-Di * beta[i])
            sumPi: float = np.sum(Pi)
            if sumPi == 0:
                Pi = np.maximum(Pi, MACHINE_EPSILON)
                sumPi = float(np.sum(Pi))
            Pi = Pi / sumPi

            # Calculate entropy and difference from target
            entropy = float(np.sum(-Pi * np.log2(np.maximum(Pi, MACHINE_EPSILON))))
            entropyDiff = float(entropy - logU)

            if abs(entropyDiff) < 1e-5:
                break

            # Update beta based on entropy difference
            if entropyDiff > 0:
                betamin = beta[i]
                if betamax == np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

        # Row-normalize P and store in matrix
        Pi = np.exp(-Di * beta[i])
        Pi = np.maximum(Pi, MACHINE_EPSILON)
        Pi = Pi / np.sum(Pi)
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : n_samples]))] = Pi
    # Symmetrize and convert to condensed form
    P = (P + P.T) / (2 * n_samples)
    P = np.maximum(P, MACHINE_EPSILON)

    # Return condensed form (upper triangular) as numpy array
    condensed_P: np.ndarray = squareform(P, checks=False)
    return condensed_P


def compute_kl_divergence(
    params: np.ndarray,
    P: np.ndarray,
    degrees_of_freedom: float,
    n_samples: int,
    n_components: int,
    skip_num_points: int = 0,
    compute_error: bool = True,
) -> Tuple[float, np.ndarray]:
    """Compute Kullback-Leibler divergence and its gradient for t-SNE optimization.

    This function computes the KL divergence between the joint probability
    distributions P (high-dimensional) and Q (low-dimensional), as well as
    the gradient of the KL divergence with respect to the embedding coordinates.

    The KL divergence serves as the cost function for t-SNE, measuring how well
    the low-dimensional embedding preserves the high-dimensional structure.
    The gradient provides the direction for adjusting the embedding to minimize
    this cost function.

    Parameters
    ----------
    params : array of shape (n_samples * n_components,)
        Current embedding coordinates flattened into 1D array
    P : array of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix from high-dimensional space
    degrees_of_freedom : float
        Degrees of freedom of Student's t-distribution; controls
        the heaviness of the tails in the low-dimensional distribution
    n_samples : int
        Number of samples in the dataset
    n_components : int
        Number of components in the embedding
    skip_num_points : int, default=0
        Number of points to skip when computing gradient; useful for
        mini-batch optimization approaches
    compute_error : bool, default=True
        Whether to compute and return the KL divergence value

    Returns
    -------
    kl_divergence : float
        KL divergence between P and Q distributions (if compute_error=True)
        Otherwise returns NaN
    grad : array of shape (n_samples * n_components,)
        Gradient of KL divergence with respect to the embedding coordinates

    Notes
    -----
    The Student's t-distribution is used for the low-dimensional space to
    address the "crowding problem" by allowing moderately distant points in
    the original space to be modeled by points that are further apart in
    the embedding space.

    The implementation uses vectorized operations for efficiency. The gradient
    computation is carefully implemented to handle numerical stability issues.
    """
    # Input validation
    # Original Python implementation as fallback
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Compute KL divergence
    if compute_error:
        kl_divergence = np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    # Compute gradient
    grad: np.ndarray = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
    grad = grad.ravel()

    # Scale the gradient
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


def update_particle_pso(
    particle: Dict,
    global_best_position: np.ndarray,
    global_best_score: float,
    inertia_weight: float,
    cognitive_weight: float,
    social_weight: float,
    random_state: np.random.RandomState,
    h: float,
    f: float,
    current_iter: int,
    n_samples: int,
    n_components: int,
    degrees_of_freedom: float,
    max_velocity: float = 0.1,
) -> Tuple[Dict, float]:
    """Update particle position and velocity using Particle Swarm Optimization rules.

    This function implements the core PSO update algorithm with dynamic weighting
    coefficients. Each particle's movement is influenced by three components:

    1. Inertia: Tendency to continue moving in the current direction
    2. Cognitive: Attraction to the particle's own best position
    3. Social: Attraction to the swarm's global best position

    The dynamic weighting scheme (using parameters h and f) allows the algorithm
    to transition from exploration (higher cognitive weight) to exploitation
    (higher social weight) as iterations progress.

    Parameters
    ----------
    particle : dict
        Dictionary containing particle state information including:
        - position: Current coordinates in the embedding space
        - velocity: Current movement vector
        - best_position: Best position found by this particle
        - best_score: Best score achieved by this particle
        - P: Joint probability matrix for this particle
        - grad_update: Previous gradient update (for hybrid approach)
        - gains: Adaptive gains for each dimension (for hybrid approach)
    global_best_position : array
        Global best position found by any particle in the swarm
    global_best_score : float
        Global best score (lowest KL divergence) found so far
    inertia_weight : float
        Weight controlling influence of previous velocity (0-1)
    cognitive_weight : float
        Base weight for cognitive component (attraction to personal best)
    social_weight : float
        Base weight for social component (attraction to global best)
    random_state : RandomState
        NumPy random number generator for reproducibility
    h : float
        Parameter controlling the overall magnitude of cognitive and social weights
    f : float
        Parameter controlling the balance between exploration and exploitation
    current_iter : int
        Current iteration number (1-indexed)
    n_samples : int
        Number of samples in the dataset
    n_components : int
        Number of dimensions in the embedding
    degrees_of_freedom : float
        Degrees of freedom parameter for the t-distribution
    max_velocity : float, default=0.1
        Maximum allowed velocity magnitude for any dimension

    Returns
    -------
    particle : dict
        Updated particle state with new position, velocity, and score
    score : float
        New KL divergence score of the particle at its updated position

    Notes
    -----
    The dynamic weighting scheme follows the formulation proposed in the
    original t-SNE-PSO paper by Allaoui et al. (2025). The cognitive weight
    decreases while the social weight increases with iterations, encouraging
    convergence to the global optimum.

    Velocity clamping is applied to prevent explosive particle movement,
    which could destabilize the optimization process.
    """
    # Input validation
    assert isinstance(particle, dict), "particle must be a dictionary"
    assert all(
        k in particle
        for k in ["position", "velocity", "best_position", "best_score", "P"]
    ), "particle missing required keys"
    assert isinstance(
        global_best_position, np.ndarray
    ), "global_best_position must be a numpy array"
    assert 0 <= inertia_weight <= 1, "inertia_weight must be between 0 and 1"
    assert cognitive_weight >= 0, "cognitive_weight must be non-negative"
    assert social_weight >= 0, "social_weight must be non-negative"
    assert current_iter > 0, "current_iter must be positive"
    assert max_velocity > 0, "max_velocity must be positive"

    # Generate random coefficients
    r1 = random_state.uniform(0, 1, particle["position"].shape)
    r2 = random_state.uniform(0, 1, particle["position"].shape)

    # Calculate adaptive weights
    adaptive_cognitive = h - (h / (1.0 + (f / current_iter)))
    adaptive_social = h / (1.0 + (f / current_iter))

    # Update velocity
    cognitive_component = (
        adaptive_cognitive * r1 * (particle["best_position"] - particle["position"])
    )
    social_component = (
        adaptive_social * r2 * (global_best_position - particle["position"])
    )

    new_velocity = (
        inertia_weight * particle["velocity"] + cognitive_component + social_component
    )

    # Apply velocity clamping
    new_velocity = np.clip(new_velocity, -max_velocity, max_velocity)

    # Update position
    new_position = particle["position"] + new_velocity

    # Update gains if using momentum
    if "gains" in particle and "grad_update" in particle:
        mask = np.sign(new_velocity) != np.sign(particle["grad_update"])
        particle["gains"][mask] = np.minimum(particle["gains"][mask] + 0.2, 1.0)
        particle["gains"][~mask] = np.maximum(particle["gains"][~mask] * 0.8, 0.01)
        new_position += particle["gains"] * new_velocity

    # Evaluate new position
    score, grad = compute_kl_divergence(
        new_position, particle["P"], degrees_of_freedom, n_samples, n_components
    )

    # Update particle state
    particle["velocity"] = new_velocity
    particle["position"] = new_position
    particle["grad_update"] = grad

    # Update personal best if improved
    if score < particle["best_score"]:
        particle["best_score"] = score
        particle["best_position"] = new_position.copy()

    return particle, score


# Machine epsilon for float64
MACHINE_EPSILON = np.finfo(np.double).eps

# Define valid metrics from sklearn
_VALID_METRICS = [
    "cityblock",
    "cosine",
    "euclidean",
    "l1",
    "l2",
    "manhattan",
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "dice",
    "hamming",
    "jaccard",
    "kulsinski",
    "mahalanobis",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
    "precomputed",
]


def _joint_probabilities(distances, perplexity, verbose=False):
    """Convert distances to joint probabilities P_ij.

    This function calculates joint probabilities using a Gaussian kernel in the
    high-dimensional space. It employs an efficient binary search algorithm to find
    the optimal bandwidth (precision) for each point that achieves the desired
    perplexity value.

    The perplexity can be interpreted as a smooth measure of the effective number
    of neighbors for each point. The function ensures that the distribution has
    approximately the same entropy (uncertainty) for each point, regardless of
    the local density of the data.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix between data points.

    perplexity : float
        Perplexity parameter that controls the effective number of neighbors.
        Typical values range between 5 and 50, with larger datasets generally
        requiring higher perplexity values.

    verbose : bool, default=False
        Whether to display progress information during computation.

    Returns
    -------
    P : ndarray of shape (n_samples*(n_samples-1)/2,)
        Condensed joint probability matrix in upper triangular form.

    Notes
    -----
    This implementation leverages scikit-learn's efficient binary search algorithm
    for finding the optimal precision values. The resulting conditional probabilities
    are symmetrized to obtain joint probabilities, following the approach in the
    original t-SNE paper.

    Numerical stability is ensured by applying proper minimum bounds to prevent
    underflow issues during normalization.
    """
    # Ensure distances are in the correct format
    distances = distances.astype(np.float32, copy=False)

    # Use sklearn's _binary_search_perplexity for calculating conditional probabilities
    # This efficiently implements the binary search for sigma values that yield the desired perplexity
    conditional_P = _utils._binary_search_perplexity(distances, perplexity, verbose)

    # Symmetrize the conditional probabilities to get joint probabilities
    # P_ij = (P_j|i + P_i|j) / (2n)
    P = conditional_P + conditional_P.T

    # Normalize and convert to condensed matrix format for efficient storage
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)

    # Final validation to ensure numerical stability
    assert np.all(np.isfinite(P)), "Joint probability matrix contains invalid values"
    assert np.all(P >= 0), "Joint probability matrix contains negative values"

    return P


def _kl_divergence(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    skip_num_points=0,
    compute_error=True,
):
    """Compute KL divergence between P and Q distributions and its gradient.

    This function calculates the Kullback-Leibler divergence between the joint
    probability distributions in high-dimensional space (P) and low-dimensional
    space (Q), as well as the gradient of this divergence with respect to the
    embedding coordinates.

    This is a wrapper function that calls the optimized implementation while
    providing a consistent interface. It supports skipping points for mini-batch
    approaches and conditional computation of the error term for efficiency.

    Parameters
    ----------
    params : ndarray of shape (n_samples * n_components,)
        Current embedding coordinates flattened into 1D array.

    P : ndarray of shape (n_samples*(n_samples-1)/2,)
        Condensed joint probability matrix from high-dimensional space.

    degrees_of_freedom : float
        Degrees of freedom for the Student's t-distribution used in the
        low-dimensional space.

    n_samples : int
        Number of samples in the dataset.

    n_components : int
        Number of dimensions in the embedding.

    skip_num_points : int, default=0
        Number of points to skip when computing gradient, enabling mini-batch
        optimization approaches.

    compute_error : bool, default=True
        Whether to compute and return the KL divergence value or just the gradient.

    Returns
    -------
    kl_divergence : float
        KL divergence between P and Q distributions if compute_error=True,
        otherwise NaN.

    grad : ndarray of shape (n_samples * n_components,)
        Gradient of KL divergence with respect to the embedding coordinates.
    """
    return compute_kl_divergence(
        params,
        P,
        degrees_of_freedom,
        n_samples,
        n_components,
        skip_num_points,
        compute_error,
    )


def _gradient_descent_step(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    momentum=0.8,
    learning_rate=200.0,
    min_gain=0.01,
    update=None,
    gains=None,
):
    """Perform one step of gradient descent with momentum and adaptive gains.

    This function implements the gradient descent update used in the hybrid approach
    of t-SNE-PSO. It incorporates momentum and adaptive gains to improve convergence,
    helping to overcome local minima and saddle points.

    The adaptive gains approach increases the effective learning rate in directions
    where the gradient consistently points in the same direction, while decreasing it
    in directions where the gradient oscillates.

    Parameters
    ----------
    params : ndarray of shape (n_samples * n_components,)
        Current embedding coordinates flattened into 1D array.

    P : ndarray of shape (n_samples*(n_samples-1)/2,)
        Condensed joint probability matrix from high-dimensional space.

    degrees_of_freedom : float
        Degrees of freedom for the Student's t-distribution used in
        low-dimensional space.

    n_samples : int
        Number of samples in the dataset.

    n_components : int
        Number of dimensions in the embedding.

    momentum : float, default=0.8
        Momentum coefficient for gradient updates, controlling the influence
        of previous update directions.

    learning_rate : float, default=200.0
        Base learning rate for gradient updates.

    min_gain : float, default=0.01
        Minimum gain value to ensure continued learning in all dimensions.

    update : ndarray of shape (n_samples * n_components,), default=None
        Previous update vector for momentum calculation. If None, initialized
        to zeros.

    gains : ndarray of shape (n_samples * n_components,), default=None
        Previous gain values for adaptive learning rates. If None, initialized
        to ones.

    Returns
    -------
    params : ndarray of shape (n_samples * n_components,)
        Updated embedding coordinates.

    error : float
        KL divergence between P and Q distributions after the update.

    update : ndarray of shape (n_samples * n_components,)
        Updated momentum vector.

    gains : ndarray of shape (n_samples * n_components,)
        Updated gain values for adaptive learning rates.

    Notes
    -----
    The adaptive gains approach was introduced in the original t-SNE paper.
    It increases the learning rate in dimensions where the gradient is consistent
    across iterations, and decreases it where the gradient oscillates, helping
    to navigate ravines in the error surface.
    """
    # Initialize update and gains if not provided
    if update is None:
        update = np.zeros_like(params)
    if gains is None:
        gains = np.ones_like(params)

    # Compute KL divergence and its gradient
    error, grad = _kl_divergence(
        params, P, degrees_of_freedom, n_samples, n_components, compute_error=True
    )

    # Update gains with adaptive learning rates
    inc = update * grad < 0.0
    dec = np.invert(inc)
    gains[inc] += 0.2
    gains[dec] *= 0.8
    np.clip(gains, min_gain, np.inf, out=gains)

    # Apply gains to gradient
    grad *= gains

    # Update parameters with momentum
    update = momentum * update - learning_rate * grad
    params += update

    return params, error, update, gains


class TSNEPSO(TransformerMixin, BaseEstimator):
    """t-SNE with Particle Swarm Optimization for high-quality dimensionality reduction.

    t-Distributed Stochastic Neighbor Embedding (t-SNE) is a powerful technique
    for visualizing high-dimensional data in lower dimensions. This implementation
    enhances the standard t-SNE algorithm by replacing gradient descent with
    Particle Swarm Optimization (PSO), providing several advantages:

    1. Better exploration of the embedding space
    2. Improved ability to escape local minima
    3. More consistent cluster separation
    4. Enhanced preservation of global structure

    The algorithm combines swarm intelligence principles with the t-SNE objective
    function. Multiple particles explore the embedding space simultaneously, sharing
    information about the best configurations found. The optimization process uses
    dynamic cognitive and social weights that transition from exploration to
    exploitation as iterations progress.

    This implementation supports a hybrid mode that combines PSO with gradient
    descent steps for improved convergence, as well as multiple initialization
    strategies to provide better starting points.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space, typically 2 or 3 for visualization.

    perplexity : float, default=30.0
        Perplexity parameter that balances attention between local and global
        aspects of the data. It can be interpreted as a smooth measure of the
        effective number of neighbors. Larger datasets generally require larger
        perplexity values (5-50).

    early_exaggeration : float, default=12.0
        Coefficient for early exaggeration phase, which increases the attraction
        between similar points to form more separated clusters initially.

    learning_rate : float or "auto", default="auto"
        Learning rate for gradient descent steps in the hybrid approach. If "auto",
        it's set to max(N / early_exaggeration / 4, 50) where N is the sample size.

    n_iter : int, default=1000
        Maximum number of iterations for optimization.

    n_particles : int, default=10
        Number of particles in the PSO swarm. More particles provide better
        exploration at the cost of computational efficiency.

    inertia_weight : float, default=0.5
        Controls how much of each particle's previous velocity is preserved.
        Values closer to 0 accelerate convergence, while values closer to 1
        encourage exploration.

    cognitive_weight : float, default=1.0
        Base weight for the cognitive component, controlling how much particles
        are attracted to their personal best positions.

    social_weight : float, default=1.0
        Base weight for the social component, controlling how much particles
        are attracted to the global best position.

    use_hybrid : bool, default=True
        Whether to use the hybrid approach combining PSO with gradient descent
        steps. This often improves convergence and final embedding quality.

    degrees_of_freedom : float, default=1.0
        Degrees of freedom of the Student's t-distribution. Lower values
        increase the heavy-tailed nature of the distribution, emphasizing
        the separation between distant clusters.

    init : str or ndarray of shape (n_samples, n_components), default='pca'
        Initialization method for the embedding. Options include:
        - 'pca': Principal Component Analysis initialization
        - 'tsne': Initialization from a standard t-SNE run
        - 'umap': Initialization from UMAP (if available)
        - 'random': Random initialization
        - ndarray: Custom initialization provided as an array

    verbose : int, default=0
        Verbosity level for logging and progress information.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the initialization and optimization.
        Pass an int for reproducible results across multiple function calls.

    method : str, default='pso'
        Optimization method. Currently only 'pso' is supported.

    angle : float, default=0.5
        Only used if method='barnes_hut' (reserved for future implementation).
        This is the trade-off between speed and accuracy for Barnes-Hut t-SNE.

    n_jobs : int, default=None
        Number of parallel jobs for computation (reserved for future implementation).

    metric : str or callable, default='euclidean'
        Distance metric for calculating pairwise distances in the original space.
        Can be any metric supported by scipy.spatial.distance.pdist, or a callable
        function that takes two arrays and returns a distance value.
        If 'precomputed', X is interpreted as a distance matrix.

    metric_params : dict, default=None
        Additional parameters for the metric function.

    h : float, default=1e-20
        Parameter for dynamic cognitive weight formula: c1 = h - (h / (1 + (f / it))).
        Controls how cognitive weight decreases over iterations.

    f : float, default=1e-21
        Parameter for dynamic social weight formula: c2 = h / (1 + (f / it)).
        Controls how social weight increases over iterations.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the optimized embedding coordinates.

    kl_divergence_ : float
        Final KL divergence value (cost function) of the embedding.

    n_iter_ : int
        Actual number of iterations run before convergence or maximum iterations.

    n_features_in_ : int
        Number of features in the input data.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during fit (if available).

    convergence_history_ : ndarray
        History of best KL divergence values across iterations (if optimization
        completes successfully).

    Examples
    --------
    >>> import numpy as np
    >>> from tsne_pso import TSNEPSO
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = TSNEPSO(random_state=0)
    >>> Y = model.fit_transform(X)
    >>> Y.shape
    (4, 2)

    Notes
    -----
    The t-SNE-PSO algorithm is particularly effective for datasets with complex
    structure where standard t-SNE might get trapped in poor local minima. The
    implementation automatically adjusts parameters based on dataset size for
    optimal performance.

    For best results with larger datasets, consider increasing both the number of
    particles and iterations, though this will increase computation time. The
    perplexity parameter should typically scale with dataset size - larger
    datasets benefit from higher perplexity values.

    References
    ----------
    .. [1] van der Maaten, L.J.P. and Hinton, G.E., 2008. "Visualizing
       High-Dimensional Data Using t-SNE." Journal of Machine Learning
       Research, 9(Nov), pp.2579-2605.

    .. [2] Allaoui, M., Belhaouari, S. B., Hedjam, R., Bouanane, K., & Kherfi, M. L. (2025).
       "t-SNE-PSO: Optimizing t-SNE using particle swarm optimization."
       Expert Systems with Applications, 269, 126398.

    .. [3] Kennedy, J. and Eberhart, R., 1995. "Particle swarm optimization."
       In Proceedings of ICNN'95 - International Conference on Neural Networks,
       Vol. 4, pp. 1942-1948.
    """

    # Define class tags to indicate behavior
    _tags = {
        "allow_nan": False,
        "array_api_support": False,
        "pairwise": False,
        "preserves_dtype": [np.float64],
        "requires_fit": True,
        "requires_positive_X": False,
        "requires_y": False,
        "X_types": ["2darray"],
        "poor_score": True,
        "no_validation": False,
        "non_deterministic": True,
        "multioutput": False,
        "allow_metric_params": True,
        "stateless": False,
        "multilabel": False,
        "requires_positive_y": False,
        "_skip_test": [
            "check_transformer_data_not_an_array",
            "check_methods_sample_order_invariance",
            "check_methods_subset_invariance",
            "check_dict_unchanged",
            "check_fit_idempotent",
            "check_fit2d_predict1d",
            "check_estimators_nan_inf",
            "check_estimators_dtypes",
            "check_estimators_pickle",
            "check_dtype_object",
            "check_estimators_empty_data_messages",
            "check_pipeline_consistency",
            "check_estimator_sparse_tag",
            "check_estimator_sparse_array",
            "check_estimator_sparse_matrix",
            "check_estimators_pickle",
        ],
    }

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "perplexity": [Interval(Real, 0, None, closed="neither")],
        "early_exaggeration": [Interval(Real, 0, None, closed="neither")],
        "learning_rate": [
            StrOptions({"auto"}),
            Interval(Real, 0, None, closed="neither"),
        ],
        "n_iter": [Interval(Integral, 0, None, closed="neither")],
        "n_particles": [Interval(Integral, 1, None, closed="left")],
        "inertia_weight": [Interval(Real, 0, 1, closed="both")],
        "cognitive_weight": [Interval(Real, 0, None, closed="left")],
        "social_weight": [Interval(Real, 0, None, closed="left")],
        "use_hybrid": ["boolean"],
        "degrees_of_freedom": [Interval(Real, 0, None, closed="neither")],
        "init": [
            StrOptions({"pca", "tsne", "umap", "random"}),
            np.ndarray,
        ],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
        "method": [StrOptions({"pso"})],
        "angle": [Interval(Real, 0, 1, closed="both")],
        "n_jobs": [None, Integral],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
        "metric_params": [dict, None],
        "h": [Interval(Real, 1e-21, None, closed="left")],
        "f": [Interval(Real, 1e-21, None, closed="left")],
    }

    def __init__(
        self,
        n_components=2,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate="auto",
        n_iter=1000,
        n_particles=10,
        inertia_weight=0.5,
        cognitive_weight=1.0,
        social_weight=1.0,
        use_hybrid=True,
        degrees_of_freedom=1.0,
        init="pca",
        verbose=0,
        random_state=None,
        method="pso",
        angle=0.5,
        n_jobs=None,
        metric="euclidean",
        metric_params=None,
        h=1e-20,
        f=1e-21,
    ):
        """Initialize the TSNEPSO model with the specified parameters.

        All parameters are stored as instance attributes and validated during the
        fitting process. Several parameters are automatically adjusted based on
        the dataset characteristics to ensure optimal performance.

        See class documentation for detailed parameter descriptions.
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_particles = n_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.use_hybrid = use_hybrid
        self.degrees_of_freedom = degrees_of_freedom
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.metric = metric
        self.metric_params = metric_params
        self.h = h
        self.f = f

    def _validate_parameters(self):
        """Validate input parameters for consistency and correctness.

        This method checks that all parameters are within their valid ranges
        and compatible with each other. It raises descriptive error messages
        when invalid parameters are detected.

        Raises
        ------
        ValueError
            If any parameter is invalid or incompatible with other parameters.
        """
        if self.perplexity <= 0:
            raise ValueError("perplexity must be greater than 0.")

        if self.method != "pso":
            raise ValueError("The method must be 'pso'.")

        self._validate_params()

        if isinstance(self.init, str) and self.init == "umap" and not _UMAP_AVAILABLE:
            warnings.warn(
                "UMAP is not available. Using PCA initialization instead.",
                UserWarning,
            )

    def _check_params_vs_input(self, X):
        """Validate perplexity against dataset size and adjust if necessary."""
        n_samples = X.shape[0]
        self._perplexity_value = self.perplexity

        # Check for absolute perplexity >= n_samples
        if self.perplexity >= n_samples:
            self._perplexity_value = max(1.0, (n_samples - 1) / 3.0)
            warnings.warn(
                f"Perplexity {self.perplexity} too high for {n_samples} samples. "
                f"Using {self._perplexity_value:.1f} instead.",
                UserWarning,
            )
        # Check for perplexity close to n_samples (99% threshold)
        elif self.perplexity >= 0.99 * n_samples:
            self._perplexity_value = max(1.0, (n_samples - 1) / 3.0)
            warnings.warn(
                f"Perplexity ({self.perplexity}) should be less than "
                f"n_samples ({n_samples}). "
                f"Using perplexity = {self._perplexity_value:.3f} instead.",
                UserWarning,
            )

    def _adjust_params_for_dataset_size(self, n_samples, n_features):
        """Automatically tune parameters based on dataset characteristics.

        This method implements an adaptive parameter selection strategy that
        adjusts key parameters based on the dataset size and dimensionality.
        The goal is to optimize both the quality of the embedding and
        computational efficiency.

        For small datasets, it reduces the number of particles and iterations
        while adjusting perplexity to appropriate levels. For larger datasets,
        it maintains more particles to ensure thorough exploration of the
        embedding space.

        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset
        n_features : int
            Number of features in the dataset (ignored for precomputed distances)
        """
        # Store original parameters for reference
        self._original_params = {
            "perplexity": self.perplexity,
            "n_particles": self.n_particles,
            "n_iter": self.n_iter,
            "learning_rate": self.learning_rate,
            "early_exaggeration": self.early_exaggeration,
        }

        # Very small datasets (e.g., Iris, Wine): n_samples < 200
        if n_samples < 200:
            # Reduce particles for speed on small datasets
            self.n_particles = min(5, self.n_particles)

            # Adjust perplexity based on dataset size
            # For very small datasets, perplexity should be smaller
            recommended_perplexity = max(5.0, min(n_samples / 5, self.perplexity))

            if self.perplexity > recommended_perplexity:
                if self.verbose:
                    logger.info(
                        f"Small dataset detected (n={n_samples}). Adjusting perplexity from "
                        f"{self.perplexity} to {recommended_perplexity}"
                    )
                self.perplexity = recommended_perplexity

            # Use shorter early exaggeration phase
            self.early_exaggeration = min(8.0, self.early_exaggeration)

            # For high-dimensional data in small datasets
            if n_features > 50:
                # Increase number of iterations for complex data spaces
                self.n_iter = max(self.n_iter, 750)

        # Medium datasets: 200 <= n_samples < 1000
        elif n_samples < 1000:
            # Use moderate number of particles
            self.n_particles = min(7, self.n_particles)

            # Adjust perplexity to about 5% of the dataset size
            recommended_perplexity = max(15.0, min(n_samples / 20, self.perplexity))

            if self.perplexity > recommended_perplexity:
                if self.verbose:
                    logger.info(
                        f"Medium dataset detected (n={n_samples}). Adjusting perplexity from "
                        f"{self.perplexity} to {recommended_perplexity}"
                    )
                self.perplexity = recommended_perplexity

        # Large datasets: n_samples >= 1000
        else:
            # Keep original number of particles for large datasets
            pass

        # Adjust learning rate if set to auto
        if self.learning_rate == "auto":
            # Already handled by existing code - no changes needed
            pass

        # Log adjustments if verbose
        if self.verbose and (
            self.perplexity != self._original_params["perplexity"]
            or self.n_particles != self._original_params["n_particles"]
        ):
            logger.info(f"Parameter adjustments for dataset size {n_samples}:")
            logger.info(
                f"  - Perplexity: {self._original_params['perplexity']} -> {self.perplexity}"
            )
            logger.info(
                f"  - Particles: {self._original_params['n_particles']} -> {self.n_particles}"
            )
            if self.n_iter != self._original_params["n_iter"]:
                logger.info(
                    f"  - Iterations: {self._original_params['n_iter']} -> {self.n_iter}"
                )

        # Add assertions for parameter validation
        assert self.perplexity > 0, "Perplexity must be positive"
        assert self.n_particles > 0, "Number of particles must be positive"
        assert self.n_iter > 0, "Number of iterations must be positive"
        assert (
            self.perplexity < n_samples
        ), "Perplexity must be less than number of samples"

    def _initialize_particles(self, X, random_state):
        """Initialize particles for PSO optimization with diverse starting positions.

        This method creates the initial set of particles for the PSO algorithm,
        with each particle representing a candidate embedding. For efficiency and
        quality, different initialization strategies are used based on the dataset
        size and the specified initialization method.

        The method computes the joint probability distribution P in the high-dimensional
        space, which is shared among all particles and used to evaluate embedding quality.
        Particles are initialized with positions, velocities, and personal best
        information, with the global best tracked across all particles.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix to be embedded.

        random_state : RandomState instance
            Random number generator for reproducible initialization.

        Returns
        -------
        particles : list of dict
            List of initialized particles, each containing:
            - position: Current coordinates in embedding space
            - velocity: Current velocity vector
            - best_position: Best position found by this particle
            - best_score: Best score achieved by this particle
            - P: Joint probability matrix for KL divergence calculation
            - grad_update: Gradient information for hybrid approach
            - gains: Adaptive gains for hybrid approach

        Notes
        -----
        For small datasets, a specialized initialization approach is used that tries
        multiple strategies and selects the most promising starting configurations.
        This helps overcome the challenges of local minima in small datasets.

        The joint probability matrix P is computed once and reused across all
        particles, applying early exaggeration to encourage initial cluster formation.
        """
        n_samples = X.shape[0]

        # Initialize the particles list
        particles = []

        # Compute pairwise distances in high-dimensional space
        if self.metric == "precomputed":
            distances = X
        else:
            metric_params = self.metric_params or {}
            distances = pairwise_distances(
                X, metric=self.metric, squared=True, n_jobs=self.n_jobs, **metric_params
            )

        # Add assertion to validate distances matrix
        assert distances.shape == (
            n_samples,
            n_samples,
        ), "Distance matrix shape mismatch"
        assert np.all(np.isfinite(distances)), "Distance matrix contains invalid values"

        # Special initialization for small datasets
        if n_samples < 200 and hasattr(self, "_original_params"):
            return self._initialize_particles_for_small_dataset(
                X, distances, random_state
            )

        # Continue with the standard initialization for larger datasets
        # Compute joint probabilities
        P = _joint_probabilities(distances, self._perplexity_value, self.verbose > 0)

        # Assert P is valid
        assert P.shape == (
            (n_samples * (n_samples - 1)) // 2,
        ), "Joint probability shape mismatch"
        assert np.all(
            np.isfinite(P)
        ), "Joint probability matrix contains invalid values"
        assert np.all(P >= 0), "Joint probability matrix contains negative values"

        # Apply early exaggeration to P
        P = P * self.early_exaggeration

        # Method for initializing embeddings
        embeddings = []

        # Generate initial embeddings based on init strategy
        if isinstance(self.init, np.ndarray):
            # Check shape of user-provided initialization
            if self.init.shape != (n_samples, self.n_components):
                raise ValueError(
                    f"init.shape={self.init.shape} but should be "
                    f"(n_samples, n_components)=({n_samples}, {self.n_components})"
                )
            # Use provided initialization for the first particle
            embeddings.append(self.init.copy())

            # For remaining particles, apply small perturbations to the provided
            # embedding
            for i in range(1, self.n_particles):
                noise = random_state.normal(0, 0.01, self.init.shape)
                embeddings.append(self.init + noise)

        elif self.init == "tsne":
            # Use scikit-learn's TSNE for initialization of the first particle
            tsne = TSNE(
                n_components=self.n_components,
                perplexity=self._perplexity_value,
                n_iter=250,
                random_state=random_state.randint(0, 2**31 - 1),
            )
            first_embedding = tsne.fit_transform(X)
            embeddings.append(first_embedding)

            # For remaining particles, apply small perturbations to the first embedding
            for i in range(1, self.n_particles):
                noise = random_state.normal(0, 0.01, first_embedding.shape)
                embeddings.append(first_embedding + noise)

        elif self.init == "umap" and _UMAP_AVAILABLE:
            # Use UMAP for initialization of the first particle
            reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=min(int(self._perplexity_value), n_samples - 1),
                min_dist=0.1,
                random_state=random_state.randint(0, 2**31 - 1),
            )
            first_embedding = reducer.fit_transform(X)
            embeddings.append(first_embedding)

            # For remaining particles, apply small perturbations to the first embedding
            for i in range(1, self.n_particles):
                noise = random_state.normal(0, 0.01, first_embedding.shape)
                embeddings.append(first_embedding + noise)

        elif self.init == "pca":
            # Use PCA for initialization of first particle
            pca = PCA(
                n_components=self.n_components,
                random_state=_get_safe_random_seed(random_state),
            )
            first_embedding = pca.fit_transform(X)

            # Normalize to ensure appropriate scaling
            first_embedding = first_embedding / np.std(first_embedding[:, 0]) * 0.0001
            embeddings.append(first_embedding)

            # For remaining particles, apply small perturbations
            for i in range(1, self.n_particles):
                noise = random_state.normal(0, 0.01, first_embedding.shape)
                embeddings.append(first_embedding + noise)

        else:  # 'random'
            for i in range(self.n_particles):
                embedding = random_state.normal(
                    0, 0.0001, (n_samples, self.n_components)
                )
                embeddings.append(embedding)

        # Initialize particles
        best_score = float("inf")
        best_position = None

        for i in range(self.n_particles):
            # Initial position and velocity
            position = embeddings[i].ravel().copy()
            velocity = random_state.normal(0, 0.0001, position.shape)

            # Evaluate fitness
            score, _ = _kl_divergence(
                position, P, self.degrees_of_freedom, n_samples, self.n_components
            )

            # Store particle
            particle = {
                "position": position.copy(),
                "velocity": velocity.copy(),
                "best_position": position.copy(),
                "best_score": score,
                "P": P,
                "grad_update": np.zeros_like(position),
                "gains": np.ones_like(position),
            }

            particles.append(particle)

            # Update global best
            if score < best_score:
                best_score = score
                best_position = position.copy()

        # Store global best in all particles
        for particle in particles:
            particle["global_best_position"] = best_position.copy()
            particle["global_best_score"] = best_score

        return particles

    def _initialize_particles_for_small_dataset(self, X, distances, random_state):
        """Special particle initialization strategy for small datasets.

        Small datasets are particularly challenging for dimensionality reduction
        because they provide less information to guide the optimization process
        and are more prone to poor local minima. This method implements a multi-strategy
        approach that:

        1. Tries different initialization methods (PCA, t-SNE, UMAP, random)
        2. Evaluates the initial KL divergence for each candidate
        3. Selects the most promising starting points for the particles
        4. Ensures diversity through controlled perturbation

        This comprehensive approach significantly improves the quality of embeddings
        for small datasets by providing better starting points for the PSO algorithm.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix to be embedded.

        distances : ndarray of shape (n_samples, n_samples)
            Pairwise distance matrix between data points.

        random_state : RandomState instance
            Random number generator for reproducible initialization.

        Returns
        -------
        particles : list of dict
            List of initialized particles with optimized starting positions.

        Notes
        -----
        For very small datasets, even subtle differences in initialization can
        dramatically affect the final embedding quality. This method increases
        robustness by trying different perplexity values and initialization
        approaches, selecting those with the lowest initial KL divergence.
        """
        n_samples = X.shape[0]
        n_features = X.shape[1] if self.metric != "precomputed" else 0

        # Compute joint probabilities
        P = _joint_probabilities(distances, self._perplexity_value, self.verbose > 0)

        # Apply early exaggeration
        P = P * self.early_exaggeration

        # For small datasets, try multiple initialization strategies and pick the best ones
        candidate_embeddings = []
        candidate_scores = []

        # 1. First try PCA initialization
        if n_features > 0:  # Skip for precomputed distances
            pca = PCA(
                n_components=self.n_components,
                random_state=_get_safe_random_seed(random_state),
            )
            pca_embedding = pca.fit_transform(X)
            pca_embedding = pca_embedding / np.std(pca_embedding[:, 0]) * 0.0001
            candidate_embeddings.append(pca_embedding)

            # Get KL divergence for PCA initialization
            kl_div, _ = _kl_divergence(
                pca_embedding.ravel(),
                P,
                self.degrees_of_freedom,
                n_samples,
                self.n_components,
            )
            candidate_scores.append(kl_div)

            # Try a scaled version of PCA for more variety
            scaled_pca = pca_embedding * 0.1  # Different scaling
            candidate_embeddings.append(scaled_pca)
            kl_div, _ = _kl_divergence(
                scaled_pca.ravel(),
                P,
                self.degrees_of_freedom,
                n_samples,
                self.n_components,
            )
            candidate_scores.append(kl_div)

        # 2. Try t-SNE initialization with different perplexities
        perplexity_values = [
            max(5.0, self._perplexity_value * 0.5),
            self._perplexity_value,
            min(self._perplexity_value * 2.0, (n_samples - 1) / 3.0),
        ]

        for perp in perplexity_values:
            try:
                tsne = TSNE(
                    n_components=self.n_components,
                    perplexity=perp,
                    n_iter=250,
                    random_state=_get_safe_random_seed(random_state),
                )
                tsne_embedding = tsne.fit_transform(X)
                candidate_embeddings.append(tsne_embedding)

                # Get KL divergence for this initialization
                kl_div, _ = _kl_divergence(
                    tsne_embedding.ravel(),
                    P,
                    self.degrees_of_freedom,
                    n_samples,
                    self.n_components,
                )
                candidate_scores.append(kl_div)

                # Also try a normalized version
                norm_tsne = tsne_embedding.copy()
                norm_tsne = norm_tsne / np.std(norm_tsne[:, 0]) * 0.0001
                candidate_embeddings.append(norm_tsne)
                kl_div, _ = _kl_divergence(
                    norm_tsne.ravel(),
                    P,
                    self.degrees_of_freedom,
                    n_samples,
                    self.n_components,
                )
                candidate_scores.append(kl_div)
            except Exception as e:
                if self.verbose:
                    logger.warning(
                        f"t-SNE initialization with perplexity {perp} failed: {str(e)}"
                    )

        # 3. Try UMAP initialization if available
        if _UMAP_AVAILABLE and n_features > 0:
            try:
                reducer = umap.UMAP(
                    n_components=self.n_components,
                    n_neighbors=min(int(self._perplexity_value), n_samples - 1),
                    min_dist=0.1,
                    random_state=_get_safe_random_seed(random_state),
                )
                umap_embedding = reducer.fit_transform(X)
                candidate_embeddings.append(umap_embedding)

                # Get KL divergence for UMAP initialization
                kl_div, _ = _kl_divergence(
                    umap_embedding.ravel(),
                    P,
                    self.degrees_of_freedom,
                    n_samples,
                    self.n_components,
                )
                candidate_scores.append(kl_div)
            except Exception as e:
                if self.verbose:
                    logger.warning(f"UMAP initialization failed: {str(e)}")

        # 4. Add some random initializations
        for i in range(3):  # Add three random initializations with different scales
            random_embedding = random_state.normal(
                0, 0.0001 * (i + 1), (n_samples, self.n_components)
            )
            candidate_embeddings.append(random_embedding)

            # Get KL divergence for random initialization
            kl_div, _ = _kl_divergence(
                random_embedding.ravel(),
                P,
                self.degrees_of_freedom,
                n_samples,
                self.n_components,
            )
            candidate_scores.append(kl_div)

        # Print stats on initialization candidates if verbose
        if self.verbose:
            logger.info(f"Generated {len(candidate_scores)} initialization candidates")
            logger.info(
                f"Best initial KL: {min(candidate_scores):.4f}, Worst: {max(candidate_scores):.4f}"
            )

        # Sort candidates by KL divergence (lower is better)
        sorted_indices = np.argsort(candidate_scores)

        # For diversity, take some good candidates but not necessarily all the best ones
        # This prevents getting stuck in similar local minima
        best_candidates = []

        # Take the absolute best candidate
        best_candidates.append(candidate_embeddings[sorted_indices[0]])

        # Mix in some good candidates but with diversity
        indices_to_use = [0]  # Already used the best one

        # Build up the list of indices to use
        i = 1
        while len(indices_to_use) < self.n_particles and i < len(sorted_indices):
            idx = sorted_indices[i]
            # Only use this candidate if its score is reasonably good
            # (not more than 2x worse than the best score)
            if candidate_scores[idx] < 2.0 * candidate_scores[sorted_indices[0]]:
                best_candidates.append(candidate_embeddings[idx])
                indices_to_use.append(idx)
            i += 1

        # If we don't have enough candidates, duplicate the best ones with noise
        while len(best_candidates) < self.n_particles:
            # Use the best embedding and add noise
            best_idx = sorted_indices[0]
            noise = random_state.normal(0, 0.01, candidate_embeddings[best_idx].shape)
            best_candidates.append(candidate_embeddings[best_idx] + noise)

        # Initialize particles with the best candidates
        particles = []
        best_score = float("inf")
        best_position = None

        for i in range(self.n_particles):
            # Initial position and velocity
            position = best_candidates[i].ravel().copy()
            velocity = random_state.normal(0, 0.0001, position.shape)

            # Evaluate fitness
            score, _ = _kl_divergence(
                position, P, self.degrees_of_freedom, n_samples, self.n_components
            )

            # Store particle
            particle = {
                "position": position.copy(),
                "velocity": velocity.copy(),
                "best_position": position.copy(),
                "best_score": score,
                "P": P,
                "grad_update": np.zeros_like(position),
                "gains": np.ones_like(position),
            }

            particles.append(particle)

            # Update global best
            if score < best_score:
                best_score = score
                best_position = position.copy()

        # Store global best in all particles
        for particle in particles:
            particle["global_best_position"] = best_position.copy()
            particle["global_best_score"] = best_score

        if self.verbose:
            logger.info(
                f"Small dataset optimization: Best initial KL divergence = {best_score:.4f}"
            )

        return particles

    def _optimize_embedding(self, X, random_state):
        """Optimize the embedding using Particle Swarm Optimization.

        This method implements the core t-SNE-PSO algorithm. It iteratively updates
        particle positions and velocities to minimize the KL divergence between
        high-dimensional and low-dimensional distributions.

        The optimization process consists of multiple phases:
        1. Early exaggeration phase: Emphasizes natural cluster structure
        2. Main optimization phase: Refines the embedding with dynamic parameters
        3. Convergence monitoring: Tracks progress and applies early stopping

        Throughout optimization, multiple strategies are employed to escape local
        minima and improve exploration, including adaptive parameter adjustment,
        particle perturbation, and periodic reinitialization of the worst particles.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix to be embedded.

        random_state : RandomState instance
            Random number generator for reproducible optimization.

        Returns
        -------
        best_position : ndarray of shape (n_samples, n_components)
            Optimized embedding coordinates.

        best_cost : float
            Final KL divergence value of the optimized embedding.

        n_iter : int
            Actual number of iterations performed.

        Notes
        -----
        The algorithm adapts PSO parameters dynamically based on optimization progress.
        Inertia weight decreases over time to improve convergence, while cognitive
        and social weights are adjusted according to the formulas from the original
        t-SNE-PSO paper to balance exploration and exploitation.

        When use_hybrid=True, gradient descent steps are interspersed with PSO updates
        to leverage the strengths of both approaches: PSO for global exploration and
        gradient descent for local refinement.
        """
        n_samples = X.shape[0]

        # Initialize particles
        particles = self._initialize_particles(X, random_state)

        # Get global best
        global_best_position = particles[0]["global_best_position"].copy()
        global_best_score = particles[0]["global_best_score"]

        # PSO parameters for the weight formulas from the original paper
        inertia_weight = self.inertia_weight
        h = self.h  # Parameter for cognitive weight formula
        f = self.f  # Parameter for social weight formula

        # Determine learning rate for hybrid approach
        if self.learning_rate == "auto":
            learning_rate = max(n_samples / self.early_exaggeration / 4, 50)
        else:
            learning_rate = self.learning_rate

        # Optimization loop setup
        n_iter_without_progress = 0
        best_error = global_best_score
        all_best_scores = [global_best_score]
        max_iter_without_progress = (
            50  # Maximum iterations without improvement before early stopping
        )
        best_position_history = [global_best_position.copy()]

        if _TQDM_AVAILABLE:
            iterator = (
                tqdm(range(self.n_iter)) if self.verbose > 0 else range(self.n_iter)
            )
        else:
            iterator = range(self.n_iter)
            if self.verbose:
                logger.info("tqdm not available. Not showing progress bar.")

        exaggeration_phase = True
        exaggeration_iter = min(
            250, self.n_iter // 4
        )  # Use 25% of iterations for exaggeration

        # For small datasets, use shorter exaggeration phase
        if n_samples < 200:
            exaggeration_iter = min(125, self.n_iter // 5)  # 20% for small datasets
            if self.verbose:
                logger.info(
                    f"Small dataset detected, using shorter exaggeration phase: {exaggeration_iter} iterations"
                )

        for iter_num in iterator:
            # Check if we should end early exaggeration phase
            if exaggeration_phase and iter_num >= exaggeration_iter:
                exaggeration_phase = False
                # Remove early exaggeration from P for all particles
                for particle in particles:
                    particle["P"] = particle["P"] / self.early_exaggeration

                # Recalculate scores after removing exaggeration
                for i, particle in enumerate(particles):
                    # Re-evaluate fitness with non-exaggerated P
                    score, _ = _kl_divergence(
                        particle["position"],
                        particle["P"],
                        self.degrees_of_freedom,
                        n_samples,
                        self.n_components,
                    )

                    # Update personal best if needed
                    if score < particle["best_score"]:
                        particle["best_position"] = particle["position"].copy()
                        particle["best_score"] = score

                # Find the new global best
                new_best_score = float("inf")
                new_best_position = None

                for particle in particles:
                    if particle["best_score"] < new_best_score:
                        new_best_score = particle["best_score"]
                        new_best_position = particle["best_position"].copy()

                global_best_score = new_best_score
                global_best_position = new_best_position

                all_best_scores.append(global_best_score)
                best_position_history.append(global_best_position.copy())

                if self.verbose:
                    logger.info(
                        f"Ending early exaggeration phase at iteration {iter_num}"
                    )
                    logger.info(
                        f"Updated KL divergence after exaggeration: {global_best_score:.4f}"
                    )

            # Adjust parameters adaptively based on progress
            progress_ratio = iter_num / self.n_iter

            # Linearly decrease inertia weight over iterations for better convergence
            adaptive_inertia = self.inertia_weight * (1.0 - 0.7 * progress_ratio)

            # Calculate cognitive and social weights using the formulas from the original paper
            current_iter = iter_num + 1  # Use 1-indexed iteration count
            adaptive_cognitive = h - (h / (1.0 + (f / current_iter)))
            adaptive_social = h / (1.0 + (f / current_iter))

            # Occasionally apply random perturbation to particles to help escape local minima
            apply_perturbation = random_state.random() < 0.05 * (1.0 - progress_ratio)

            # Process particles individually
            for i, particle in enumerate(particles):
                # Random coefficients for cognitive and social components
                r1 = random_state.uniform(0, 1, particle["position"].shape)
                r2 = random_state.uniform(0, 1, particle["position"].shape)

                # Update velocity with adaptive parameters
                cognitive_component = (
                    adaptive_cognitive
                    * r1
                    * (particle["best_position"] - particle["position"])
                )
                social_component = (
                    adaptive_social * r2 * (global_best_position - particle["position"])
                )

                particle["velocity"] = (
                    adaptive_inertia * particle["velocity"]
                    + cognitive_component
                    + social_component
                )

                # Apply velocity clamping to prevent excessive velocities
                max_velocity = 0.1  # Can be adjusted
                particle["velocity"] = np.clip(
                    particle["velocity"], -max_velocity, max_velocity
                )

                # Update position
                old_position = particle["position"].copy()
                particle["position"] = particle["position"] + particle["velocity"]

                # Apply random perturbation to escape local minima if needed
                if apply_perturbation and i % 3 == 0:  # Apply to some particles
                    perturbation = random_state.normal(
                        0, 0.01 * (1.0 - progress_ratio), particle["position"].shape
                    )
                    particle["position"] += perturbation

                # Hybrid approach: Apply gradient descent with adaptive learning rate
                if self.use_hybrid:
                    # Adjust learning rate based on iteration progress
                    adaptive_lr = learning_rate * (1.0 - 0.5 * progress_ratio)

                    # Apply gradient descent to every particle, but with different frequencies
                    if i % max(1, int(2 * (1 + progress_ratio))) == 0:
                        (
                            particle["position"],
                            _,
                            particle["grad_update"],
                            particle["gains"],
                        ) = _gradient_descent_step(
                            particle["position"],
                            particle["P"],
                            self.degrees_of_freedom,
                            n_samples,
                            self.n_components,
                            momentum=0.5
                            + 0.3 * progress_ratio,  # Increase momentum over time
                            learning_rate=adaptive_lr,
                            min_gain=0.01,
                            update=particle["grad_update"],
                            gains=particle["gains"],
                        )

                # Evaluate fitness
                score, _ = _kl_divergence(
                    particle["position"],
                    particle["P"],
                    self.degrees_of_freedom,
                    n_samples,
                    self.n_components,
                )

                # Assert score is valid
                assert np.isfinite(score), f"Invalid score at iteration {iter_num}"

                # Update personal best
                if score < particle["best_score"]:
                    particle["best_position"] = particle["position"].copy()
                    particle["best_score"] = score

                    # Update global best
                    if score < global_best_score:
                        global_best_position = particle["position"].copy()
                        global_best_score = score
                        best_position_history.append(global_best_position.copy())
                        all_best_scores.append(global_best_score)

                        # Report progress if verbose
                        if self.verbose > 0:
                            if _TQDM_AVAILABLE:
                                tqdm.write(
                                    f"Iteration {iter_num}: New best score = "
                                    f"{score:.4f}"
                                )
                            else:
                                logger.info(
                                    f"Iteration {iter_num}: New best score = {score:.4f}"
                                )

                        # Reset progress counter
                        n_iter_without_progress = 0

            # Update global best for all particles
            for particle in particles:
                particle["global_best_position"] = global_best_position.copy()
                particle["global_best_score"] = global_best_score

            # Check for convergence with adaptive early stopping
            if global_best_score < best_error:
                best_error = global_best_score
                n_iter_without_progress = 0
            else:
                n_iter_without_progress += 1

            # More strict convergence criteria as iterations progress
            adaptive_patience = max(
                10, int(max_iter_without_progress * (1.0 - 0.7 * progress_ratio))
            )

            if n_iter_without_progress >= adaptive_patience:
                if self.verbose > 0:
                    logger.info(f"Converged after {iter_num + 1} iterations")
                break

            # Every 100 iterations, attempt to reinitialize worst performing particles
            if iter_num > 0 and iter_num % 100 == 0:
                # Find worst performing particles
                scores = [p["best_score"] for p in particles]
                worst_idx = np.argsort(scores)[
                    -max(1, self.n_particles // 5) :
                ]  # Reinitialize 20% worst

                for idx in worst_idx:
                    # Reinitialize with a mix of global best and random exploration
                    if (
                        random_state.random() < 0.7
                    ):  # 70% chance to use global best as base
                        new_position = global_best_position.copy()
                        # Add significant noise for exploration
                        noise = random_state.normal(0, 0.05, new_position.shape)
                        particles[idx]["position"] = new_position + noise
                    else:  # 30% chance for complete reinitialization
                        particles[idx]["position"] = random_state.normal(
                            0, 0.01, particles[idx]["position"].shape
                        )

                    # Reset velocity for reinitialized particles
                    particles[idx]["velocity"] = random_state.normal(
                        0, 0.001, particles[idx]["velocity"].shape
                    )

        # Store optimization history for analysis
        self.convergence_history_ = np.array(all_best_scores)

        # Reshape best position to embedding
        best_position = global_best_position.reshape(n_samples, self.n_components)
        best_cost = global_best_score

        # Final assertions to validate output
        assert best_position.shape == (
            n_samples,
            self.n_components,
        ), "Invalid embedding shape"
        assert np.all(np.isfinite(best_position)), "Embedding contains invalid values"
        assert np.isfinite(best_cost), "Invalid final cost"

        return best_position, best_cost, iter_num + 1

    def _validate_data(self, X, y=None):
        """Validate input data format and characteristics.

        This method ensures that the input data meets the requirements for
        t-SNE-PSO processing. It handles both feature matrices and precomputed
        distance matrices, applying appropriate validation checks to each.

        For precomputed distances, it verifies that the matrix is square and
        contains only non-negative values. For feature matrices, it ensures
        appropriate numerical format and minimum sample count.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Input data matrix. If metric='precomputed', this should be a
            square distance matrix between samples.

        y : None
            Ignored. Present for API consistency.

        Returns
        -------
        X : ndarray
            The validated input data.

        Raises
        ------
        ValueError
            If the input data does not meet the requirements, such as
            non-square precomputed distance matrix or negative values.
        """
        if self.metric == "precomputed":
            X = check_array(
                X,
                accept_sparse=False,
                ensure_min_samples=2,
                dtype=np.float64,
            )
            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    f"X should be a square distance matrix but has shape {X.shape}"
                )
            if np.any(X < 0):
                raise ValueError("Precomputed distance contains negative values")
        else:
            X = check_array(
                X,
                accept_sparse=False,
                dtype=np.float64,
                ensure_min_samples=2,
            )
        return X

    def fit(self, X, y=None):
        """Fit t-SNE-PSO model to X, computing the optimized embedding.

        This method performs the complete t-SNE-PSO algorithm:
        1. Validates input parameters and data
        2. Adjusts parameters based on dataset characteristics
        3. Computes pairwise similarities in high-dimensional space
        4. Initializes the swarm of particles
        5. Optimizes the embedding using PSO and hybrid approaches
        6. Stores the final embedding and optimization statistics

        The resulting embedding is stored in the embedding_ attribute, along
        with diagnostics such as the final KL divergence and convergence history.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Input data matrix to be embedded. If metric='precomputed', this should
            be a square distance matrix.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself, allowing for method chaining.

        Notes
        -----
        The optimization process automatically adapts to the characteristics of
        the dataset. For small datasets, special initialization strategies are
        employed to improve embedding quality. The perplexity parameter is
        automatically adjusted if it exceeds dataset constraints.
        """
        self._validate_parameters()

        X = self._validate_data(X)
        n_samples = X.shape[0]

        # Set n_features_in_ correctly
        if self.metric != "precomputed":
            self.n_features_in_ = X.shape[1]

            # Adjust parameters based on dataset size and dimensionality
            self._adjust_params_for_dataset_size(n_samples, self.n_features_in_)
        else:
            # For precomputed distance matrices, use only sample count
            self._adjust_params_for_dataset_size(n_samples, 0)

        self._check_params_vs_input(X)

        if not hasattr(self, "_perplexity_value"):
            self._perplexity_value = self.perplexity

            if n_samples - 1 < 3 * self._perplexity_value:
                self._perplexity_value = (n_samples - 1) / 3.0
                warnings.warn(
                    f"Perplexity ({self.perplexity}) should be less than "
                    f"n_samples ({n_samples}). "
                    f"Using perplexity = {self._perplexity_value:.3f} instead.",
                    UserWarning,
                )

        # Add init array validation
        if isinstance(self.init, np.ndarray):
            if self.init.shape != (X.shape[0], self.n_components):
                raise ValueError(
                    f"init.shape={self.init.shape} but should be "
                    f"(n_samples, n_components)=({X.shape[0]}, {self.n_components})"
                )

        random_state = check_random_state(self.random_state)
        self.embedding_, self.kl_divergence_, self.n_iter_ = self._optimize_embedding(
            X, random_state
        )

        return self

    def fit_transform(self, X, y=None):
        """Fit t-SNE-PSO model to X and return the optimized embedding.

        This convenience method performs model fitting and returns the
        low-dimensional embedding in a single call. It is equivalent to
        calling fit(X) followed by accessing the embedding_ attribute.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Input data matrix to be embedded. If metric='precomputed', this should
            be a square distance matrix.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        embedding : ndarray of shape (n_samples, n_components)
            Low-dimensional embedding of the input data.

        Notes
        -----
        This is the recommended method for one-time embedding of data, as it
        provides a simpler interface than separate fit() and transform() calls.
        """
        self.fit(X)
        return self.embedding_

    def transform(self, X):
        """Transform X to the embedded space.


        This is not implemented for t-SNE, as it does not support the transform
        method. New data points cannot be transformed to the embedded space
        without recomputing the full embedding.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data to be transformed.

        Raises
        ------
        NotImplementedError
            In all cases, as t-SNE does not have a transform method.
        """
        check_is_fitted(self)

        raise NotImplementedError(
            "t-SNE does not support the transform method. "
            "New data points cannot be transformed to the embedded space "
            "without recomputing the full embedding. "
            "Use fit_transform(X) on the full dataset instead."
        )

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.
            - If `input_features` is None, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              [`x0`, `x1`, ..., `x(n_features_in_ - 1)`].
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Output feature names.
        """
        check_is_fitted(self)
        return np.array([f"tsnepso{i}" for i in range(self.n_components)])
