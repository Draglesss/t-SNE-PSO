# Authors: Allaoui, Mebarka <mebarka.allaoui@gmail.com>
#          Belhaouari, Samir Brahim
#          Hedjam, Rachid
#          Bouanane, Khadra
#          Kherfi, Mohammed Lamine
#          Otmane Fatteh <fattehotmane@hotmail.com>
# License: BSD 3 clause
#
# This implementation is based on the paper:
# Allaoui, M., Belhaouari, S. B., Hedjam, R., Bouanane, K., & Kherfi, M. L. (2025).
# t-SNE-PSO: Optimizing t-SNE using particle swarm optimization.
# Expert Systems with Applications, 269, 126398.

import warnings
from numbers import Integral, Real

import numpy as np
from scipy.spatial.distance import pdist, squareform

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.manifold import _utils

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

# Machine epsilon for float64
MACHINE_EPSILON = np.finfo(np.double).eps

# Define valid metrics from sklearn
_VALID_METRICS = [
    'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
    'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming',
    'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto',
    'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
    'yule', 'precomputed'
]


def _joint_probabilities(distances, perplexity, verbose=False):
    """Convert distances to joint probabilities P_ij.

    This function calculates the joint probabilities based on pairwise distances,
    implementing the approach described in the original t-SNE paper by van der Maaten
    and Hinton (2008) and adapted for t-SNE-PSO by Allaoui et al. (2025).

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix.

    perplexity : float
        Perplexity parameter (related to the number of nearest neighbors).
        Larger datasets usually require a larger perplexity. Consider
        selecting a value between 5 and 50.

    verbose : bool, default=False
        Whether to print progress messages.

    Returns
    -------
    P : ndarray of shape (n_samples*(n_samples-1)/2,)
        Condensed joint probability matrix.
    """
    # Ensure distances are in the correct format
    distances = distances.astype(np.float32, copy=False)
    
    # Use sklearn's _binary_search_perplexity for calculating conditional probabilities
    # This efficiently implements the binary search for sigma values that yield the desired perplexity
    conditional_P = _utils._binary_search_perplexity(
        distances, perplexity, verbose
    )
    
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

    Parameters
    ----------
    params : ndarray of shape (n_samples * n_components,)
        Flattened array of current embeddings.

    P : ndarray of shape (n_samples*(n_samples-1)/2,)
        Condensed joint probability matrix from high-dimensional space.

    degrees_of_freedom : float
        Degrees of freedom of the Student's t-distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    skip_num_points : int, default=0
        Number of points to skip in gradient computation.

    compute_error : bool, default=True
        Whether to compute the KL divergence.

    Returns
    -------
    kl_divergence : float
        KL divergence between P and Q.

    grad : ndarray of shape (n_samples * n_components,)
        Gradient of the KL divergence with respect to the embedding.
    """
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
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
    grad = grad.ravel()

    # Scale the gradient
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


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

    Parameters
    ----------
    params : ndarray of shape (n_samples * n_components,)
        Flattened array of current embeddings.

    P : ndarray of shape (n_samples*(n_samples-1)/2,)
        Condensed joint probability matrix from high-dimensional space.

    degrees_of_freedom : float
        Degrees of freedom of the Student's t-distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    momentum : float, default=0.8
        Momentum for gradient descent.

    learning_rate : float, default=200.0
        Learning rate for gradient descent.

    min_gain : float, default=0.01
        Minimum gain for gradient descent.

    update : ndarray of shape (n_samples * n_components,), default=None
        Previous update for momentum calculation.

    gains : ndarray of shape (n_samples * n_components,), default=None
        Previous gains for adaptive learning rates.

    Returns
    -------
    params : ndarray of shape (n_samples * n_components,)
        Updated embeddings.

    error : float
        KL divergence between P and Q.

    update : ndarray of shape (n_samples * n_components,)
        Updated gradients.

    gains : ndarray of shape (n_samples * n_components,)
        Updated gains.
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
    """t-SNE with Particle Swarm Optimization.

    t-Distributed Stochastic Neighbor Embedding (t-SNE) with Particle Swarm
    Optimization (PSO) for the optimization step instead of gradient descent.
    This approach can be more effective at avoiding local minima and often
    produces embeddings with better cluster separation.

    This implementation is based on the paper:
    Allaoui, M., Belhaouari, S. B., Hedjam, R., Bouanane, K., & Kherfi, M. L. (2025).
    "t-SNE-PSO: Optimizing t-SNE using particle swarm optimization."
    Expert Systems with Applications, 269, 126398.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.

    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors used.
        Larger datasets usually require a larger perplexity. Consider
        selecting a value between 5 and 50.

    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in
        the embedded space. Larger values ensure more widely separated
        embedding clusters. Only used during the early exaggeration phase.

    learning_rate : float or "auto", default="auto"
        The learning rate for t-SNE optimization. If "auto", the learning
        rate is set to max(N / early_exaggeration / 4, 50) where N is the
        sample size. Only used during gradient descent steps in the hybrid
        PSO approach.

    n_iter : int, default=1000
        Maximum number of iterations for optimization.

    n_particles : int, default=10
        Number of particles for PSO optimization. Larger values can provide
        better exploration at the cost of computational efficiency.

    inertia_weight : float, default=0.5
        Inertia weight for PSO. Controls how much of the previous velocity
        is preserved. Values closer to 0 accelerate convergence, while values
        closer to 1 encourage exploration.

    cognitive_weight : float, default=1.0
        Cognitive weight for PSO. Controls how much particles are influenced
        by their personal best position.

    social_weight : float, default=1.0
        Social weight for PSO. Controls how much particles are influenced
        by the global best position.

    use_hybrid : bool, default=True
        Whether to use hybrid PSO with gradient descent steps. When True,
        alternates between PSO updates and gradient descent steps for
        improved convergence.

    degrees_of_freedom : float, default=1.0
        Degrees of freedom of the Student's t-distribution. Lower values
        emphasize the separation between clusters.

    init : str or ndarray of shape (n_samples, n_components), default='pca'
        Initialization method. Valid options are:
        - 'pca': Principal Component Analysis initialization
        - 'tsne': Initialization from a standard t-SNE run
        - 'umap': Initialization from UMAP (if available)
        - 'random': Random initialization
        - ndarray: ndarray of shape (n_samples, n_components) to use for initialization

    verbose : int, default=0
        Verbosity level. If greater than 0, progress messages are printed.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator for initialization.
        Pass an int for reproducible results across multiple function calls.

    method : str, default='pso'
        Method to use for optimization. Currently only 'pso' is supported.

    angle : float, default=0.5
        Only used if method='barnes_hut'. This is the trade-off between speed
        and accuracy for Barnes-Hut T-SNE. 'angle' is the angular size (referred
        to as theta in [3]) of a distant node as measured from a point.

    n_jobs : int, default=None
        The number of parallel jobs to run for computation. -1 means using all
        processors. Currently not used (placeholder for future implementation).

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    h : float, default=1e-20
        Parameter for dynamic cognitive weight formula: c1 = h - (h / (1 + (f / it))).
        Controls the balance between exploration and exploitation during optimization.
        Used in the original t-SNE-PSO paper.

    f : float, default=1e-21
        Parameter for dynamic social weight formula: c2 = h / (1 + (f / it)).
        Controls the balance between exploration and exploitation during optimization.
        Used in the original t-SNE-PSO paper.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the embedding vectors.

    kl_divergence_ : float
        Final KL divergence value (cost function).

    n_iter_ : int
        Number of iterations run.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    References
    ----------
    .. [1] van der Maaten, L.J.P. and Hinton, G.E., 2008. "Visualizing
       High-Dimensional Data Using t-SNE." Journal of Machine Learning
       Research, 9(Nov), pp.2579-2605.

    .. [2] Kennedy, J. and Eberhart, R., 1995. "Particle swarm optimization."
       In Proceedings of ICNN'95 - International Conference on Neural Networks,
       Vol. 4, pp. 1942-1948.

    .. [3] Shi, Y. and Eberhart, R., 1998. "A modified particle swarm
       optimizer." In 1998 IEEE International Conference on Evolutionary
       Computation Proceedings, pp. 69-73.

    Examples
    --------
    >>> import numpy as np
    >>> from tsne_pso import TSNEPSO
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = TSNEPSO(random_state=0)
    >>> Y = model.fit_transform(X)
    >>> Y.shape
    (4, 2)
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
        "h": [Interval(Real, 0, None, closed="left")],
        "f": [Interval(Real, 0, None, closed="left")],
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
        self.h = h  # Parameter for dynamic cognitive weight formula
        self.f = f  # Parameter for dynamic social weight formula

    def _validate_parameters(self):
        """Validate input parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid.
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
        """Check if perplexity is smaller than number of samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        """
        n_samples = X.shape[0]
        # We issue a warning if perplexity is too close to n_samples
        perplexity_threshold = 0.99 * n_samples  # 99% of n_samples
        
        if self.perplexity >= n_samples:
            # Adjust perplexity to be slightly less than n_samples
            self._perplexity_value = max(1.0, (n_samples - 1) / 3.0)
            warnings.warn(
                f"Perplexity ({self.perplexity}) should be less than "
                f"n_samples ({n_samples}). "
                f"Using perplexity = {self._perplexity_value:.3f} instead.",
                UserWarning,
            )
        elif self.perplexity >= perplexity_threshold:
            # Warning for perplexity close to n_samples
            self._perplexity_value = max(1.0, (n_samples - 1) / 3.0)
            warnings.warn(
                f"Perplexity ({self.perplexity}) should be less than "
                f"n_samples ({n_samples}). "
                f"Using perplexity = {self._perplexity_value:.3f} instead.",
                UserWarning,
            )
        else:
            self._perplexity_value = self.perplexity

    def _initialize_particles(self, X, random_state):
        """Initialize particles for PSO optimization.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        random_state : RandomState instance
            Random number generator.

        Returns
        -------
        particles : list of dict
            List of particles with their positions, velocities, personal best positions,
            personal best scores, and evaluations of the fitness function.
        """
        n_samples = X.shape[0]
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
        assert distances.shape == (n_samples, n_samples), "Distance matrix shape mismatch"
        assert np.all(np.isfinite(distances)), "Distance matrix contains invalid values"

        # Compute joint probabilities
        P = _joint_probabilities(distances, self._perplexity_value, self.verbose > 0)
        
        # Assert P is valid
        assert P.shape == ((n_samples * (n_samples - 1)) // 2,), "Joint probability shape mismatch"
        assert np.all(np.isfinite(P)), "Joint probability matrix contains invalid values"
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
                random_state=random_state.randint(0, 2**32 - 1),
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
                random_state=random_state.randint(0, 2**32 - 1),
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
                random_state=random_state.randint(0, 2**32 - 1),
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

    def _optimize_embedding(self, X, random_state):
        """Optimize embedding using Particle Swarm Optimization.

        This function implements the core t-SNE-PSO algorithm as described in Allaoui et al. (2025).
        It uses Particle Swarm Optimization with dynamic cognitive and social weights
        to minimize the KL divergence between high-dimensional and low-dimensional distributions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        random_state : RandomState instance
            Random number generator.

        Returns
        -------
        best_position : ndarray of shape (n_samples, n_components)
            Optimized embedding.

        best_cost : float
            Final KL divergence value.

        n_iter : int
            Number of iterations run.
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
        max_iter_without_progress = 50  # Maximum iterations without improvement before early stopping
        best_position_history = [global_best_position.copy()]

        if _TQDM_AVAILABLE:
            iterator = tqdm(range(self.n_iter)) if self.verbose > 0 else range(self.n_iter)
        else:
            iterator = range(self.n_iter)
            if self.verbose:
                print("tqdm not available. Not showing progress bar.")

        exaggeration_phase = True
        exaggeration_iter = min(250, self.n_iter // 4)  # Use 25% of iterations for exaggeration

        for iter_num in iterator:
            # Check if we should end early exaggeration phase
            if exaggeration_phase and iter_num >= exaggeration_iter:
                exaggeration_phase = False
                # Remove early exaggeration from P for all particles
                for particle in particles:
                    particle["P"] = particle["P"] / self.early_exaggeration
                if self.verbose:
                    print(f"Ending early exaggeration phase at iteration {iter_num}")
            
            # Adjust parameters adaptively based on progress
            progress_ratio = iter_num / self.n_iter
            
            # Linearly decrease inertia weight over iterations for better convergence
            adaptive_inertia = self.inertia_weight * (1.0 - 0.7 * progress_ratio)
            
            # Calculate cognitive and social weights using the formulas from the original paper:
            # - Cognitive weight (c1): Measures particle's attraction to its personal best position
            # - Social weight (c2): Measures particle's attraction to the global best position
            # The dynamic adjustment helps balance exploration and exploitation during optimization
            current_iter = iter_num + 1  # Use 1-indexed iteration count
            adaptive_cognitive = h - (h / (1.0 + (f / current_iter)))
            adaptive_social = h / (1.0 + (f / current_iter))
            
            # Occasionally apply random perturbation to particles to help escape local minima
            # Probability decreases as iterations progress (simulated annealing approach)
            apply_perturbation = random_state.random() < 0.05 * (1.0 - progress_ratio)

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
                    perturbation = random_state.normal(0, 0.01 * (1.0 - progress_ratio), 
                                                    particle["position"].shape)
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
                            momentum=0.5 + 0.3 * progress_ratio,  # Increase momentum over time
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
                                print(
                                    f"Iteration {iter_num}: New best score = "
                                    f"{score:.4f}"
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
                adaptive_patience = max(10, int(max_iter_without_progress * (1.0 - 0.7 * progress_ratio)))
                
                if n_iter_without_progress >= adaptive_patience:
                    if self.verbose > 0:
                        print(f"Converged after {iter_num + 1} iterations")
                    break
                    
            # Every 100 iterations, attempt to reinitialize worst performing particles
            if iter_num > 0 and iter_num % 100 == 0:
                # Find worst performing particles
                scores = [p["best_score"] for p in particles]
                worst_idx = np.argsort(scores)[-max(1, self.n_particles // 5):]  # Reinitialize 20% worst
                
                for idx in worst_idx:
                    # Reinitialize with a mix of global best and random exploration
                    if random_state.random() < 0.7:  # 70% chance to use global best as base
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
        assert best_position.shape == (n_samples, self.n_components), "Invalid embedding shape"
        assert np.all(np.isfinite(best_position)), "Embedding contains invalid values"
        assert np.isfinite(best_cost), "Invalid final cost"

        return best_position, best_cost, iter_num + 1

    def _validate_data(self, X, y=None):
        """Validate the input data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        y : None
            Ignored.

        Returns
        -------
        X : ndarray
            The validated input.
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
        """Fit t-SNE model to X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._validate_parameters()

        X = self._validate_data(X)
        n_samples = X.shape[0]
        
        # Set n_features_in_ correctly
        if self.metric != "precomputed":
            self.n_features_in_ = X.shape[1]
        
        self._check_params_vs_input(X)

        if not hasattr(self, "_perplexity_value"):
            self._perplexity_value = self.perplexity

            if n_samples - 1 < 3 * self._perplexity_value:
                self._perplexity_value = (n_samples - 1) / 3.0
                warnings.warn(
                    f"Perplexity is too large for the number of samples. "
                    f"Using perplexity = {self._perplexity_value:.3f} instead.",
                    UserWarning,
                )

        random_state = check_random_state(self.random_state)
        self.embedding_, self.kl_divergence_, self.n_iter_ = self._optimize_embedding(
            X, random_state
        )

        return self

    def fit_transform(self, X, y=None):
        """Fit t-SNE model to X and return the embedding.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        embedding : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
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
