"""
Test script for the enhanced TSNE-PSO implementation.

This script compares the performance of the enhanced TSNE-PSO against
standard t-SNE on multiple datasets, measuring KL divergence and
visualization quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from time import time

from tsne_pso import TSNEPSO

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Setup plotting
plt.style.use('seaborn-v0_8-whitegrid')

def run_comparison(dataset_name, X, y, perplexity=30.0, n_iter=500):
    """
    Run a comparison between TSNE-PSO and standard t-SNE.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset for display purposes.
    X : ndarray
        Input data to embed.
    y : ndarray
        Target labels for coloring.
    perplexity : float, default=30.0
        Perplexity parameter for both algorithms.
    n_iter : int, default=500
        Maximum number of iterations.
    
    Returns
    -------
    results : dict
        Dictionary containing results and timing information.
    """
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name} | Shape: {X.shape} | Classes: {len(np.unique(y))}")
    print(f"{'='*50}")
    
    # Scale the data
    X_scaled = StandardScaler().fit_transform(X)
    
    results = {}
    
    # Run TSNE-PSO
    print("\nRunning TSNE-PSO...")
    tsne_pso = TSNEPSO(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        n_particles=10,
        inertia_weight=0.5,
        cognitive_weight=1.0,
        social_weight=1.0,
        use_hybrid=True,
        verbose=1,
        random_state=RANDOM_STATE
    )
    
    t0 = time()
    embedding_pso = tsne_pso.fit_transform(X_scaled)
    pso_time = time() - t0
    
    results['tsne_pso'] = {
        'embedding': embedding_pso,
        'kl_divergence': tsne_pso.kl_divergence_,
        'time': pso_time
    }
    
    # Run standard t-SNE
    print("\nRunning standard t-SNE...")
    tsne_std = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=RANDOM_STATE
    )
    
    t0 = time()
    embedding_std = tsne_std.fit_transform(X_scaled)
    std_time = time() - t0
    
    results['tsne_std'] = {
        'embedding': embedding_std,
        'kl_divergence': tsne_std.kl_divergence_,
        'time': std_time
    }
    
    # Print results
    print("\nResults:")
    print(f"TSNE-PSO KL divergence: {tsne_pso.kl_divergence_:.6f} | Time: {pso_time:.2f}s")
    if hasattr(tsne_pso, 'convergence_history_'):
        print(f"TSNE-PSO convergence points: {len(tsne_pso.convergence_history_)}")
    print(f"Standard t-SNE KL divergence: {tsne_std.kl_divergence_:.6f} | Time: {std_time:.2f}s")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot TSNE-PSO
    scatter0 = axes[0].scatter(embedding_pso[:, 0], embedding_pso[:, 1], 
                              c=y, cmap='viridis', alpha=0.8, s=40)
    axes[0].set_title(f'TSNE-PSO | {dataset_name} | KL={tsne_pso.kl_divergence_:.4f}')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    fig.colorbar(scatter0, ax=axes[0], label='Class')
    
    # Plot standard t-SNE
    scatter1 = axes[1].scatter(embedding_std[:, 0], embedding_std[:, 1], 
                              c=y, cmap='viridis', alpha=0.8, s=40)
    axes[1].set_title(f'Standard t-SNE | {dataset_name} | KL={tsne_std.kl_divergence_:.4f}')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    fig.colorbar(scatter1, ax=axes[1], label='Class')
    
    plt.tight_layout()
    plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot convergence history if available
    if hasattr(tsne_pso, 'convergence_history_'):
        plt.figure(figsize=(10, 6))
        plt.plot(tsne_pso.convergence_history_, '-o', linewidth=2, markersize=4)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(f'TSNE-PSO Convergence | {dataset_name}')
        plt.xlabel('Improvement Step')
        plt.ylabel('KL Divergence')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_convergence.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    return results


def main():
    """Run the comparison on multiple datasets."""
    datasets = {
        'Iris': load_iris(),
        'Wine': load_wine(),
        'Digits': load_digits()
    }
    
    all_results = {}
    
    # Full test on small datasets
    for name, dataset in datasets.items():
        if name != 'Digits':
            # Small datasets - use full data
            X, y = dataset.data, dataset.target
            all_results[name] = run_comparison(name, X, y, 
                                               perplexity=min(30.0, X.shape[0] / 5),
                                               n_iter=500)
    
    # Subsample larger datasets
    if 'Digits' in datasets:
        print("\nProcessing Digits dataset (subsampled)...")
        X, y = datasets['Digits'].data, datasets['Digits'].target
        
        # Subsample to speed up testing
        np.random.seed(RANDOM_STATE)
        sample_size = 500
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sub, y_sub = X[indices], y[indices]
        
        all_results['Digits'] = run_comparison('Digits (subsampled)', X_sub, y_sub, 
                                              perplexity=min(30.0, sample_size / 5),
                                              n_iter=500)
    
    # Compare all results
    print("\nSummary of results:")
    print(f"{'Dataset':<20} {'TSNE-PSO KL':<15} {'Standard KL':<15} {'Improvement':<15} {'TSNE-PSO Time':<15} {'Standard Time':<15}")
    print('-' * 100)
    
    for name, result in all_results.items():
        pso_kl = result['tsne_pso']['kl_divergence']
        std_kl = result['tsne_std']['kl_divergence']
        improvement = (std_kl - pso_kl) / std_kl * 100  # Percentage improvement
        
        pso_time = result['tsne_pso']['time']
        std_time = result['tsne_std']['time']
        
        print(f"{name:<20} {pso_kl:<15.6f} {std_kl:<15.6f} {improvement:<15.2f}% {pso_time:<15.2f}s {std_time:<15.2f}s")


if __name__ == "__main__":
    main() 