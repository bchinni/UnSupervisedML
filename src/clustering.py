"""
GMM clustering with silhouette score optimization.

Supports:
- Automatic selection of optimal cluster number
- Multiple covariance structures
- Cluster probability outputs
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


class GMMClusterer:
    """Handles Gaussian Mixture Model clustering."""
    
    def __init__(
        self,
        init_params: str = 'kmeans',
        max_iter: int = 100,
        n_init: int = 10,
        random_state: int = 42
    ):
        """
        Initialize GMM clusterer.
        
        Args:
            init_params: Initialization method ('kmeans' or 'random')
            max_iter: Maximum iterations for EM algorithm
            n_init: Number of initializations
            random_state: Random seed
        """
        self.init_params = init_params
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.model = None
        self.best_params = None
        
    def fit_predict(
        self,
        X: np.ndarray,
        n_components: int = 3,
        covariance_type: str = 'full'
    ) -> np.ndarray:
        """
        Fit GMM and predict cluster labels.
        
        Args:
            X: Input data
            n_components: Number of clusters
            covariance_type: Type of covariance parameters
            
        Returns:
            Cluster labels
        """
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            init_params=self.init_params,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state
        )
        
        logger.info(f"Fitting GMM with {n_components} components and "
                   f"{covariance_type} covariance")
        
        labels = self.model.fit_predict(X)
        
        logger.info(f"GMM fitting complete. Found {len(np.unique(labels))} clusters")
        
        return labels
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Input data
            
        Returns:
            Cluster labels
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit_predict first.")
        
        return self.model.predict(X)
    
    def get_cluster_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Get cluster membership probabilities.
        
        Args:
            X: Input data
            
        Returns:
            Probability matrix (n_samples, n_components)
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit_predict first.")
        
        return self.model.predict_proba(X)
    
    def optimize_clustering(
        self,
        X: np.ndarray,
        n_components_range: List[int] = None,
        covariance_types: List[str] = None,
        return_all_scores: bool = True
    ) -> Tuple[GaussianMixture, np.ndarray, Optional[pd.DataFrame]]:
        """
        Find optimal number of clusters and covariance type using silhouette score.
        
        Args:
            X: Input data
            n_components_range: Range of cluster numbers to test
            covariance_types: List of covariance types to test
            return_all_scores: Whether to return all scores
            
        Returns:
            Tuple of (best_model, best_labels, scores_df)
        """
        n_components_range = n_components_range or list(range(2, 11))
        covariance_types = covariance_types or ['full', 'tied', 'diag', 'spherical']
        
        logger.info(f"Optimizing GMM with {len(n_components_range)} cluster options and "
                   f"{len(covariance_types)} covariance types")
        
        results = []
        best_score = -1
        best_model = None
        best_labels = None
        best_params = None
        
        for n_components in n_components_range:
            for covariance_type in covariance_types:
                try:
                    # Fit model
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        init_params=self.init_params,
                        max_iter=self.max_iter,
                        n_init=self.n_init,
                        random_state=self.random_state
                    )
                    
                    labels = gmm.fit_predict(X)
                    
                    # Calculate metrics
                    silhouette = silhouette_score(X, labels)
                    bic = gmm.bic(X)
                    aic = gmm.aic(X)
                    
                    result = {
                        'n_components': n_components,
                        'covariance_type': covariance_type,
                        'silhouette_score': silhouette,
                        'bic': bic,
                        'aic': aic,
                        'converged': gmm.converged_
                    }
                    results.append(result)
                    
                    logger.debug(f"n_components={n_components}, covariance={covariance_type}, "
                               f"silhouette={silhouette:.3f}, BIC={bic:.2f}")
                    
                    # Update best
                    if silhouette > best_score:
                        best_score = silhouette
                        best_model = gmm
                        best_labels = labels
                        best_params = {
                            'n_components': n_components,
                            'covariance_type': covariance_type
                        }
                
                except Exception as e:
                    logger.warning(f"Failed for n_components={n_components}, "
                                 f"covariance={covariance_type}: {e}")
                    continue
        
        self.model = best_model
        self.best_params = best_params
        
        logger.info(f"Best configuration: {best_params['n_components']} clusters with "
                   f"{best_params['covariance_type']} covariance")
        logger.info(f"Best silhouette score: {best_score:.3f}")
        
        scores_df = pd.DataFrame(results) if return_all_scores else None
        
        return best_model, best_labels, scores_df
    
    def get_cluster_centers(self) -> np.ndarray:
        """
        Get cluster centers (means).
        
        Returns:
            Cluster centers
        """
        if self.model is None:
            raise ValueError("Model has not been fitted.")
        
        return self.model.means_
    
    def get_cluster_covariances(self) -> np.ndarray:
        """
        Get cluster covariances.
        
        Returns:
            Covariance matrices
        """
        if self.model is None:
            raise ValueError("Model has not been fitted.")
        
        return self.model.covariances_
    
    def get_model_info(self) -> dict:
        """
        Get information about the fitted model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            raise ValueError("Model has not been fitted.")
        
        info = {
            'n_components': self.model.n_components,
            'covariance_type': self.model.covariance_type,
            'n_iter': self.model.n_iter_,
            'converged': self.model.converged_,
            'n_features': self.model.means_.shape[1]
        }
        
        return info


def create_silhouette_matrix(
    scores_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a pivot table of silhouette scores for visualization.
    
    Args:
        scores_df: DataFrame with optimization results
        
    Returns:
        Pivot table with n_components as rows and covariance_type as columns
    """
    pivot = scores_df.pivot_table(
        index='n_components',
        columns='covariance_type',
        values='silhouette_score'
    )
    
    return pivot


def compare_clustering_methods(
    X: np.ndarray,
    n_components: int,
    methods: List[str] = None
) -> pd.DataFrame:
    """
    Compare different clustering initialization methods.
    
    Args:
        X: Input data
        n_components: Number of clusters
        methods: List of initialization methods
        
    Returns:
        Comparison results
    """
    methods = methods or ['kmeans', 'random']
    results = []
    
    for method in methods:
        for cov_type in ['full', 'tied', 'diag', 'spherical']:
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=cov_type,
                    init_params=method,
                    random_state=42
                )
                
                labels = gmm.fit_predict(X)
                silhouette = silhouette_score(X, labels)
                bic = gmm.bic(X)
                
                results.append({
                    'init_method': method,
                    'covariance_type': cov_type,
                    'silhouette_score': silhouette,
                    'bic': bic,
                    'converged': gmm.converged_
                })
                
            except Exception as e:
                logger.warning(f"Failed for {method}/{cov_type}: {e}")
                continue
    
    return pd.DataFrame(results)


def get_cluster_statistics(X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Calculate statistics for each cluster.
    
    Args:
        X: Input data
        labels: Cluster labels
        
    Returns:
        DataFrame with cluster statistics
    """
    stats = []
    
    for label in np.unique(labels):
        cluster_data = X[labels == label]
        
        stats.append({
            'cluster': label,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(X) * 100,
            'mean_dist_to_center': np.mean(
                np.linalg.norm(cluster_data - cluster_data.mean(axis=0), axis=1)
            )
        })
    
    return pd.DataFrame(stats)