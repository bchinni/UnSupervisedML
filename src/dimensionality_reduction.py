"""
UMAP dimensionality reduction with hyperparameter optimization.

Supports:
- Grid search for optimal UMAP parameters
- Multiple distance metrics
- Custom evaluation metrics (within/between cluster SSD)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import umap
from itertools import product
import logging

logger = logging.getLogger(__name__)


class UMAPOptimizer:
    """Handles UMAP dimensionality reduction and optimization."""
    
    def __init__(
        self,
        n_components: int = 3,
        metric: str = 'euclidean',
        init: str = 'spectral',
        min_dist: float = 0.0,
        random_state: int = 42
    ):
        """
        Initialize UMAP optimizer.
        
        Args:
            n_components: Number of dimensions for embedding
            metric: Distance metric
            init: Initialization method
            min_dist: Minimum distance between points
            random_state: Random seed
        """
        self.n_components = n_components
        self.metric = metric
        self.init = init
        self.min_dist = min_dist
        self.random_state = random_state
        self.reducer = None
        self.best_params = None
        
    def fit_transform(
        self,
        X: np.ndarray,
        n_neighbors: int = 15,
        learning_rate: float = 1.0,
        spread: float = 1.0
    ) -> np.ndarray:
        """
        Fit UMAP and transform data.
        
        Args:
            X: Input data
            n_neighbors: Size of local neighborhood
            learning_rate: Learning rate for optimization
            spread: Effective scale of embedded points
            
        Returns:
            Transformed data
        """
        self.reducer = umap.UMAP(
            n_components=self.n_components,
            metric=self.metric,
            n_neighbors=n_neighbors,
            learning_rate=learning_rate,
            spread=spread,
            min_dist=self.min_dist,
            init=self.init,
            random_state=self.random_state
        )
        
        logger.info(f"Fitting UMAP with n_neighbors={n_neighbors}, "
                   f"learning_rate={learning_rate}, spread={spread}")
        
        X_transformed = self.reducer.fit_transform(X)
        
        logger.info(f"UMAP transformation complete. Output shape: {X_transformed.shape}")
        
        return X_transformed
    
    def transform(self, X: np.ndarray, params: Optional[Dict] = None) -> np.ndarray:
        """
        Transform new data using fitted UMAP or with specific parameters.
        
        Args:
            X: Input data
            params: Optional dictionary of UMAP parameters
            
        Returns:
            Transformed data
        """
        if params is not None:
            return self.fit_transform(
                X,
                n_neighbors=params.get('n_neighbors', 15),
                learning_rate=params.get('learning_rate', 1.0),
                spread=params.get('spread', 1.0)
            )
        
        if self.reducer is None:
            raise ValueError("UMAP has not been fitted. Call fit_transform first or provide params.")
        
        return self.reducer.transform(X)
    
    def optimize_parameters(
        self,
        X: np.ndarray,
        n_neighbors_range: List[int] = None,
        learning_rate_range: List[float] = None,
        spread_range: List[float] = None,
        eval_metric: str = 'ssd',
        return_all_results: bool = True
    ) -> Tuple[Dict, Optional[pd.DataFrame]]:
        """
        Perform grid search to find optimal UMAP parameters.
        
        Args:
            X: Input data
            n_neighbors_range: List of n_neighbors to test
            learning_rate_range: List of learning_rate to test
            spread_range: List of spread to test
            eval_metric: Evaluation metric ('ssd' for sum of squared distances)
            return_all_results: Whether to return all results or just best
            
        Returns:
            Tuple of (best_params, results_df)
        """
        # Default ranges
        n_neighbors_range = n_neighbors_range or [15, 20, 25, 30, 35, 40, 45, 50]
        learning_rate_range = learning_rate_range or [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        spread_range = spread_range or [0.5, 1.0, 1.5, 2.0]
        
        logger.info(f"Starting UMAP parameter optimization with {len(n_neighbors_range) * len(learning_rate_range) * len(spread_range)} combinations")
        
        results = []
        best_score = np.inf if eval_metric == 'ssd' else -np.inf
        best_params = None
        
        # Grid search
        for n_neighbors, learning_rate, spread in product(
            n_neighbors_range, learning_rate_range, spread_range
        ):
            try:
                # Transform data
                X_umap = self.fit_transform(X, n_neighbors, learning_rate, spread)
                
                # Calculate evaluation metric
                if eval_metric == 'ssd':
                    # Calculate variance (lower is better for compact embedding)
                    score = np.sum(np.var(X_umap, axis=0))
                else:
                    raise ValueError(f"Unknown metric: {eval_metric}")
                
                result = {
                    'n_neighbors': n_neighbors,
                    'learning_rate': learning_rate,
                    'spread': spread,
                    'score': score
                }
                results.append(result)
                
                # Update best
                if score < best_score:
                    best_score = score
                    best_params = {
                        'n_neighbors': n_neighbors,
                        'learning_rate': learning_rate,
                        'spread': spread
                    }
                
                logger.debug(f"n_neighbors={n_neighbors}, learning_rate={learning_rate}, "
                           f"spread={spread}, score={score:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed for n_neighbors={n_neighbors}, "
                             f"learning_rate={learning_rate}, spread={spread}: {e}")
                continue
        
        self.best_params = best_params
        logger.info(f"Best parameters: {best_params} with score={best_score:.4f}")
        
        results_df = pd.DataFrame(results) if return_all_results else None
        
        return best_params, results_df
    
    def optimize_with_clustering(
        self,
        X: np.ndarray,
        clusterer,
        n_neighbors_range: List[int] = None,
        learning_rate_range: List[float] = None,
        spread_range: List[float] = None,
        n_clusters_range: List[int] = None,
        covariance_types: List[str] = None
    ) -> Tuple[Dict, Dict, pd.DataFrame]:
        """
        Optimize UMAP parameters jointly with clustering parameters.
        
        Args:
            X: Input data
            clusterer: Clustering object with optimize_clustering method
            n_neighbors_range: List of n_neighbors to test
            learning_rate_range: List of learning_rate to test
            spread_range: List of spread to test
            n_clusters_range: List of cluster numbers to test
            covariance_types: List of covariance types to test
            
        Returns:
            Tuple of (best_umap_params, best_cluster_params, results_df)
        """
        # Default ranges
        n_neighbors_range = n_neighbors_range or [20, 30, 40, 50]
        learning_rate_range = learning_rate_range or [0.1, 0.5, 0.9]
        spread_range = spread_range or [0.5, 1.0, 1.5]
        n_clusters_range = n_clusters_range or list(range(2, 6))
        covariance_types = covariance_types or ['full', 'tied', 'diag', 'spherical']
        
        logger.info(f"Starting joint UMAP-clustering optimization")
        
        results = []
        best_silhouette = -1
        best_umap_params = None
        best_cluster_params = None
        
        for n_neighbors, learning_rate, spread in product(
            n_neighbors_range, learning_rate_range, spread_range
        ):
            try:
                # Transform with UMAP
                X_umap = self.fit_transform(X, n_neighbors, learning_rate, spread)
                
                # Optimize clustering on this embedding
                _, _, cluster_scores = clusterer.optimize_clustering(
                    X_umap,
                    n_components_range=n_clusters_range,
                    covariance_types=covariance_types
                )
                
                # Get best clustering result
                best_cluster_result = cluster_scores.loc[cluster_scores['silhouette_score'].idxmax()]
                
                result = {
                    'n_neighbors': n_neighbors,
                    'learning_rate': learning_rate,
                    'spread': spread,
                    'best_n_clusters': best_cluster_result['n_components'],
                    'best_covariance_type': best_cluster_result['covariance_type'],
                    'silhouette_score': best_cluster_result['silhouette_score'],
                    'bic': best_cluster_result['bic']
                }
                results.append(result)
                
                # Update best overall
                if best_cluster_result['silhouette_score'] > best_silhouette:
                    best_silhouette = best_cluster_result['silhouette_score']
                    best_umap_params = {
                        'n_neighbors': n_neighbors,
                        'learning_rate': learning_rate,
                        'spread': spread
                    }
                    best_cluster_params = {
                        'n_components': int(best_cluster_result['n_components']),
                        'covariance_type': best_cluster_result['covariance_type']
                    }
                
                logger.info(f"UMAP: n_neighbors={n_neighbors}, lr={learning_rate}, spread={spread} | "
                          f"Clustering: {best_cluster_result['n_components']} clusters, "
                          f"silhouette={best_cluster_result['silhouette_score']:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed for n_neighbors={n_neighbors}, lr={learning_rate}, "
                             f"spread={spread}: {e}")
                continue
        
        self.best_params = best_umap_params
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Best UMAP params: {best_umap_params}")
        logger.info(f"Best clustering params: {best_cluster_params}")
        logger.info(f"Best silhouette score: {best_silhouette:.3f}")
        
        return best_umap_params, best_cluster_params, results_df


def calculate_ssd(X: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Calculate within-cluster and between-cluster sum of squared distances.
    
    Args:
        X: Data points
        labels: Cluster labels
        
    Returns:
        Tuple of (within_cluster_ssd, between_cluster_ssd)
    """
    within_cluster_ssd = 0
    overall_mean = X.mean(axis=0)
    between_cluster_ssd = 0
    
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        cluster_center = cluster_points.mean(axis=0)
        
        # Within-cluster SSD
        within_cluster_ssd += ((cluster_points - cluster_center) ** 2).sum()
        
        # Between-cluster SSD
        n_points = cluster_points.shape[0]
        between_cluster_ssd += n_points * ((cluster_center - overall_mean) ** 2).sum()
    
    return within_cluster_ssd, between_cluster_ssd