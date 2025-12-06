"""
Unit tests for dimensionality reduction module.
"""

import pytest
import numpy as np
from sklearn.datasets import make_blobs
from src.dimensionality_reduction import UMAPOptimizer, calculate_ssd


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    X, y = make_blobs(n_samples=200, n_features=10, centers=3, random_state=42)
    return X, y


class TestUMAPOptimizer:
    
    def test_initialization(self):
        """Test UMAP optimizer initialization."""
        optimizer = UMAPOptimizer(
            n_components=3,
            metric='euclidean',
            random_state=42
        )
        
        assert optimizer.n_components == 3
        assert optimizer.metric == 'euclidean'
        assert optimizer.random_state == 42
        assert optimizer.reducer is None
    
    def test_fit_transform_basic(self, sample_data):
        """Test basic fit_transform functionality."""
        X, _ = sample_data
        optimizer = UMAPOptimizer(n_components=3, random_state=42)
        
        X_transformed = optimizer.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], 3)
        assert optimizer.reducer is not None
    
    def test_fit_transform_2d(self, sample_data):
        """Test 2D transformation."""
        X, _ = sample_data
        optimizer = UMAPOptimizer(n_components=2, random_state=42)
        
        X_transformed = optimizer.fit_transform(X)
        
        assert X_transformed.shape == (X.shape[0], 2)
    
    def test_fit_transform_parameters(self, sample_data):
        """Test fit_transform with custom parameters."""
        X, _ = sample_data
        optimizer = UMAPOptimizer(n_components=3, random_state=42)
        
        X_transformed = optimizer.fit_transform(
            X,
            n_neighbors=20,
            learning_rate=0.5,
            spread=1.5
        )
        
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 3
    
    def test_transform_with_params(self, sample_data):
        """Test transform with parameter dictionary."""
        X, _ = sample_data
        optimizer = UMAPOptimizer(n_components=3, random_state=42)
        
        params = {
            'n_neighbors': 20,
            'learning_rate': 0.5,
            'spread': 1.0
        }
        
        X_transformed = optimizer.transform(X, params)
        
        assert X_transformed.shape == (X.shape[0], 3)
    
    def test_transform_without_fit_raises_error(self, sample_data):
        """Test that transform without fit raises error."""
        X, _ = sample_data
        optimizer = UMAPOptimizer(n_components=3, random_state=42)
        
        with pytest.raises(ValueError, match="UMAP has not been fitted"):
            optimizer.transform(X)
    
    def test_optimize_parameters(self, sample_data):
        """Test parameter optimization."""
        X, _ = sample_data
        optimizer = UMAPOptimizer(n_components=3, random_state=42)
        
        best_params, results_df = optimizer.optimize_parameters(
            X,
            n_neighbors_range=[15, 30],
            learning_rate_range=[0.5, 1.0],
            spread_range=[1.0],
            return_all_results=True
        )
        
        assert best_params is not None
        assert 'n_neighbors' in best_params
        assert 'learning_rate' in best_params
        assert 'spread' in best_params
        assert results_df is not None
        assert len(results_df) == 4  # 2 x 2 x 1 combinations
        assert 'score' in results_df.columns
    
    def test_optimize_parameters_no_results(self, sample_data):
        """Test optimization without returning all results."""
        X, _ = sample_data
        optimizer = UMAPOptimizer(n_components=3, random_state=42)
        
        best_params, results_df = optimizer.optimize_parameters(
            X,
            n_neighbors_range=[15],
            learning_rate_range=[1.0],
            spread_range=[1.0],
            return_all_results=False
        )
        
        assert best_params is not None
        assert results_df is None
    
    def test_different_metrics(self, sample_data):
        """Test different distance metrics."""
        X, _ = sample_data
        
        metrics = ['euclidean', 'manhattan', 'cosine']
        
        for metric in metrics:
            optimizer = UMAPOptimizer(
                n_components=2,
                metric=metric,
                random_state=42
            )
            X_transformed = optimizer.fit_transform(X, n_neighbors=15)
            assert X_transformed.shape == (X.shape[0], 2)
    
    def test_optimize_with_clustering(self, sample_data):
        """Test joint optimization with clustering."""
        X, _ = sample_data
        from src.clustering import GMMClusterer
        
        optimizer = UMAPOptimizer(n_components=3, random_state=42)
        clusterer = GMMClusterer(random_state=42)
        
        best_umap, best_cluster, results = optimizer.optimize_with_clustering(
            X,
            clusterer,
            n_neighbors_range=[15, 30],
            learning_rate_range=[1.0],
            spread_range=[1.0],
            n_clusters_range=[2, 3],
            covariance_types=['full']
        )
        
        assert best_umap is not None
        assert best_cluster is not None
        assert 'n_components' in best_cluster
        assert 'covariance_type' in best_cluster
        assert not results.empty
        assert 'silhouette_score' in results.columns


class TestCalculateSSD:
    
    def test_calculate_ssd_basic(self):
        """Test SSD calculation."""
        X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        labels = np.array([0, 0, 1, 1])
        
        within_ssd, between_ssd = calculate_ssd(X, labels)
        
        assert within_ssd > 0
        assert between_ssd > 0
        assert isinstance(within_ssd, (int, float))
        assert isinstance(between_ssd, (int, float))
    
    def test_calculate_ssd_single_cluster(self):
        """Test SSD with single cluster."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        labels = np.array([0, 0, 0])
        
        within_ssd, between_ssd = calculate_ssd(X, labels)
        
        assert within_ssd > 0
        assert between_ssd == 0  # No between-cluster variance
    
    def test_calculate_ssd_perfect_separation(self):
        """Test SSD with perfectly separated clusters."""
        X = np.array([[0, 0], [0, 0], [10, 10], [10, 10]])
        labels = np.array([0, 0, 1, 1])
        
        within_ssd, between_ssd = calculate_ssd(X, labels)
        
        assert within_ssd == 0  # Perfect clusters
        assert between_ssd > 0


class TestEdgeCases:
    
    def test_small_sample_size(self):
        """Test UMAP with very small sample size."""
        X = np.random.randn(10, 5)
        optimizer = UMAPOptimizer(n_components=2, random_state=42)
        
        # Should still work but might get warnings
        X_transformed = optimizer.fit_transform(X, n_neighbors=5)
        assert X_transformed.shape == (10, 2)
    
    def test_high_dimensional_data(self):
        """Test UMAP with high-dimensional data."""
        X = np.random.randn(100, 100)
        optimizer = UMAPOptimizer(n_components=3, random_state=42)
        
        X_transformed = optimizer.fit_transform(X)
        assert X_transformed.shape == (100, 3)
    
    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same random_state."""
        X, _ = sample_data
        
        optimizer1 = UMAPOptimizer(n_components=3, random_state=42)
        X_transformed1 = optimizer1.fit_transform(X, n_neighbors=15)
        
        optimizer2 = UMAPOptimizer(n_components=3, random_state=42)
        X_transformed2 = optimizer2.fit_transform(X, n_neighbors=15)
        
        # Results should be very similar (allowing for minor numerical differences)
        assert np.allclose(X_transformed1, X_transformed2, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])