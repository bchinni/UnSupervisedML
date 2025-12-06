"""
Unit tests for clustering module.
"""

import pytest
import numpy as np
from sklearn.datasets import make_blobs
from src.clustering import GMMClusterer, create_silhouette_matrix, get_cluster_statistics


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    X, y = make_blobs(n_samples=300, n_features=10, centers=3, random_state=42)
    return X, y


class TestGMMClusterer:
    
    def test_initialization(self):
        """Test clusterer initialization."""
        clusterer = GMMClusterer(random_state=42)
        assert clusterer.random_state == 42
        assert clusterer.init_params == 'kmeans'
        assert clusterer.model is None
    
    def test_fit_predict(self, sample_data):
        """Test basic fit_predict functionality."""
        X, _ = sample_data
        clusterer = GMMClusterer(random_state=42)
        labels = clusterer.fit_predict(X, n_components=3)
        
        assert labels.shape[0] == X.shape[0]
        assert len(np.unique(labels)) == 3
        assert clusterer.model is not None
    
    def test_get_cluster_probabilities(self, sample_data):
        """Test cluster probability outputs."""
        X, _ = sample_data
        clusterer = GMMClusterer(random_state=42)
        labels = clusterer.fit_predict(X, n_components=3)
        probs = clusterer.get_cluster_probabilities(X)
        
        assert probs.shape == (X.shape[0], 3)
        assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probs >= 0) and np.all(probs <= 1)  # Valid probabilities
    
    def test_optimize_clustering(self, sample_data):
        """Test clustering optimization."""
        X, _ = sample_data
        clusterer = GMMClusterer(random_state=42)
        
        model, labels, scores_df = clusterer.optimize_clustering(
            X,
            n_components_range=[2, 3, 4],
            covariance_types=['full', 'diag']
        )
        
        assert model is not None
        assert labels.shape[0] == X.shape[0]
        assert not scores_df.empty
        assert 'silhouette_score' in scores_df.columns
        assert 'n_components' in scores_df.columns
        assert clusterer.best_params is not None
    
    def test_get_model_info(self, sample_data):
        """Test model info extraction."""
        X, _ = sample_data
        clusterer = GMMClusterer(random_state=42)
        clusterer.fit_predict(X, n_components=3)
        
        info = clusterer.get_model_info()
        
        assert 'n_components' in info
        assert 'covariance_type' in info
        assert 'converged' in info
        assert info['n_components'] == 3
        assert info['n_features'] == X.shape[1]


class TestHelperFunctions:
    
    def test_create_silhouette_matrix(self, sample_data):
        """Test silhouette matrix creation."""
        X, _ = sample_data
        clusterer = GMMClusterer(random_state=42)
        
        _, _, scores_df = clusterer.optimize_clustering(
            X,
            n_components_range=[2, 3],
            covariance_types=['full', 'diag']
        )
        
        matrix = create_silhouette_matrix(scores_df)
        
        assert not matrix.empty
        assert matrix.shape[0] == 2  # 2 n_components values
        assert matrix.shape[1] == 2  # 2 covariance types
    
    def test_get_cluster_statistics(self, sample_data):
        """Test cluster statistics calculation."""
        X, _ = sample_data
        clusterer = GMMClusterer(random_state=42)
        labels = clusterer.fit_predict(X, n_components=3)
        
        stats = get_cluster_statistics(X, labels)
        
        assert len(stats) == 3
        assert 'cluster' in stats.columns
        assert 'size' in stats.columns
        assert 'percentage' in stats.columns
        assert np.isclose(stats['percentage'].sum(), 100.0, atol=0.1)


if __name__ == '__main__':
    pytest.main([__file__])