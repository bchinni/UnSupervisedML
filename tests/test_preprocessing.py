"""
Unit tests for preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import DataPreprocessor


@pytest.fixture
def sample_dataframe():
    """Generate sample dataframe for testing."""
    np.random.seed(42)
    data = {
        'id': [f'sample_{i}' for i in range(100)],
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'feature4': np.random.randn(100),
        'feature5': np.random.randn(100),
        'categorical1': np.random.choice(['A', 'B', 'C'], 100),
        'categorical2': np.random.choice([0, 1], 100),
        'outcome': np.random.choice([0, 1], 100)
    }
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[0:5, 'feature1'] = np.nan
    df.loc[10:15, 'feature2'] = np.nan
    
    return df


class TestDataPreprocessor:
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor.replace_values == ['.', 'NaT', '', ' ', 'nan', 'NaN']
        assert preprocessor.scaler is None
    
    def test_custom_replace_values(self):
        """Test custom replace values."""
        preprocessor = DataPreprocessor(replace_values=['NA', 'missing'])
        assert preprocessor.replace_values == ['NA', 'missing']
    
    def test_filter_missing_features(self, sample_dataframe):
        """Test filtering features with high missing values."""
        df = sample_dataframe.copy()
        # Add feature with 50% missing
        df['feature_missing'] = np.nan
        df.loc[0:49, 'feature_missing'] = np.random.randn(50)
        
        preprocessor = DataPreprocessor()
        df_filtered = preprocessor.filter_missing_features(df, threshold=0.3)
        
        assert 'feature_missing' not in df_filtered.columns
        assert 'feature1' in df_filtered.columns  # Only ~5% missing
    
    def test_scale_features_standard(self, sample_dataframe):
        """Test standard scaling."""
        preprocessor = DataPreprocessor()
        df = sample_dataframe[['feature1', 'feature2', 'feature3']].dropna()
        
        df_scaled = preprocessor.scale_features(df, method='standard')
        
        # Check mean ~0 and std ~1
        assert np.allclose(df_scaled.mean(), 0, atol=1e-10)
        assert np.allclose(df_scaled.std(), 1, atol=1e-10)
    
    def test_scale_features_minmax(self, sample_dataframe):
        """Test min-max scaling."""
        preprocessor = DataPreprocessor()
        df = sample_dataframe[['feature1', 'feature2', 'feature3']].dropna()
        
        df_scaled = preprocessor.scale_features(df, method='minmax')
        
        # Check values in [0, 1]
        assert df_scaled.min().min() >= 0
        assert df_scaled.max().max() <= 1
    
    def test_scale_features_robust(self, sample_dataframe):
        """Test robust scaling."""
        preprocessor = DataPreprocessor()
        df = sample_dataframe[['feature1', 'feature2', 'feature3']].dropna()
        
        df_scaled = preprocessor.scale_features(df, method='robust')
        
        assert df_scaled.shape == df.shape
        assert preprocessor.scaler is not None
    
    def test_scale_features_exclude_cols(self, sample_dataframe):
        """Test scaling with excluded columns."""
        preprocessor = DataPreprocessor()
        df = sample_dataframe[['feature1', 'feature2', 'feature3']].dropna()
        
        df_scaled = preprocessor.scale_features(
            df, 
            method='standard',
            exclude_cols=['feature3']
        )
        
        # feature3 should be unchanged
        assert df_scaled['feature3'].equals(df['feature3'])
        # Other features should be scaled
        assert not df_scaled['feature1'].equals(df['feature1'])
    
    def test_classify_columns(self, sample_dataframe):
        """Test column classification."""
        preprocessor = DataPreprocessor()
        numerical, categorical = preprocessor.classify_columns(
            sample_dataframe, 
            threshold=10
        )
        
        assert 'id' in numerical  # Unique for each row
        assert 'feature1' in numerical
        assert 'categorical1' in categorical
        assert 'categorical2' in categorical
    
    def test_filter_by_outcome(self, sample_dataframe):
        """Test filtering by outcome."""
        preprocessor = DataPreprocessor()
        
        # Add missing outcome
        df = sample_dataframe.copy()
        df.loc[0:5, 'outcome'] = np.nan
        
        df_filtered = preprocessor.filter_by_outcome(
            df, 
            outcome_col='outcome',
            drop_missing=True
        )
        
        assert len(df_filtered) == 94  # 100 - 6 missing
        assert df_filtered['outcome'].isnull().sum() == 0
    
    def test_filter_by_outcome_value(self, sample_dataframe):
        """Test filtering by specific outcome value."""
        preprocessor = DataPreprocessor()
        
        df_filtered = preprocessor.filter_by_outcome(
            sample_dataframe,
            outcome_col='outcome',
            outcome_value=1
        )
        
        assert all(df_filtered['outcome'] == 1)
        assert len(df_filtered) < len(sample_dataframe)
    
    def test_select_numeric_features(self, sample_dataframe):
        """Test selecting numeric features."""
        preprocessor = DataPreprocessor()
        
        df_numeric = preprocessor.select_numeric_features(sample_dataframe)
        
        assert 'categorical1' not in df_numeric.columns
        assert 'feature1' in df_numeric.columns
        assert all(df_numeric.dtypes.apply(lambda x: np.issubdtype(x, np.number)))
    
    def test_select_numeric_features_exclude(self, sample_dataframe):
        """Test selecting numeric features with exclusions."""
        preprocessor = DataPreprocessor()
        
        df_numeric = preprocessor.select_numeric_features(
            sample_dataframe,
            exclude_cols=['outcome']
        )
        
        assert 'outcome' not in df_numeric.columns
        assert 'feature1' in df_numeric.columns
    
    def test_prepare_for_clustering(self, sample_dataframe):
        """Test preparing data for clustering."""
        preprocessor = DataPreprocessor()
        
        feature_df, metadata_df = preprocessor.prepare_for_clustering(
            sample_dataframe,
            id_col='id',
            outcome_col='outcome'
        )
        
        assert 'id' not in feature_df.columns
        assert 'outcome' not in feature_df.columns
        assert 'id' in metadata_df.columns
        assert 'outcome' in metadata_df.columns
        assert len(feature_df) == len(metadata_df)
    
    def test_get_data_summary(self, sample_dataframe):
        """Test data summary generation."""
        preprocessor = DataPreprocessor()
        
        summary = preprocessor.get_data_summary(sample_dataframe)
        
        assert 'shape' in summary
        assert 'n_samples' in summary
        assert 'n_features' in summary
        assert 'missing_counts' in summary
        assert summary['n_samples'] == 100
        assert summary['n_features'] == 9


class TestEdgeCases:
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame()
        
        summary = preprocessor.get_data_summary(df)
        assert summary['n_samples'] == 0
        assert summary['n_features'] == 0
    
    def test_no_numeric_columns(self):
        """Test handling dataframe with no numeric columns."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z']
        })
        
        df_scaled = preprocessor.scale_features(df)
        assert df_scaled.equals(df)  # Should return unchanged
    
    def test_all_missing_column(self):
        """Test handling column with all missing values."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [np.nan, np.nan, np.nan]
        })
        
        df_filtered = preprocessor.filter_missing_features(df, threshold=0.5)
        assert 'feature2' not in df_filtered.columns
        assert 'feature1' in df_filtered.columns


if __name__ == '__main__':
    pytest.main([__file__])