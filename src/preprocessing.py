"""
Data preprocessing utilities for unsupervised clustering pipeline.

Handles:
- Data loading and cleaning
- Missing value handling
- Feature scaling
- Column type classification
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing for clustering analysis."""
    
    def __init__(self, replace_values: Optional[List] = None):
        """
        Initialize preprocessor.
        
        Args:
            replace_values: List of values to replace with NaN
        """
        self.replace_values = replace_values or ['.', 'NaT', '', ' ', 'nan', 'NaN']
        self.scaler = None
        
    def load_and_clean(
        self, 
        filepath: str, 
        sep: str = ',',
        lowercase_columns: bool = True,
        drop_duplicates: bool = True
    ) -> pd.DataFrame:
        """
        Load and perform initial cleaning on dataframe.
        
        Args:
            filepath: Path to data file
            sep: Delimiter (default: ',')
            lowercase_columns: Whether to lowercase column names
            drop_duplicates: Whether to drop duplicate rows
            
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Loading data from {filepath}")
        
        # Determine file extension
        if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath, sep=sep)
        
        # Lowercase columns
        if lowercase_columns:
            df.columns = df.columns.str.lower().str.strip()
        
        # Replace missing value indicators
        for val in self.replace_values:
            df = df.replace(val, np.nan)
        
        # Drop duplicates
        if drop_duplicates:
            original_len = len(df)
            df = df.drop_duplicates()
            if len(df) < original_len:
                logger.info(f"Dropped {original_len - len(df)} duplicate rows")
        
        logger.info(f"Loaded dataframe with shape {df.shape}")
        return df
    
    def filter_missing_features(
        self, 
        df: pd.DataFrame, 
        threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Remove features with high proportion of missing values.
        
        Args:
            df: Input dataframe
            threshold: Maximum proportion of missing values (0-1)
            
        Returns:
            Filtered dataframe
        """
        missing_prop = df.isnull().sum() / len(df)
        cols_to_keep = missing_prop[missing_prop <= threshold].index.tolist()
        cols_dropped = missing_prop[missing_prop > threshold].index.tolist()
        
        if cols_dropped:
            logger.info(f"Dropped {len(cols_dropped)} columns with >{threshold*100}% missing values")
            logger.debug(f"Dropped columns: {cols_dropped}")
        
        return df[cols_to_keep]
    
    def scale_features(
        self, 
        df: pd.DataFrame,
        method: str = 'standard',
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input dataframe
            method: Scaling method ('standard', 'minmax', 'robust')
            exclude_cols: Columns to exclude from scaling
            
        Returns:
            Scaled dataframe
        """
        exclude_cols = exclude_cols or []
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
        
        if not cols_to_scale:
            logger.warning("No numeric columns to scale")
            return df
        
        # Initialize scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Scale features
        df_scaled = df.copy()
        df_scaled[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        
        logger.info(f"Scaled {len(cols_to_scale)} features using {method} scaling")
        
        return df_scaled
    
    def classify_columns(
        self, 
        df: pd.DataFrame, 
        threshold: int = 10
    ) -> Tuple[List[str], List[str]]:
        """
        Classify columns as numerical or categorical based on unique value count.
        
        Args:
            df: Input dataframe
            threshold: Minimum unique values for numerical classification
            
        Returns:
            Tuple of (numerical_columns, categorical_columns)
        """
        numerical_cols = []
        categorical_cols = []
        
        for column in df.columns:
            unique_count = df[column].nunique()
            if unique_count >= threshold:
                numerical_cols.append(column)
            else:
                categorical_cols.append(column)
        
        logger.info(f"Classified {len(numerical_cols)} numerical and "
                   f"{len(categorical_cols)} categorical columns")
        
        return numerical_cols, categorical_cols
    
    def filter_by_outcome(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        outcome_value: Optional[Union[int, float, str]] = None,
        drop_missing: bool = True
    ) -> pd.DataFrame:
        """
        Filter dataframe based on outcome column.
        
        Args:
            df: Input dataframe
            outcome_col: Name of outcome column
            outcome_value: Specific outcome value to filter (None = all non-missing)
            drop_missing: Whether to drop rows with missing outcome
            
        Returns:
            Filtered dataframe
        """
        df_filtered = df.copy()
        
        if drop_missing:
            df_filtered = df_filtered.dropna(subset=[outcome_col])
            logger.info(f"Dropped {len(df) - len(df_filtered)} rows with missing outcome")
        
        if outcome_value is not None:
            df_filtered = df_filtered[df_filtered[outcome_col] == outcome_value]
            logger.info(f"Filtered to {len(df_filtered)} rows with outcome={outcome_value}")
        
        return df_filtered.reset_index(drop=True)
    
    def select_numeric_features(
        self, 
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Select only numeric features.
        
        Args:
            df: Input dataframe
            exclude_cols: Columns to exclude
            
        Returns:
            DataFrame with only numeric columns
        """
        exclude_cols = exclude_cols or []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_keep = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Selected {len(cols_to_keep)} numeric features")
        
        return df[cols_to_keep]
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with summary statistics
        """
        missing_counts = df.isnull().sum()
        
        summary = {
            'shape': df.shape,
            'n_samples': df.shape[0],
            'n_features': df.shape[1],
            'missing_counts': missing_counts[missing_counts > 0].to_dict(),
            'missing_percent': (missing_counts[missing_counts > 0] / len(df) * 100).to_dict(),
            'dtypes': df.dtypes.value_counts().to_dict(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        
        logger.info(f"Data summary: {summary['n_samples']} samples, "
                   f"{summary['n_features']} features, "
                   f"{summary['memory_usage_mb']} MB")
        
        return summary
    
    def prepare_for_clustering(
        self,
        df: pd.DataFrame,
        id_col: Optional[str] = None,
        outcome_col: Optional[str] = None,
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for clustering by separating features from metadata.
        
        Args:
            df: Input dataframe
            id_col: Sample identifier column
            outcome_col: Outcome/label column
            exclude_cols: Additional columns to exclude
            
        Returns:
            Tuple of (feature_df, metadata_df)
        """
        exclude_cols = exclude_cols or []
        metadata_cols = []
        
        if id_col:
            metadata_cols.append(id_col)
        if outcome_col:
            metadata_cols.append(outcome_col)
        
        metadata_cols.extend(exclude_cols)
        
        # Get metadata
        existing_metadata = [col for col in metadata_cols if col in df.columns]
        metadata_df = df[existing_metadata].copy() if existing_metadata else pd.DataFrame()
        
        # Get features
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        feature_df = df[feature_cols].copy()
        
        logger.info(f"Prepared {len(feature_cols)} features and "
                   f"{len(existing_metadata)} metadata columns")
        
        return feature_df, metadata_df


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config