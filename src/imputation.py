"""
Missing data imputation utilities.

Supports:
- Iterative imputation (MICE)
- Simple imputation (mean, median)
- K-nearest neighbors imputation
"""

import pandas as pd
import numpy as np
from typing import Optional, Union
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.tree import DecisionTreeRegressor
import logging

logger = logging.getLogger(__name__)


class DataImputer:
    """Handles missing data imputation."""
    
    def __init__(
        self,
        method: str = 'iterative',
        max_iter: int = 10,
        random_state: int = 42,
        n_neighbors: int = 5
    ):
        """
        Initialize imputer.
        
        Args:
            method: Imputation method ('iterative', 'mean', 'median', 'knn')
            max_iter: Maximum iterations for iterative imputation
            random_state: Random seed
            n_neighbors: Number of neighbors for KNN imputation
        """
        self.method = method
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.imputer = None
        self._initialize_imputer()
        
    def _initialize_imputer(self):
        """Initialize the appropriate imputer based on method."""
        if self.method == 'iterative':
            self.imputer = IterativeImputer(
                estimator=DecisionTreeRegressor(
                    max_features="sqrt",
                    random_state=self.random_state
                ),
                missing_values=np.nan,
                max_iter=self.max_iter,
                verbose=1,
                random_state=self.random_state
            )
            logger.info("Initialized iterative imputer with DecisionTreeRegressor")
            
        elif self.method == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
            logger.info("Initialized mean imputer")
            
        elif self.method == 'median':
            self.imputer = SimpleImputer(strategy='median')
            logger.info("Initialized median imputer")
            
        elif self.method == 'knn':
            self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
            logger.info(f"Initialized KNN imputer with {self.n_neighbors} neighbors")
            
        else:
            raise ValueError(f"Unknown imputation method: {self.method}")
    
    def fit_transform(
        self, 
        df: pd.DataFrame,
        return_dataframe: bool = True
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit imputer and transform data.
        
        Args:
            df: Input dataframe with missing values
            return_dataframe: Whether to return DataFrame (True) or array (False)
            
        Returns:
            Imputed data
        """
        logger.info(f"Starting {self.method} imputation...")
        
        # Check for missing values
        n_missing = df.isnull().sum().sum()
        if n_missing == 0:
            logger.info("No missing values found, returning original data")
            return df if return_dataframe else df.values
        
        logger.info(f"Found {n_missing} missing values across {df.isnull().any().sum()} features")
        
        # Store column names and index
        columns = df.columns
        index = df.index
        
        # Perform imputation
        imputed_data = self.imputer.fit_transform(df)
        
        logger.info("Imputation completed successfully")
        
        if return_dataframe:
            return pd.DataFrame(imputed_data, columns=columns, index=index)
        else:
            return imputed_data
    
    def transform(
        self, 
        df: pd.DataFrame,
        return_dataframe: bool = True
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform new data using fitted imputer.
        
        Args:
            df: Input dataframe
            return_dataframe: Whether to return DataFrame (True) or array (False)
            
        Returns:
            Imputed data
        """
        if self.imputer is None:
            raise ValueError("Imputer has not been fitted. Call fit_transform first.")
        
        columns = df.columns
        index = df.index
        
        imputed_data = self.imputer.transform(df)
        
        if return_dataframe:
            return pd.DataFrame(imputed_data, columns=columns, index=index)
        else:
            return imputed_data
    
    def get_imputation_stats(self, df_original: pd.DataFrame, df_imputed: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics about imputed values.
        
        Args:
            df_original: Original dataframe with missing values
            df_imputed: Imputed dataframe
            
        Returns:
            DataFrame with imputation statistics per feature
        """
        stats = []
        
        for col in df_original.columns:
            n_missing = df_original[col].isnull().sum()
            if n_missing > 0:
                original_mean = df_original[col].mean()
                imputed_mean = df_imputed[col].mean()
                original_std = df_original[col].std()
                imputed_std = df_imputed[col].std()
                
                # Get imputed values
                mask = df_original[col].isnull()
                imputed_values = df_imputed.loc[mask, col]
                
                stats.append({
                    'feature': col,
                    'n_missing': n_missing,
                    'pct_missing': round(n_missing / len(df_original) * 100, 2),
                    'original_mean': round(original_mean, 4),
                    'imputed_mean': round(imputed_mean, 4),
                    'original_std': round(original_std, 4),
                    'imputed_std': round(imputed_std, 4),
                    'imputed_values_mean': round(imputed_values.mean(), 4),
                    'imputed_values_std': round(imputed_values.std(), 4)
                })
        
        return pd.DataFrame(stats)


def check_missing_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing data patterns.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with missing data analysis
    """
    missing_info = []
    
    for col in df.columns:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            missing_info.append({
                'feature': col,
                'n_missing': n_missing,
                'pct_missing': round(n_missing / len(df) * 100, 2),
                'dtype': str(df[col].dtype)
            })
    
    missing_df = pd.DataFrame(missing_info)
    
    if not missing_df.empty:
        missing_df = missing_df.sort_values('pct_missing', ascending=False)
        logger.info(f"Found missing values in {len(missing_df)} features")
    else:
        logger.info("No missing values found")
    
    return missing_df


def impute_by_group(
    df: pd.DataFrame,
    group_col: str,
    method: str = 'mean'
) -> pd.DataFrame:
    """
    Perform group-wise imputation.
    
    Args:
        df: Input dataframe
        group_col: Column to group by
        method: Imputation method ('mean' or 'median')
        
    Returns:
        Imputed dataframe
    """
    df_imputed = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if group_col in numeric_cols:
        numeric_cols.remove(group_col)
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if method == 'mean':
                df_imputed[col] = df.groupby(group_col)[col].transform(
                    lambda x: x.fillna(x.mean())
                )
            elif method == 'median':
                df_imputed[col] = df.groupby(group_col)[col].transform(
                    lambda x: x.fillna(x.median())
                )
            else:
                raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Performed group-wise {method} imputation by '{group_col}'")
    
    return df_imputed