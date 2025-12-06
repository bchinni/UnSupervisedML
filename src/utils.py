"""
Utility functions for the clustering pipeline.

Includes:
- File I/O operations
- Model saving/loading
- Results export
- Logging configuration
"""

import pandas as pd
import numpy as np
import pickle
import yaml
import json
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure logging for the pipeline.
    
    Args:
        log_file: Path to log file (None for console only)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict, output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logging.info(f"Saved configuration to {output_path}")


def save_model(model: Any, filepath: str):
    """
    Save model to pickle file.
    
    Args:
        model: Model object
        filepath: Path to save model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logging.info(f"Saved model to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load model from pickle file.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    logging.info(f"Loaded model from {filepath}")
    return model


class ResultsExporter:
    """Handles exporting of clustering results."""
    
    def __init__(self, output_dir: str = 'results/'):
        """
        Initialize results exporter.
        
        Args:
            output_dir: Base output directory
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def save_clustering_results(
        self,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        sample_ids: Optional[np.ndarray] = None,
        filename: str = 'cluster_labels.csv'
    ):
        """
        Save clustering results to CSV.
        
        Args:
            labels: Cluster labels
            probabilities: Cluster probabilities
            sample_ids: Sample identifiers
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, 'clustering', filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create dataframe
        if sample_ids is not None:
            df = pd.DataFrame({'sample_id': sample_ids, 'cluster': labels})
        else:
            df = pd.DataFrame({'cluster': labels})
        
        # Add probabilities if provided
        if probabilities is not None:
            n_clusters = probabilities.shape[1]
            for i in range(n_clusters):
                df[f'prob_cluster_{i}'] = probabilities[:, i]
        
        df.to_csv(output_path, index=False)
        logging.info(f"Saved clustering results to {output_path}")
        
    def save_umap_embedding(
        self,
        X_umap: np.ndarray,
        sample_ids: Optional[np.ndarray] = None,
        filename: str = 'umap_embedding.csv'
    ):
        """
        Save UMAP embedding to CSV.
        
        Args:
            X_umap: UMAP coordinates
            sample_ids: Sample identifiers
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, 'umap', filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create column names
        n_dims = X_umap.shape[1]
        columns = [f'UMAP{i+1}' for i in range(n_dims)]
        
        df = pd.DataFrame(X_umap, columns=columns)
        
        if sample_ids is not None:
            df.insert(0, 'sample_id', sample_ids)
        
        df.to_csv(output_path, index=False)
        logging.info(f"Saved UMAP embedding to {output_path}")
        
    def save_optimization_results(
        self,
        results_df: pd.DataFrame,
        stage: str,
        filename: Optional[str] = None
    ):
        """
        Save optimization results to Excel.
        
        Args:
            results_df: Results dataframe
            stage: Pipeline stage ('umap' or 'clustering')
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            filename = f'{stage}_optimization_results.xlsx'
        
        output_path = os.path.join(self.output_dir, stage, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results_df.to_excel(output_path, index=False)
        logging.info(f"Saved {stage} optimization results to {output_path}")
        
    def save_summary_report(
        self,
        config: Dict,
        umap_params: Dict,
        cluster_params: Dict,
        n_samples: int,
        n_features: int,
        cluster_sizes: Dict,
        filename: str = 'summary_report.json'
    ):
        """
        Save summary report of pipeline run.
        
        Args:
            config: Pipeline configuration
            umap_params: Best UMAP parameters
            cluster_params: Best clustering parameters
            n_samples: Number of samples
            n_features: Number of features
            cluster_sizes: Cluster size distribution
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = {
            'timestamp': self.timestamp,
            'data_info': {
                'n_samples': n_samples,
                'n_features': n_features
            },
            'config': config,
            'best_parameters': {
                'umap': umap_params,
                'clustering': cluster_params
            },
            'results': {
                'cluster_sizes': cluster_sizes
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logging.info(f"Saved summary report to {output_path}")


def create_directory_structure(base_dir: str = 'results/'):
    """
    Create standard directory structure for results.
    
    Args:
        base_dir: Base directory
    """
    dirs = [
        os.path.join(base_dir, 'figures'),
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, 'umap'),
        os.path.join(base_dir, 'clustering'),
        os.path.join(base_dir, 'logs'),
        'data/raw',
        'data/processed'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    logging.info(f"Created directory structure under {base_dir}")


def validate_config(config: Dict) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ['data', 'preprocessing', 'umap', 'gmm']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    # Validate data section
    if 'input_file' not in config['data']:
        raise ValueError("Missing 'input_file' in data config")
    
    # Validate preprocessing
    valid_scaling = ['standard', 'minmax', 'robust']
    if config['preprocessing'].get('scaling') not in valid_scaling:
        raise ValueError(f"Invalid scaling method. Must be one of {valid_scaling}")
    
    # Validate UMAP
    if config['umap']['n_components'] < 2:
        raise ValueError("UMAP n_components must be >= 2")
    
    # Validate GMM
    if not config['gmm']['n_components_range']:
        raise ValueError("GMM n_components_range cannot be empty")
    
    logging.info("Configuration validated successfully")
    return True


def get_timestamp() -> str:
    """
    Get formatted timestamp string.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"