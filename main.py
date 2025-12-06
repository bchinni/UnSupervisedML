"""
Main pipeline orchestrator for UMAP + GMM clustering analysis.

Usage:
    python main.py --config config/config.yaml
    python main.py --config config/config.yaml --step preprocessing
    python main.py --config config/config.yaml --step umap_optimization
    python main.py --config config/config.yaml --step clustering
"""

import argparse
import time
import numpy as np
from pathlib import Path

from src.preprocessing import DataPreprocessor
from src.imputation import DataImputer, check_missing_patterns
from src.dimensionality_reduction import UMAPOptimizer
from src.clustering import GMMClusterer
from src.visualization import ClusterVisualizer, save_all_plots
from src.utils import (
    setup_logging, load_config, save_model, ResultsExporter,
    create_directory_structure, validate_config, format_duration
)


def run_preprocessing(config: dict, logger):
    """Run data preprocessing step."""
    logger.info("="*80)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("="*80)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_and_clean(
        config['data']['input_file'],
        sep=config['data'].get('separator', ',')
    )
    
    # Print data summary
    summary = preprocessor.get_data_summary(df)
    logger.info(f"Data summary: {summary}")
    
    # Prepare for clustering
    feature_df, metadata_df = preprocessor.prepare_for_clustering(
        df,
        id_col=config['data'].get('id_col'),
        outcome_col=config['data'].get('outcome_col'),
        exclude_cols=config['preprocessing'].get('exclude_cols', [])
    )
    
    # Filter features with too many missing values
    if config['preprocessing'].get('missing_threshold'):
        feature_df = preprocessor.filter_missing_features(
            feature_df,
            threshold=config['preprocessing']['missing_threshold']
        )
    
    # Check missing patterns
    missing_info = check_missing_patterns(feature_df)
    if not missing_info.empty:
        logger.info(f"\nMissing data patterns:\n{missing_info.head(10)}")
    
    # Imputation
    if feature_df.isnull().sum().sum() > 0:
        logger.info("\nPerforming imputation...")
        imputer = DataImputer(
            method=config['preprocessing']['imputation_method'],
            random_state=config.get('random_state', 42)
        )
        feature_df = imputer.fit_transform(feature_df)
        
        # Save imputer
        save_model(imputer, 'results/models/imputer.pkl')
    
    # Select numeric features
    feature_df = preprocessor.select_numeric_features(feature_df)
    
    # Scale features
    feature_df_scaled = preprocessor.scale_features(
        feature_df,
        method=config['preprocessing']['scaling']
    )
    
    # Save scaler
    save_model(preprocessor.scaler, 'results/models/scaler.pkl')
    
    # Save processed data
    feature_df_scaled.to_csv('data/processed/features_scaled.csv', index=False)
    if not metadata_df.empty:
        metadata_df.to_csv('data/processed/metadata.csv', index=False)
    
    logger.info(f"\nPreprocessing complete. Final shape: {feature_df_scaled.shape}")
    
    return feature_df_scaled, metadata_df


def run_umap_optimization(config: dict, X_scaled, logger):
    """Run UMAP optimization step."""
    logger.info("="*80)
    logger.info("STEP 2: UMAP OPTIMIZATION")
    logger.info("="*80)
    
    # Initialize optimizer
    optimizer = UMAPOptimizer(
        n_components=config['umap']['n_components'],
        metric=config['umap']['metric'],
        init=config['umap'].get('init', 'spectral'),
        min_dist=config['umap'].get('min_dist', 0.0),
        random_state=config.get('random_state', 42)
    )
    
    # Optimize parameters
    logger.info("Starting UMAP parameter optimization...")
    best_params, results_df = optimizer.optimize_parameters(
        X_scaled.values,
        n_neighbors_range=config['umap']['n_neighbors'],
        learning_rate_range=config['umap']['learning_rate'],
        spread_range=config['umap']['spread']
    )
    
    logger.info(f"\nBest UMAP parameters: {best_params}")
    
    # Transform with best parameters
    X_umap = optimizer.transform(X_scaled.values, best_params)
    
    # Save results
    exporter = ResultsExporter()
    exporter.save_umap_embedding(X_umap, filename='best_umap_embedding.csv')
    exporter.save_optimization_results(results_df, 'umap')
    save_model(optimizer.reducer, 'results/models/umap_model.pkl')
    
    logger.info(f"UMAP transformation complete. Output shape: {X_umap.shape}")
    
    return X_umap, best_params


def run_clustering(config: dict, X_umap, logger):
    """Run GMM clustering step."""
    logger.info("="*80)
    logger.info("STEP 3: GMM CLUSTERING")
    logger.info("="*80)
    
    # Initialize clusterer
    clusterer = GMMClusterer(
        init_params=config['gmm'].get('init_params', 'kmeans'),
        random_state=config.get('random_state', 42)
    )
    
    # Optimize clustering
    logger.info("Starting GMM clustering optimization...")
    best_model, labels, scores_df = clusterer.optimize_clustering(
        X_umap,
        n_components_range=config['gmm']['n_components_range'],
        covariance_types=config['gmm']['covariance_types']
    )
    
    logger.info(f"\nBest clustering parameters: {clusterer.best_params}")
    
    # Get cluster probabilities
    probs = clusterer.get_cluster_probabilities(X_umap)
    
    # Calculate cluster statistics
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip([int(x) for x in unique], [int(x) for x in counts]))
    logger.info(f"\nCluster sizes: {cluster_sizes}")
    
    # Save results
    exporter = ResultsExporter()
    exporter.save_clustering_results(labels, probs, filename='cluster_labels.csv')
    exporter.save_optimization_results(scores_df, 'clustering')
    save_model(best_model, 'results/models/gmm_model.pkl')
    
    logger.info("Clustering complete.")
    
    return labels, probs, scores_df, clusterer.best_params, cluster_sizes


def run_visualization(X_umap, labels, scores_df, logger):
    """Run visualization step."""
    logger.info("="*80)
    logger.info("STEP 4: VISUALIZATION")
    logger.info("="*80)
    
    save_all_plots(X_umap, labels, scores_df)
    
    logger.info("Visualization complete. Plots saved to results/figures/")


def run_full_pipeline(config_path: str):
    """Run the complete clustering pipeline."""
    
    # Setup
    create_directory_structure()
    logger = setup_logging('results/logs/pipeline.log')
    
    logger.info("="*80)
    logger.info("UNSUPERVISED ML CLUSTERING PIPELINE")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Load and validate config
    config = load_config(config_path)
    validate_config(config)
    
    try:
        # Step 1: Preprocessing
        X_scaled, metadata = run_preprocessing(config, logger)
        
        # Step 2: UMAP
        X_umap, umap_params = run_umap_optimization(config, X_scaled, logger)
        
        # Step 3: Clustering
        labels, probs, scores_df, cluster_params, cluster_sizes = run_clustering(
            config, X_umap, logger
        )
        
        # Step 4: Visualization
        run_visualization(X_umap, labels, scores_df, logger)
        
        # Save summary report
        exporter = ResultsExporter()
        exporter.save_summary_report(
            config=config,
            umap_params=umap_params,
            cluster_params=cluster_params,
            n_samples=len(X_scaled),
            n_features=X_scaled.shape[1],
            cluster_sizes=cluster_sizes
        )
        
        # Final summary
        duration = time.time() - start_time
        logger.info("="*80)
        logger.info(f"PIPELINE COMPLETE - Duration: {format_duration(duration)}")
        logger.info("="*80)
        logger.info(f"Results saved to: results/")
        logger.info(f"- UMAP embedding: results/umap/best_umap_embedding.csv")
        logger.info(f"- Cluster labels: results/clustering/cluster_labels.csv")
        logger.info(f"- Figures: results/figures/")
        logger.info(f"- Models: results/models/")
        logger.info(f"- Summary: results/summary_report.json")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


def run_single_step(config_path: str, step: str):
    """Run a single pipeline step."""
    
    logger = setup_logging('results/logs/pipeline.log')
    config = load_config(config_path)
    validate_config(config)
    
    if step == 'preprocessing':
        X_scaled, metadata = run_preprocessing(config, logger)
        
    elif step == 'umap_optimization':
        # Load preprocessed data
        X_scaled = pd.read_csv('data/processed/features_scaled.csv')
        X_umap, umap_params = run_umap_optimization(config, X_scaled, logger)
        
    elif step == 'clustering':
        # Load UMAP embedding
        import pandas as pd
        umap_df = pd.read_csv('results/umap/best_umap_embedding.csv')
        X_umap = umap_df.values
        labels, probs, scores_df, cluster_params, cluster_sizes = run_clustering(
            config, X_umap, logger
        )
        
    elif step == 'visualization':
        # Load results
        import pandas as pd
        umap_df = pd.read_csv('results/umap/best_umap_embedding.csv')
        labels_df = pd.read_csv('results/clustering/cluster_labels.csv')
        scores_df = pd.read_excel('results/clustering/clustering_optimization_results.xlsx')
        
        X_umap = umap_df.values
        labels = labels_df['cluster'].values
        
        run_visualization(X_umap, labels, scores_df, logger)
        
    else:
        raise ValueError(f"Unknown step: {step}. Must be one of: "
                        "preprocessing, umap_optimization, clustering, visualization")


def main():
    parser = argparse.ArgumentParser(
        description='UMAP + GMM Clustering Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--step',
        type=str,
        default=None,
        choices=['preprocessing', 'umap_optimization', 'clustering', 'visualization'],
        help='Run only a specific step (default: run full pipeline)'
    )
    
    args = parser.parse_args()
    
    if args.step:
        run_single_step(args.config, args.step)
    else:
        run_full_pipeline(args.config)


if __name__ == '__main__':
    main()