"""
Visualization utilities for clustering analysis.

Supports:
- 3D scatter plots of UMAP embeddings
- Silhouette score heatmaps
- Cluster distribution plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ClusterVisualizer:
    """Handles visualization of clustering results."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not available, using default")
        
    def plot_3d_clusters(
        self,
        X_umap: np.ndarray,
        labels: np.ndarray,
        title: str = "3D UMAP Clustering Visualization",
        save_path: Optional[str] = None,
        alpha: float = 0.6,
        s: int = 20
    ):
        """
        Create 3D scatter plot of UMAP embedding with cluster colors.
        
        Args:
            X_umap: UMAP-transformed data (n_samples, 3)
            labels: Cluster labels
            title: Plot title
            save_path: Path to save figure
            alpha: Point transparency
            s: Point size
        """
        if X_umap.shape[1] != 3:
            raise ValueError(f"Expected 3D data, got shape {X_umap.shape}")
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter plot
        scatter = ax.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
            X_umap[:, 2],
            c=labels,
            cmap='viridis',
            s=s,
            alpha=alpha,
            edgecolors='none'
        )
        
        # Labels and title
        ax.set_xlabel("UMAP1", fontsize=12)
        ax.set_ylabel("UMAP2", fontsize=12)
        ax.set_zlabel("UMAP3", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Legend
        legend = ax.legend(*scatter.legend_elements(), 
                          loc="upper right", 
                          title="Cluster",
                          fontsize=10)
        plt.setp(legend.get_title(), fontsize=11)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 3D plot to {save_path}")
        
        plt.show()
        
    def plot_2d_clusters(
        self,
        X_umap: np.ndarray,
        labels: np.ndarray,
        dims: Tuple[int, int] = (0, 1),
        title: str = "2D UMAP Clustering Visualization",
        save_path: Optional[str] = None,
        alpha: float = 0.6,
        s: int = 30
    ):
        """
        Create 2D scatter plot of UMAP embedding.
        
        Args:
            X_umap: UMAP-transformed data
            labels: Cluster labels
            dims: Dimensions to plot (default: first two)
            title: Plot title
            save_path: Path to save figure
            alpha: Point transparency
            s: Point size
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        scatter = ax.scatter(
            X_umap[:, dims[0]],
            X_umap[:, dims[1]],
            c=labels,
            cmap='viridis',
            s=s,
            alpha=alpha,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_xlabel(f"UMAP{dims[0]+1}", fontsize=12)
        ax.set_ylabel(f"UMAP{dims[1]+1}", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 2D plot to {save_path}")
        
        plt.show()
        
    def plot_silhouette_heatmap(
        self,
        scores_df: pd.DataFrame,
        title: str = "Silhouette Scores for Different GMM Configurations",
        save_path: Optional[str] = None,
        cmap: str = 'viridis',
        annot: bool = True,
        fmt: str = '.3f'
    ):
        """
        Create heatmap of silhouette scores.
        
        Args:
            scores_df: DataFrame with columns [n_components, covariance_type, silhouette_score]
            title: Plot title
            save_path: Path to save figure
            cmap: Color map
            annot: Whether to annotate cells
            fmt: Format for annotations
        """
        # Create pivot table
        pivot = scores_df.pivot_table(
            index='n_components',
            columns='covariance_type',
            values='silhouette_score'
        )
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(
            pivot,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            ax=ax,
            cbar_kws={'label': 'Silhouette Score'},
            linewidths=0.5
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Covariance Type", fontsize=12)
        ax.set_ylabel("Number of Components (Clusters)", fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved heatmap to {save_path}")
        
        plt.show()
        
    def plot_cluster_distribution(
        self,
        labels: np.ndarray,
        title: str = "Cluster Size Distribution",
        save_path: Optional[str] = None
    ):
        """
        Create bar plot of cluster sizes.
        
        Args:
            labels: Cluster labels
            title: Plot title
            save_path: Path to save figure
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bars = ax.bar(unique, counts, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel("Cluster", fontsize=12)
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(unique)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved distribution plot to {save_path}")
        
        plt.show()
        
    def plot_umap_optimization_results(
        self,
        results_df: pd.DataFrame,
        metric: str = 'score',
        title: str = "UMAP Parameter Optimization Results",
        save_path: Optional[str] = None
    ):
        """
        Visualize UMAP optimization results.
        
        Args:
            results_df: DataFrame with optimization results
            metric: Metric to plot
            title: Plot title
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: n_neighbors vs score
        for spread in results_df['spread'].unique():
            data = results_df[results_df['spread'] == spread]
            axes[0].plot(data['n_neighbors'], data[metric], 
                        marker='o', label=f'spread={spread}')
        axes[0].set_xlabel('n_neighbors', fontsize=11)
        axes[0].set_ylabel(metric, fontsize=11)
        axes[0].set_title('Effect of n_neighbors', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: learning_rate vs score
        for spread in results_df['spread'].unique():
            data = results_df[results_df['spread'] == spread]
            axes[1].plot(data['learning_rate'], data[metric], 
                        marker='o', label=f'spread={spread}')
        axes[1].set_xlabel('learning_rate', fontsize=11)
        axes[1].set_ylabel(metric, fontsize=11)
        axes[1].set_title('Effect of learning_rate', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: spread vs score
        for n_neighbors in sorted(results_df['n_neighbors'].unique())[::2]:
            data = results_df[results_df['n_neighbors'] == n_neighbors]
            axes[2].plot(data['spread'], data[metric], 
                        marker='o', label=f'n_neighbors={n_neighbors}')
        axes[2].set_xlabel('spread', fontsize=11)
        axes[2].set_ylabel(metric, fontsize=11)
        axes[2].set_title('Effect of spread', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved optimization plot to {save_path}")
        
        plt.show()
        
    def plot_cluster_probabilities(
        self,
        probs: np.ndarray,
        sample_indices: Optional[np.ndarray] = None,
        title: str = "Cluster Membership Probabilities",
        save_path: Optional[str] = None
    ):
        """
        Visualize cluster membership probabilities.
        
        Args:
            probs: Probability matrix (n_samples, n_clusters)
            sample_indices: Indices to plot (default: first 50)
            title: Plot title
            save_path: Path to save figure
        """
        if sample_indices is None:
            sample_indices = np.arange(min(50, len(probs)))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(probs[sample_indices].T, aspect='auto', cmap='viridis')
        
        ax.set_xlabel("Sample Index", fontsize=12)
        ax.set_ylabel("Cluster", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_yticks(range(probs.shape[1]))
        
        plt.colorbar(im, ax=ax, label='Probability')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved probability plot to {save_path}")
        
        plt.show()


def save_all_plots(
    X_umap: np.ndarray,
    labels: np.ndarray,
    scores_df: pd.DataFrame,
    output_dir: str = 'results/figures/'
):
    """
    Save all standard plots.
    
    Args:
        X_umap: UMAP embedding
        labels: Cluster labels
        scores_df: Clustering scores
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    viz = ClusterVisualizer()
    
    # 3D clusters
    if X_umap.shape[1] >= 3:
        viz.plot_3d_clusters(X_umap, labels, 
                            save_path=f'{output_dir}/clusters_3d.png')
    
    # 2D clusters
    viz.plot_2d_clusters(X_umap, labels,
                        save_path=f'{output_dir}/clusters_2d.png')
    
    # Silhouette heatmap
    viz.plot_silhouette_heatmap(scores_df,
                               save_path=f'{output_dir}/silhouette_heatmap.png')
    
    # Cluster distribution
    viz.plot_cluster_distribution(labels,
                                 save_path=f'{output_dir}/cluster_distribution.png')
    
    logger.info(f"Saved all plots to {output_dir}")