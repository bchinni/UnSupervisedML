# Unsupervised ML: UMAP + GMM Clustering Pipeline

A modular Python pipeline for unsupervised clustering analysis combining UMAP dimensionality reduction with Gaussian Mixture Model (GMM) clustering, optimized using silhouette scores.

## Project Overview

This pipeline implements:
- **UMAP dimensionality reduction** with hyperparameter optimization
- **Gaussian Mixture Model clustering** with silhouette score optimization
- **3D visualization** of UMAP embeddings and cluster assignments

## Project Structure

```
UnsupervisedML/
├── data/                       # Place your datasets here (not tracked)
│   ├── raw/                   # Raw input files
│   └── processed/             # Intermediate outputs
├── notebooks/                  # Jupyter notebooks for exploration
│   └── demo_clustering.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Data cleaning, encoding, scaling
│   ├── imputation.py           # Iterative imputation utilities
│   ├── dimensionality_reduction.py  # UMAP optimization
│   ├── clustering.py           # GMM clustering with silhouette optimization
│   ├── visualization.py        # 3D plots, heatmaps
│   └── utils.py                # Helper functions (I/O, metrics)
├── tests/                      # pytest unit tests
│   ├── test_preprocessing.py
│   ├── test_clustering.py
│   └── test_dimensionality_reduction.py
├── results/                    # Output directory (gitignored)
│   ├── figures/
│   ├── models/
│   └── metrics/
├── config/
│   └── config.yaml            # Configuration parameters
├── requirements.txt
├── environment.yml            # Conda environment
├── main.py                    # Main pipeline orchestrator
├── README.md
└── .gitignore
```

## Installation

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/UnsupervisedML.git
cd UnsupervisedML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda

```bash
# Create environment from file
conda env create -f environment.yml
conda activate unsupervised-ml
```

## Quick Start

### 1. Prepare Your Data

Place your dataset in the `data/raw/` directory:
- CSV or TSV file with numeric features
- Each row is a sample, each column is a feature
- Optional: outcome/label column for validation

### 2. Configure Parameters

Edit `config/config.yaml` to set your parameters:

```yaml
data:
  input_file: 'data/raw/your_data.csv'
  outcome_col: null  # Optional outcome column name
  id_col: 'sample_id'  # Sample identifier column

preprocessing:
  missing_threshold: 0.3  # Drop features with >30% missing
  scaling: 'standard'  # 'standard', 'minmax', or 'robust'
  imputation_method: 'iterative'  # 'iterative', 'mean', 'median'

umap:
  n_components: 3
  metric: 'euclidean'  # 'euclidean', 'manhattan', 'cosine', 'braycurtis', etc.
  n_neighbors: [15, 20, 25, 30, 35, 40, 45, 50]
  learning_rate: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
  spread: [0.5, 1.0, 1.5, 2.0]
  min_dist: 0.0
  init: 'spectral'
  random_state: 42

gmm:
  n_components_range: [2, 3, 4, 5, 6, 7, 8, 9, 10]
  covariance_types: ['full', 'tied', 'diag', 'spherical']
  init_params: 'kmeans'
  random_state: 42
```

### 3. Run the Pipeline

```bash
# Full pipeline
python main.py --config config/config.yaml

# Individual steps
python main.py --step preprocessing
python main.py --step umap_optimization
python main.py --step clustering
```

## Pipeline Steps

### Step 1: Data Preprocessing

```python
from src.preprocessing import DataPreprocessor
from src.imputation import DataImputer

# Load and clean data
preprocessor = DataPreprocessor()
df_clean = preprocessor.load_and_clean('data/raw/your_data.csv')

# Handle missing values
imputer = DataImputer(method='iterative')
df_imputed = imputer.fit_transform(df_clean)

# Scale features
df_scaled = preprocessor.scale_features(df_imputed, method='standard')
```

### Step 2: UMAP Optimization

```python
from src.dimensionality_reduction import UMAPOptimizer

# Initialize optimizer
optimizer = UMAPOptimizer(n_components=3, metric='euclidean')

# Grid search for best parameters
best_params, results_df = optimizer.optimize_parameters(
    X_scaled, 
    n_neighbors_range=[20, 30, 40, 50],
    learning_rate_range=[0.1, 0.5, 0.9],
    spread_range=[0.5, 1.0, 1.5]
)

# Transform with best parameters
X_umap = optimizer.transform(X_scaled, best_params)
```

### Step 3: GMM Clustering

```python
from src.clustering import GMMClusterer

# Initialize clusterer
clusterer = GMMClusterer()

# Optimize number of clusters and covariance type
best_model, labels, scores = clusterer.optimize_clustering(
    X_umap,
    n_components_range=range(2, 11),
    covariance_types=['full', 'tied', 'diag', 'spherical']
)

# Get cluster probabilities
cluster_probs = clusterer.get_cluster_probabilities(X_umap)
```

### Step 4: Visualization

```python
from src.visualization import ClusterVisualizer

# Initialize visualizer
visualizer = ClusterVisualizer()

# 3D scatter plot
visualizer.plot_3d_clusters(X_umap, labels, 
                            save_path='results/figures/clusters_3d.png')

# Silhouette score heatmap
visualizer.plot_silhouette_heatmap(scores, 
                                   save_path='results/figures/silhouette_heatmap.png')
```

## Output Files

After running the pipeline, you'll find:

- `results/umap/optimization_results.xlsx` - UMAP hyperparameter search results
- `results/umap/best_umap_embedding.csv` - Final UMAP coordinates
- `results/clustering/cluster_labels.csv` - Final cluster assignments and probabilities
- `results/clustering/silhouette_scores.xlsx` - GMM optimization metrics
- `results/figures/umap_3d_clusters.png` - 3D visualization of clusters
- `results/figures/silhouette_heatmap.png` - GMM performance heatmap
- `results/models/best_umap_model.pkl` - Saved UMAP model
- `results/models/best_gmm_model.pkl` - Saved GMM model

## Key Features

### 1. UMAP Hyperparameter Optimization
- Grid search over multiple parameters
- Custom metrics: within/between cluster sum of squared distances
- Reproducible with fixed random seeds
- Support for multiple distance metrics

### 2. GMM Clustering with Silhouette Optimization
- Automatic selection of optimal cluster number
- Multiple covariance structures tested
- K-means initialization for stability
- Cluster probability outputs

### 3. Comprehensive Visualization
- Interactive 3D scatter plots
- Silhouette score heatmaps
- UMAP parameter comparison plots
- Cluster distribution analysis

## Configuration Options

### UMAP Parameters

- `n_components`: Embedding dimensions (default: 3 for visualization)
- `metric`: Distance metric
  - 'euclidean': Standard Euclidean distance
  - 'manhattan': City-block distance
  - 'cosine': Cosine similarity
  - 'braycurtis': Bray-Curtis dissimilarity
  - Many others supported
- `n_neighbors`: Local neighborhood size (affects local vs global structure)
- `learning_rate`: Optimization learning rate
- `spread`: Scale of embedded points
- `min_dist`: Minimum distance between points in embedding

### GMM Parameters

- `n_components`: Number of clusters to test
- `covariance_type`: Covariance structure
  - 'full': Each component has its own general covariance matrix
  - 'tied': All components share the same general covariance matrix
  - 'diag': Each component has its own diagonal covariance matrix
  - 'spherical': Each component has its own single variance
- `init_params`: Initialization method ('kmeans' or 'random')

### Preprocessing Options

- `scaling`: Feature scaling method
  - 'standard': Zero mean, unit variance
  - 'minmax': Scale to [0, 1]
  - 'robust': Robust to outliers
- `imputation_method`: Missing value handling
  - 'iterative': Multivariate imputation
  - 'mean': Mean imputation
  - 'median': Median imputation
  - 'knn': K-nearest neighbors imputation

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_clustering.py

# Run with coverage
pytest --cov=src tests/
```

## Advanced Usage

### Custom Distance Metrics

```python
from src.dimensionality_reduction import UMAPOptimizer

# Use custom metric
optimizer = UMAPOptimizer(n_components=3, metric='correlation')
X_umap = optimizer.fit_transform(X_scaled)
```

### Export Results

```python
from src.utils import ResultsExporter

exporter = ResultsExporter(output_dir='results/')
exporter.save_clustering_results(
    labels=labels,
    probabilities=cluster_probs,
    sample_ids=sample_ids
)
```

### Load Pre-trained Models

```python
from src.utils import load_model

# Load saved models
umap_model = load_model('results/models/best_umap_model.pkl')
gmm_model = load_model('results/models/best_gmm_model.pkl')

# Transform new data
X_new_umap = umap_model.transform(X_new_scaled)
new_labels = gmm_model.predict(X_new_umap)
```

## Example Workflow

```python
# Complete example
from src.preprocessing import DataPreprocessor
from src.imputation import DataImputer
from src.dimensionality_reduction import UMAPOptimizer
from src.clustering import GMMClusterer
from src.visualization import ClusterVisualizer

# 1. Load and preprocess
preprocessor = DataPreprocessor()
df = preprocessor.load_and_clean('data/raw/data.csv')

imputer = DataImputer()
df_imputed = imputer.fit_transform(df)

X_scaled = preprocessor.scale_features(df_imputed)

# 2. UMAP optimization
umap_opt = UMAPOptimizer()
best_params, _ = umap_opt.optimize_parameters(X_scaled)
X_umap = umap_opt.transform(X_scaled, best_params)

# 3. GMM clustering
clusterer = GMMClusterer()
model, labels, scores = clusterer.optimize_clustering(X_umap)

# 4. Visualize
viz = ClusterVisualizer()
viz.plot_3d_clusters(X_umap, labels)
viz.plot_silhouette_heatmap(scores)


## Troubleshooting

### Common Issues

**Issue**: UMAP optimization taking too long  
**Solution**: Reduce the parameter grid size or use fewer parameter combinations

**Issue**: GMM convergence warnings  
**Solution**: Try different `covariance_type` or ensure data is properly scaled

**Issue**: Memory errors with large datasets  
**Solution**: Reduce `n_neighbors` or process data in batches

**Issue**: Poor cluster separation  
**Solution**: Try different UMAP metrics or adjust `n_neighbors` and `min_dist`

## Acknowledgments

- UMAP: McInnes, L., Healy, J., & Melville, J. (2018)
- Scikit-learn contributors
- Gaussian Mixture Models: Pedregosa et al. (2011)