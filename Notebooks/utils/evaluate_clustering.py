from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap.umap_ as umap
import pandas as pd

import warnings
import logging
from sklearn.exceptions import ConvergenceWarning

def evaluate_clustering_no_umap(dataframe, exclude_cols=['Local_Authority_Code', 'Local_Authority_Name']):
    """
    Evaluates clustering algorithms on a tabular dataset without applying any dimensionality reduction.
    
    This function:
    1. Preprocesses the data by selecting numeric features, imputing missing values with KNN, and scaling.
    2. Applies a grid search over clustering parameters for KMeans, Gaussian Mixture Models (GMM), and DBSCAN.
    3. Computes three evaluation metrics for each model configuration: Silhouette, Calinski-Harabasz, and Davies-Bouldin.
    4. Normalizes the scores and calculates a composite mean score to rank the configurations.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input dataset containing numeric features to cluster.

    exclude_cols : list of str, optional
        Column names to drop before clustering. These typically contain identifiers or non-numeric metadata.

    Returns
    -------
    results_df : pandas.DataFrame
        A DataFrame containing performance metrics, model parameters, and labels for each clustering configuration,
        sorted by the mean of normalized scores.

    X_scaled : ndarray of shape (n_samples, n_features)
        The preprocessed and scaled feature matrix used for clustering.
    """

    # Step 1: Prepare feature matrix by excluding identifiers and selecting numeric columns only
    X = dataframe.drop(columns=exclude_cols, errors='ignore')
    X = X.select_dtypes(include='number').dropna()

    # Step 2: Impute missing values using KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)

    # Step 3: Standardize the data (zero mean, unit variance)
    X_scaled = StandardScaler().fit_transform(X_imputed)

    # Step 4: Define a grid of clustering algorithms and their hyperparameters
    clustering_algorithms = {
        'KMeans': {
            'model': KMeans,
            'params': {'n_clusters': range(4, 16, 1)}  # Try 4 to 14 clusters
        },
        'GaussianMixture': {
            'model': GaussianMixture,
            'params': {'n_components': range(4, 16, 1), 'covariance_type': ['full', 'tied', 'diag']}
        },
        'DBSCAN': {
            'model': DBSCAN,
            'params': {'eps': [0.3, 0.5, 1.0], 'min_samples': [4, 5, 10, 15]}
        }
    }

    results = []

    # Iterate over each clustering algorithm and its parameter grid
    for name, cfg in clustering_algorithms.items():
        Model = cfg['model']
        grid = ParameterGrid(cfg['params'])

        for params in grid:
            try:
                # Instantiate model with hyperparameters
                if name in ['KMeans', 'GaussianMixture']:
                    model = Model(**params, n_init=20)  # Use multiple initializations for stability
                else:
                    model = Model(**params)

                # Fit and obtain cluster labels
                if name == 'GaussianMixture':
                    labels = model.fit(X_scaled).predict(X_scaled)
                else:
                    labels = model.fit_predict(X_scaled)

                # Ignore configurations that produce fewer than 2 clusters
                unique_labels = set(labels)
                n_clusters = len(unique_labels - {-1}) if -1 in unique_labels else len(unique_labels)
                if n_clusters < 2:
                    continue

                # Compute cluster quality metrics
                sil = silhouette_score(X_scaled, labels)
                ch = calinski_harabasz_score(X_scaled, labels)
                db = davies_bouldin_score(X_scaled, labels)

                # Store results
                results.append({
                    'labels': labels,
                    'algorithm': name,
                    'params': params,
                    'n_clusters': n_clusters,
                    'silhouette_score': sil,
                    'calinski_harabasz': ch,
                    'davies_bouldin': db
                })

            except Exception as e:
                print(f"{name} failed for {params}: {e}")

    # Step 5: Normalize and combine evaluation metrics for comparison
    results_df = pd.DataFrame(results)

    # Scale the three metrics to [0, 1] for comparability
    scaler = MinMaxScaler()
    results_df[['silhouette_score_scaled',
                'calinski_harabasz_scaled',
                'davies_bouldin_scaled']] = scaler.fit_transform(
        results_df[['silhouette_score', 'calinski_harabasz', 'davies_bouldin']]
    )

    # Invert Davies-Bouldin since lower is better
    results_df['davies_bouldin_scaled'] = 1 - results_df['davies_bouldin_scaled']

    # Compute an overall mean score
    results_df['mean_score'] = results_df[[
        'silhouette_score_scaled',
        'calinski_harabasz_scaled',
        'davies_bouldin_scaled'
    ]].mean(axis=1)

    # Sort by mean score (higher = better)
    results_df = results_df.sort_values(by='mean_score', ascending=False)

    return results_df, X_scaled

def evaluate_clustering(dataframe, exclude_cols=['Local_Authority_Code', 'Local_Authority_Name']):
    """
    Evaluates multiple clustering models on a dataset after UMAP-based dimensionality reduction.

    This function performs the following steps:
    1. Preprocesses the dataset: excludes identifier columns, selects numeric features, imputes missing values,
       and standardizes the data.
    2. Applies UMAP with different numbers of output dimensions (n_components).
    3. For each UMAP projection, fits clustering models (KMeans, Gaussian Mixture, DBSCAN) with a parameter grid.
    4. Evaluates each configuration using three clustering quality metrics: silhouette score,
       Calinski-Harabasz index, and Davies-Bouldin index.
    5. Normalizes the metrics, inverts Davies-Bouldin (since lower is better), and computes a composite mean score.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input data containing features for clustering.

    exclude_cols : list of str, optional
        Column names to exclude before feature selection. These typically contain IDs or labels not meant for clustering.

    Returns
    -------
    results_df : pandas.DataFrame
        DataFrame containing clustering results, parameters, metrics (raw and normalized), and sorted by Calinski-Harabasz.
    """
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger('sklearn.mixture').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    if exclude_cols is None:
        exclude_cols = []

    # Step 1: Select numeric columns, excluding any specified
    X = dataframe.select_dtypes(include='number').drop(columns=exclude_cols, errors='ignore')

    if X.empty:
        raise ValueError("No numeric columns found after exclusions. Cannot proceed.")

    if X.shape[0] == 0:
        raise ValueError("No samples available in the input data. Check earlier filtering steps.")

    # Step 1: Feature selection and cleaning
    X = dataframe.drop(columns=exclude_cols, errors='ignore')  # Drop any known metadata columns
    X = X.select_dtypes(include='number').dropna()             # Keep numeric columns and drop rows with missing values

    # Step 2: Impute any remaining missing values using KNN
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)

    # Step 3: Standardize features (zero mean, unit variance)
    X_scaled = StandardScaler().fit_transform(X_imputed)

    # Step 4: Define clustering algorithms and their hyperparameter grids
    clustering_algorithms = {
        'KMeans': {
            'model': KMeans,
            'params': {'n_clusters': range(4, 16)}
        },
        # 'GaussianMixture': {
        #     'model': GaussianMixture,
        #     'params': {'n_components': range(4, 16), 'covariance_type': ['full', 'tied', 'diag']}
        # },
        # 'DBSCAN': {
        #     'model': DBSCAN,
        #     'params': {'eps': [0.3, 0.5, 1.0], 'min_samples': [4, 5, 10, 15]}
        # }
    }

    results = []

    # Step 5: Iterate over different UMAP output dimensions
    for n_comp in range(2, 16):
        reducer = umap.UMAP(n_components=n_comp, random_state=1)
        X_umap = reducer.fit_transform(X_scaled)  # Apply UMAP reduction

        # Step 6: Iterate over each clustering algorithm
        for name, cfg in clustering_algorithms.items():
            Model = cfg['model']
            grid = ParameterGrid(cfg['params'])

            # Step 7: Grid search over model parameters
            for params in grid:
                try:
                    # Instantiate clustering model
                    if name == 'KMeans':
                        model = Model(**params, n_init=20, random_state=1)
                    elif name == 'GaussianMixture':
                        model = Model(**params, n_init=20, random_state=1)
                    else:
                        model = Model(**params)

                    # Fit and predict clusters
                    if name == 'GaussianMixture':
                        labels = model.fit(X_umap).predict(X_umap)
                    else:
                        labels = model.fit_predict(X_umap)

                    # Handle outliers (-1 label in DBSCAN)
                    n_clusters = len(set(labels) - {-1}) if -1 in labels else len(set(labels))
                    if n_clusters < 2:
                        continue  # Skip degenerate clusterings

                    # Step 8: Compute clustering metrics
                    sil = silhouette_score(X_umap, labels)
                    ch = calinski_harabasz_score(X_umap, labels)
                    db = davies_bouldin_score(X_umap, labels)

                    # Store result
                    results.append({
                        'algorithm': name,
                        'params': params,
                        'n_components': n_comp,
                        'n_clusters': n_clusters,
                        'silhouette_score': sil,
                        'calinski_harabasz': ch,
                        'davies_bouldin': db,
                        'labels': labels
                    })

                except Exception as e:
                    print(f"{name} failed for n_components={n_comp} with params {params}: {e}")

    # Step 9: Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Step 10: Normalize scores for fair comparison
    scaler = MinMaxScaler()
    results_df[['silhouette_score_scaled', 'calinski_harabasz_scaled', 'davies_bouldin_scaled']] = scaler.fit_transform(
        results_df[['silhouette_score', 'calinski_harabasz', 'davies_bouldin']]
    )

    # Step 11: Invert Davies-Bouldin (lower is better)
    results_df['davies_bouldin_scaled'] = 1 - results_df['davies_bouldin_scaled']

    # Step 12: Compute mean score across all metrics
    results_df['mean_score'] = results_df[[
        'silhouette_score_scaled',
        'calinski_harabasz_scaled',
        'davies_bouldin_scaled'
    ]].mean(axis=1)

    # Step 13: Sort by Calinski-Harabasz (or change to 'mean_score' as needed)
    results_df = results_df.sort_values(by='silhouette_score_scaled', ascending=False)

    return results_df
