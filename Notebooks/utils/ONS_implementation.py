"""
ONS_implementation.py

This module replicates the ONS Local Authority clustering methodology exactly as described in:
https://www.ons.gov.uk/peoplepopulationandcommunity/wellbeing/methodologies/clusteringlocalauthoritiesagainstsubnationalindicatorsmethodology
And https://github.com/ONSdigital/Subnational-Statistics-and-Analysis/tree/main/clustering_and_nearest_neighbours

This assumes the dataset is already preprocessed to match the ONS structure, extracted from:
https://www.ons.gov.uk/explore-local-statistics/indicators/gross-disposable-household-income-per-head (bottom of the page, under "accompanying dataset (ODS, 4MB))
(Refer to our Datasets_Wrangling_LocalIndicators.ipynb for initial data wrangling)

This also assumes you have access to the Clustered Local Authorities data from ONS, extracted from:
https://www.ons.gov.uk/peoplepopulationandcommunity/wellbeing/datasets/clusteringlocalauthoritiesagainstsubnationalindicatorsengland

Key Features:
-------------
- Uses fixed k values as published by ONS
- Excludes specified Local Authority codes per view and labels them as Cluster = -1 as published by ONS
- Applies preprocessing: cleaning, winsorization, z-score standardization as published by ONS
- Applies PCA and KMeans with a fixed random seed for reproducibility as published by ONS
- Visualises clustering structure using external plots_and_visualisations()

Expected Input:
---------------
Each view is a DataFrame containing:
- 'Area Level'
- 'Area Code'
- 'Area'
- Several numeric indicator columns
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from plots_and_visualisations import plot_best_embedding, radar_plot  # External visualisation module


# Fixed ONS parameters per view

# Known optimal number of clusters per view (ONS-determined)
ONS_K_VALUES = {
    'economic': 4,
    'connectivity': 4,
    'educational_attainment': 8,
    'skills': 4,
    'health': 4,
    'wellbeing': 4
}

# Local Authority codes excluded from clustering per view (ONS-defined)
ONS_EXCLUSIONS = {
    'economic': ['E06000053', 'E09000001'],
    'connectivity': ['E06000053', 'E06000057', 'E06000058', 'E06000059',
                     'E06000060', 'E06000061', 'E06000062', 'E08000037'],
    'educational_attainment': ['E06000017', 'E06000053', 'E06000060',
                                'E06000061', 'E06000062', 'E09000001'],
    'skills': ['E06000053', 'E07000166', 'E09000001'],
    'health': ['E06000022', 'E06000049', 'E06000050', 'E06000053', 'E06000060',
               'E06000061', 'E06000062', 'E07000234', 'E07000236', 'E07000237',
               'E07000238', 'E07000239', 'E08000001', 'E08000002', 'E09000001',
               'E09000010', 'E09000025', 'E09000032'],
    'wellbeing': ['E06000053', 'E09000001']
}

# Preprocessing

def preprocess_view(df: pd.DataFrame, view_name: str) -> pd.DataFrame:
    """
    Preprocesses a single view dataframe for clustering.

    This includes:
    - Removing known problematic or non-numeric entries (e.g. 'na')
    - Converting all values to numeric (non-numeric coerced to NaN)
    - Dropping columns with more than 50% missing values
    - Winsorizing outliers at the 1st and 99th percentiles
    - Dropping rows with any missing values after winsorization
    - Standardizing all numeric features using z-score
    - Keeping 'Area Code' as the index

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe for a view, must include 'Area Code' and numeric indicators.

    view_name : str
        Name of the view (used for logging or debugging).

    Returns
    -------
    pd.DataFrame
        Cleaned, winsorized, and standardized dataframe with Area Code as index.
    """
    df = df.copy()

    # Identify columns
    meta_cols = ['Area Code', 'Area', 'Area Level']
    feature_cols = [col for col in df.columns if col not in meta_cols]

    # Replace string variants of missing with actual NaN
    df[feature_cols] = df[feature_cols].replace(r'(?i)^na$', np.nan, regex=True)

    # Convert all to numeric (coerce errors to NaN)
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    # Winsorize (clip outliers at 1st and 99th percentile)
    for col in feature_cols:
        if col in df.columns:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)

    # Drop rows with any remaining missing data
    df = df.dropna()

    # Standardize numeric features
    numeric_cols = [col for col in df.columns if col not in meta_cols]
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)

    # Add Area Code as index (if not already)
    df_scaled['Area Code'] = df['Area Code'].values
    df_scaled = df_scaled.set_index('Area Code')

    return df_scaled


def apply_pca(df_scaled: pd.DataFrame, random_seed: int = 1, variance_threshold: float = 0.25) -> np.ndarray:
    """
    Applies PCA to a standardized DataFrame, retaining enough components
    to explain at least the given proportion of total variance.

    Parameters
    ----------
    df_scaled : pd.DataFrame
        Standardized numeric dataframe, with Area Code as index.

    random_seed : int
        Seed for reproducibility.

    variance_threshold : float
        Minimum proportion of variance to retain (e.g., 0.25 for 25%).

    Returns
    -------
    np.ndarray
        PCA-transformed data with the selected number of components (minimum 2).
    """
    # Fit full PCA to determine cumulative variance explained
    pca_full = PCA(n_components=None, random_state=random_seed)
    X_full = pca_full.fit_transform(df_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    # Choose number of components to retain the required variance
    n_components = max(2, np.searchsorted(cum_var, variance_threshold) + 1)

    # Re-run PCA with selected number of components
    pca_final = PCA(n_components=n_components, random_state=random_seed)
    X_pca = pca_final.fit_transform(df_scaled)

    return X_pca


# Clustering utilies

def run_ons_pipeline_gridsearch_k(
    df: pd.DataFrame,
    view_name: str,
    k_range: range,
    variance_threshold: float = 0.25,
    random_state: int = 1,
    n_init: int = 1000,
    verbose: bool = True
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int, float, dict]:
    """
    Runs the ONS clustering pipeline on a single view, performing:
    preprocessing → PCA → KMeans clustering → silhouette-based grid search for best k.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input view containing 'Area Code' and numeric indicators.

    view_name : str
        Name of the view (used in logs and results).

    k_range : range
        Range of k values to try in KMeans (e.g., range(4, 16)).

    variance_threshold : float
        Variance to retain in PCA (default 0.25 for 25%).

    random_state : int
        Random seed for reproducibility.

    n_init : int
        Number of initializations for KMeans.

    verbose : bool
        If True, prints silhouette scores per k.

    Returns
    -------
    df_clusters : pd.DataFrame
        DataFrame with Area Code and final cluster labels.

    X_pca : np.ndarray
        PCA-reduced representation of the data.

    labels : np.ndarray
        Cluster labels assigned to each sample.

    best_k : int
        Optimal number of clusters.

    best_score : float
        Silhouette score of the best k.

    silhouette_scores : dict
        Dictionary mapping k to silhouette score.
    """
    # Preprocess view
    df_scaled = preprocess_view(df, view_name)

    # PCA
    X_pca = apply_pca(df_scaled, random_seed=random_state, variance_threshold=variance_threshold)

    # Grid search for best k
    best_score = -1
    best_k = None
    best_labels = None
    silhouette_scores = {}

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        silhouette_scores[k] = score

        if verbose:
            print(f"[{view_name}] k={k} | Silhouette = {score:.3f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    # Create final cluster output
    df_clusters = pd.DataFrame({
        "Area Code": df_scaled.index,
        "Cluster": best_labels
    })

    return df_clusters, X_pca, best_labels, best_k, best_score, silhouette_scores


def run_ons_on_views(
    views: dict,
    variance_threshold: float = 0.25,
    random_state: int = 1,
    n_init: int = 1000,
    verbose: bool = True
) -> tuple[dict, dict]:
    """
    Runs the ONS clustering pipeline on all views using fixed k-values per view.

    Parameters
    ----------
    views : dict
        Dictionary of {view_name: DataFrame} for each thematic view.

    variance_threshold : float
        Variance to retain in PCA (default is 0.25 to match ONS).

    random_state : int
        Random seed for reproducibility.

    n_init : int
        Number of initializations for KMeans.

    verbose : bool
        If True, print clustering details per view.

    Returns
    -------
    results_by_view : dict
        For each view, a dictionary with:
        - 'clusters': DataFrame with Area Code and cluster labels
        - 'X_pca': PCA-reduced features
        - 'labels': cluster assignments
        - 'k': number of clusters used

    silhouette_scores_by_view : dict
        Dictionary mapping view name to silhouette score.
    """
    results_by_view = {}
    silhouette_scores_by_view = {}

    for view_name, df in views.items():
        if view_name not in ONS_K_VALUES:
            continue  # skip views without fixed k

        k = ONS_K_VALUES[view_name]
        if verbose:
            print(f"\n[{view_name}] Running fixed-k clustering (k={k})")

        # Preprocess
        df_scaled = preprocess_view(df, view_name)

        # PCA
        X_pca = apply_pca(df_scaled, random_seed=random_state, variance_threshold=variance_threshold)

        # KMeans
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X_pca)

        # Silhouette
        sil = silhouette_score(X_pca, labels)
        silhouette_scores_by_view[view_name] = sil

        if verbose:
            print(f"[{view_name}] Silhouette = {sil:.3f} on {len(labels)} areas")

        # Package results
        df_clusters = pd.DataFrame({
            "Area Code": df_scaled.index,
            "Cluster": labels
        })

        results_by_view[view_name] = {
            "clusters": df_clusters,
            "X_pca": X_pca,
            "labels": labels,
            "k": k
        }
        
        plot_best_embedding(X_pca, labels)
        radar_plot(view_name, views, results_by_view, preprocess_view)

    return results_by_view, silhouette_scores_by_view

# Integration strategies for Headline Model

def run_ons_headline_model(
    views: dict,
    selected_metrics: dict,
    exclusions: list,
    k: int = 4,
    variance_threshold: float = 0.25,
    random_state: int = 1,
    n_init: int = 1000,
    verbose: bool = True
):
    """
    Implements the ONS-style headline model using one representative metric from each view.

    Parameters
    ----------
    views : dict
        Dictionary mapping view name to raw DataFrame (must include 'Area Code').

    selected_metrics : dict
        Dictionary mapping view name to the column name of the metric to include.

    exclusions : list
        List of area codes to exclude from clustering (they are assigned -1).

    k : int
        Number of clusters to form.

    variance_threshold : float
        Minimum variance to retain in PCA.

    random_state : int
        Random seed for PCA and KMeans reproducibility.

    n_init : int
        Number of KMeans initializations.

    verbose : bool
        Whether to print logging info and show UMAP plots.

    Returns
    -------
    final_df : pd.DataFrame
        DataFrame with 'Area Code' and final cluster assignments (including -1 for exclusions).

    best_model : sklearn.cluster.KMeans
        Fitted KMeans model on PCA-reduced data.

    sil_score : float
        Silhouette score on the included areas.

    silhouette_by_k : dict
        Dictionary of silhouette scores (single entry with key = k).
    """

    # Merge selected columns from each view
    merged = None
    for view, metric in selected_metrics:
        df = views[view][["Area Code", metric]].copy()
        df = df.rename(columns={metric: f"{view}_metric"})
        merged = df if merged is None else merged.merge(df, on="Area Code", how="inner")

    # Exclude predefined areas and drop rows with missing values
    included = merged[~merged["Area Code"].isin(exclusions)].dropna()
    excluded = merged[merged["Area Code"].isin(exclusions)]

    # Extract features and standardize
    X = included.drop(columns=["Area Code"]).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    X_pca = apply_pca(
        pd.DataFrame(X_scaled, index=included["Area Code"]),
        random_seed=random_state,
        variance_threshold=variance_threshold
    )

    # KMeans clustering
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)

    # Prepare results
    included = included.copy()
    included["HeadlineCluster"] = labels

    if not excluded.empty:
        excluded = excluded[["Area Code"]].copy()
        excluded["HeadlineCluster"] = -1
        final_df = pd.concat([included[["Area Code", "HeadlineCluster"]], excluded])
    else:
        final_df = included[["Area Code", "HeadlineCluster"]]

    final_df = final_df.sort_values("Area Code").reset_index(drop=True)

    # Plot embedding
    if verbose:
        print(f"[Headline] Silhouette = {sil:.3f} | Excluded = {len(exclusions)}")
        plot_best_embedding(X_pca, labels)

        # Prepare merged dataframe with raw metrics
        merged_df = None
        for view, metric in selected_metrics:
            df = views[view][["Area Code", metric]].copy()
            df = df.rename(columns={metric: f"{view}_metric"})
            merged_df = df if merged_df is None else merged_df.merge(df, on="Area Code", how="inner")

        # Add cluster labels
        merged_df = merged_df.merge(
    final_df[["Area Code", "HeadlineCluster"]],
    on="Area Code", how="left"
).rename(columns={"HeadlineCluster": "Cluster"})
        # Ensure 'Area Code' is in the columns, not index
        df_for_plot = preprocess_view(merged_df, "headline").copy()
        if "Area Code" not in df_for_plot.columns and df_for_plot.index.name == "Area Code":
            df_for_plot = df_for_plot.reset_index()

        # Ensure no pre-existing 'Cluster' columns
        df_for_plot = preprocess_view(merged_df, "headline").copy()
        df_for_plot = df_for_plot.reset_index() if df_for_plot.index.name == "Area Code" else df_for_plot
        df_for_plot = df_for_plot.drop(columns=[col for col in df_for_plot.columns if col.startswith("Cluster")], errors="ignore")

        # Merge in the cluster labels
        df_for_plot = df_for_plot.merge(
            merged_df[["Area Code", "Cluster"]],
            on="Area Code", how="left"
        )
        
        # Now ready for plotting
        radar_plot(
            "headline",
            {"headline": df_for_plot},
            {"headline": {"clusters": df_for_plot[["Area Code", "Cluster"]]}},
            preprocess_view=lambda x, _: x
        )

    silhouette_by_k = {k: sil}
    return final_df, model, sil, silhouette_by_k


def run_ons_headline_model_intermediate(
    views: dict,
    k_range: range,
    variance_threshold: float = 0.25,
    random_state: int = 1,
    n_init: int = 1000,
    verbose: bool = True
):
    """
    Applies intermediate integration by reducing each view separately with PCA,
    then concatenating the resulting latent spaces before clustering.

    Parameters
    ----------
    views : dict
        Dictionary of {view_name: raw DataFrame} per thematic view.

    k_range : range
        Range of k values to search for the best clustering.

    variance_threshold : float
        Minimum proportion of variance to retain during PCA per view.

    random_state : int
        Random seed for reproducibility.

    n_init : int
        Number of KMeans initializations.

    verbose : bool
        If True, prints detailed processing and silhouette scores.

    Returns
    -------
    final_df : pd.DataFrame
        DataFrame with Area Codes and assigned cluster labels.

    best_model : sklearn.cluster.KMeans
        Trained KMeans model with the best silhouette score.

    best_score : float
        Best silhouette score found during grid search.

    silhouette_scores : dict
        Dictionary of {k: silhouette score}.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Preprocess all views
    preprocessed_views = {
        view_name: preprocess_view(df, view_name)
        for view_name, df in views.items()
    }

    # Align views on shared Area Codes
    common_area_codes = set.intersection(*(set(df.index) for df in preprocessed_views.values()))
    if verbose:
        print(f"[Intermediate] Shared Area Codes across all views: {len(common_area_codes)}")

    # Apply PCA independently to each view and collect latent spaces
    latent_spaces = []
    for view_name, df_scaled in preprocessed_views.items():
        df_view = df_scaled.loc[sorted(common_area_codes)]
        X_pca = apply_pca(df_view, random_seed=random_state, variance_threshold=variance_threshold)

        if verbose:
            print(f"[{view_name}] Latent space shape: {X_pca.shape}")
        latent_spaces.append(X_pca)

    # Concatenate latent spaces (intermediate integration)
    X_combined = np.concatenate(latent_spaces, axis=1)

    # Grid search over k for best silhouette score
    best_score = -1
    best_k = None
    best_model = None
    best_labels = None
    silhouette_scores = {}

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X_combined)
        sil = silhouette_score(X_combined, labels)
        silhouette_scores[k] = sil

        if verbose:
            print(f"[Intermediate] k = {k} | Silhouette = {sil:.3f}")

        if sil > best_score:
            best_score = sil
            best_k = k
            best_model = model
            best_labels = labels

    # Construct final result dataframe
    final_df = pd.DataFrame({
        "Area Code": sorted(common_area_codes),
        "HeadlineCluster": best_labels
    })

    # Plot best embedding
    if verbose:
        plot_best_embedding(X_combined, labels=best_labels)

        # Prepare merged raw dataframe for radar_plot
        merged_df = None
        for name, v in views.items():
            df = preprocess_view(v, name).loc[sorted(common_area_codes)]
            df.columns = [f"{name}_{c}" for c in df.columns]
            merged_df = df if merged_df is None else pd.concat([merged_df, df], axis=1)

        # Create cluster dataframe for plotting
        cluster_df = pd.DataFrame({
            "Area Code": sorted(common_area_codes),
            "Cluster": best_labels
        })

        # Merge cluster assignments into merged_df
        merged_df = merged_df.reset_index()
        merged_df = merged_df.merge(cluster_df, on="Area Code", how="inner")

        # Plot radar
        radar_plot(
            "intermediate",
            {"intermediate": merged_df},
            {"intermediate": {"clusters": cluster_df}},
            preprocess_view
        )

    return final_df, best_model, best_score, silhouette_scores


def run_ons_headline_model_late(
    views: dict,
    k_range: range,
    variance_threshold: float = 0.25,
    random_state: int = 1,
    n_init: int = 1000,
    verbose: bool = True
):
    """
    Performs late integration clustering by independently clustering each view
    and then concatenating their cluster assignments for final evaluation.

    Parameters
    ----------
    views : dict
        Dictionary of {view_name: raw DataFrame} per thematic view.

    k_range : range
        Range of k values to search for best clustering for each view.

    variance_threshold : float
        Minimum proportion of variance to retain during PCA per view.

    random_state : int
        Random seed for reproducibility.

    n_init : int
        Number of KMeans initializations.

    verbose : bool
        If True, prints detailed processing and silhouette scores.

    Returns
    -------
    final_df : pd.DataFrame
        DataFrame with Area Codes and combined cluster labels.

    best_k_dict : dict
        Dictionary with {view_name: best_k}.

    final_silhouette : float
        Silhouette score of combined clustering representation.

    silhouette_scores_by_view : dict
        Silhouette scores per k for each view.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Preprocess each view
    preprocessed_views = {
        view_name: preprocess_view(df, view_name)
        for view_name, df in views.items()
    }

    # Align on common Area Codes
    common_area_codes = sorted(set.intersection(*(set(df.index) for df in preprocessed_views.values())))
    if verbose:
        print(f"[Late Integration] Shared Area Codes: {len(common_area_codes)}")

    # Cluster each view independently
    cluster_label_df = pd.DataFrame(index=common_area_codes)
    best_k_dict = {}
    silhouette_scores_by_view = {}

    for view_name, df_scaled in preprocessed_views.items():
        df_view = df_scaled.loc[common_area_codes]
        X_pca = apply_pca(df_view, random_seed=random_state, variance_threshold=variance_threshold)

        best_score = -1
        best_k = None
        best_labels = None
        silhouette_scores = {}

        for k in k_range:
            model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
            labels = model.fit_predict(X_pca)
            sil = silhouette_score(X_pca, labels)
            silhouette_scores[k] = sil

            if sil > best_score:
                best_score = sil
                best_k = k
                best_labels = labels

            if verbose:
                print(f"[{view_name}] k = {k} | Silhouette = {sil:.3f}")

        # Store best k and labels
        cluster_label_df[f"{view_name}_cluster"] = best_labels
        best_k_dict[view_name] = best_k
        silhouette_scores_by_view[view_name] = silhouette_scores

        if verbose:
            print(f"[{view_name}] Best k = {best_k} | Silhouette = {best_score:.3f}")

    # Compute silhouette score on combined cluster labels
    X_combined = cluster_label_df.values
    final_model = KMeans(n_clusters=4, random_state=random_state, n_init=n_init)
    final_labels = final_model.fit_predict(X_combined)
    final_silhouette = silhouette_score(X_combined, final_labels)

    if verbose:
        print(f"[Late Integration] Final silhouette score from combined cluster labels = {final_silhouette:.3f}")

    #  Final output
    final_df = cluster_label_df.copy()
    final_df["HeadlineCluster"] = final_labels

    # Plot
    plot_best_embedding(X_combined, final_labels)
    final_df.index.name = "Area Code"
    cluster_df = final_df.reset_index()[["Area Code", "HeadlineCluster"]].rename(
    columns={"HeadlineCluster": "Cluster"}
)
    
    # Use raw concatenated view as input
    merged_df = None
    for name, v in views.items():
        df = preprocess_view(v, name).loc[common_area_codes]
        df.columns = [f"{name}_{c}" for c in df.columns]
        merged_df = df if merged_df is None else pd.concat([merged_df, df], axis=1)

    # Add cluster assignments
    merged_df = merged_df.reset_index()
    merged_df["Cluster"] = cluster_df.set_index("Area Code").loc[merged_df["Area Code"], "Cluster"].values

    # Plot radar
    radar_plot("late", {"late": merged_df}, {"late": {"clusters": cluster_df}}, preprocess_view)

    return final_df, best_k_dict, final_silhouette, silhouette_scores_by_view


# Output processing utilities

def export_cluster_labels(results_by_view: dict, headline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines all cluster labels from each view and the headline model
    into a single DataFrame per Local Authority.

    Parameters
    ----------
    results_by_view : dict
        Dictionary with view names as keys and each value containing a 'clusters' DataFrame
        with columns ['Area Code', 'Cluster'].

    headline_df : pd.DataFrame
        DataFrame with columns ['Area Code', 'HeadlineCluster'] from run_ons_headline_model().

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['Area Code', 'economic', 'connectivity', ..., 'wellbeing', 'HeadlineCluster']
    """
    merged = None

    for view, view_data in results_by_view.items():
        df = view_data['clusters'][['Area Code', 'Cluster']].rename(columns={'Cluster': view})
        merged = df if merged is None else merged.merge(df, on='Area Code', how='outer')

    # Merge headline cluster
    full = merged.merge(headline_df, on='Area Code', how='outer')

    # Optional: sort for readability
    full = full.sort_values('Area Code').reset_index(drop=True)

    return full

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd

def compare_label_consistency(our_df: pd.DataFrame, ons_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares clustering consistency between our labels and ONS labels across all models.

    Parameters
    ----------
    our_df : pd.DataFrame
        DataFrame with our cluster labels. Columns: ['Area Code', 'economic', ..., 'HeadlineCluster']

    ons_df : pd.DataFrame
        ONS official clustering results. Expected columns:
        ['Local_Authority_Code', ..., 'Economic model Code', ..., 'Headline Model Code']

    Returns
    -------
    pd.DataFrame
        A summary DataFrame with ARI and NMI for each model.
    """
    # Mapping between our column names and ONS column names
    col_map = {
        'economic': 'Economic model Code',
        'connectivity': 'Connectivity model Code',
        'educational_attainment': 'Educational attainment model Code',
        'skills': 'Skills model Code',
        'health': 'Health model Code',
        'wellbeing': 'Well-being model Code',
        'HeadlineCluster': 'Headline Model Code'
    }

    # Merge both DataFrames on Area Code
    merged = our_df.merge(
        ons_df,
        left_on='Area Code',
        right_on='Local_Authority_Code',
        how='inner'
    )

    results = []
    for our_col, ons_col in col_map.items():
        # Drop rows with -1 or NaN in either column
        sub_df = merged[[our_col, ons_col]].dropna()
        sub_df = sub_df[(sub_df[our_col] != -1) & (sub_df[ons_col] != -1)]

        if sub_df.empty:
            ari = nmi = None
        else:
            ari = adjusted_rand_score(sub_df[ons_col], sub_df[our_col])
            nmi = normalized_mutual_info_score(sub_df[ons_col], sub_df[our_col])

        results.append({
            'Model': our_col.replace('_', ' ').title(),
            'Adjusted Rand Index': ari,
            'Normalized Mutual Information': nmi,
            'Compared Pairs': len(sub_df)
        })

    return pd.DataFrame(results).sort_values('Model').reset_index(drop=True)
