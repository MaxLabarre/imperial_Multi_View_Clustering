from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Tuple

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
from sklearn.metrics import silhouette_score

from data_preprocessing import preprocess_view, exclude_las, get_combined_views_union, _standardize_columns, DEFAULT_META_COLS

# Similarity-based Methods utilities

def compute_masked_cosine_similarity(X: np.ndarray) -> np.ndarray:
    """
    Compute masked cosine similarity for an array with possible NaNs.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features) with possible NaNs.

    Returns
    -------
    np.ndarray
        Symmetric similarity matrix S (n_samples x n_samples) with values in [-1, 1].
        If two rows have no overlapping non-NaN features, similarity is 0.0.
    """
    n = X.shape[0]
    S = np.zeros((n, n), dtype=float)

    for i in range(n):
        xi = X[i]
        for j in range(i, n):
            xj = X[j]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            if np.sum(mask) == 0:
                sim = 0.0
            else:
                xi_m = xi[mask]
                xj_m = xj[mask]
                denom = (np.linalg.norm(xi_m) * np.linalg.norm(xj_m)) + 1e-8
                sim = float(np.dot(xi_m, xj_m) / denom)
            S[i, j] = S[j, i] = sim
    return S

def build_union_feature_matrix(views: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the early-integration union feature matrix from multiple views.

    Parameters
    ----------
    views : dict[str, pd.DataFrame]
        Mapping view name -> DataFrame, each containing 'Local_Authority_Code' and numeric columns.

    Returns
    -------
    X_union : pd.DataFrame
        Union-aligned (rows = Local_Authority_Code) feature matrix with possible NaNs.
    area_codes : list[str]
        Row index (Local_Authority_Code) in order.
    """
    X_union, area_codes = get_combined_views_union(views)  # uses standard pipeline
    X_union = X_union.apply(pd.to_numeric, errors="coerce")  # defensive: numeric only
    return X_union, list(area_codes)

# Early Integration - Masked Cosine Similarity

def early_masked_cosine_similarity(
    views: Dict[str, pd.DataFrame],
    normalize_rows: bool = True,
    zero_diagonal: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Early integration via masked cosine similarity computed on the concatenated union matrix.

    Parameters
    ----------
    views : dict[str, pd.DataFrame]
        Mapping view name -> DataFrame with 'Local_Authority_Code' + numeric columns.
    normalize_rows : bool, default=True
        If True, row-normalize the similarity matrix (row-stochastic).
    zero_diagonal : bool, default=True
        If True, zero the diagonal before row-normalization.

    Returns
    -------
    S : np.ndarray
        (n x n) masked-cosine similarity computed on the union feature matrix.
    area_codes : list[str]
        Ordered Local_Authority_Code list for S rows/cols.
    """
    X_union, area_codes = build_union_feature_matrix(views)
    S = compute_masked_cosine_similarity(X_union.values.astype(float))
    if normalize_rows:
        S = normalize_similarity(S, zero_diagonal=zero_diagonal)
    return S, area_codes


def run_early_mcs_pca_kmeans(
    views: Dict[str, pd.DataFrame],
    k_range: Iterable[int] = range(4, 16),
    n_pca_range: Iterable[int] = range(2, 6),
    n_init: int | str = "auto",
    normalize_rows: bool = True,
    zero_diagonal: bool = True,
    verbose: bool = True,
):
    """
    One-call Early Integration:
    Concatenate views → masked cosine similarity → (optional row-normalize) →
    KernelPCA → KMeans grid search. Returns everything needed downstream.

    Parameters
    ----------
    views : dict[str, pd.DataFrame]
        View name -> DataFrame with 'Local_Authority_Code' and numeric columns.
    k_range : iterable[int], default=range(4, 16)
        Candidate K values for KMeans.
    n_pca_range : iterable[int], default=range(2, 6)
        KernelPCA embedding dimensions to try.
    n_init : int | str, default="auto"
        KMeans restarts. Use an int (e.g., 50) if sklearn < 1.4.
    normalize_rows : bool, default=True
        Row-normalize similarity to reduce degree effects.
    zero_diagonal : bool, default=True
        Zero diagonal before row-normalization to avoid self-sim dominance.
    verbose : bool, default=True
        Print grid progress and best result.

    Returns
    -------
    out : dict
        {
          # Similarity + indexing
          "S": np.ndarray (n x n),                     # early-integration similarity
          "area_codes": list[str],

          # Clustering
          "labels": np.ndarray (n,),
          "embedding": np.ndarray (n x n_pca_best),
          "silhouette": float,
          "best_params": tuple[int, int],              # (n_pca_best, k_best)
          "assignments_df": pd.DataFrame,              # Local_Authority_Code → Cluster
          "grid_results": pd.DataFrame                 # (PCA_n_components, K, Silhouette)
        }
    """
    # Early integration similarity
    S, area_codes = early_masked_cosine_similarity(
        views=views,
        normalize_rows=normalize_rows,
        zero_diagonal=zero_diagonal,
    )

    # Cluster (KernelPCA + KMeans) on this similarity
    # Reuse the generic clustering helper
    S_psd = make_psd((S + S.T) / 2.0)
    best, results_df = cluster_from_similarity(
        S=S_psd,
        index=area_codes,
        k_range=k_range,
        n_pca_range=n_pca_range,
        n_init=n_init,
        verbose=verbose,
    )

    out = {
        "S": S,  # return the normalized S used (for plotting/diagnostics)
        "area_codes": area_codes,
        "labels": best["labels"],
        "embedding": best["embedding"],
        "silhouette": best["sil"],
        "best_params": best["params"],
        "assignments_df": best["result_df"],
        "grid_results": results_df,
    }
    return out

# Intermediate Integration - Multi-View Similarity Fusion
def normalize_similarity(S: np.ndarray, zero_diagonal: bool = True) -> np.ndarray:
    """
    Row-normalize a similarity matrix to be row-stochastic.

    Parameters
    ----------
    S : np.ndarray
        Square similarity matrix (n x n).
    zero_diagonal : bool, default=True
        If True, set the diagonal to 0 before normalizing to avoid self-similarity dominance.

    Returns
    -------
    np.ndarray
        Row-stochastic matrix, where each row sums to ~1.
    """
    S = S.copy()
    if zero_diagonal:
        np.fill_diagonal(S, 0.0)
    row_sums = S.sum(axis=1, keepdims=True) + 1e-8
    return S / row_sums


def make_psd(K: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Project a symmetric matrix onto the PSD cone by clipping negative eigenvalues.

    Parameters
    ----------
    K : np.ndarray
        Symmetric matrix.
    eps : float, default=1e-8
        Minimum eigenvalue after clipping.

    Returns
    -------
    np.ndarray
        PSD matrix of the same shape as K.
    """
    K = (K + K.T) / 2.0
    w, V = np.linalg.eigh(K)
    w_clipped = np.clip(w, a_min=eps, a_max=None)
    return (V * w_clipped) @ V.T


# Fusion + contributions
def multiview_similarity_fusion(
    views: Dict[str, pd.DataFrame],
    normalize_rows: bool = True,
    zero_diagonal: bool = True,
    return_all_matrices: bool = False,
    meta_cols: Iterable[str] | None = None,
) -> Tuple[np.ndarray, List[str], List[np.ndarray], List[str]]:
    """
    Build per-view masked-cosine similarity matrices and fuse them by simple mean.

    Parameters
    ----------
    views : dict[str, pd.DataFrame]
        Mapping from view name to DataFrame. Each frame should contain
        'Local_Authority_Code' and numeric variables (others will be dropped).
    normalize_rows : bool, default=True
        If True, row-normalize each view's similarity matrix before fusion.
    zero_diagonal : bool, default=True
        If True, set diagonal to 0 before row-normalization.
    return_all_matrices : bool, default=False
        If True, return list of per-view similarity matrices alongside the fused matrix.
    meta_cols : iterable[str] or None, default=None
        Set of metadata columns to drop; defaults to DEFAULT_META_COLS.

    Returns
    -------
    S_fused : np.ndarray
        Fused similarity matrix (n x n).
    all_indices : list[str]
        Sorted list of Local_Authority_Code indices used in S_fused.
    similarity_matrices : list[np.ndarray]
        Per-view similarity matrices in the same order as view_names.
    view_names : list[str]
        Names of the views in the order they were processed.
    """
    meta_cols = set(meta_cols) if meta_cols is not None else set(DEFAULT_META_COLS)

    cleaned_views = {}
    all_indices = set()

    # Standardize & align
    for name, df in views.items():
        df_clean = preprocess_view(df.copy())
        df_clean = exclude_las(df_clean.copy(), name)
        if "Local_Authority_Code" not in df_clean.columns:
            raise ValueError(f"'Local_Authority_Code' missing after preprocessing for view '{name}'.")
        df_clean = df_clean.set_index("Local_Authority_Code")
        df_clean = df_clean.drop(columns=[c for c in meta_cols if c in df_clean.columns], errors="ignore")
        df_clean = df_clean.apply(pd.to_numeric, errors="coerce")
        df_clean = df_clean.sort_index()
        cleaned_views[name] = df_clean
        all_indices.update(df_clean.index)

    all_indices = sorted(all_indices)
    view_names = list(cleaned_views.keys())
    similarity_matrices = []

    # Compute per-view similarities on the union (allow NaNs)
    for name in view_names:
        X = cleaned_views[name].reindex(all_indices).values.astype(float)
        S = compute_masked_cosine_similarity(X)
        if normalize_rows:
            S = normalize_similarity(S, zero_diagonal=zero_diagonal)
        similarity_matrices.append(S)

    # Simple mean fusion
    S_fused = np.mean(similarity_matrices, axis=0)

    if return_all_matrices:
        return S_fused, all_indices, similarity_matrices, view_names
    else:
        return S_fused, all_indices, [], view_names


def compute_per_view_contributions(
    similarity_matrices: List[np.ndarray],
    S_fused: np.ndarray,
    eps: float = 1e-8,
    return_mode: str = "both",
):
    """
    Compute contributions of each view to the fused similarity matrix.

    Parameters
    ----------
    similarity_matrices : list[np.ndarray]
        List of per-view similarity matrices, all (n x n).
    S_fused : np.ndarray
        Fused similarity matrix (n x n).
    eps : float, default=1e-8
        Small constant to avoid division by zero.
    return_mode : {"global", "per_la", "both"}, default="both"
        Whether to return global averages, per-LA matrix, or both.

    Returns
    -------
    If "global": np.ndarray of shape (n_views,)
        Global average contribution per view.
    If "per_la": np.ndarray of shape (n_samples, n_views)
        Per-LA contribution matrix (row: LA, col: view).
    If "both": tuple (global_contribs, per_la_matrix)
        global_contribs: (n_views,)
        per_la_matrix : (n_samples, n_views)
    """
    n_views = len(similarity_matrices)
    n_samples = S_fused.shape[0]

    per_la_matrix = np.zeros((n_samples, n_views), dtype=float)
    denom = S_fused + eps

    for v, S_v in enumerate(similarity_matrices):
        ratio = S_v / denom  # elementwise
        per_la_matrix[:, v] = ratio.mean(axis=1)

    global_contribs = per_la_matrix.mean(axis=0)

    if return_mode == "global":
        return global_contribs
    elif return_mode == "per_la":
        return per_la_matrix
    elif return_mode == "both":
        return global_contribs, per_la_matrix
    else:
        raise ValueError("Invalid return_mode. Use 'global', 'per_la', or 'both'.")


# Clustering (KernelPCA + KMeans)
def cluster_from_similarity(
    S: np.ndarray,
    index: List[str],
    k_range: Iterable[int] = range(4, 16),
    n_pca_range: Iterable[int] = range(2, 6),
    n_init: int | str = "auto",
    verbose: bool = True,
):
    """
    Grid-search KernelPCA dimensionality and KMeans K on a fused similarity matrix.

    Parameters
    ----------
    S : np.ndarray
        Fused similarity (n x n). Will be symmetrized and projected to PSD.
    index : list[str]
        Ordered identifiers (e.g., Local_Authority_Code) matching S rows/cols.
    k_range : iterable[int], default=range(4, 16)
        Values of K to try.
    n_pca_range : iterable[int], default=range(2, 6)
        KernelPCA output dimensions to try.
    n_init : int | str, default="auto"
        KMeans restarts.
    verbose : bool, default=True
        Print progress and the best setting.

    Returns
    -------
    best : dict
        {
          "embedding": np.ndarray (n x n_pca_best),
          "labels": np.ndarray (n,),
          "silhouette": float,
          "sil": float,  # legacy alias
          "params": (n_pca_best, k_best),
          "result_df": pd.DataFrame with columns ["Local_Authority_Code","Cluster"]
        }
    results_df : pd.DataFrame
        Rows with ["PCA_n_components","K","Silhouette"] for the grid.
    """
    # Ensure symmetric & PSD for kernel methods
    S = (S + S.T) / 2.0
    S = make_psd(S)

    results = []
    best = {"silhouette": -1.0, "sil": -1.0}

    for n_pca in n_pca_range:
        kpca = KernelPCA(
            n_components=n_pca,
            kernel="precomputed",
            random_state=19042022,
            remove_zero_eig=True,
        )
        embedding = kpca.fit_transform(S)

        for k in k_range:
            if k < 2 or k > embedding.shape[0]:
                if verbose:
                    print(f"Skipping PCA={n_pca}, K={k} (n={embedding.shape[0]})")
                continue

            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=n_init,
                random_state=19042022,
            )
            labels = km.fit_predict(embedding)

            try:
                sil = silhouette_score(embedding, labels)
            except ValueError:
                sil = -1.0

            if verbose:
                print(f"n={len(index)}, PCA={n_pca}, K={k} → Silhouette: {sil:.4f}")

            results.append({"PCA_n_components": n_pca, "K": k, "Silhouette": sil})
            if sil > best["silhouette"]:
                best = {
                    "embedding": embedding,
                    "labels": labels,
                    "silhouette": sil,
                    "sil": sil,  # legacy alias
                    "params": (n_pca, k),
                    "result_df": pd.DataFrame(
                        {"Local_Authority_Code": index, "Cluster": labels}
                    ),
                }

    results_df = pd.DataFrame(results)
    if verbose and best["silhouette"] >= 0:
        n_pca_best, k_best = best["params"]
        print(f"\nBest: PCA={n_pca_best}, K={k_best}, Silhouette={best['silhouette']:.4f}")

    return best, results_df


# One-call wrapper
def run_mvsf_pca_kmeans(
    views: Dict[str, pd.DataFrame],
    k_range: Iterable[int] = range(4, 16),
    n_pca_range: Iterable[int] = range(2, 6),
    n_init: int | str = "auto",
    normalize_rows: bool = True,
    zero_diagonal: bool = True,
    verbose: bool = True,
):
    """
    One-call Multi-View Similarity Fusion → (optional row-normalization) → Contributions →
    KernelPCA → KMeans grid search, returning everything needed for reporting/plotting.

    Parameters
    ----------
    views : dict[str, pd.DataFrame]
        Mapping view name → DataFrame, each containing 'Local_Authority_Code' + numeric columns.
    k_range : iterable[int], default=range(4, 16)
        KMeans K values to search.
    n_pca_range : iterable[int], default=range(2, 6)
        KernelPCA output dimensions to search.
    n_init : int | str, default="auto"
        KMeans restarts.
    normalize_rows : bool, default=True
        Row-normalize each view’s similarity before fusion.
    zero_diagonal : bool, default=True
        Zero the diagonal before row-normalization.
    verbose : bool, default=True
        Print progress and best parameters.

    Returns
    -------
    out : dict
        {
          # Fusion + inputs for plots
          "S_fused": np.ndarray (n x n),
          "area_codes": list[str],
          "similarity_matrices": list[np.ndarray],   # per-view, in order of view_names
          "view_names": list[str],

          # Contributions
          "per_view_contributions_global": np.ndarray (n_views,),
          "per_view_contributions_per_la": np.ndarray (n x n_views),

          # Clustering
          "labels": np.ndarray (n,),
          "embedding": np.ndarray (n x n_pca_best),
          "silhouette": float,
          "best_params": tuple[int, int],            # (n_pca_best, k_best)
          "assignments_df": pd.DataFrame,            # Local_Authority_Code → Cluster
          "grid_results": pd.DataFrame               # grid search results
        }
    """
    # Fusion
    S_fused, area_codes, similarity_matrices, view_names = multiview_similarity_fusion(
        views=views,
        normalize_rows=normalize_rows,
        zero_diagonal=zero_diagonal,
        return_all_matrices=True,
    )

    # per-view contributions
    per_view_contributions_global, per_view_contributions_per_la = compute_per_view_contributions(
        similarity_matrices=similarity_matrices,
        S_fused=S_fused,
        return_mode="both"
    )

    # Clustering (KernelPCA + KMeans)
    best, results_df = cluster_from_similarity(
        S=S_fused,
        index=area_codes,
        k_range=k_range,
        n_pca_range=n_pca_range,
        n_init=n_init,
        verbose=verbose,
    )

    # Unified output (includes convenience copies)
    out = {
        "S_fused": S_fused,
        "similarity_matrices": similarity_matrices,               # list[np.ndarray], one per view (n×n)
        "area_codes": area_codes,                                 # list[str]
        "view_names": view_names,                                 # list[str]

        "per_view_contributions_global": per_view_contributions_global,  # (V,)
        "per_view_contributions_per_la": per_view_contributions_per_la,  # (n, V)

        "best": best,                                             # {'embedding','labels','silhouette','params','result_df'}
        "grid_results": results_df,                               # DataFrame: PCA_n_components, K, Silhouette

        # convenience copies for easy unpacking downstream
        "embedding": best["embedding"],                           # (n, n_pca_best)
        "labels": best["labels"],                                 # (n,)
        "best_params": best["params"],                            # (n_pca_best, k_best)
        "assignments_df": best["result_df"],                      # tidy cluster assignments
    }
    return out

# Late Integration - Co-Association Similarity Ensemble

def cluster_per_view(
    views: Dict[str, pd.DataFrame],
    ons_k_values: Dict[str, int],
    n_init: int | str = "auto",
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Cluster each view independently with KMeans (using view-specific k from ons_k_values).

    Steps per view:
      1) Standardize column names (ensures 'Local_Authority_Code').
      2) preprocess_view (drops meta, winsorize/clean as pipeline defines).
      3) exclude_las(view_name) (ONS exclusions).
      4) Set index to 'Local_Authority_Code', coerce numeric, drop rows with any NaN.
      5) Robust scale features; KMeans with provided k.

    Parameters
    ----------
    views : dict[str, pd.DataFrame]
        Mapping view name -> DataFrame with 'Local_Authority_Code' and numeric features.
    ons_k_values : dict[str, int]
        Mapping from lowercased view name -> target number of clusters (k).
    n_init : int | str, default="auto"
        KMeans n_init (use an int like 50 if sklearn < 1.4).
    verbose : bool, default=True
        Print skips and basic progress messages.

    Returns
    -------
    clusterings : dict[str, dict]
        Per-view results:
        {
          view_name: {
            "labels": np.ndarray (n_view,),
            "area_codes": list[str] (length n_view)
          },
          ...
        }
    """
    clusterings: Dict[str, dict] = {}

    for name, df in views.items():
        k = ons_k_values.get(name.lower())
        if k is None:
            if verbose:
                print(f"[{name}] skipped: no k value defined in ons_k_values")
            continue

        # Standardize columns
        df = _standardize_columns(df)

        # Preprocess (handles meta; uses pipeline rules)
        df_clean = preprocess_view(df, meta_cols=DEFAULT_META_COLS)

        # Exclusions & indexing
        df_clean = exclude_las(df_clean.copy(), name)
        id_col = "Local_Authority_Code"
        if id_col not in df_clean.columns:
            raise KeyError(f"[{name}] '{id_col}' not found after preprocessing.")
        df_clean = df_clean.set_index(id_col)

        # Numeric only, drop remaining meta & rows with any NaN
        df_clean = df_clean.drop(
            columns=[c for c in DEFAULT_META_COLS if c in df_clean.columns],
            errors="ignore",
        )
        df_clean = df_clean.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

        if df_clean.shape[0] < k:
            if verbose:
                print(f"[{name}] skipped: not enough rows ({df_clean.shape[0]}) for k={k}")
            continue

        # Scale + KMeans
        X = RobustScaler().fit_transform(df_clean.values)
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=19042022)
        labels = km.fit_predict(X)

        clusterings[name] = {
            "labels": labels.astype(int),
            "area_codes": df_clean.index.tolist(),
        }

    return clusterings


def build_coassociation_matrix(
    clusterings: Dict[str, dict],
    all_indices: List[str],
) -> np.ndarray:
    """
    Build an n x n co-association matrix from per-view clusterings.

    S_ij = (# of views where i and j appear AND share the same cluster) /
           (# of views where i and j both appear)

    If a pair never co-appears in any view, S_ij = 0.

    Parameters
    ----------
    clusterings : dict[str, dict]
        Output of cluster_per_view.
    all_indices : list[str]
        Sorted union of all Local_Authority_Code across views.

    Returns
    -------
    S : np.ndarray
        Symmetric (n x n) co-association similarity in [0, 1].
    """
    n = len(all_indices)
    index_to_pos = {code: i for i, code in enumerate(all_indices)}
    agree = np.zeros((n, n), dtype=float)
    count = np.zeros((n, n), dtype=float)

    for clus in clusterings.values():
        codes = clus["area_codes"]
        labels = clus["labels"]
        pos = np.array([index_to_pos[c] for c in codes], dtype=int)

        # increment counts for all co-present pairs
        count[np.ix_(pos, pos)] += 1.0

        # same-cluster mask
        same = (labels[:, None] == labels[None, :]).astype(float)
        agree[np.ix_(pos, pos)] += same

    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.true_divide(agree, count)
        S[~np.isfinite(S)] = 0.0  # where count == 0

    # symmetry & unit diagonal
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 1.0)
    return S


def cluster_from_coassociation(
    S: np.ndarray,
    index: List[str],
    k_range: Iterable[int] = range(4, 16),
    n_pca_range: Iterable[int] = range(2, 6),
    n_init: int | str = "auto",
    verbose: bool = True,
):
    """
    Grid-search KernelPCA (precomputed) + KMeans on a co-association similarity.

    Parameters
    ----------
    S : np.ndarray
        Co-association similarity (n x n), values in [0,1].
    index : list[str]
        Ordered Local_Authority_Code list.
    k_range : iterable[int], default=range(4, 16)
        K values for KMeans.
    n_pca_range : iterable[int], default=range(2, 6)
        Embedding dims for KernelPCA.
    n_init : int | str, default="auto"
        KMeans restarts.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    best : dict
        {
          "embedding": np.ndarray (n x n_pca_best),
          "labels": np.ndarray (n,),
          "sil": float,
          "params": (n_pca_best, k_best),
          "result_df": pd.DataFrame[Local_Authority_Code, Cluster]
        }
    results_df : pd.DataFrame
        Grid rows: [PCA_n_components, K, Silhouette]
    """
    # Ensure symmetry and positive semidefiniteness for kernel usage
    S = (S + S.T) / 2.0
    # PSD projection (eigenvalue clipping)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-8, None)
    S_psd = (V * w) @ V.T

    rows = []
    best = {"sil": -1.0}

    for n_pca in n_pca_range:
        kpca = KernelPCA(
            n_components=n_pca,
            kernel="precomputed",
            random_state=19042022,
            remove_zero_eig=True,
        )
        embedding = kpca.fit_transform(S_psd)

        for k in k_range:
            if k < 2 or k > embedding.shape[0]:
                if verbose:
                    print(f"Skipping PCA={n_pca}, K={k} (n={embedding.shape[0]})")
                continue

            km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=19042022)
            labels = km.fit_predict(embedding)

            try:
                sil = silhouette_score(embedding, labels)
            except ValueError:
                sil = -1.0

            if verbose:
                print(f"n={len(index)}, PCA={n_pca}, K={k} → Silhouette: {sil:.4f}")

            rows.append({"PCA_n_components": n_pca, "K": k, "Silhouette": sil})

            if sil > best["sil"]:
                best = {
                    "embedding": embedding,
                    "labels": labels,
                    "sil": sil,
                    "params": (n_pca, k),
                    "result_df": pd.DataFrame(
                        {"Local_Authority_Code": index, "Cluster": labels}
                    ),
                }

    results_df = pd.DataFrame(rows)
    if verbose and best["sil"] >= 0:
        npca, kk = best["params"]
        print(f"\nBest: PCA={npca}, K={kk}, Silhouette={best['sil']:.4f}")

    return best, results_df


def run_late_coassoc_pca_kmeans(
    views: Dict[str, pd.DataFrame],
    ons_k_values: Dict[str, int],
    k_range: Iterable[int] = range(4, 16),
    n_pca_range: Iterable[int] = range(2, 6),
    n_init: int | str = "auto",
    verbose: bool = True,
):
    """
    One-call Late Integration:
      per-view KMeans → co-association similarity → KernelPCA → KMeans grid search.

    Parameters
    ----------
    views : dict[str, pd.DataFrame]
        Mapping view name -> DataFrame with 'Local_Authority_Code' and numeric columns.
    ons_k_values : dict[str, int]
        Lowercased view name -> k (cluster count) for that view.
    k_range : iterable[int], default=range(4, 16)
        KMeans K values to search on the embedded co-association.
    n_pca_range : iterable[int], default=range(2, 6)
        KernelPCA output dimensions to search.
    n_init : int | str, default="auto"
        KMeans restarts.
    verbose : bool, default=True
        Print progress and the best result.

    Returns
    -------
    out : dict
        {
          # Per-view results
          "clusterings": dict[str, {"labels": np.ndarray, "area_codes": list[str]}],

          # Co-association
          "S_coassoc": np.ndarray (n x n),
          "area_codes": list[str],             # union of all LAs, sorted

          # Clustering on coassoc
          "labels": np.ndarray (n,),
          "embedding": np.ndarray (n x n_pca_best),
          "silhouette": float,
          "best_params": tuple[int, int],      # (n_pca_best, k_best)
          "assignments_df": pd.DataFrame,      # LA → Cluster
          "grid_results": pd.DataFrame
        }
    """
    # Per-view KMeans
    clusterings = cluster_per_view(views, ons_k_values=ons_k_values, n_init=n_init, verbose=verbose)

    # Union of all area codes, sorted
    all_indices = sorted(set().union(*[c["area_codes"] for c in clusterings.values()]))

    # Co-association similarity
    S_coassoc = build_coassociation_matrix(clusterings, all_indices)

    # KernelPCA + KMeans grid search
    best, results_df = cluster_from_coassociation(
        S=S_coassoc,
        index=all_indices,
        k_range=k_range,
        n_pca_range=n_pca_range,
        n_init=n_init,
        verbose=verbose,
    )

    out = {
        "clusterings": clusterings,
        "S_coassoc": S_coassoc,
        "area_codes": all_indices,
        "labels": best["labels"],
        "embedding": best["embedding"],
        "silhouette": best["sil"],
        "best_params": best["params"],
        "assignments_df": best["result_df"],
        "grid_results": results_df,
    }
    return out