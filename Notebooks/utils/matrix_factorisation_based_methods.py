from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, normalize

from data_preprocessing import (
    get_combined_views_union,  # preferred union builder (returns X_union, index)
    _standardize_columns,
    preprocess_view,
    exclude_las,
    DEFAULT_META_COLS,
)

RANDOM_STATE = 19042022


# Masked Non-Negative Matrix Factorisation

def masked_nmf(
    X: np.ndarray,
    mask: np.ndarray,
    rank: int,
    max_iter: int = 200,
    tol: float = 1e-4,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mask-aware Nonnegative Matrix Factorization via multiplicative updates.

    Parameters
    ----------
    X : np.ndarray (n x m)
        Data array with NaNs replaced by 0.0 (values outside the mask are ignored).
    mask : np.ndarray (n x m)
        Boolean or {0,1} matrix, where 1 = observed, 0 = missing.
    rank : int
        Target rank (latent dimension k).
    max_iter : int, default=200
        Maximum number of multiplicative updates.
    tol : float, default=1e-4
        Relative improvement tolerance on masked Frobenius objective for early stopping.
    random_state : int, default=RANDOM_STATE
        RNG seed for reproducible initialization.

    Returns
    -------
    G : np.ndarray (n x k)
        Nonnegative row factors (embedding of samples).
    H : np.ndarray (m x k)
        Nonnegative column factors (loadings of variables).
    """
    rng = np.random.default_rng(random_state)
    eps = 1e-8

    X = np.asarray(X, dtype=float)
    M = (mask.astype(float) if mask.dtype != float else mask).copy()

    n, m = X.shape
    # Nonnegative random init
    G = np.abs(rng.random((n, rank)))
    H = np.abs(rng.random((m, rank)))

    MX = M * X
    prev_obj = np.inf

    for it in range(max_iter):
        X_hat = G @ H.T
        X_hat = np.clip(X_hat, eps, None)  # avoid zero denominators

        # Updates
        numer_G = MX @ H
        denom_G = (M * X_hat) @ H + eps
        G *= numer_G / denom_G

        numer_H = MX.T @ G
        denom_H = (M * X_hat).T @ G + eps
        H *= numer_H / denom_H

        # Early stopping check (every 10 iterations)
        if (it + 1) % 10 == 0 or it == max_iter - 1:
            X_hat = G @ H.T
            obj = float(np.sqrt(np.sum((M * (X - X_hat)) ** 2)))
            if abs(prev_obj - obj) <= tol * max(1.0, prev_obj):
                break
            prev_obj = obj

    # Sanitize any numerical artifacts
    G[~np.isfinite(G)] = 0.0
    H[~np.isfinite(H)] = 0.0
    return G, H


def _sanitize_for_pca(G: np.ndarray) -> np.ndarray:
    """
    Ensure no NaN/Inf before PCA. Sklearn PCA will center internally.
    """
    Z = np.asarray(G, dtype=float).copy()
    Z[~np.isfinite(Z)] = 0.0
    return Z


def _cap_pca_components(Z: np.ndarray, requested_range: Iterable[int]) -> List[int]:
    """
    Return a list of valid n_components values based on numerical rank and array shape.
    """
    n, d = Z.shape
    try:
        r = int(np.linalg.matrix_rank(Z))
    except Exception:
        r = min(n, d)
    max_pca = max(1, min(n, d, r))
    return [int(p) for p in requested_range if 0 < int(p) <= max_pca] or [min(2, max_pca)]


def _make_psd(K: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Project a symmetric matrix onto the PSD cone by clipping negative eigenvalues.
    """
    K = (K + K.T) / 2.0
    w, V = np.linalg.eigh(K)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T

# Early integration: Masked NMF (MNMF)

def run_early_mmf_pca_kmeans(
    views: Dict[str, pd.DataFrame],
    rank_range: Iterable[int] = range(2, 6),
    n_pca_range: Iterable[int] = range(2, 6),
    k_range: Iterable[int] = range(4, 16),
    nmf_max_iter: int = 200,
    nmf_tol: float = 1e-4,
    n_init: int | str = "auto",
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
):
    """
    Early integration pipeline using Masked NMF on the union feature matrix (with NaNs),
    followed by PCA and KMeans grid search scored by silhouette.

    Returns a dict with unified keys:
      "S_coassoc", "area_codes", "labels", "embedding",
      "assignments_df", "grid_results", "clusterings"
    """
    # 1) Build union feature matrix
    X_union, area_codes = get_combined_views_union(views)
    X_values = X_union.values.astype(float)

    # 2) Mask + zero-fill for MNMF
    mask = ~np.isnan(X_values)
    X_filled = np.nan_to_num(X_values, nan=0.0)

    best_score = -1.0
    best = {}

    # Keep only ONE row per (n_pca, k): best across ranks
    best_by_pk: dict[tuple[int, int], dict] = {}

    for rank in rank_range:
        # 3) MNMF → G, H
        G, H = masked_nmf(
            X=X_filled,
            mask=mask,
            rank=int(rank),
            max_iter=nmf_max_iter,
            tol=nmf_tol,
            random_state=random_state,
        )

        # 4) Sanitize for PCA and derive latent dim
        Z = _sanitize_for_pca(G)          # <-- THIS was missing before
        latent_dim = Z.shape[1]

        for n_pca in n_pca_range:
            # Guard: cannot ask PCA for more components than latent_dim
            if n_pca > latent_dim:
                if verbose:
                    print(f"Skipping Rank={rank}, PCA={n_pca} (Z dim={latent_dim})")
                continue

            emb = PCA(n_components=int(n_pca), random_state=random_state).fit_transform(Z)

            for k in k_range:
                # Guard: valid k
                if k < 2 or k > emb.shape[0]:
                    if verbose:
                        print(f"Skipping Rank={rank}, PCA={n_pca}, K={k} (n={emb.shape[0]})")
                    continue

                km = KMeans(n_clusters=int(k), init="k-means++", n_init=n_init, random_state=random_state)
                labels = km.fit_predict(emb)

                try:
                    sil = silhouette_score(emb, labels)
                except ValueError:
                    sil = -1.0

                if verbose:
                    print(f"n={len(area_codes)}, Rank={rank}, PCA={n_pca}, K={k} → Silhouette={sil:.4f}")

                # Dedup grid: keep best silhouette for this (n_pca, k), store winning Rank
                key = (int(n_pca), int(k))
                if (key not in best_by_pk) or (sil > best_by_pk[key]["Silhouette"]):
                    best_by_pk[key] = {"Silhouette": float(sil), "Rank": int(rank)}

                # Track global best as before
                if sil > best_score:
                    best_score = sil
                    best = {
                        "rank": int(rank),
                        "n_pca": int(n_pca),
                        "k": int(k),
                        "embedding": emb,
                        "labels": labels,
                        "silhouette": float(sil),
                        "assignments_df": pd.DataFrame(
                            {"Local_Authority_Code": area_codes, "Cluster": labels}
                        ),
                        "X_union": X_union,
                        "area_codes": list(area_codes),
                    }

    # Build deduped grid results: one row per (PCA_n_components, K) with the winning Rank
    grid_results = pd.DataFrame(
        [
            {"PCA_n_components": p, "K": k, "Silhouette": rec["Silhouette"], "Rank": rec["Rank"]}
            for (p, k), rec in best_by_pk.items()
        ]
    ).sort_values(["PCA_n_components", "K"], ignore_index=True)

    if best:
        if verbose:
            print(
                f"\nBest (EARLY-MMF): Rank={best['rank']}, PCA={best['n_pca']}, "
                f"K={best['k']}, Silhouette={best['silhouette']:.4f}"
            )
        return {
            "area_codes": best["area_codes"],
            "labels": best["labels"],
            "embedding": best["embedding"],
            "assignments_df": best["assignments_df"],
            "grid_results": grid_results,   # deduped by (n_pca, k) with Rank column
        }
    else:
        if verbose:
            print("\nNo valid clustering result found (EARLY-MMF).")
        return {
            "area_codes": list(X_union.index) if 'X_union' in locals() else [],
            "labels": np.array([]),
            "embedding": np.empty((0, 0)),
            "assignments_df": pd.DataFrame(columns=["Local_Authority_Code", "Cluster"]),
            "grid_results": grid_results,   # may be empty
        }


# Intermediate integration: Per-View NMF + concatenated G matrices (PVNMF)

def run_intermediate_pvnmf_pca_kmeans(
    views: Dict[str, pd.DataFrame],
    n_components: int = 2,                            # default per-view NMF rank if no range is provided
    n_components_range: Iterable[int] | None = None,
    rank_range: Iterable[int] | None = None,
    n_pca_range: Iterable[int] = range(2, 6),
    k_range: Iterable[int] = range(4, 16),
    n_init: int | str = "auto",
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
):
    """
    Intermediate integration via per-view NMF (PVNMF) with deduped grid by (n_pca, k).

    Returns extras needed by the factorization report:
      - 'view_names'
      - 'per_view_embeddings'           : list[np.ndarray], each shape (n, r_view) aligned to area_codes
      - 'similarity_matrices'           : list[np.ndarray], cosine similarities per view (n x n), diag=1
      - 'S_fused'                       : (n x n) central similarity = average of per-view similarities
      - 'per_view_contributions_per_la' : (n x V), rows sum to 1
      - 'best_params'                   : {'Rank', 'n_pca', 'k'}
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Backwards-compatible rank range resolution
    if rank_range is not None:
        ranks_to_try = list(rank_range)
    elif n_components_range is not None:
        ranks_to_try = list(n_components_range)
    else:
        ranks_to_try = [n_components]

    from sklearn.decomposition import NMF
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity

    best_global = {"sil": -1.0}
    best_by_pk: dict[tuple[int, int], dict] = {}  # (n_pca, k) -> {"Silhouette": float, "Rank": int}

    best_area_codes: list[str] | None = None
    best_assignments_df: pd.DataFrame | None = None
    best_embedding: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    best_rank: int | None = None

    # We will also retain artifacts for the best (global) configuration to feed the report
    best_view_names: list[str] | None = None
    best_per_view_embeddings: list[np.ndarray] | None = None
    best_similarity_matrices: list[np.ndarray] | None = None
    best_S_fused: np.ndarray | None = None
    best_contribs: np.ndarray | None = None

    for r in ranks_to_try:
        # Per-view NMF for rank r
        G_frames = []
        per_view_latents = []     # aligned per-view G_v (as np.ndarray)
        view_names = []
        area_codes_union = set()

        for view_name, df in views.items():
            df = _standardize_columns(df.copy())
            df_clean = preprocess_view(df, meta_cols=DEFAULT_META_COLS)
            df_clean = exclude_las(df_clean.copy(), view_name)

            if "Local_Authority_Code" not in df_clean.columns:
                raise KeyError(f"[{view_name}] 'Local_Authority_Code' missing after preprocessing.")
            df_clean = df_clean.set_index("Local_Authority_Code")

            # Numeric only; drop rows with any NaN for this view
            df_clean = df_clean.drop(columns=[c for c in DEFAULT_META_COLS if c in df_clean.columns], errors="ignore")
            df_clean = df_clean.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

            if df_clean.empty:
                if verbose:
                    print(f"[{view_name}] skipped: empty after cleaning.")
                continue

            X = df_clean.values
            if (X < 0).any():
                if verbose:
                    print(f"[{view_name}] skipped: contains negatives (NMF requires nonnegative).")
                continue

            r_view = min(int(r), min(X.shape))
            try:
                model = NMF(n_components=r_view, init="nndsvda", random_state=random_state)
                G_v = model.fit_transform(X)
            except ValueError:
                model = NMF(n_components=r_view, init="random", random_state=random_state)
                G_v = model.fit_transform(X)

            cols = [f"{view_name}_comp_{i+1}" for i in range(r_view)]
            G_df = pd.DataFrame(G_v, index=df_clean.index, columns=cols).sort_index()
            G_frames.append(G_df)
            per_view_latents.append(G_df.values)  # aligned later
            view_names.append(view_name)
            area_codes_union.update(G_df.index)

        if not G_frames:
            if verbose:
                print(f"[PVNMF] No valid views processed for rank={r}. Skipping.")
            continue

        # Align to global area codes; build concatenated latent
        area_codes = sorted(area_codes_union)
        # align each view latent to area_codes
        aligned_latents = []
        for G_df in G_frames:
            G_al = G_df.reindex(area_codes).fillna(0.0).values
            aligned_latents.append(G_al)

        # Keep a list of per-view embeddings (aligned) for the report
        per_view_embeddings = [arr.copy() for arr in aligned_latents]

        # Concatenate across views (column-wise) and L2-normalize rows for clustering
        G_concat_df = pd.concat([g.reindex(area_codes) for g in G_frames], axis=1).fillna(0.0)
        G_concat = normalize(G_concat_df.values, norm="l2")

        # Build per-view cosine similarities and a central S_fused
        S_list = []
        for Z in per_view_embeddings:
            # cosine sim in [0,1] if Z nonnegative & nonzero; guard zeros
            S = cosine_similarity(Z)
            np.fill_diagonal(S, 1.0)
            S_list.append(S)
        # central similarity as simple average
        S_fused = np.mean(S_list, axis=0)
        S_fused = 0.5 * (S_fused + S_fused.T)
        np.fill_diagonal(S_fused, 1.0)

        # Per-LA per-view contributions (inverse deviation from central)
        # for each i, d_v = ||S_v[i,:] - S_fused[i,:]||_2; score_v = 1/(d_v+eps); contrib = score/sum(score)
        eps = 1e-12
        V = len(S_list)
        n = len(area_codes)
        contrib = np.zeros((n, V), dtype=float)
        for i in range(n):
            diffs = np.array([np.linalg.norm(Sv[i, :] - S_fused[i, :]) for Sv in S_list], dtype=float)
            scores = 1.0 / (diffs + eps)
            contrib[i, :] = scores / scores.sum()

        # PCA/KMeans grid for this rank r
        for n_pca in n_pca_range:
            emb = PCA(n_components=int(n_pca), random_state=random_state).fit_transform(G_concat)

            for k in k_range:
                if k < 2 or k > emb.shape[0]:
                    if verbose:
                        print(f"Skipping PVNMF r={r}, PCA={n_pca}, K={k} (n={emb.shape[0]})")
                    continue

                km = KMeans(n_clusters=int(k), init="k-means++", n_init=n_init, random_state=random_state)
                labels = km.fit_predict(emb)
                try:
                    sil = silhouette_score(emb, labels)
                except ValueError:
                    sil = -1.0

                if verbose:
                    print(f"n={len(area_codes)}, PVNMF Rank={r}, PCA={n_pca}, K={k} → Silhouette={sil:.4f}")

                # Dedup by (n_pca, k): keep best silhouette; store winning Rank
                key = (int(n_pca), int(k))
                if (key not in best_by_pk) or (sil > best_by_pk[key]["Silhouette"]):
                    best_by_pk[key] = {"Silhouette": float(sil), "Rank": int(r)}

                # Track global best and cache artifacts for the report
                if sil > best_global["sil"]:
                    best_global = {"sil": float(sil), "params": (int(n_pca), int(k)), "Rank": int(r)}
                    best_area_codes = area_codes
                    best_embedding = emb
                    best_labels = labels
                    best_assignments_df = pd.DataFrame(
                        {"Local_Authority_Code": area_codes, "Cluster": labels}
                    )
                    best_rank = int(r)

                    best_view_names = view_names
                    best_per_view_embeddings = per_view_embeddings
                    best_similarity_matrices = S_list
                    best_S_fused = S_fused
                    best_contribs = contrib

    # Deduped grid results
    grid_results = pd.DataFrame(
        [
            {"PCA_n_components": p, "K": k, "Silhouette": rec["Silhouette"], "Rank": rec["Rank"]}
            for (p, k), rec in best_by_pk.items()
        ]
    ).sort_values(["PCA_n_components", "K"], ignore_index=True)

    if best_area_codes is None:
        if verbose:
            print("\nNo valid clustering result found (INTERMEDIATE-PVNMF).")
        return {
            "area_codes": [],
            "labels": np.array([]),
            "embedding": np.empty((0, 0)),
            "assignments_df": pd.DataFrame(columns=["Local_Authority_Code", "Cluster"]),
            "grid_results": grid_results,
        }

    n_pca_best, k_best = best_global["params"]
    if verbose:
        print(f"\nBest (INTERMEDIATE-PVNMF): Rank={best_rank}, PCA={n_pca_best}, "
              f"K={k_best}, Silhouette={best_global['sil']:.4f}")

    # Final, safe packing for the factorization report
    out = {
        "area_codes": best_area_codes,
        "labels": best_labels,
        "embedding": best_embedding,
        "assignments_df": best_assignments_df,
        "grid_results": grid_results,  # one row per (n_pca, k), with Rank column
        "best_params": {"Rank": best_rank, "n_pca": n_pca_best, "k": k_best},

        # Extras used by plot_factorization_report:
        "view_names": best_view_names or [],
        "per_view_embeddings": best_per_view_embeddings or [],
        "similarity_matrices": best_similarity_matrices or [],
        "S_fused": best_S_fused if best_S_fused is not None else None,
        "per_view_contributions_per_la": best_contribs if best_contribs is not None else None,
    }
    return out



# Late integration: Co-association Factorization Ensemble (CFE)

def build_coassociation_matrix(clusterings: Dict[str, dict], all_indices: List[str]) -> np.ndarray:
    """
    Build an n x n co-association matrix from per-view clusterings.

    S_ij = (# of views where i and j appear AND share the same cluster) /
           (# of views where i and j both appear)
    Pairs that never co-appear have 0 similarity.
    """
    n = len(all_indices)
    idx = {code: i for i, code in enumerate(all_indices)}
    agree = np.zeros((n, n), dtype=float)
    count = np.zeros((n, n), dtype=float)

    for clus in clusterings.values():
        codes = clus["area_codes"]
        labels = clus["labels"]
        pos = np.array([idx[c] for c in codes], dtype=int)
        count[np.ix_(pos, pos)] += 1.0
        same = (labels[:, None] == labels[None, :]).astype(float)
        agree[np.ix_(pos, pos)] += same

    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.true_divide(agree, count)
        S[~np.isfinite(S)] = 0.0
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 1.0)
    return S


def run_late_nmf_coassoc_pca_kmeans(
    views: Dict[str, pd.DataFrame],
    n_components: int = 2,
    n_components_range: Iterable[int] | None = None,
    rank_range: Iterable[int] | None = None,
    n_pca_range: Iterable[int] = range(2, 6),
    k_range: Iterable[int] = range(4, 16),
    n_init: int | str = "auto",
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
):
    """
    Late integration via NMF + Co-association + PCA/KMeans.
    Dedupes results so grid_results has one row per (n_pca, k), with winning Rank.
    """

    # Backwards-compatible rank range resolution
    if rank_range is not None:
        ranks_to_try = list(rank_range)
    elif n_components_range is not None:
        ranks_to_try = list(n_components_range)
    else:
        ranks_to_try = [n_components]

    best_global = {"sil": -1.0}
    best_by_pk: dict[tuple[int, int], dict] = {}  # (n_pca, k) -> {"Silhouette": float, "Rank": int}

    best_area_codes: list[str] | None = None
    best_assignments_df: pd.DataFrame | None = None
    best_embedding: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    best_S_coassoc: np.ndarray | None = None
    best_clusterings: dict | None = None

    for r in ranks_to_try:
        # Per-view NMF clusterings
        clusterings = {}
        all_indices = set()

        for view_name, df in views.items():
            df = _standardize_columns(df.copy())
            df_clean = preprocess_view(df, meta_cols=DEFAULT_META_COLS)
            df_clean = exclude_las(df_clean.copy(), view_name)

            if "Local_Authority_Code" not in df_clean.columns:
                raise KeyError(f"[{view_name}] 'Local_Authority_Code' missing after preprocessing.")
            df_clean = df_clean.set_index("Local_Authority_Code")

            df_clean = df_clean.drop(columns=[c for c in DEFAULT_META_COLS if c in df_clean.columns], errors="ignore")
            df_clean = df_clean.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

            if df_clean.empty:
                if verbose:
                    print(f"[{view_name}] skipped: empty after cleaning.")
                continue

            X = df_clean.values
            if (X < 0).any():
                if verbose:
                    print(f"[{view_name}] skipped: negatives present (NMF requires nonnegatives).")
                continue

            r_view = min(int(r), min(X.shape))
            try:
                from sklearn.decomposition import NMF
                model = NMF(n_components=r_view, init="nndsvda", random_state=random_state)
                G_v = model.fit_transform(X)
            except ValueError:
                model = NMF(n_components=r_view, init="random", random_state=random_state)
                G_v = model.fit_transform(X)

            km = KMeans(n_clusters=r_view, n_init=n_init, random_state=random_state)
            labels = km.fit_predict(G_v)

            clusterings[view_name] = {"labels": labels, "area_codes": df_clean.index.tolist()}
            all_indices.update(df_clean.index)

        if not clusterings:
            if verbose:
                print(f"[LATE-NMF] No valid views processed for rank={r}. Skipping.")
            continue

        all_codes = sorted(all_indices)

        # Build co-association similarity
        S_coassoc = build_coassociation_matrix(clusterings, all_codes)

        # PCA + KMeans grid
        # Center S for kernel PCA
        n = S_coassoc.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K_centered = H @ S_coassoc @ H

        for n_pca in n_pca_range:
            emb = PCA(n_components=int(n_pca), random_state=random_state).fit_transform(K_centered)

            for k in k_range:
                if k < 2 or k > emb.shape[0]:
                    if verbose:
                        print(f"Skipping LATE-NMF r={r}, PCA={n_pca}, K={k} (n={emb.shape[0]})")
                    continue

                km = KMeans(n_clusters=int(k), n_init=n_init, random_state=random_state)
                labels = km.fit_predict(emb)
                try:
                    sil = silhouette_score(emb, labels)
                except ValueError:
                    sil = -1.0

                if verbose:
                    print(f"n={len(all_codes)}, LATE-NMF Rank={r}, PCA={n_pca}, K={k} → Silhouette={sil:.4f}")

                key = (int(n_pca), int(k))
                if (key not in best_by_pk) or (sil > best_by_pk[key]["Silhouette"]):
                    best_by_pk[key] = {"Silhouette": float(sil), "Rank": int(r)}

                if sil > best_global["sil"]:
                    best_global = {"sil": float(sil), "params": (int(n_pca), int(k)), "Rank": int(r)}
                    best_area_codes = all_codes
                    best_embedding = emb
                    best_labels = labels
                    best_assignments_df = pd.DataFrame(
                        {"Local_Authority_Code": all_codes, "Cluster": labels}
                    )
                    best_S_coassoc = S_coassoc
                    best_clusterings = clusterings

    # Deduped grid results
    grid_results = pd.DataFrame(
        [
            {"PCA_n_components": p, "K": k, "Silhouette": rec["Silhouette"], "Rank": rec["Rank"]}
            for (p, k), rec in best_by_pk.items()
        ]
    ).sort_values(["PCA_n_components", "K"], ignore_index=True)

    if best_area_codes is None:
        if verbose:
            print("\nNo valid clustering result found (LATE-NMF).")
        return {
            "area_codes": [],
            "labels": np.array([]),
            "embedding": np.empty((0, 0)),
            "assignments_df": pd.DataFrame(columns=["Local_Authority_Code", "Cluster"]),
            "grid_results": grid_results,
        }

    n_pca_best, k_best = best_global["params"]
    if verbose:
        print(f"\nBest (LATE-NMF): Rank={best_global['Rank']}, PCA={n_pca_best}, K={k_best}, Silhouette={best_global['sil']:.4f}")

    return {
        "S_coassoc": best_S_coassoc,
        "area_codes": best_area_codes,
        "labels": best_labels,
        "embedding": best_embedding,
        "assignments_df": best_assignments_df,
        "grid_results": grid_results,
        "clusterings": best_clusterings,
    }
    
