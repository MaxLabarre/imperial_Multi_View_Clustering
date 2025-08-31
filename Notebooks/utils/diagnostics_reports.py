fromyoufrom __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_samples, silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Abbreviation helper for long view names
def _shorten_view_name(name: str) -> str:
    if name == "educational_attainment":
        return "edu.attainment"
    return name

def _shorten_names(names: List[str]) -> List[str]:
    return [_shorten_view_name(n) for n in names]

# Missingness helpers (defaults)
def _default_compute_global_missingness(views_dict: Dict[str, pd.DataFrame]) -> pd.Series:
    frames = []
    for df in views_dict.values():
        if "Area Code" not in df.columns:
            raise KeyError("Expected 'Area Code' in each view DataFrame.")
        frames.append(df.set_index("Area Code"))
    combined = pd.concat(frames, axis=1)
    return combined.isna().mean(axis=1)

def _default_stratify_missingness(missingness: pd.Series, method="quantiles") -> pd.Series:
    if method == "quantiles":
        ranked = missingness.rank(method="average") / len(missingness)
        return pd.cut(ranked, bins=[0, 0.25, 0.5, 0.75, 1.0],
                      labels=["Q1", "Q2", "Q3", "Q4"], include_lowest=True)
    elif method == "fixed":
        bins = [0, 0.05, 0.2, 1.0]
        labels = ["Low", "Medium", "High"]
        return pd.cut(missingness, bins=bins, labels=labels, include_lowest=True)
    else:
        raise ValueError(f"Unsupported stratification method: {method}")

# Helper: robustly extract (S, area_codes) from fusion output
def _extract_S_and_index(fusion_ret):
    """
    Accepts tuple/list/dict returns from multiview_similarity_fusion and
    extracts (S, area_codes). Works with:
      - (S, area_codes)
      - (S, area_codes, ...)
      - {"S_fused": S, "area_codes": idx, ...}
      - {"S": S, "index": idx}  (fallback)
    """
    # dict-like
    if isinstance(fusion_ret, dict):
        if "S_fused" in fusion_ret and "area_codes" in fusion_ret:
            return fusion_ret["S_fused"], fusion_ret["area_codes"]
        if "S" in fusion_ret and ("area_codes" in fusion_ret or "index" in fusion_ret):
            return fusion_ret["S"], fusion_ret.get("area_codes", fusion_ret.get("index"))
        # try common alternatives
        for s_key in ("S_fused", "S", "similarity", "S_coassoc"):
            if s_key in fusion_ret:
                S = fusion_ret[s_key]
                break
        else:
            raise ValueError("Could not find similarity matrix in fusion return dict.")
        for i_key in ("area_codes", "index", "indices", "ids"):
            if i_key in fusion_ret:
                idx = fusion_ret[i_key]
                break
        else:
            raise ValueError("Could not find index/area_codes in fusion return dict.")
        return S, idx

    # tuple/list-like
    if isinstance(fusion_ret, (tuple, list)):
        if len(fusion_ret) < 2:
            raise ValueError("Fusion function must return at least (S, area_codes).")
        return fusion_ret[0], fusion_ret[1]

    raise TypeError("Unsupported fusion return type; expected dict or (tuple/list).")

# Helper: robustly extract best params from grid_results DataFrame
def _get_best_params_from_grid(grid: pd.DataFrame):
    """Robustly extract best params from a grid_results df."""
    metric_candidates = ["silhouette", "Silhouette", "Silhouette_consensus", "Silhouette_embedding", "score"]
    metric_col = next((c for c in metric_candidates if c in grid.columns), None)
    if metric_col is None:
        raise KeyError(f"No metric column found in grid. Available: {list(grid.columns)}")

    def pick(cands): 
        return next((c for c in cands if c in grid.columns), None)

    k_col   = pick(["k", "K"])
    pca_col = pick(["n_pca", "PCA_n_components", "PCA"])
    r_col   = pick(["rank", "Rank", "n_components"])

    best_row = grid.sort_values(metric_col, ascending=False).iloc[0]
    return {
        "metric_col": metric_col,
        "metric": float(best_row[metric_col]),
        "k": int(best_row[k_col]) if k_col else None,
        "n_pca": int(best_row[pca_col]) if pca_col else None,
        "rank": int(best_row[r_col]) if r_col else None,
        "row": best_row,
    }

# Cross-view agreement (NMI & ARI)
def compute_clustering_agreement_both(
    similarity_matrices=None,
    view_names=None,
    n_components: int = 3,
    k_clusters: int = 5,
    labels_per_view: dict | None = None,
):
    """
    Compute pairwise NMI/ARI between views.

    Two modes:
      • labels_per_view provided  -> use those labels directly (no PCA/KMeans).
      • else use similarity_matrices + PCA + KMeans to derive labels per view.
    """
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    if labels_per_view is not None:
        # Direct mode: labels_per_view = {view_name: (n,) int array}
        view_names = list(labels_per_view.keys()) if view_names is None else view_names
        labels_dict = {vn: np.asarray(labels_per_view[vn]) for vn in view_names}
    else:
        # Similarity mode: derive per-view labels via PCA->KMeans
        if similarity_matrices is None or view_names is None:
            raise ValueError("Provide labels_per_view OR (similarity_matrices AND view_names).")
        mats = [similarity_matrices[vn] for vn in view_names]
        labels_dict = {}
        for vn, S in zip(view_names, mats):
            n = S.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            Kc = H @ ((S + S.T) / 2.0) @ H
            emb = PCA(n_components=n_components, random_state=19042022).fit_transform(Kc)
            lab = KMeans(n_clusters=k_clusters, n_init=1000, random_state=19042022).fit_predict(emb)
            labels_dict[vn] = lab

    # Build NMI/ARI matrices
    nmi = pd.DataFrame(index=view_names, columns=view_names, dtype=float)
    ari = pd.DataFrame(index=view_names, columns=view_names, dtype=float)
    for i, vi in enumerate(view_names):
        for j, vj in enumerate(view_names):
            nmi.iloc[i, j] = normalized_mutual_info_score(labels_dict[vi], labels_dict[vj])
            ari.iloc[i, j] = adjusted_rand_score(labels_dict[vi], labels_dict[vj])
    return nmi, ari


# Flexible helpers for reading many grid formats
def _pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _infer_best_from_grid(grid: pd.DataFrame) -> dict:
    """
    Infer best params from a grid_results DataFrame with flexible column names.
    Returns dict with keys: k, n_pca (may be None), rank (may be None), metric.
    """
    # Column names that may appear
    metric_col = _pick_first_col(grid, ["Silhouette", "silhouette", "silhouette_score"])
    if metric_col is None:
        raise KeyError("grid_results has no silhouette metric column.")

    # Find best row
    best_idx = grid[metric_col].astype(float).idxmax()
    best_row = grid.loc[best_idx]

    # Extract params with fallbacks
    k_col     = _pick_first_col(grid, ["K", "k"])
    pca_col   = _pick_first_col(grid, ["PCA_n_components", "n_pca", "pca_dim", "Embedding Dim"])
    rank_col  = _pick_first_col(grid, ["Rank", "rank", "Latent Dim", "latent_dim"])

    best = {
        "k":     int(best_row[k_col]) if k_col is not None else None,
        "n_pca": int(best_row[pca_col]) if pca_col is not None else None,
        "rank":  int(best_row[rank_col]) if rank_col is not None else None,
        "metric": float(best_row[metric_col]),
    }
    return best

def _infer_best_params_from_out(method_out: dict) -> dict:
    """
    Try best_params first; otherwise derive from grid_results.
    Normalizes keys to: k, n_pca, rank, metric.
    """
    # If a normalized best_params already exists, reuse it
    if "best_params" in method_out and isinstance(method_out["best_params"], dict):
        bp = method_out["best_params"]
        # harmonize keys and add metric if missing
        k     = bp.get("k") or bp.get("K")
        n_pca = bp.get("n_pca") or bp.get("PCA_n_components") or bp.get("pca_dim") or bp.get("Embedding Dim")
        rank  = bp.get("rank") or bp.get("Rank") or bp.get("Latent Dim") or bp.get("latent_dim")
        metric = bp.get("sil") or bp.get("silhouette") or bp.get("silhouette_score") or bp.get("metric")

        out = {
            "k": int(k) if k is not None else None,
            "n_pca": int(n_pca) if n_pca is not None else None,
            "rank": int(rank) if rank is not None else None,
            "metric": float(metric) if metric is not None else None,
        }
        # if metric still missing but we have embedding+labels, compute it
        if out["metric"] is None and ("embedding" in method_out and "labels" in method_out):
            try:
                out["metric"] = float(silhouette_score(method_out["embedding"], method_out["labels"]))
            except Exception:
                pass
        # If k/rank/n_pca still missing, try falling back to grid
        if (out["k"] is None or out["rank"] is None or out["n_pca"] is None) and ("grid_results" in method_out):
            inferred = _infer_best_from_grid(method_out["grid_results"])
            out["k"]     = out["k"]     if out["k"]     is not None else inferred["k"]
            out["n_pca"] = out["n_pca"] if out["n_pca"] is not None else inferred["n_pca"]
            out["rank"]  = out["rank"]  if out["rank"]  is not None else inferred["rank"]
            out["metric"] = out["metric"] if out["metric"] is not None else inferred["metric"]
        return out

    # Else infer directly from grid_results
    if "grid_results" in method_out and isinstance(method_out["grid_results"], pd.DataFrame):
        return _infer_best_from_grid(method_out["grid_results"])

    # Final fallback: if we have embedding+labels, at least return metric
    if "embedding" in method_out and "labels" in method_out:
        try:
            m = float(silhouette_score(method_out["embedding"], method_out["labels"]))
        except Exception:
            m = None
        return {"k": None, "n_pca": None, "rank": None, "metric": m}

    raise KeyError("Cannot infer best parameters: no best_params, no usable grid_results, and no embedding+labels.")

def ablation_importance_from_out(
    method_out: dict,
    views: dict,
    mode: str = "view",
    multiview_similarity_fusion=None,   # similarity path
    cluster_from_similarity=None,       # similarity path
    run_intermediate_fn=None,           # (unused here)
    run_late_fn=None,                   # LATE pipeline path
    n_init: int = 100,
    verbose: bool = True,
    # optional overrides; if None we try to infer once from `method_out`
    k: int | None = None,
    n_pca: int | None = None,
    rank: int | None = None,
) -> pd.DataFrame:
    """
    Drop-one-view ablation that supports:
      • similarity path (fusion + cluster_from_similarity)
      • LATE path (run_late_fn)

    Interfaces expected:
      SIMILARITY PATH
        multiview_similarity_fusion(views_subset) -> {"S_fused": S (n×n), "area_codes": list}
        cluster_from_similarity(S, area_codes, k_range=[k], n_pca_range=[n_pca], ...) -> (best, grid_df)
        grid_df must have a metric column: one of ["Silhouette","silhouette","score","Metric","metric"].

      LATE PATH
        run_late_fn(views=views_subset, rank_range=[rank], n_pca_range=[n_pca], k_range=[k], n_init=..., verbose=False)
          -> out (dict or tuple) that contains either:
             - out["metric"]  (float), or
             - out["grid_results"] DataFrame with one of the metric columns above.
    """
    # helpers (no array truthiness)
    def _best_metric_from_grid(grid: pd.DataFrame) -> float | None:
        if isinstance(grid, pd.DataFrame) and not grid.empty:
            for col in ("Silhouette", "silhouette", "score", "Metric", "metric"):
                if col in grid.columns:
                    s = pd.to_numeric(grid[col], errors="coerce")
                    if s.notna().any():
                        return float(s.max())
        return None

    def _metric_from_out(out) -> float | None:
        """Try out['metric'] first; else look for a grid DF inside common keys."""
        if isinstance(out, dict):
            m = out.get("metric", None)
            if isinstance(m, (int, float)) and not pd.isna(m):
                return float(m)
            for key in ("grid_results", "grid", "results"):
                if key in out and isinstance(out[key], pd.DataFrame):
                    bm = _best_metric_from_grid(out[key])
                    if bm is not None:
                        return bm
            return None
        if isinstance(out, (tuple, list)) and len(out) >= 2 and isinstance(out[1], pd.DataFrame):
            return _best_metric_from_grid(out[1])
        return None

    def _labels_and_grid(obj):
        """For similarity path: return (labels, grid_df_or_none)."""
        if isinstance(obj, dict):
            labels = obj.get("labels")
            grid = None
            for key in ("grid", "grid_results", "results"):
                if key in obj:
                    grid = obj[key]
                    break
            return labels, grid
        if isinstance(obj, (tuple, list)):
            if len(obj) >= 2:
                return obj[0], obj[1]
            return obj[0], None
        return obj, None

    # choose effective params (overrides > inferred)
    try:
        inferred = _infer_best_params_from_out(method_out)  # helper, if defined in this module
    except Exception:
        inferred = {}
    k_eff     = int(k)     if k     is not None else inferred.get("k")
    n_pca_eff = int(n_pca) if n_pca is not None else inferred.get("n_pca")
    rank_eff  = int(rank)  if rank  is not None else inferred.get("rank")

    # ===================== SIMILARITY PATH =====================
    if multiview_similarity_fusion is not None and cluster_from_similarity is not None:
        if k_eff is None or n_pca_eff is None:
            raise KeyError("k/n_pca not set. Pass k=..., n_pca=... or ensure method_out has best_params/grid_results.")

        base_ret = multiview_similarity_fusion(views)
        # use safe extractor if defined in this module; else expect dict
        try:
            S_full, acodes = _extract_S_and_index(base_ret)  # helper
        except Exception:
            if not (isinstance(base_ret, dict) and "S_fused" in base_ret and "area_codes" in base_ret):
                raise KeyError("multiview_similarity_fusion must return {'S_fused': S, 'area_codes': ...}.")
            S_full, acodes = base_ret["S_fused"], base_ret["area_codes"]

        _, grid_full = cluster_from_similarity(S_full, acodes, k_range=[k_eff], n_pca_range=[n_pca_eff], verbose=False)
        baseline = _best_metric_from_grid(grid_full)
        if baseline is None:
            raise ValueError("Could not read baseline metric from cluster_from_similarity grid.")

        if verbose:
            print(f"[ablation] similarity-based (k={k_eff}, n_pca={n_pca_eff}), baseline={baseline:.4f}")

        rows = []
        for vname in list(views.keys()):
            vsub = {vn: df for vn, df in views.items() if vn != vname}
            if not vsub:
                rows.append({"Item Removed": vname, "Silhouette": np.nan, "Δ vs. Baseline": np.nan})
                continue
            try:
                sub_ret = multiview_similarity_fusion(vsub)
                try:
                    S_sub, ac_sub = _extract_S_and_index(sub_ret)
                except Exception:
                    S_sub, ac_sub = sub_ret["S_fused"], sub_ret["area_codes"]
                _, grid_sub = cluster_from_similarity(S_sub, ac_sub, k_range=[k_eff], n_pca_range=[n_pca_eff], verbose=False)
                sil = _best_metric_from_grid(grid_sub)
            except Exception as e:
                if verbose:
                    print(f"[ablation] remove '{vname}' failed: {e}")
                sil = np.nan
            delta = sil - baseline if (isinstance(sil, (int, float)) and not pd.isna(sil)) else np.nan
            rows.append({"Item Removed": vname, "Silhouette": sil, "Δ vs. Baseline": delta})
        return pd.DataFrame(rows).sort_values("Δ vs. Baseline")

    # LATE PATH
    if run_late_fn is not None:
        # for LATE we need rank & k (and optionally n_pca if used in late runner)
        if k_eff is None and "k" in getattr(method_out, "get", lambda *_: {})("best_params", {}):
            k_eff = method_out["best_params"]["k"]
        if n_pca_eff is None and "n_pca" in getattr(method_out, "get", lambda *_: {})("best_params", {}):
            n_pca_eff = method_out["best_params"]["n_pca"]
        if rank_eff is None and "rank" in getattr(method_out, "get", lambda *_: {})("best_params", {}):
            rank_eff = method_out["best_params"]["rank"]

        if k_eff is None or rank_eff is None:
            raise KeyError("For late path, need 'rank' and 'k'. Pass overrides or set method_out['best_params'].")

        def _run(views_subset: dict):
            kwargs = {"views": views_subset, "n_init": n_init, "verbose": False}
            # only pass ranges that are known (late fn should accept these)
            if rank_eff is not None:
                kwargs["rank_range"] = [int(rank_eff)]
            if n_pca_eff is not None:
                kwargs["n_pca_range"] = [int(n_pca_eff)]
            if k_eff is not None:
                kwargs["k_range"] = [int(k_eff)]
            return run_late_fn(**kwargs)

        # Baseline on ALL views
        out_full = _run(views)
        baseline = _metric_from_out(out_full)
        if baseline is None:
            raise ValueError("run_late_fn must return either {'metric': float} or a grid DF with a metric column.")
        if verbose:
            print(f"[ablation] LATE (rank={rank_eff}, n_pca={n_pca_eff}, k={k_eff}), baseline={baseline:.4f}")

        # Drop-one-view
        rows = []
        for vname in list(views.keys()):
            vsub = {vn: df for vn, df in views.items() if vn != vname}
            if not vsub:
                rows.append({"Item Removed": vname, "Silhouette": np.nan, "Δ vs. Baseline": np.nan})
                continue
            try:
                out_sub = _run(vsub)
                sil = _metric_from_out(out_sub)
            except Exception as e:
                if verbose:
                    print(f"[ablation] remove '{vname}' failed: {e}")
                sil = None
            sil_val = float(sil) if isinstance(sil, (int, float)) and not pd.isna(sil) else np.nan
            delta = sil_val - baseline if not pd.isna(sil_val) else np.nan
            rows.append({"Item Removed": vname, "Silhouette": sil_val, "Δ vs. Baseline": delta})
        return pd.DataFrame(rows).sort_values("Δ vs. Baseline")

    raise ValueError("Provide either (multiview_similarity_fusion & cluster_from_similarity) or run_late_fn.")



# View Ablation bar plot
def plot_ablation_impact(df: pd.DataFrame, title: str = "Ablation Impact on Silhouette Score"):
    # Diverging colormap
    cmap = LinearSegmentedColormap.from_list("custom_diverging", ["#0202CD", "#FFFFFF", "#C71585"], N=256)

    vmin = df["Δ vs. Baseline"].min()
    vmax = df["Δ vs. Baseline"].max()
    max_abs = max(abs(vmin), abs(vmax))
    norm = mpl.colors.Normalize(vmin=-max_abs, vmax=+max_abs)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    bar_colors = [cmap(norm(val)) for val in df["Δ vs. Baseline"]]

    fig, ax = plt.subplots(figsize=(10, 0.4 * max(1, len(df))))
    sns.barplot(data=df, x="Δ vs. Baseline", y="Item Removed", palette=bar_colors, edgecolor="black", ax=ax)
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel("Change in Silhouette Score")
    ax.set_ylabel("Removed View or Feature")
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Δ vs. Baseline")

    plt.tight_layout()
    plt.show()
    return fig

# Main Silhouette Report
def plot_silhouette_report(
    embedding: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    views_dict: Optional[Dict[str, pd.DataFrame]] = None,
    area_codes: Optional[List[str]] = None,
    ablation_df_view: Optional[pd.DataFrame] = None,
    similarity_matrices: Optional[Dict[str, np.ndarray]] = None,
    labels_per_view: Optional[Dict[str, np.ndarray]] = None,
    view_names: Optional[List[str]] = None,
    views_palette: Optional[List[str]] = None,
    clusters_palette: Optional[List[str]] = None,
    strat_method: str = "quantiles",
    figsize: Tuple[int, int] = (14, 9),
    dpi: int = 300,
    compute_global_missingness_fn=None,
    stratify_missingness_fn=None,
    nmi_ari_components: int = 3,
    nmi_ari_k: int = 5,
    nmi_ari_min_overlap: int = 50,
    show_overlap_counts: bool = False,
    *,
    method_out: Optional[Dict] = None,
    # Optional: compute ablation here if given the callables
    multiview_similarity_fusion=None,
    cluster_from_similarity=None,
    box_ylim: tuple[float, float] | None = None,
):
    """
    Plot a 2x2 silhouette report: 
    (1) Grid search results (lineplot), 
    (2) silhouette histogram, 
    (3) per-view contributions (boxplot), 
    (4) scatter embedding.

    Parameters
    ----------
    grid_results : pd.DataFrame
        Grid search results with ["PCA_n_components","K","Silhouette"].
    best : dict
        Best clustering result from cluster_from_similarity.
    method_name : str
        Label for the method (used in titles).
    view_names : list[str], optional
        Names of the views for labeling contributions.
    per_view_contributions : pd.DataFrame, optional
        DataFrame with columns ["view","contribution"] for ablation/per-view importances.
    figsize : tuple[int, int], default=(14, 10)
        Size of the full figure.
    box_ylim : tuple[float, float], optional
        y-limits for the bottom-left boxplot. If None, matplotlib auto-scales.
    """
    # Extract from method_out if provided
    if method_out is not None:
        # embedding / labels / area_codes
        if embedding is None:
            embedding = method_out.get("embedding", None)
        if labels is None:
            labels = method_out.get("labels", None)
        if area_codes is None:
            area_codes = method_out.get("area_codes", None)

        # similarity matrices (MVSF and similar)
        if similarity_matrices is None:
            # try dict already
            sim_dict = method_out.get("similarity_matrices_dict", None)
            if sim_dict is None:
                # else try list + names
                sim_list = method_out.get("similarity_matrices", None)
                vnames = method_out.get("view_names", None)
                if sim_list is not None and vnames is not None:
                    similarity_matrices = {vn: S for vn, S in zip(vnames, sim_list)}
                else:
                    similarity_matrices = None
            else:
                similarity_matrices = sim_dict

        if view_names is None and method_out.get("view_names", None) is not None:
            view_names = method_out["view_names"]

        # if ablation df not provided but we can compute it here, do so
        if (ablation_df_view is None) and (views_dict is not None) and (multiview_similarity_fusion is not None) and (cluster_from_similarity is not None):
            ablation_df_view = ablation_importance_from_out(
                method_out,
                views=views_dict,
                multiview_similarity_fusion=multiview_similarity_fusion,
                cluster_from_similarity=cluster_from_similarity,
                mode="view",
                verbose=False
            )

    # Basic validation
    if embedding is None or labels is None or area_codes is None:
        raise ValueError("embedding, labels, and area_codes must be provided (explicitly or via method_out).")

    labels = np.asarray(labels)
    unique_clusters = sorted(np.unique(labels))
    if len(unique_clusters) < 2:
        raise ValueError("Silhouette requires at least 2 clusters.")

    if views_palette is None:
        views_palette = ["#4B0082", "#E55526", "#008080", "#708090", "#2253C3", "#FF99CC", "#1D0E2A", "#AFF9A2", "#FFFF00"]
    if clusters_palette is None:
        clusters_palette = ["#0202CD", "#C71585", "#006400", "#8B4513", "#1D0E2A", "#4B0082", "#E55526", "#008080",
                            "#708090", "#2253C3", "#FF99CC", "#1D0E2A", "#AFF9A2"]

    # cluster color mapping
    def cluster_color(c):
        idx = int(c) if str(c).isdigit() else unique_clusters.index(c)
        return clusters_palette[idx % len(clusters_palette)]

    # view color mapping: use view_names order
    if view_names is None and similarity_matrices is not None:
        view_names = list(similarity_matrices.keys())
    view_colors = {vn: views_palette[i % len(views_palette)] for i, vn in enumerate(view_names or [])}
    view_names_short = _shorten_names(view_names or [])

    # silhouettes on embedding
    sil_vals = silhouette_samples(embedding, labels)
    sil_avg = silhouette_score(embedding, labels)

    # missingness helpers
    compute_missing = compute_global_missingness_fn or _default_compute_global_missingness
    stratify_fn = stratify_missingness_fn or _default_stratify_missingness

    # figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], wspace=0.25, hspace=0.35)

    # Top-left: Silhouette plot
    ax1 = fig.add_subplot(gs[0, 0])
    y_lower = 10
    for c in unique_clusters:
        vals = np.sort(sil_vals[labels == c])
        size = len(vals)
        ax1.fill_betweenx(np.arange(y_lower, y_lower + size), 0, vals,
                          facecolor=cluster_color(c), edgecolor=cluster_color(c), alpha=0.8, linewidth=0)
        ax1.text(-0.05, y_lower + 0.5 * size, str(c), va="center", fontsize=9)
        y_lower += size + 10
    ax1.axvline(x=sil_avg, color="black", linestyle="--", linewidth=1)
    ax1.set_xlim([-0.1, 1.0])
    ax1.set_xlabel("Silhouette Coefficient")
    ax1.set_ylabel("Cluster")
    ax1.set_title(f"Silhouette Plot (avg={sil_avg:.3f})", fontsize=12, fontweight='bold')
    ax1.set_yticks([])

    # Top-right: View ablation (if provided/computed)
    ax2 = fig.add_subplot(gs[0, 1])
    if ablation_df_view is not None and not ablation_df_view.empty:
        dfv = ablation_df_view.copy()
        dfv = dfv.assign(Item_Short=dfv["Item Removed"].map(lambda v: _shorten_view_name(str(v))))
        dfv = dfv.sort_values("Δ vs. Baseline", ascending=False)
        bar_colors = [view_colors.get(v, "#999999") for v in dfv["Item Removed"]]
        sns.barplot(data=dfv, y="Item_Short", x="Δ vs. Baseline",
                    palette=bar_colors, edgecolor="black", ax=ax2)
        ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_title("View Ablation Impact on Silhouette", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Δ Silhouette vs. Baseline")
        ax2.set_ylabel("")
        ax2.invert_yaxis()
        ax2.grid(axis="x", alpha=0.3)
    else:
        ax2.axis("off")

    # Bottom-left: Silhouette by Missingness
    ax3 = fig.add_subplot(gs[1, 0])
    if views_dict is not None:
        missingness = compute_missing(views_dict).reindex(pd.Index(area_codes))
        missingness = missingness.fillna(0.0)
        strata = stratify_fn(missingness, method=strat_method)
        sil_df = pd.DataFrame({"Area Code": area_codes, "Silhouette": sil_vals,
                               "Missingness": missingness.values, "Stratum": strata.values})
        sns.boxplot(x="Stratum", y="Silhouette", data=sil_df, palette="Set2", ax=ax3)
        sns.stripplot(x="Stratum", y="Silhouette", data=sil_df, color="gray", size=3, alpha=0.5, ax=ax3)
        ax3.set_ylim(box_ylim if box_ylim is not None else (0, 1))
        ax3.set_title("Silhouette by Missingness Stratum", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Missingness Stratum")
        ax3.set_ylabel("Silhouette Score")
        ax3.grid(axis="y", alpha=0.2)
    else:
        ax3.axis("off")
        sil_df = pd.DataFrame({"Area Code": area_codes, "Silhouette": sil_vals})

    # Bottom-right: Cross-View Agreement (NMI & ARI)
    outer_ax = fig.add_subplot(gs[1, 1]); outer_ax.axis("off")

    # Decide whether we can show the agreement panel
    have_labels = labels_per_view is not None and len(labels_per_view) > 0
    have_sims   = (similarity_matrices is not None) and (view_names is not None)

    if have_labels or have_sims:
        # Compute NMI/ARI from per-view labels if provided; otherwise from per-view similarities
        # 1) If labels_per_view is provided, align and compute
        if have_labels:
            # Align to area_codes and drop LAs that miss ANY view label (e.g. -1)
            labels_df = pd.DataFrame({"Area Code": area_codes}).set_index("Area Code")
            for vn, lab in labels_per_view.items():
                s = pd.Series(lab, index=area_codes, name=vn)
                labels_df[vn] = s
            # mask rows with any -1 or NaN
            valid_mask = labels_df.notna().all(axis=1)
            for vn in labels_df.columns:
                valid_mask &= (labels_df[vn] != -1)
            labels_aligned = labels_df.loc[valid_mask]
            if labels_aligned.shape[0] >= 2 and labels_aligned.shape[1] >= 2:
                # Build dict back for compute fn
                labels_per_view_aligned = {vn: labels_aligned[vn].to_numpy(int) for vn in labels_aligned.columns}
                # Compute agreement (labels take precedence)
                nmi_df, ari_df = compute_clustering_agreement_both(
                    similarity_matrices=None,
                    view_names=list(labels_per_view_aligned.keys()),
                    n_components=nmi_ari_components,   # ignored when using labels
                    k_clusters=nmi_ari_k,              # ignored when using labels
                    labels_per_view=labels_per_view_aligned
                )
                show_panel = True
                # For display use short names
                view_names_short = _shorten_names(list(labels_per_view_aligned.keys()))
            else:
                show_panel = False
        else:
            show_panel = False

        # 2) If we still don't have labels-based panel, try similarity-based panel
        if not show_panel and have_sims:
            nmi_df, ari_df = compute_clustering_agreement_both(
                similarity_matrices=similarity_matrices,
                view_names=view_names,
                n_components=nmi_ari_components,
                k_clusters=nmi_ari_k,
                labels_per_view=None
            )
            show_panel = True
            view_names_short = _shorten_names(view_names)

        # 3) Draw the panel (if computed)
        if show_panel:
            gs_inner = gs[1, 1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.25)
            ax_nmi = fig.add_subplot(gs_inner[0, 0])
            ax_ari = fig.add_subplot(gs_inner[1, 0])

            # Reindex to shortened names for display only
            nmi_df_disp = nmi_df.copy()
            ari_df_disp = ari_df.copy()
            nmi_df_disp.index = view_names_short; nmi_df_disp.columns = view_names_short
            ari_df_disp.index = view_names_short; ari_df_disp.columns = view_names_short

            ax_nmi.set_title("Cross-View Agreement on Cluster Assignments", fontsize=11, fontweight='bold', pad=6)
            sns.heatmap(nmi_df_disp, annot=True, fmt=".2f", cmap="Blues", cbar=False, vmin=0, vmax=1, ax=ax_nmi)
            ax_nmi.set_xticklabels([]); ax_nmi.set_xlabel(""); ax_nmi.set_ylabel("")
            ax_nmi.set_yticklabels(ax_nmi.get_yticklabels(), rotation=0)

            sns.heatmap(ari_df_disp, annot=True, fmt=".2f", cmap="Purples", cbar=False, vmin=0, vmax=1, ax=ax_ari)
            ax_ari.set_xlabel(""); ax_ari.set_ylabel("")
            ax_ari.set_xticklabels(ax_ari.get_xticklabels(), rotation=45, ha="right")
            ax_ari.set_yticklabels(ax_ari.get_yticklabels(), rotation=0)

            ax_nmi.annotate("NMI", xy=(1.02, 0.5), xycoords='axes fraction',
                            rotation=270, va='center', ha='center', fontsize=10, weight='bold')
            ax_ari.annotate("ARI", xy=(1.02, 0.5), xycoords='axes fraction',
                            rotation=270, va='center', ha='center', fontsize=10, weight='bold')
        else:
            ax_placeholder = fig.add_subplot(gs[1, 1]); ax_placeholder.axis("off")
    else:
        ax_placeholder = fig.add_subplot(gs[1, 1]); ax_placeholder.axis("off")



    plt.tight_layout()
    return fig, sil_df


def plot_fusion_report(
    method_out: dict,
    views_custom_hex_palette: List[str],
    clusters_custom_hex_palette: List[str],
    linkage_method: str = "average",
    figsize: tuple[float, float] = (13.5, 9.0),
    dpi: int = 300,
    heatmap_cmap: str = "viridis",
    savepath: Optional[str] = None,
    *,
    normalize_contrib: bool = True,    # row-normalize per-LA contributions so plots stay in [0,1]
    cluster_order: str = "numeric",    # "numeric" | "appearance" | "size_desc" | "error_asc" | "error_desc"
    show: bool = True,                 # control display to avoid double plotting
):
    """
    Plot a multi-panel Fusion Report for a fused similarity matrix.

    Panels:
    (1) Top: Per-view contributions (boxplots + jitter).
    (2) Left: Slim cluster strip (ordered by hierarchical leaves).
    (3) Center: Fused similarity heatmap (ordered by hierarchical linkage).
    (4) Right: Fusion error per cluster (mean ± std), order configurable.

    Parameters
    ----------
    method_out : dict
        Output from run_mvsf_pca_kmeans, containing:
        - "S_fused": (n x n) fused similarity matrix
        - "area_codes": list[str]
        - "similarity_matrices": list[np.ndarray], each (n x n)
        - "labels": np.ndarray (n,)
        - "per_view_contributions_per_la": np.ndarray (n x V)
        - "view_names": list[str]
    ...
    normalize_contrib : bool, default True
        If True, row-normalize per-view contributions (per LA) to sum to 1, then clip to [0,1].
    cluster_order : {"numeric","appearance","size_desc","error_asc","error_desc"}, default "numeric"
        Controls the order of clusters in the right-hand bar chart.
    show : bool, default True
        If True, calls plt.show() to display the figure.

    Returns
    -------
    dict
        {
          "order": reordered indices of entities (leaves),
          "cluster_order": list of cluster ids in right-hand panel,
          "figure": matplotlib Figure
        }
    """

    # Unpack from method_out (TOP-LEVEL keys)
    S_fused = method_out["S_fused"]
    area_codes = method_out["area_codes"]
    similarity_matrices = method_out["similarity_matrices"]
    labels = method_out["labels"]
    per_view_contributions = method_out["per_view_contributions_per_la"]
    view_names = method_out["view_names"]

    n = S_fused.shape[0]
    V = per_view_contributions.shape[1]
    assert S_fused.shape == (n, n)
    assert len(area_codes) == n and labels.shape[0] == n
    assert len(view_names) == V and all(m.shape == (n, n) for m in similarity_matrices)

    # Linkage order for rows/cols (no dendrogram drawn)
    D = 1.0 - S_fused
    np.fill_diagonal(D, 0.0)
    Z = sch.linkage(squareform(D, checks=False), method=linkage_method, optimal_ordering=True)
    leaves = sch.dendrogram(Z, no_plot=True)["leaves"]

    S_ord = S_fused[np.ix_(leaves, leaves)]
    labels_ord = labels[leaves]

    # Stable color mapping by numeric cluster id
    def color_by_cluster_id(c):
        try:
            idx = int(c)
        except Exception:
            uniq = sorted(pd.unique(labels))
            idx = uniq.index(c)
        return clusters_custom_hex_palette[idx % len(clusters_custom_hex_palette)]

    # Fusion error per cluster (compute first, reorder later)
    errs = sum(np.linalg.norm(vS - S_fused, axis=1) for vS in similarity_matrices) / len(similarity_matrices)
    err_df = pd.DataFrame({"Cluster": labels, "err": errs})
    # group by numeric cluster id (fallback if not numeric)
    try:
        err_df["Cluster_num"] = err_df["Cluster"].astype(int)
    except Exception:
        uniq = sorted(pd.unique(labels))
        mapping = {c: i for i, c in enumerate(uniq)}
        err_df["Cluster_num"] = err_df["Cluster"].map(mapping)

    err_summary = (
        err_df.groupby("Cluster_num", as_index=False)
              .agg(mean=("err", "mean"), std=("err", "std"))
              .rename(columns={"Cluster_num": "Cluster"})
    )

    # Choose display order for right bar chart
    if cluster_order == "numeric":
        order = sorted(err_summary["Cluster"])
    elif cluster_order == "size_desc":
        counts = pd.Series(labels).astype(int, errors="ignore").value_counts()
        counts.index = counts.index.astype(int, errors="ignore")
        order = counts.sort_values(ascending=False).index.tolist()
    elif cluster_order == "error_asc":
        order = err_summary.sort_values("mean", ascending=True)["Cluster"].tolist()
    elif cluster_order == "error_desc":
        order = err_summary.sort_values("mean", ascending=False)["Cluster"].tolist()
    elif cluster_order == "appearance":
        # first appearance in dendrogram leaf order
        first = pd.Series(labels_ord).reset_index().drop_duplicates(0)
        order = [int(c) if str(c).isdigit() else c for c in first[0].tolist()]
    else:
        order = sorted(err_summary["Cluster"])

    err_summary = err_summary.set_index("Cluster").loc[order].reset_index()

    # Figure grid
    plt.close("all")
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(
        nrows=2, ncols=3,
        height_ratios=[4.0, 11.5],
        width_ratios=[0.35, 14.0, 3.25],
        left=0.06, right=0.985, top=0.92, bottom=0.12, wspace=0.16, hspace=0.16
    )

    # TOP per-view contributions (normalize/clamp so % is valid)
    PVC = np.asarray(per_view_contributions, dtype=float)
    if normalize_contrib:
        rs = PVC.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        PVC = PVC / rs
    PVC = np.clip(PVC, 0.0, 1.0)

    ax_top = fig.add_subplot(gs[0, :])
    dfc = (pd.DataFrame(PVC, columns=view_names, index=area_codes)
             .rename_axis("Area Code").reset_index()
             .melt(id_vars="Area Code", var_name="View", value_name="Contribution"))
    view_colors = {vn: views_custom_hex_palette[i % len(views_custom_hex_palette)] for i, vn in enumerate(view_names)}

    for i, vn in enumerate(view_names):
        data = dfc.loc[dfc["View"] == vn, "Contribution"].values
        color = view_colors[vn]
        ax_top.boxplot(
            data, positions=[i + 0.2], widths=0.36, patch_artist=True,
            boxprops=dict(facecolor='none', color=color, linewidth=2),
            capprops=dict(color=color, linewidth=2),
            whiskerprops=dict(color=color, linewidth=2),
            flierprops=dict(marker='o', color=color, alpha=0.3),
            medianprops=dict(color=color, linewidth=2)
        )
        jitter_x = np.random.normal(i - 0.2, 0.035, size=len(data))
        ax_top.scatter(jitter_x, data, color=color, alpha=0.6, s=16, edgecolor='gray', linewidth=0.4)
        ax_top.scatter(i + 0.2, np.median(data), marker='+', s=90, color=color, linewidths=2, zorder=3)
    ax_top.set_xticks([]); ax_top.set_xlabel("")
    ax_top.set_ylabel("Contribution to\nFused Similarity (%)", fontweight='bold', fontsize=10)
    ax_top.yaxis.set_major_formatter(PercentFormatter(1.0))
    # ax_top.set_ylim(0.0, 1.0)
    ax_top.grid(axis="y", alpha=0.2)
    handles = [Patch(facecolor='none', edgecolor=view_colors[vn], label=vn, linewidth=2) for vn in view_names]
    ax_top.legend(handles=handles, ncol=1, frameon=False, fontsize=9,
                  title="Views", title_fontsize=10,
                  loc="center left", bbox_to_anchor=(1.01, 0.5),
                  borderaxespad=0.0, handlelength=1.8, columnspacing=1.0)

    # LEFT slim cluster strip (colors stable by cluster id)
    ax_strip = fig.add_subplot(gs[1, 0])
    strip_rgb = np.array([to_rgb(color_by_cluster_id(c)) for c in labels_ord], float).reshape(-1, 1, 3)
    ax_strip.imshow(strip_rgb, aspect="auto", interpolation="nearest")
    ax_strip.set_xticks([]); ax_strip.set_yticks([])
    for sp in ax_strip.spines.values():
        sp.set_visible(False)
    ax_strip.set_ylabel("Clusters", fontweight='bold', fontsize=10, rotation=90, labelpad=4)

    # CENTER similarity heatmap (ordered by leaves)
    ax_h = fig.add_subplot(gs[1, 1])
    im = ax_h.imshow(S_ord, cmap=heatmap_cmap, aspect="auto", interpolation="nearest")
    ax_h.set_title("Local Authority-level Fused similarity (leaf-ordered)", fontsize=12, pad=6, fontweight='bold')
    ax_h.set_xticks([]); ax_h.set_yticks([])
    change_idx = np.where(labels_ord[:-1] != labels_ord[1:])[0]
    for ci in change_idx:
        ax_h.axhline(ci + 0.5, color="white", lw=0.6, alpha=0.6)
        ax_h.axvline(ci + 0.5, color="white", lw=0.6, alpha=0.6)
    cax = inset_axes(ax_h, width="70%", height="10%", loc="lower center",
                     bbox_to_anchor=(0.0, -0.1, 1.0, 1.0),
                     bbox_transform=ax_h.transAxes, borderpad=0)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Similarity", fontsize=8, labelpad=2)

    # RIGHT: fusion error per cluster (ordered)
    ax_r = fig.add_subplot(gs[1, 2])
    y = np.arange(len(err_summary))
    err_colors = [color_by_cluster_id(c) for c in err_summary["Cluster"]]
    ax_r.barh(y=y, width=err_summary["mean"].values,
              xerr=err_summary["std"].values,
              color=err_colors, edgecolor="black", linewidth=0.5, capsize=3)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([str(c) for c in err_summary["Cluster"]], fontsize=9)
    ax_r.invert_yaxis()
    ax_r.set_xlabel("Average Fusion Error", fontsize=10)
    ax_r.set_title("Fusion Error per Cluster", fontweight='bold', fontsize=10, pad=4)
    ax_r.grid(axis="x", alpha=0.2)

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return {"order": leaves, "cluster_order": order, "figure": fig}

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
from typing import List, Optional, Dict, Tuple

# helpers
def _cosine_sim(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Cosine similarity on rows; safe + symmetric + unit diagonal."""
    X = np.asarray(X, float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + eps
    Z = X / nrm
    S = Z @ Z.T
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S

def _compute_contributions_from_Slist(S_list: List[np.ndarray], S_ref: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Return per-LA contributions (n x V): for each view v, c_i,v = mean_j S_v[i,j] / (S_ref[i,j]+eps)
    Then rows are renormalized to sum to 1.
    """
    n = S_ref.shape[0]
    V = len(S_list)
    C = np.zeros((n, V), float)
    denom = np.where(S_ref <= 0, eps, S_ref)
    for v, Sv in enumerate(S_list):
        R = Sv / denom
        C[:, v] = np.nanmean(R, axis=1)
    # renormalize rows
    row_sums = C.sum(axis=1, keepdims=True) + eps
    C = C / row_sums
    return C

def _pick_central_matrix(method_out: dict, embedding: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Choose the matrix to display in the center panel:
    - S_fused (if present)
    - S_coassoc (if present)
    - else cosine on embedding
    """
    if "S_fused" in method_out and method_out["S_fused"] is not None:
        return method_out["S_fused"], "S_fused"
    if "S_coassoc" in method_out and method_out["S_coassoc"] is not None:
        return method_out["S_coassoc"], "S_coassoc"
    return _cosine_sim(embedding), "S_from_embedding"

def _collect_per_view_similarity(method_out: dict) -> Tuple[List[np.ndarray], List[str]]:
    """
    Retrieve per-view similarity matrices in a flexible way:
    - if 'similarity_matrices' present → use them directly.
    - elif 'per_view_embeddings' present → cosine on each latent to get S_v.
      (expects a list aligned to 'view_names')
    - else → return empty list.
    """
    if "similarity_matrices" in method_out and method_out["similarity_matrices"] is not None:
        # could be list or dict keyed by view_names
        sims = method_out["similarity_matrices"]
        if isinstance(sims, dict):
            view_names = list(sims.keys())
            S_list = [sims[vn] for vn in view_names]
        else:
            view_names = method_out.get("view_names", [f"view_{i}" for i in range(len(sims))])
            S_list = sims
        return S_list, view_names

    if "per_view_embeddings" in method_out and method_out["per_view_embeddings"] is not None:
        Z_list = method_out["per_view_embeddings"]
        view_names = method_out.get("view_names", [f"view_{i}" for i in range(len(Z_list))])
        S_list = [_cosine_sim(Z) for Z in Z_list]
        return S_list, view_names

    return [], method_out.get("view_names", [])

def plot_factorization_report(
    method_out: dict,
    views_custom_hex_palette: List[str],
    clusters_custom_hex_palette: List[str],
    linkage_method: str = "average",
    figsize: tuple[float, float] = (13.5, 9.0),
    dpi: int = 300,
    heatmap_cmap: str = "magma",
    top_panel: str = "auto",
    savepath: Optional[str] = None,
):
    """
    Factorization Report (mirrors Fusion Report) — works for:
      • Early MNMF / Masked NMF (embedding+labels required; per-view matrices optional)
      • Intermediate PV-NMF (per-view latent blocks available)
      • Late Co-association Factorization Ensemble (S_coassoc provided)

    Required in method_out:
      - 'area_codes' : list[str]
      - 'labels'     : (n,)
      - 'embedding'  : (n, d)
    Optional in method_out:
      - 'S_fused' or 'S_coassoc' : (n, n) central matrix to display
      - 'similarity_matrices'    : list[np.ndarray] or dict[name->S] per view
      - 'per_view_embeddings'    : list[np.ndarray] (Z_v per view); used to derive S_v
      - 'view_names'             : list[str]
      - 'per_view_contributions_per_la' : (n, V) if already computed

    Panels:
      (1) Top: per-view contributions (boxplots + jitter, rows normalized to 100%)
      (2) Left: cluster strip (ordered by linkage leaves)
      (3) Center: heatmap of central matrix (ordered)
      (4) Right: “factorization error” per cluster = mean±std over i in cluster of
                 ||S_v[i,:] - S_central[i,:]||_2 averaged over views.

    Returns dict with {"order", "cluster_order", "figure"}.
    """
    # unpack required
    area_codes = method_out["area_codes"]
    labels     = np.asarray(method_out["labels"])
    embedding  = np.asarray(method_out["embedding"])

    n = len(area_codes)
    assert embedding.shape[0] == n and labels.shape[0] == n

    # central matrix
    S_central, central_name = _pick_central_matrix(method_out, embedding)
    S_central = np.asarray(S_central, float)
    S_central = 0.5 * (S_central + S_central.T)
    np.fill_diagonal(S_central, 1.0)

    # per-view similarities + contributions
    S_list, view_names = _collect_per_view_similarity(method_out)

    if S_list:
        # ensure shapes and symmetry
        S_list = [0.5 * (S + S.T) for S in S_list]
        for S in S_list:
            np.fill_diagonal(S, 1.0)

    if "per_view_contributions_per_la" in method_out and method_out["per_view_contributions_per_la"] is not None:
        C = np.asarray(method_out["per_view_contributions_per_la"], float)
    else:
        if not S_list:
            # fall back: single “view” from embedding; contributions = 1.0
            view_names = view_names or ["latent"]
            C = np.ones((n, 1), float)
        else:
            C = _compute_contributions_from_Slist(S_list, S_central)

    V = C.shape[1]
    if not view_names or len(view_names) != V:
        view_names = [f"view_{i}" for i in range(V)]

    # order by linkage on central matrix
    D = 1.0 - S_central
    np.fill_diagonal(D, 0.0)
    Z = sch.linkage(squareform(D, checks=False), method=linkage_method, optimal_ordering=True)
    leaves = sch.dendrogram(Z, no_plot=True)["leaves"]

    S_ord = S_central[np.ix_(leaves, leaves)]
    labels_ord = labels[leaves]

    # cluster appearance sequence
    cluster_sequence = pd.Index(pd.unique(labels_ord))
    k = len(cluster_sequence)

    # fusion/factorization error per cluster
    if S_list:
        errs = np.mean([np.linalg.norm(Sv - S_central, axis=1) for Sv in S_list], axis=0)
    else:
        # no per-view mats: define error vs global mean profile
        mean_row = S_central.mean(axis=0, keepdims=True)
        errs = np.linalg.norm(S_central - mean_row, axis=1)

    err_df = pd.DataFrame({"Cluster": labels, "err": errs})
    err_df["Cluster"] = pd.Categorical(err_df["Cluster"], categories=list(cluster_sequence), ordered=True)
    err_summary = err_df.groupby("Cluster").agg(mean=("err","mean"), std=("err","std")).reset_index()

    # figure
    plt.close("all")
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(
        nrows=2, ncols=3,
        height_ratios=[4.0, 11.5],
        width_ratios=[0.35, 14.0, 3.25],
        left=0.06, right=0.985, top=0.92, bottom=0.12, wspace=0.16, hspace=0.16
    )

    # (1) Top: per-view contributions
    ax_top = fig.add_subplot(gs[0, :])

    # Decide which top panel to render
    has_contribs = _has_useful_contribs(method_out, S_list, C)
    central_is_coassoc = ("S_coassoc" in method_out and method_out["S_coassoc"] is not None)

    panel_mode = top_panel
    if top_panel == "auto":
        if has_contribs:
            panel_mode = "contrib"
        elif central_is_coassoc or (not S_list):
            panel_mode = "consensus_margin"
        else:
            panel_mode = "silhouette"

    if panel_mode == "contrib":
        # contributions code (unchanged)
        dfc = (pd.DataFrame(C, columns=view_names, index=area_codes)
                .rename_axis("Area Code").reset_index()
                .melt(id_vars="Area Code", var_name="View", value_name="Contribution"))
        view_colors = {vn: views_custom_hex_palette[i % len(views_custom_hex_palette)] for i, vn in enumerate(view_names)}
        for i, vn in enumerate(view_names):
            data = dfc.loc[dfc["View"] == vn, "Contribution"].values
            color = view_colors[vn]
            ax_top.boxplot(
                data, positions=[i + 0.2], widths=0.36, patch_artist=True,
                boxprops=dict(facecolor='none', color=color, linewidth=2),
                capprops=dict(color=color, linewidth=2),
                whiskerprops=dict(color=color, linewidth=2),
                flierprops=dict(marker='o', color=color, alpha=0.25),
                medianprops=dict(color=color, linewidth=2)
            )
            rng = np.random.default_rng(19042022)
            jitter_x = rng.normal(i - 0.2, 0.035, size=len(data))
            ax_top.scatter(jitter_x, data, color=color, alpha=0.55, s=14, edgecolor='gray', linewidth=0.4)
            ax_top.scatter(i + 0.2, np.median(data), marker='+', s=90, color=color, linewidths=2, zorder=3)
        ax_top.set_xticks([]); ax_top.set_xlabel("")
        ax_top.set_ylabel("Contribution to\nCentral Similarity (%)", fontweight='bold', fontsize=10)
        ax_top.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax_top.set_ylim(0, 1)
        ax_top.grid(axis="y", alpha=0.2)
        handles = [Patch(facecolor='none', edgecolor=view_colors[vn], label=vn, linewidth=2) for vn in view_names]
        ax_top.legend(handles=handles, ncol=1, frameon=False, fontsize=9,
                    title="Views", title_fontsize=10,
                    loc="center left", bbox_to_anchor=(1.01, 0.5))

    elif panel_mode == "consensus_margin":
        # consensus margin per cluster, colored with clusters_custom_hex_palette
        within, rival, margin = _consensus_margin_per_item(S_central, labels)
        dfm = pd.DataFrame({"Cluster": labels, "Margin": margin})

        # Keep the cluster appearance order and make categories STRINGS to sync with seaborn's palette keys
        order = [str(c) for c in cluster_sequence]
        dfm["Cluster"] = pd.Categorical(dfm["Cluster"].astype(str), categories=order, ordered=True)

        # Build palette dict whose KEYS are the string categories shown on the x-axis
        def _cluster_idx(s: str) -> int:
            try:
                return int(s)  # works for "0","10","-1" etc.
            except Exception:
                return order.index(s)

        cluster_palette = {
            c: clusters_custom_hex_palette[_cluster_idx(c) % len(clusters_custom_hex_palette)]
            for c in order
        }
        # Optional: special color for unassigned cluster "-1"
        # if "-1" in cluster_palette: cluster_palette["-1"] = "#9e9e9e"

        import seaborn as sns
        sns.boxplot(
            data=dfm, x="Cluster", y="Margin",
            order=order, palette=cluster_palette,
            showcaps=True, fliersize=2, width=0.5, ax=ax_top
        )

        # (Optional) overlay jittered points in matching colors (keeps boxplot style but shows density)
        rng = np.random.default_rng(19042022)
        for i, c in enumerate(order):
            vals = dfm.loc[dfm["Cluster"] == c, "Margin"].to_numpy()
            if len(vals) == 0:
                continue
            jitter_x = rng.normal(i, 0.06, size=len(vals))
            ax_top.scatter(
                jitter_x, vals, s=12, alpha=0.45,
                color=cluster_palette[c], edgecolor="gray", linewidths=0.3, zorder=2
            )

        ax_top.axhline(0.0, ls="--", lw=1, color="gray", alpha=0.7)
        ax_top.set_title("Consensus Margin per Cluster (Within − Best Rival)", fontsize=11, fontweight="bold")
        ax_top.set_ylabel("Margin"); ax_top.set_xlabel("")
        ax_top.grid(axis="y", alpha=0.2)


    elif panel_mode == "silhouette":
        # === fallback: silhouette samples over the embedding ===
        from sklearn.metrics import silhouette_samples
        try:
            s = silhouette_samples(embedding, labels)
            dfs = pd.DataFrame({"Cluster": labels, "Silhouette": s})
            dfs["Cluster"] = pd.Categorical(dfs["Cluster"], categories=list(cluster_sequence), ordered=True)

            import seaborn as sns
            sns.violinplot(data=dfs, x="Cluster", y="Silhouette", inner=None, cut=0, ax=ax_top)
            sns.boxplot(data=dfs, x="Cluster", y="Silhouette", whis=1.5, width=0.25,
                        showcaps=True, fliersize=2, ax=ax_top)
            ax_top.axhline(0.0, ls="--", lw=1, color="gray", alpha=0.7)
            ax_top.set_title("Silhouette by Cluster (embedding space)", fontsize=11, fontweight="bold")
            ax_top.set_xlabel(""); ax_top.set_ylabel("Silhouette")
            ax_top.grid(axis="y", alpha=0.2)
        except Exception:
            ax_top.axis("off")
            ax_top.text(0.5, 0.5, "No silhouette available", ha="center", va="center")
    else:
        # Safety
        ax_top.axis("off")

    # helper: color-by-cluster (0 -> first palette color)
    def color_by_cluster_id(c):
        try: idx = int(c)
        except Exception:
            idx = list(cluster_sequence).index(c)
        return clusters_custom_hex_palette[idx % len(clusters_custom_hex_palette)]

    # (2) Left slim cluster strip (ordered)
    ax_strip = fig.add_subplot(gs[1, 0])
    strip_rgb = np.array([to_rgb(color_by_cluster_id(c)) for c in labels_ord], float).reshape(-1, 1, 3)
    ax_strip.imshow(strip_rgb, aspect="auto", interpolation="nearest")
    ax_strip.set_xticks([]); ax_strip.set_yticks([])
    for sp in ax_strip.spines.values():
        sp.set_visible(False)
    ax_strip.set_ylabel("Clusters", fontweight='bold', fontsize=10, rotation=90, labelpad=4)

    # (3) Center heatmap of S_central (ordered)
    ax_h = fig.add_subplot(gs[1, 1])
    im = ax_h.imshow(S_ord, cmap=heatmap_cmap, aspect="auto", interpolation="nearest")
    ax_h.set_xticks([]); ax_h.set_yticks([])
    # draw boundaries between cluster blocks
    change_idx = np.where(labels_ord[:-1] != labels_ord[1:])[0]
    for ci in change_idx:
        ax_h.axhline(ci + 0.5, color="white", lw=0.6, alpha=0.6)
        ax_h.axvline(ci + 0.5, color="white", lw=0.6, alpha=0.6)
    # colorbar below
    cax = inset_axes(ax_h, width="70%", height="10%",
                     loc="lower center",
                     bbox_to_anchor=(0.0, -0.1, 1.0, 1.0),
                     bbox_transform=ax_h.transAxes, borderpad=0)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=8)
    cb.set_label(f"Similarity ({central_name})", fontsize=8, labelpad=2)

    # (4) Right: factorization error per cluster (ordered by appearance)
    ax_r = fig.add_subplot(gs[1, 2])
    y = np.arange(k)
    err_colors = [color_by_cluster_id(c) for c in err_summary["Cluster"]]
    ax_r.barh(y=y, width=err_summary["mean"].values,
              xerr=err_summary["std"].values,
              color=err_colors, edgecolor="black", linewidth=0.5, capsize=3)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([str(c) for c in err_summary["Cluster"]], fontsize=9)
    ax_r.invert_yaxis()  # keep first cluster (as it appears in strip) at the top
    ax_r.set_xlabel("Avg. Deviation from Central Similarity", fontsize=10)
    ax_r.set_title("Factorization Error per Cluster", fontweight='bold', fontsize=10, pad=4)
    ax_r.grid(axis="x", alpha=0.2)

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return {"order": leaves, "cluster_order": list(cluster_sequence), "figure": fig}

def _consensus_margin_per_item(S: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-item: within-cluster consensus, best rival-cluster consensus, and margin = within - rival.
    Uses the provided central similarity S (n x n) and labels (n,).
    """
    S = np.asarray(S, float)
    labels = np.asarray(labels)
    n = S.shape[0]
    uniq = sorted(pd.unique(labels))
    masks = {g: (labels == g) for g in uniq}

    within = np.zeros(n, float)
    rival = np.zeros(n, float)
    for i in range(n):
        gi = labels[i]
        mask_in = masks[gi].copy()
        mask_in[i] = False
        denom_in = max(mask_in.sum(), 1)
        within[i] = S[i, mask_in].mean() if denom_in > 0 else 0.0

        means = []
        for g, m in masks.items():
            if g == gi: 
                continue
            denom = max(m.sum(), 1)
            means.append(S[i, m].mean() if denom > 0 else 0.0)
        rival[i] = max(means) if means else 0.0

    margin = within - rival
    return within, rival, margin


def _has_useful_contribs(method_out: dict, S_list: list[np.ndarray], C: np.ndarray | None) -> bool:
    """
    True if per-view contributions are meaningful (i.e., we actually have >1 view).
    """
    if C is not None and C.shape[1] > 1:
        return True
    if S_list and len(S_list) > 1:
        return True
    return False
