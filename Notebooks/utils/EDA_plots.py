from typing import Any, List, Dict, Iterable, Optional, Mapping, Callable, Sequence, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from scipy.stats import zscore, entropy

from data_preprocessing import DEFAULT_META_COLS as META_COLS
import textwrap

def plot_distribution(
    df: pd.DataFrame,
    title: str,
    max_features: int = 5,
    *,
    columns: Optional[Sequence[str]] = None,
    bins: int = 30,
    kde: bool = True,
    truncate_label_len: int = 20,
    show: bool = True,
) -> Tuple[matplotlib.figure.Figure, List[matplotlib.axes.Axes], List[str]]:
    """
    Plot 1-row KDE/hist distributions for selected numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    title : str
        Figure title.
    max_features : int, default 5
        Max number of features to plot (when `columns` is None).
    columns : sequence[str] or None, default None
        Explicit list of numeric columns to plot (order preserved). If None,
        the first `max_features` numeric columns are used.
    bins : int, default 30
        Histogram bins.
    kde : bool, default True
        Whether to overlay KDE on the histogram.
    truncate_label_len : int, default 20
        Truncate axis/subplot titles to this many characters (with ellipsis).
    show : bool, default True
        If True, calls plt.show().

    Returns
    -------
    fig, axes, used_columns : (Figure, list[Axes], list[str])
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric columns found in dataframe.")

    if columns is None:
        used_columns = numeric_df.columns[:max_features].tolist()
    else:
        # Intersect in the given order
        numeric_cols = set(numeric_df.columns)
        used_columns = [c for c in columns if c in numeric_cols][:max_features]

    n_features = len(used_columns)
    if n_features == 0:
        raise ValueError("No valid numeric columns selected for plotting.")

    fig_width = max(3 * n_features, 3)
    fig, axes = plt.subplots(1, n_features, figsize=(fig_width, 3))

    if n_features == 1:
        axes = [axes]  # make iterable

    def _short(label: str) -> str:
        return (label[:truncate_label_len] + "...") if len(label) > truncate_label_len else label

    for i, col in enumerate(used_columns):
        data = numeric_df[col].dropna()
        if data.empty:
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[i].set_axis_off()
            continue

        sns.histplot(data, kde=kde, bins=bins, ax=axes[i])
        axes[i].set_title(_short(col), fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].tick_params(axis="both", which="both", length=0, labelsize=8)

    fig.suptitle(f"{title} - Distributions (First {n_features} Features)", fontsize=16)
    fig.tight_layout(pad=1.5, rect=[0, 0, 1, 0.92])

    if show:
        plt.show()

    return fig, axes, used_columns


def plot_correlation(
    df: pd.DataFrame,
    title: str,
    *,
    method: str = "pearson",
    annotate: bool = False,
    fmt: str = ".2f",
    cmap: str = "coolwarm",
    truncate_label_len: int = 20,
    mask_upper: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    linewidths: float = 0.5,
    cbar_shrink: float = 0.8,
    show: bool = True,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, pd.DataFrame]:
    """
    Generate a correlation heatmap for all numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    title : str
        Figure title prefix.
    method : {'pearson','spearman','kendall'}, default 'pearson'
        Correlation method.
    annotate : bool, default False
        If True, write correlation values in cells.
    fmt : str, default '.2f'
        Annotation format string (used when annotate=True).
    cmap : str, default 'coolwarm'
        Heatmap colormap.
    truncate_label_len : int, default 20
        Truncate axis labels to this many characters (with ellipsis).
    mask_upper : bool, default False
        If True, mask the upper triangle of the matrix.
    vmin, vmax : float or None
        Color scale min/max. If None, determined from data.
    linewidths : float, default 0.5
        Grid line width between cells.
    cbar_shrink : float, default 0.8
        Colorbar shrink factor.
    show : bool, default True
        If True, calls plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    corr : pd.DataFrame
        Full correlation matrix (untruncated labels).
    """
    num = df.select_dtypes(include="number")
    if num.shape[1] == 0:
        raise ValueError("No numeric columns found for correlation heatmap.")

    corr = num.corr(method=method)

    # Create display copy with truncated labels
    def _short(label: str) -> str:
        return (label[:truncate_label_len] + "...") if len(label) > truncate_label_len else label

    display_corr = corr.copy()
    display_corr.index = [ _short(c) for c in corr.index ]
    display_corr.columns = [ _short(c) for c in corr.columns ]

    # Optional masking of upper triangle
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(display_corr, dtype=bool))

    # Dynamic sizing
    n = display_corr.shape[1]
    fig_size = max(6.0, 0.5 * n)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        display_corr,
        mask=mask,
        cmap=cmap,
        annot=annotate,
        fmt=fmt if annotate else "",
        square=True,
        cbar_kws={"shrink": cbar_shrink},
        linewidths=linewidths,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )

    ax.set_title(f"{title} - Correlation Heatmap", fontsize=14)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax, corr

    
def compute_cluster_feature_stats(
    views: Dict[str, pd.DataFrame],
    labels: Iterable,
    area_codes: Iterable,
    top_k: int = 10,
    meta_cols: Optional[Iterable[str]] = ("Area Code", "Area", "Area Name", "Local Authority", "LA Name"),
    histogram_bins: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute feature importance summaries for clusters using:
      (1) max absolute z-score of cluster means (per feature), and
      (2) mean KL divergence of cluster-wise histograms vs overall (per feature).
    Also returns exemplar Local Authorities per cluster (up to top_k, de-duplicated).

    Parameters
    ----------
    views : dict[str, pd.DataFrame]
        Mapping of view name -> dataframe. Each dataframe must contain either an
        'Area Code' column or use 'Area Code' as the index. If present, an 'Area'
        column will be harvested to label areas in outputs.
    labels : Iterable
        Cluster labels (one per area code), order-aligned with `area_codes`.
    area_codes : Iterable
        Area codes corresponding to `labels`. These will be used to align/subset.
    top_k : int, default 10
        Number of top-ranked features/LAs to return.
    meta_cols : Iterable[str] or None, default ("Area Code","Area","Area Name","Local Authority","LA Name")
        Columns to drop from feature matrices (except we retain 'Area' separately).
    histogram_bins : int, default 20
        Number of bins used for KL divergence histograms.

    Returns
    -------
    top_zscore_features : pd.DataFrame
        Columns: ['Feature', 'Max |Z-Score|', 'Cluster (at max)'] ranked desc.
    top_kl_features : pd.DataFrame
        Columns: ['Feature', 'Mean KL Divergence'] ranked desc.
    top_las_by_cluster : pd.DataFrame
        Columns: ['Cluster', 'Area Code', 'Area'] with up to top_k rows per cluster.

    Notes
    -----
    - Z-scores are computed on cluster means per feature:
          z = (μ_c - mean_c(μ_c)) / (std_c(μ_c) + eps)
      where μ_c is the mean of a feature within cluster c, and mean_c/std_c are
      computed across clusters for that feature.
    - KL averages the divergence between each cluster histogram and the overall
      distribution for that feature (simple mean across clusters).
    """
    # Build feature matrix (prefix columns by view) and collect Area names
    feature_dfs: List[pd.DataFrame] = []
    area_dfs: List[pd.DataFrame] = []

    meta_cols = set(meta_cols or ())

    for name, df in views.items():
        df = df.copy()

        # Standardize index to 'Area Code'
        if 'Area Code' in df.columns:
            df = df.set_index('Area Code')
        elif df.index.name != 'Area Code':
            # If the index is unlabeled but is Area Code, keep; otherwise we assume it's Area Code.
            df.index.name = 'Area Code'

        # Keep any available Area column for later labeling
        if 'Area' in df.columns:
            area_dfs.append(df[['Area']])

        # Drop metadata (except Area) and coerce to numeric
        drop_cols = [c for c in df.columns if c in meta_cols and c != 'Area']
        df = df.drop(columns=drop_cols, errors='ignore')
        df = df.apply(pd.to_numeric, errors='coerce')

        # Prefix feature columns with the view name
        df.columns = [f"{name}: {col}" for col in df.columns]
        feature_dfs.append(df)

    # Merge all features
    if not feature_dfs:
        raise ValueError("No feature dataframes were constructed from `views`.")
    all_features_df = pd.concat(feature_dfs, axis=1).sort_index()

    # Compose a single 'Area' column (first non-null across views)
    if area_dfs:
        area_stack = pd.concat(area_dfs, axis=1)
        all_features_df['Area'] = area_stack.bfill(axis=1).iloc[:, 0].astype(str)
    else:
        all_features_df['Area'] = pd.Series(index=all_features_df.index, dtype='object')

    # Align with labels & area_codes
    labels_s = pd.Series(list(labels), index=pd.Index(list(area_codes), name='Area Code'))
    common_idx = labels_s.index.intersection(all_features_df.index)

    if common_idx.empty:
        raise ValueError(
            "No overlap between provided `area_codes` and the views' Area Code index."
        )

    # Subset + attach labels
    all_features_df = all_features_df.loc[common_idx].copy()
    all_features_df['Cluster'] = labels_s.loc[common_idx].values

    # Cluster means (numeric only)
    numeric_cols = all_features_df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric features found after cleaning.")

    cluster_means = all_features_df.groupby('Cluster')[numeric_cols].mean(numeric_only=True)
    eps = 1e-8

    # Z-SCORE ANALYSIS (per feature across clusters)
    mu = cluster_means
    mu_bar = mu.mean(axis=0)           # per-feature mean across clusters
    mu_std = mu.std(axis=0) + eps      # per-feature std across clusters
    zscores = (mu - mu_bar) / mu_std   # shape: [n_clusters, n_features]

    # Max |z| per feature + the cluster where it occurs
    abs_z = zscores.abs()
    max_abs_by_feature = abs_z.max(axis=0).sort_values(ascending=False)
    argmax_cluster = abs_z.idxmax(axis=0).reindex(max_abs_by_feature.index)

    top_zscore_features = (
        pd.DataFrame({
            'Feature': max_abs_by_feature.index,
            'Max |Z-Score|': max_abs_by_feature.values,
            'Cluster (at max)': argmax_cluster.values
        })
        .head(top_k)
        .reset_index(drop=True)
    )

    # KL DIVERGENCE ANALYSIS  (fixed to handle duplicate-named columns)
    kl_scores = {}

    for feature in cluster_means.columns:
        # Pull the feature column; if duplicates exist (DataFrame), reduce to a single 1D series.
        feat_col = all_features_df[feature]
        if isinstance(feat_col, pd.DataFrame):
            # combine duplicates deterministically (mean across duplicate columns)
            feat_col = feat_col.mean(axis=1)

        # Build a tidy frame for this feature
        feat_df = pd.DataFrame({
            "val": pd.to_numeric(feat_col, errors="coerce"),
            "Cluster": all_features_df["Cluster"]
        }).dropna()

        if feat_df.empty:
            continue

        fmin = feat_df["val"].min()
        fmax = feat_df["val"].max()

        # Skip degenerate/undefined ranges
        if pd.isna(fmin) or pd.isna(fmax) or np.isclose(fmin, fmax):
            continue

        # Overall reference distribution
        overall, _ = np.histogram(
            feat_df["val"].values,
            bins=histogram_bins,
            range=(fmin, fmax),
            density=True
        )
        overall = overall + 1e-8

        # Per-cluster distributions → KL to overall, then mean across clusters
        def _hist(x: pd.Series) -> np.ndarray:
            h, _ = np.histogram(
                x.values,
                bins=histogram_bins,
                range=(fmin, fmax),
                density=True
            )
            return h + 1e-8

        kl_divs = (
            feat_df.groupby("Cluster")["val"]
            .apply(lambda x: entropy(_hist(x), overall))
            .astype(float)
        )
        kl_scores[feature] = float(kl_divs.mean())

    top_kl_features = (
        pd.Series(kl_scores, name="Mean KL Divergence")
        .sort_values(ascending=False)
        .head(top_k)
        .reset_index()
        .rename(columns={"index": "Feature"})
    )

    # Top Local Authorities by cluster (simple exemplars)
    la_cols = ['Area']  # keep Area name
    # Keep Area Code too for clarity
    la_df = all_features_df[la_cols].copy()
    la_df['Area Code'] = la_df.index
    top_las_by_cluster = (
        all_features_df[['Cluster']]
        .join(la_df)
        .reset_index(drop=True)
        .drop_duplicates(subset=['Cluster', 'Area Code'])
        .groupby('Cluster', group_keys=False)
        .head(top_k)
        .loc[:, ['Cluster', 'Area Code', 'Area']]
        .reset_index(drop=True)
    )

    return top_zscore_features, top_kl_features, top_las_by_cluster

def plot_missingness_and_means(
    views: Dict[str, pd.DataFrame],
    labels: Iterable,
    area_codes: Iterable,
    META_COLS: Iterable[str],
    views_custom_hex_palette: Sequence[str],
    *,
    max_feature_name_length: int = 80,
    savepath: Optional[str] = None,
    dpi: int = 300,
    annotate_threshold: int = 10,
    use_median: bool = True,
    histogram_normalize_rows: bool = True,
    sort_clusters: bool = True,
    view_shorten_map: Optional[Mapping[str, str]] = None,
    fig_width: float = 18.0,
    row_height: float = 0.25,
    min_fig_height: float = 4.0,
    max_fig_height: float = 40.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Plot (1) % missingness per feature per cluster and (2) row-wise normalized
    central tendency per feature per cluster (median by default). Returns the
    matrices used for the two heatmaps.

    Parameters
    ----------
    views : dict[name, DataFrame]
        Each DataFrame should have features by row (areas). Must include 'Area Code'
        either as a column or as the index.
    labels : Iterable
        Cluster labels aligned to `area_codes`.
    area_codes : Iterable
        Area codes aligned to `labels`. Will be intersected with available rows in views.
    META_COLS : Iterable[str]
        Columns to drop from each view before numeric conversion (e.g., ['Area Code','Area',...]).
    views_custom_hex_palette : sequence of str
        Hex colors used to shade y-tick backgrounds by view prefix.
    max_feature_name_length : int, default 80
        Truncate long feature labels to keep plots readable.
    savepath : str or None
        If provided, save the figure to this path.
    dpi : int, default 300
        Save resolution (if saving).
    annotate_threshold : int, default 10
        Only annotate heatmap cells if #clusters <= this threshold.
    use_median : bool, default True
        If True, cluster central tendency uses median; otherwise mean.
    histogram_normalize_rows : bool, default True
        If True, normalize cluster central tendency per feature using (x - min) / (max - min).
        If False, leave raw central tendency values (still safe to plot).
    sort_clusters : bool, default True
        Sort cluster columns ascending; otherwise preserve first-seen order.
    view_shorten_map : Mapping[str, str] or None
        Optional substring replacement map for view names; defaults to
        {'educational_attainment': 'edu.attainment'} if None.
    fig_width : float, default 18.0
        Width of the figure (inches).
    row_height : float, default 0.25
        Height per feature row (inches).
    min_fig_height : float, default 4.0
        Minimum figure height (inches).
    max_fig_height : float, default 40.0
        Maximum figure height (inches).

    Returns
    -------
    cluster_missingness : DataFrame (features x clusters, % missing)
    cluster_tendency_scaled : DataFrame (features x clusters, normalized if requested)

    Notes
    -----
    - Only numeric columns are kept for the cluster summaries.
    - Feature labels are prefixed as '<short_view_name>: <original_feature>'.
    - Y-tick backgrounds are lightly shaded by view to help scan across blocks.
    """
    # Build combined numeric feature matrix with prefixed column names
    meta_cols = set(META_COLS or [])
    view_shorten_map = dict(view_shorten_map or {"educational_attainment": "edu.attainment"})

    all_features: list[str] = []
    full_df = pd.DataFrame()

    for i, (view_name, df) in enumerate(views.items()):
        df = df.copy()

        # Normalize 'Area Code' as index
        if 'Area Code' in df.columns:
            df = df.set_index('Area Code')
        elif df.index.name != 'Area Code':
            df.index.name = 'Area Code'

        # Drop metadata columns and coerce to numeric
        df = df.drop(columns=[c for c in df.columns if c in meta_cols], errors='ignore')
        df = df.apply(pd.to_numeric, errors='coerce').sort_index()

        # Shorten view name (simple substring replacement map)
        short_name = view_name
        for old, new in view_shorten_map.items():
            if old in short_name:
                short_name = short_name.replace(old, new)

        # Prefix feature columns with view name
        df.columns = [f"{short_name}: {c}" for c in df.columns]

        all_features.extend(df.columns)
        full_df = pd.concat([full_df, df], axis=1)

    # No features?
    if not all_features:
        raise ValueError("No numeric features were found in the provided views after dropping META_COLS.")

    # Align to (area_codes, labels)
    labels_s = pd.Series(list(labels), index=pd.Index(list(area_codes), name="Area Code"))
    common_idx = full_df.index.intersection(labels_s.index)

    if common_idx.empty:
        raise ValueError("No overlap between `area_codes` and the union of rows across `views`.")

    full_df = full_df.loc[common_idx].copy()
    # attach aligned cluster labels
    full_df['Cluster'] = labels_s.loc[common_idx].values

    # Fix cluster order if requested (for consistent column ordering in heatmaps)
    unique_clusters = pd.unique(full_df['Cluster'])
    if sort_clusters:
        try:
            # sort but stable for mixed types by string conversion
            cluster_order = sorted(unique_clusters, key=lambda x: (str(type(x)), x))
        except Exception:
            cluster_order = list(unique_clusters)
    else:
        cluster_order = list(unique_clusters)

    full_df['Cluster'] = pd.Categorical(full_df['Cluster'], categories=cluster_order, ordered=True)

    # Determine numeric feature columns (drop all-NaN columns)
    numeric_cols = full_df.select_dtypes(include='number').columns.tolist()
    # Remove the 'Cluster' code from numeric list if codes are numeric-like
    if 'Cluster' in numeric_cols:
        numeric_cols.remove('Cluster')

    # Drop columns with all NaN across aligned rows
    numeric_cols = [c for c in numeric_cols if not full_df[c].isna().all()]
    if not numeric_cols:
        raise ValueError("After alignment, all numeric features are empty (all-NaN).")

    # Missingness (% within cluster)
    cluster_missingness = (
        full_df.groupby('Cluster')[numeric_cols]
        .apply(lambda g: g.isna().sum().astype(float) / max(len(g), 1) * 100.0)
        .T
    )

    # Cluster sizes for labels (safe lookup even if some are absent after filtering)
    cluster_sizes = full_df['Cluster'].value_counts().reindex(cluster_missingness.columns, fill_value=0)
    col_labels_missing = [f"{cl} (n={int(cluster_sizes.loc[cl])})" for cl in cluster_missingness.columns]

    # Decide whether to annotate
    annot_missing = (
        cluster_missingness.applymap(lambda x: f"{x:.1f}%")
        if cluster_missingness.shape[1] <= annotate_threshold
        else None
    )

    # Cluster central tendency (median/mean)
    if use_median:
        cluster_tendency = full_df.groupby('Cluster')[numeric_cols].median().T
        tendency_name = "Median"
    else:
        cluster_tendency = full_df.groupby('Cluster')[numeric_cols].mean().T
        tendency_name = "Mean"

    # Scale per feature row if requested (robust to constant rows)
    if histogram_normalize_rows:
        row_min = cluster_tendency.min(axis=1)
        row_max = cluster_tendency.max(axis=1)
        denom = (row_max - row_min).replace(0, np.nan)
        cluster_tendency_scaled = (cluster_tendency.sub(row_min, axis=0)).div(denom, axis=0)
        cluster_tendency_scaled = cluster_tendency_scaled.fillna(0.0)  # constants → 0
    else:
        cluster_tendency_scaled = cluster_tendency.copy()

    # Optional annotation with (central tendency ± std)
    # We compute std on original (not normalized) values:
    cluster_std = full_df.groupby('Cluster')[numeric_cols].std().T
    col_labels_means = [f"{cl} (n={int(cluster_sizes.loc[cl])})" for cl in cluster_tendency_scaled.columns]

    if cluster_tendency_scaled.shape[1] <= annotate_threshold:
        annot_means = cluster_tendency.copy()  # annotate with raw central tendency
        for r in cluster_tendency.index:
            for c in cluster_tendency.columns:
                ct = cluster_tendency.loc[r, c]
                sd = cluster_std.loc[r, c]
                annot_means.loc[r, c] = f"{ct:.2f} ({(0.0 if pd.isna(sd) else sd):.2f})"
    else:
        annot_means = None

    # Truncate feature labels (keep view prefix intact)
    def _truncate(label: str) -> str:
        return f"{label[:max_feature_name_length]}..." if len(label) > max_feature_name_length else label

    truncated_labels = [ _truncate(name) for name in cluster_tendency_scaled.index ]

    # Figure sizing
    fig_height = max(min_fig_height, min(len(numeric_cols) * row_height, max_fig_height))
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True)

    # Left heatmap: Missingness
    hm1 = sns.heatmap(
        cluster_missingness,
        cmap="Purples",
        annot=annot_missing,
        fmt="s" if annot_missing is not None else "",
        xticklabels=col_labels_missing,
        yticklabels=truncated_labels,
        cbar_kws={'label': '% Missing'},
        linewidths=0.5,
        ax=axes[0],
    )
    axes[0].set_title("Missingness (%) per Feature per Cluster")
    axes[0].set_ylabel("Features")
    axes[0].set_xlabel("Cluster Label")

    # Right heatmap: Central tendency (row-wise normalized if requested)
    hm2 = sns.heatmap(
        cluster_tendency_scaled,
        cmap="coolwarm",
        annot=annot_means,
        fmt="s" if annot_means is not None else "",
        xticklabels=col_labels_means,
        yticklabels=truncated_labels,
        cbar_kws={'label': 'Row-wise Normalized' if histogram_normalize_rows else tendency_name},
        linewidths=0.5,
        ax=axes[1],
    )
    axes[1].set_title(f"{tendency_name} (Std) Feature Values per Cluster" + (" (Row-wise Normalized)" if histogram_normalize_rows else ""))
    axes[1].set_xlabel("Cluster Label")

    # Rotate/label colorbars (defensive if seaborn/mpl change handles)
    try:
        cbar1 = hm1.collections[0].colorbar
        cbar1.set_label("% Missing", rotation=270, labelpad=15)
    except Exception:
        pass
    try:
        cbar2 = hm2.collections[0].colorbar
        cbar2.set_label("Row-wise Normalized" if histogram_normalize_rows else tendency_name, rotation=270, labelpad=15)
    except Exception:
        pass

    # Shade y-tick backgrounds by view prefix
    # Build prefix -> color map (cycled)
    prefix_to_color: Dict[str, str] = {}
    view_names_in_order = list(views.keys())
    for i, vname in enumerate(view_names_in_order):
        sname = vname
        for old, new in view_shorten_map.items():
            if old in sname:
                sname = sname.replace(old, new)
        prefix_to_color[sname] = views_custom_hex_palette[i % len(views_custom_hex_palette)]

    def _prefix_from_label(lbl: str) -> str:
        # ytick label is either "<prefix>: <feature>" or truncated version, but prefix stays intact
        return lbl.split(":")[0].strip() if ":" in lbl else lbl

    for tick in axes[0].get_yticklabels():
        prefix = _prefix_from_label(tick.get_text())
        if prefix in prefix_to_color:
            rgba = (*matplotlib.colors.to_rgb(prefix_to_color[prefix]), 0.30)
            tick.set_backgroundcolor(rgba)
            tick.set_color("black")

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")

    plt.show()

    return cluster_missingness, cluster_tendency_scaled

# config
custom_hex_palette: List[str] = [
    "#0202CD", "#C71585", "#006400", "#8B4513", "#4B0082", "#E55526", "#008080",
    "#708090", "#2253C3", "#FF99CC", "#FFFF00", "#1D0E2A", "#AFF9A2",
]

# Canonical features and their desired direction (+1 higher=better, -1 lower=better)
canonical_feature_signs: Dict[str, int] = {
    'economic: Gross Value Added (GVA) per hour worked (£)': 1,
    'economic: Gross median weekly pay (£)': 1,
    'economic: Employment rate, ages 16-64 (%)': 1,
    'economic: Gross disposable household income, per head (£)': 1,
    'connectivity: Percentage of premises with gigabit-capable broadband (%)': 1,
    'skills: Proportion of the population aged 16-64 with NVQ3+ qualification (%)': 1,
    'skills: 19+ further education and skills participation (per 100,000 population)': -1,
    'health: Female Healthy Life Expectancy (years)': 1,
    'health: Male Healthy Life Expectancy (years)': 1,
    'health: Percentage of adults that currently smoke cigarettes (%)': -1,
    'health: Proportion of children overweight or obese at reception age (%)': -1,
    'health: Proportion of children overweight or obese at Year 6 age (%)': -1,
    'health: Proportion of adults (18+) overweight or obese (%)': -1,
    'health: Proportion of cancers diagnosed at stages 1 and 2 (%)': 1,
    'wellbeing: Mean happiness yesterday scored 0 (not at all) - 10 (completely)': 1,
}

# Meta columns fallback (import if available from preprocessing module)
try:
    from data_preprocessing import DEFAULT_META_COLS as _DEFAULT_META_COLS  # type: ignore
except Exception:
    _DEFAULT_META_COLS = ['Area Code', 'Area', 'Area Level']


def _pick_index_col(df: pd.DataFrame) -> Optional[str]:
    """Heuristic to choose the area identifier column to index by."""
    if "Local_Authority_Code" in df.columns:
        return "Local_Authority_Code"
    if "Area Code" in df.columns:
        return "Area Code"
    return None


def _remap_exclude_combine(
    labels: Iterable[Any],
    exclude_clusters: Optional[Iterable[Any]] = None,
    combine_groups: Optional[Iterable[Iterable[Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[Any, str]]:
    """
    Optionally exclude some cluster ids and/or merge groups of cluster ids.

    Returns
    -------
    mask : np.ndarray of bool
        True for rows retained (after exclusions).
    remapped_labels : np.ndarray
        Labels after applying the combine-groups mapping.
    legend_names : dict
        Mapping new_label -> display string (e.g., '2+4').
    """
    labels = np.asarray(labels)
    exclude_clusters = set(exclude_clusters or [])
    combine_groups = [set(g) for g in (combine_groups or [])]

    # Validate disjoint groups
    seen: set = set()
    for g in combine_groups:
        if seen & g:
            overlap = seen & g
            raise ValueError(f"combine_groups overlap not allowed (overlap: {sorted(overlap)})")
        seen |= g

    # Exclusion mask
    mask = ~np.isin(labels, list(exclude_clusters))

    # Build mapping
    remap: Dict[Any, Any] = {}
    legend_names: Dict[Any, str] = {}

    # Map combine groups to min(member)
    for g in combine_groups:
        new_id = min(g)
        for old in g:
            remap[old] = new_id
        legend_names[new_id] = "+".join(str(x) for x in sorted(g))

    # Pass-through for remaining labels (kept and not in any group)
    kept_unique = np.unique(labels[mask])
    for lab in kept_unique:
        if lab in exclude_clusters:
            continue
        if lab not in remap:
            remap[lab] = lab
            legend_names.setdefault(lab, str(lab))

    # Apply mapping
    remapped = np.array([remap.get(l, l) for l in labels])

    # Recompute mask after mapping (exclusions still apply)
    mask = ~np.isin(labels, list(exclude_clusters))
    return mask, remapped, legend_names


def _build_feature_table(
    views: Dict[str, pd.DataFrame],
    area_codes: Iterable[Any],
    meta_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Merge all numeric features from multiple views (prefixing with view name)
    and align to the given area_codes order.
    """
    meta_cols = list(meta_cols or _DEFAULT_META_COLS)
    parts: List[pd.DataFrame] = []

    for name, df in views.items():
        d = df.copy()
        d = d.replace(r'(?i)^na$', np.nan, regex=True)
        idx_col = _pick_index_col(d)
        if idx_col is not None:
            d = d.set_index(idx_col)
        d = d.drop(columns=[c for c in meta_cols if c in d.columns], errors="ignore")
        d = d.apply(pd.to_numeric, errors="coerce")
        d = d.sort_index()
        d.columns = [f"{name}: {c}" for c in d.columns]
        parts.append(d)

    if not parts:
        raise ValueError("No view data provided.")

    all_features = pd.concat(parts, axis=1)
    all_features = all_features.reindex(list(area_codes))
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]
    return all_features


def radar_plot_clusters_on_canonical_features(
    views: Dict[str, pd.DataFrame],
    labels: Iterable[Any],
    area_codes: Iterable[Any],
    canonical_feature_signs: Mapping[str, int],
    color_palette: Optional[Sequence[str]] = None,
    exclude_clusters: Optional[Iterable[Any]] = None,
    combine_groups: Optional[Iterable[Iterable[Any]]] = None,
    meta_cols: Optional[Iterable[str]] = None,
    label_wrap: int = 35,
    *,
    # comparison controls (kept from original)
    direction_align: bool = True,
    strict_complete_cases: bool = False,
    normalization_scope: str = "subset",
    # grouped subsets (one line per group)
    canonical_clusters: Optional[Iterable[Any]] = None,
    least_deprived_clusters: Optional[Iterable[Any]] = None,
    most_deprived_clusters: Optional[Iterable[Any]] = None,
    # always include all clusters in canonical when True
    canonical_use_all: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Plot radar profiles on a set of canonical features, either:
      - grouped mode: plots Canonical / Least deprived / Most deprived medians, or
      - legacy mode: plots one line per (possibly remapped) cluster.

    Parameters
    ----------
    views : dict[str, DataFrame]
        Mapping of view name -> dataframe containing features by area.
    labels : Iterable
        Cluster labels aligned with `area_codes`.
    area_codes : Iterable
        Row ids (area codes) aligned with `labels`.
    canonical_feature_signs : dict[str, int]
        Canonical feature list and desired direction (+1 / -1). (Direction arrows only.)
    color_palette : sequence[str] or None
        Colors used in legacy per-cluster mode; defaults to `custom_hex_palette`.
    exclude_clusters : iterable or None
        Cluster ids to exclude (legacy mode).
    combine_groups : iterable[iterable] or None
        E.g., [[2,4]] to merge 2 and 4 into a single profile (legacy mode).
    meta_cols : iterable of str or None
        Metadata columns to drop in view tables (defaults to `_DEFAULT_META_COLS`).
    label_wrap : int
        Word-wrap width for axis labels.
    direction_align : bool
        (Reserved) If True, would flip negative-direction features; currently not applied.
    strict_complete_cases : bool
        If True, only rows with no missing canonical values are kept.
    normalization_scope : {'subset','complete_cases'}
        Basis for min-max scaling.
    canonical_clusters, least_deprived_clusters, most_deprived_clusters : iterable or None
        Group definitions for grouped mode.
    canonical_use_all : bool
        If True, canonical profile always covers ALL clusters present.

    Returns
    -------
    pd.DataFrame or None
        - Grouped mode: DataFrame with rows for 'canonical'/'least deprived'/'most deprived' (present ones).
        - Legacy mode: per-cluster median profile table.
        - None if nothing could be plotted.
    """
    if color_palette is None:
        color_palette = list(custom_hex_palette)
    if meta_cols is None:
        meta_cols = list(_DEFAULT_META_COLS)

    # 1) Build combined features and keep only canonical ones (preserve order)
    all_features_df = _build_feature_table(views, area_codes, meta_cols=meta_cols)
    canonical_order = [f for f in canonical_feature_signs.keys() if f in all_features_df.columns]
    missing = [f for f in canonical_feature_signs.keys() if f not in all_features_df.columns]
    if missing:
        print(f"[radar] Warning: {len(missing)} canonical features not found and will be skipped.")
    if len(canonical_order) < 3:
        print(f"[radar] Not enough canonical features found (got {len(canonical_order)}). Aborting plot.")
        return None

    subset = all_features_df[canonical_order].copy()

    # 2) Strict complete-case handling
    labels_arr = np.asarray(list(labels))
    area_arr = np.asarray(list(area_codes))
    if len(labels_arr) != len(subset):
        # Align if needed (defensive)
        subset = subset.reindex(area_arr)
        if len(subset) != len(labels_arr):
            raise ValueError("Labels and area_codes lengths must match the number of aligned rows.")
    if strict_complete_cases:
        complete_idx = subset.dropna().index
        row_mask = subset.index.isin(complete_idx)
        subset = subset.loc[row_mask]
        labels_arr = labels_arr[row_mask]
        area_arr = area_arr[row_mask]

    # 3) (Reserved) direction flip — intentionally not modifying values now
    # if direction_align:
    #     for col in subset.columns:
    #         if canonical_feature_signs.get(col, 1) == -1:
    #             subset[col] = -subset[col]

    # 4) Normalization basis
    if normalization_scope not in {"subset", "complete_cases"}:
        raise ValueError("normalization_scope must be 'subset' or 'complete_cases'")
    basis = subset.dropna() if normalization_scope == "complete_cases" and not strict_complete_cases else subset
    min_vals = basis.min()
    max_vals = basis.max()
    norm = (subset - min_vals) / (max_vals - min_vals + 1e-8)

    if len(labels_arr) != len(norm):
        raise ValueError("Length of labels does not match number of rows after preprocessing.")

    # Grouped mode?
    grouped_mode = any(x is not None for x in (canonical_clusters, least_deprived_clusters, most_deprived_clusters))

    # 5) Radar setup
    features = canonical_order
    num_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.subplots_adjust(left=0.1, right=0.85)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_xticks([])

    def _plot_line(vals_series: pd.Series, color: str, marker: str, linestyle: str, filled: bool, label: str) -> None:
        vals_list = vals_series.tolist()
        vals = vals_list + [vals_list[0]]
        ax.plot(angles, vals, color=color, linewidth=2, linestyle=linestyle,
                marker=marker, label=label, markerfacecolor=color)
        if filled:
            ax.fill(angles, vals, color=color, alpha=0.1)
        # centroid marker
        x = np.array(vals[:-1]) * np.cos(angles[:-1])
        y = np.array(vals[:-1]) * np.sin(angles[:-1])
        cx, cy = x.mean(), y.mean()
        r = float(np.sqrt(cx**2 + cy**2))
        theta = float(np.arctan2(cy, cx))
        ax.plot([theta], [r], marker="+", color=color, markersize=16, markeredgewidth=2)

    # Grouped subsets mode
    if grouped_mode:
        # Determine canonical set
        try:
            unique_ids = sorted(set(labels_arr.tolist()))
        except Exception:
            unique_ids = list(dict.fromkeys(labels_arr.tolist()))  # stable unique if mixed types
        if canonical_use_all:
            canonical_clusters = unique_ids

        # Aggregator: median over rows in group
        def _agg(ids: Optional[Iterable[Any]]) -> Optional[pd.Series]:
            if not ids:
                return None
            ids_list = list(ids)
            mask = np.isin(labels_arr, ids_list)
            if not mask.any():
                return None
            return norm.loc[mask, features].median()

        prof_canon = _agg(canonical_clusters)
        prof_aff   = _agg(least_deprived_clusters)
        prof_depr  = _agg(most_deprived_clusters)

        def _fmt(name: str, ids: Optional[Iterable[Any]]) -> str:
            if not ids:
                return name
            try:
                ids_s = ",".join(str(i) for i in sorted(set(ids)))
            except Exception:
                ids_s = ",".join(str(i) for i in dict.fromkeys(ids))  # stable unique
            return f"{name} ({ids_s})"

        if prof_canon is not None:
            _plot_line(prof_canon,  "#2253C3", 'o', '-',  True,  _fmt("Canonical", canonical_clusters))
        if prof_aff is not None:
            _plot_line(prof_aff,   "#006400", '^', '--', False, _fmt("Least deprived", least_deprived_clusters))
        if prof_depr is not None:
            _plot_line(prof_depr, "#C71585", 's', '--', False, _fmt("Most deprived", most_deprived_clusters))

    else:
        # Legacy behavior: per-cluster profiles
        pre_mask, remapped_labels, legend_names = _remap_exclude_combine(
            labels_arr, exclude_clusters=exclude_clusters, combine_groups=combine_groups
        )
        norm = norm.loc[pre_mask]
        remapped_labels = remapped_labels[pre_mask]
        norm = norm.copy()
        norm["Cluster"] = remapped_labels
        cluster_profiles = norm.groupby("Cluster").median().sort_index()

        for i, cid in enumerate(list(cluster_profiles.index)):
            row = cluster_profiles.loc[cid, features]
            color = color_palette[i % len(color_palette)]
            _plot_line(row, color, 'o', '-', True, f"Cluster {legend_names.get(cid, str(cid))}")

    # Axis labels & direction arrows
    for i, raw_label in enumerate(features):
        angle = angles[i]
        label = textwrap.fill(raw_label, label_wrap)
        align = 'left' if 0 < angle < np.pi else 'right'
        ax.text(
            angle, 0.9, label, size=11,
            horizontalalignment=align, verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.25')
        )
        # Direction arrow (visual only; does not change data)
        sign = int(canonical_feature_signs.get(raw_label, 0))
        if sign != 0:
            ax.annotate(
                '', xy=(angle, 1.02), xytext=(angle, 0.98), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                color=("#006400" if sign > 0 else "#C71585"),
                                lw=2)
            )

    ax.legend(bbox_to_anchor=(1.15, 1), loc='center left', frameon=False, fontsize=12)
    plt.show()

    # Return what we plotted
    if grouped_mode:
        out: Dict[str, pd.Series] = {}
        if 'prof_canon' in locals() and prof_canon is not None: out['canonical'] = prof_canon
        if 'prof_aff'   in locals() and prof_aff   is not None: out['least_deprived']  = prof_aff
        if 'prof_depr'  in locals() and prof_depr  is not None: out['most_deprived']  = prof_depr
        return pd.DataFrame(out).T[features] if out else None
    else:
        return cluster_profiles