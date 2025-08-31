from typing import Dict, Sequence, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import textwrap
import seaborn as sns
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap
import networkx as nx

custom_hex_palette = [
    "#0202CD", "#C71585", "#006400", "#8B4513", "#4B0082", "#E55526", "#008080",
    "#708090", "#2253C3", "#FF99CC", "#FFFF00", "#1D0E2A", "#AFF9A2",
]

def plot_best_embedding(X, labels, tsne_grid=None, umap_grid=None, isomap_grid=None):
    """
    Plots 2D visualizations of clustered data using the best t-SNE, UMAP, and Isomap embeddings,
    selected based on Calinski-Harabasz score, with clusters visualized via colors, centroids,
    and coverage circles.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input feature matrix.

    labels : ndarray of shape (n_samples,)
        Cluster labels corresponding to X.

    tsne_grid : list of int, optional
        Perplexity values to try for t-SNE. Default: [5, 10, 30, 50].

    umap_grid : list of int, optional
        n_neighbors values to try for UMAP. Default: [5, 10, 20, 30].

    isomap_grid : list of int, optional
        n_neighbors values to try for Isomap. Default: [5, 10, 20, 30].

    Returns
    -------
    None
        Displays a matplotlib figure with 3 subplots.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as mpatches
    from sklearn.manifold import TSNE, Isomap
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    import umap

    methods = {
        't-SNE': {'param_grid': tsne_grid or [5, 10, 30, 50]},
        'UMAP': {'param_grid': umap_grid or [5, 10, 20, 30]},
        'Isomap': {'param_grid': isomap_grid or [5, 10, 20, 30]}
    }

    # Define font family
    # Set the global font family to 'sans-serif'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
        
    unique_labels = np.unique(labels)

    # Ensure sufficient colors are available
    if len(custom_hex_palette) < len(unique_labels):
        raise ValueError("Not enough colors in custom_hex_palette for the number of unique labels.")

    palette = custom_hex_palette[:len(unique_labels)]
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    best_results = {}

    for i, (method, cfg) in enumerate(methods.items()):
        best_score = -1
        best_embedding = None
        best_param = None
        best_ch = best_db = best_sil = None

        for param in cfg['param_grid']:
            try:
                if method == 't-SNE':
                    emb = TSNE(n_components=2, perplexity=param, random_state=1).fit_transform(X)
                elif method == 'UMAP':
                    emb = umap.UMAP(n_components=2, n_neighbors=param, random_state=1).fit_transform(X)
                else:  # Isomap
                    emb = Isomap(n_components=2, n_neighbors=param).fit_transform(X)

                sil = silhouette_score(emb, labels)
                ch = calinski_harabasz_score(emb, labels)
                db = davies_bouldin_score(emb, labels)

                if ch > best_score:
                    best_score = ch
                    best_embedding = emb
                    best_param = param
                    best_ch = ch
                    best_db = db
                    best_sil = sil
            except Exception as e:
                print(f"{method} failed at param {param}: {e}")
                continue

        best_results[method] = {
            'embedding': best_embedding,
            'param': best_param,
            'scores': {
                'silhouette': best_sil,
                'calinski_harabasz': best_ch,
                'davies_bouldin': best_db
            }
        }

        # Plot
        ax = axes[i]
        for label in unique_labels:
            cluster_points = best_embedding[np.array(labels) == label]
            sns.kdeplot(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                levels=2,
                thresh=0.3,
                color=color_map[label],
                linewidths=1,
                fill=True,
                alpha=0.1,
                ax=ax
            )
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       label=f'Cluster {label}', s=10, alpha=0.7, color=color_map[label])

        # Centroids
        for label in unique_labels:
            points = best_embedding[np.array(labels) == label]
            centroid = points.mean(axis=0)
            ax.scatter(*centroid, color=color_map[label], marker='+', s=120,
                       edgecolor='black', linewidth=1.2, zorder=5)

        ax.set_title(f"{method} (param={best_param}, sil={best_sil:.3f}, CH={best_ch:.1f}, DB={best_db:.3f})", fontsize=10)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_xticks([]) 
        ax.set_yticks([]) 

    plt.tight_layout()
    plt.show()

    return best_results


def radar_plot(view_name, views, results_by_view, preprocess_view, color_palette=None):
    """
    Generate a radar plot for cluster profiles from a specific view.

    Parameters:
    - view_name: str
        Name of the view (e.g., 'economic').
    - views: dict
        Dictionary of view DataFrames (each keyed by view name).
    - results_by_view: dict
        Output from clustering, must contain 'clusters' key per view.
    - preprocess_view: function
        Function to clean and preprocess a single view.
    - color_palette: list of str, optional
        List of HEX colors for clusters.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import textwrap

    if color_palette is None:
        color_palette = list(custom_hex_palette)


    # Preprocess and merge clusters
    df = preprocess_view(views[view_name], view_name)
    clusters_df = results_by_view[view_name]["clusters"]

    df = df.copy().reset_index()
    df["Area Code"] = df["Area Code"].astype(str)
    clusters_df["Area Code"] = clusters_df["Area Code"].astype(str)
    df = df.merge(clusters_df.rename(columns={"Cluster": "Cluster_from_result"}), on="Area Code", how="inner")
    if "Cluster" in df.columns:
        df = df.drop(columns=["Cluster"])  # Drop any existing ambiguous Cluster
    df = df.rename(columns={"Cluster_from_result": "Cluster"})

    # Identify and normalize indicator columns
    exclude_cols = {"Area Code", "Area Level", "Cluster", "index"}
    indicator_cols = [col for col in df.columns if col not in exclude_cols]
    df_scaled = df[indicator_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    df_scaled["Cluster"] = df["Cluster"].values
    # Filter out Cluster=-1 if it exists
    df_scaled = df_scaled[df_scaled["Cluster"] != -1]
    cluster_medians = df_scaled.groupby("Cluster").median()

    # Radar axes setup
    num_vars = len(indicator_cols)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # loop closure

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.subplots_adjust(left=0.1, right=0.75)
    ax.set_theta_offset(np.pi / 2.5)  # rotated to avoid title clash
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.grid(alpha=0.15)
    ax.set_xticklabels([])
    ax.set_yticks([])

    #  Plot cluster shapes and centroids
    for i, (cluster_id, row) in enumerate(cluster_medians.iterrows()):
        values = row.tolist()
        values += values[:1]  # append only the first value once
        ax.plot(angles, values, color=color_palette[i % len(color_palette)],
                linewidth=2, label=f"Cluster {cluster_id}")
        ax.fill(angles, values, color=color_palette[i % len(color_palette)], alpha=0.1)

        # Plot centroid marker
        x_vals = np.array(values[:-1]) * np.cos(angles[:-1])
        y_vals = np.array(values[:-1]) * np.sin(angles[:-1])
        centroid_x = x_vals.mean()
        centroid_y = y_vals.mean()
        r = np.sqrt(centroid_x**2 + centroid_y**2)
        theta = np.arctan2(centroid_y, centroid_x)
        ax.plot([theta], [r], marker="+", color=color_palette[i % len(color_palette)],
                markersize=18, markeredgewidth=2)

    # Variable Labels (in polar coordinates)
    label_offset = 1.05  # slight offset beyond the outermost radius

    for i, raw_label in enumerate(indicator_cols):
        angle = angles[i]
        wrapped_label = "\n".join(textwrap.wrap(raw_label, width=25))

        alignment = 'left' if 0 < angle < np.pi else 'right'

        ax.text(
            angle,
            label_offset,
            wrapped_label,
            horizontalalignment=alignment,
            verticalalignment='center',
            fontsize=13,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3')
        )
    # Legend with view name as title
    legend = ax.legend(bbox_to_anchor=(1.25, 0.8), loc='center left', frameon=False, fontsize=14, title=f"{view_name.title()} View")
    legend.get_title().set_fontsize(15)
    
    plt.show()

def plot_silhouette(embedding, labels, title=None, color_palette=None):
    """
    Draw a silhouette plot with a custom color per cluster.

    Parameters
    ----------
    embedding : array-like, shape (n_samples, n_features)
    labels    : array-like, shape (n_samples,)
    title     : str or None
    color_palette   : list of hex colors or None (defaults to custom_hex_palette)
    """
    labels = np.asarray(labels)
    unique_clusters = sorted(np.unique(labels))
    n_clusters = len(unique_clusters)
    if n_clusters < 2:
        raise ValueError("Silhouette score requires at least 2 clusters.")

    if color_palette is None:
        color_palette = list(custom_hex_palette)

    # map each cluster label to a color (cycle if k > len(color_palette))
    color_map = {c: color_palette[i % len(color_palette)] for i, c in enumerate(unique_clusters)}

    sample_sil = silhouette_samples(embedding, labels, metric="euclidean")
    avg_sil = silhouette_score(embedding, labels, metric="euclidean")

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10  # padding at the top

    for c in unique_clusters:
        sil_c = sample_sil[labels == c]
        sil_c.sort()
        size_c = sil_c.shape[0]
        y_upper = y_lower + size_c

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            sil_c,
            color=color_map[c],
            alpha=0.9,
            linewidth=0,
        )

        ax.text(-0.05, y_lower + 0.5 * size_c, str(c), va="center")
        y_lower = y_upper + 10  # spacing between clusters

    ax.axvline(x=avg_sil, linestyle="--", color='black', linewidth=1)  # average line

    ax.set_xlim([-0.1, 1.0])
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    if title is None:
        title = f"Silhouette plot (k={n_clusters}): avg={avg_sil:.4f}"
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(-0.1, 1.0, 12))

    plt.tight_layout()
    plt.show()

    return avg_sil, sample_sil

## UMAP Graph for MVSF outputs
# Normalize inputs
def _extract_mvsf_views(method_out: dict) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Expect MVSF_out to contain:
      - 'view_names' (list[str])
      - 'similarity_matrices' (list[np.ndarray], each n x n)
      - 'area_codes' (list[str])
    Returns: (dict view->S, area_codes)
    """
    for k in ("view_names", "similarity_matrices", "area_codes"):
        if k not in method_out:
            raise KeyError(f"method_out missing '{k}'.")
    view_names = list(method_out["view_names"])
    matrices   = list(method_out["similarity_matrices"])
    area_codes = list(map(str, method_out["area_codes"]))
    if len(view_names) != len(matrices):
        raise ValueError("view_names and similarity_matrices length mismatch.")
    n = len(area_codes)
    for i, S in enumerate(matrices):
        if not (isinstance(S, np.ndarray) and S.shape == (n, n)):
            raise ValueError(f"similarity_matrices[{i}] has shape {getattr(S,'shape',None)}, expected {(n,n)}.")
    sims_by_view = {vn: S for vn, S in zip(view_names, matrices)}
    return sims_by_view, area_codes


# Fuse similarity, convert to distances, embed with UMAP
def _fuse_similarity(sims_by_view: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Simple weighted mean (defaults to uniform). Symmetrize + unit diagonal."""
    views = list(sims_by_view.keys())
    W = np.ones(len(views), float) if weights is None else np.array([weights.get(v, 1.0) for v in views], float)
    W = W / (W.sum() + 1e-12)
    S_stack = np.stack([sims_by_view[v] for v in views], axis=0)  # (m, n, n)
    S = np.tensordot(W, S_stack, axes=(0, 0))
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S

def _similarity_to_distance(S: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Turn similarity into a nonnegative, small-is-near distance. Scale-agnostic."""
    # Use (max(S)-S) then rescale to [0,1]
    m = np.nanmax(S)
    D = m - S
    D[D < 0] = 0
    D = D / (np.nanmax(D) + eps)
    np.fill_diagonal(D, 0.0)
    return D

def embed_umap_from_similarity(S: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1,
                               random_state: int = 42) -> np.ndarray:
    """Return (n,2) embedding from an n×n similarity matrix."""
    D = _similarity_to_distance(S)
    reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric="precomputed",
            random_state=random_state
        )
    X = reducer.fit_transform(D)
    return X


# Build edges from fused similarity
def build_similarity_edges(
    S_fused: np.ndarray,
    sims_by_view: Dict[str, np.ndarray],
    *,
    mode: str = "knn",
    k: int = 10,
    min_sim: float = 0.0,
    ego_center: Optional[int] = None,
) -> pd.DataFrame:
    """
    Return an edge table with columns:
      ['i','j','w','dominant_view','w_dominant','view_maxes'].
    - i,j are integer node indices (i<j)
    - w is fused similarity
    - dominant_view is argmax over views for that pair (string)
    - w_dominant is the winning view similarity
    - view_maxes is a dict of {view: S_view[i,j]} (handy for debugging)
    """
    n = S_fused.shape[0]
    assert S_fused.shape == (n, n)

    # choose candidate pairs
    pairs = set()
    if mode == "knn":
        for i in range(n):
            idx = np.argsort(S_fused[i, :])[::-1]
            idx = [j for j in idx if j != i][:max(0, k)]
            for j in idx:
                a, b = (i, j) if i < j else (j, i)
                pairs.add((a, b))
    elif mode == "threshold":
        I, J = np.where((S_fused >= min_sim) & (~np.eye(n, dtype=bool)))
        for i, j in zip(I, J):
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    elif mode == "ego":
        if ego_center is None:
            raise ValueError("ego mode requires ego_center (node index).")
        sims = S_fused[ego_center, :].copy()
        sims[ego_center] = -np.inf
        nbrs = np.argsort(sims)[::-1][:max(0, k)]
        for j in nbrs:
            a, b = (ego_center, j) if ego_center < j else (j, ego_center)
            pairs.add((a, b))
    else:
        raise ValueError("mode must be 'knn', 'threshold', or 'ego'.")

    # compute per-pair, per-view maxima and dominant view
    rows = []
    V = list(sims_by_view.keys())
    for (i, j) in pairs:
        per_view = {v: float(sims_by_view[v][i, j]) for v in V}
        dominant = max(per_view, key=per_view.get)
        rows.append({
            "i": i, "j": j,
            "w": float(S_fused[i, j]),
            "dominant_view": dominant,
            "w_dominant": per_view[dominant],
            "view_maxes": per_view
        })
    df = pd.DataFrame(rows)
    return df.sort_values("w", ascending=False).reset_index(drop=True)


# Community detection on fused graph
def detect_communities_from_edges(n_nodes: int, edges_df: pd.DataFrame, weight_col: str = "w") -> np.ndarray:
    """Greedy modularity communities on weighted, undirected graph. Returns community id per node."""
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for _, r in edges_df.iterrows():
        i, j, w = int(r["i"]), int(r["j"]), float(r[weight_col])
        if i == j: continue
        if G.has_edge(i, j):
            G[i][j]["weight"] = max(G[i][j]["weight"], w)
        else:
            G.add_edge(i, j, weight=w)
    coms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    label = np.full(n_nodes, -1, dtype=int)
    for cid, nodes in enumerate(coms):
        for u in nodes:
            label[u] = cid
    return label


def summarize_community_dominant_view(edges_df: pd.DataFrame, comm_labels: np.ndarray) -> pd.DataFrame:
    """For each community, which view dominates internal edges most often?"""
    rows = []
    for cid in np.unique(comm_labels[comm_labels >= 0]):
        mask = (comm_labels[edges_df["i"].values] == cid) & (comm_labels[edges_df["j"].values] == cid)
        sub = edges_df.loc[mask]
        if sub.empty:
            rows.append({"community": int(cid), "edges": 0, "dominant_view": None})
            continue
        counts = sub["dominant_view"].value_counts()
        dom_view = counts.idxmax()
        rows.append({
            "community": int(cid),
            "edges": int(len(sub)),
            "dominant_view": dom_view,
            **{f"count_{k}": int(v) for k, v in counts.items()}
        })
    return pd.DataFrame(rows).sort_values("community")


# Plot all nodes with UMAP + colored edges
def plot_mvsf_graph(
    *,
    method_out: dict,
    mode: str = "full", # "full" or "ego"
    ego_area_code: Optional[str] = None,
    top_k_edges: int = 10, # used for full (knn per node) and ego
    threshold: Optional[float] = None,  # if set and mode="full", use threshold mode instead of knn
    n_neighbors_umap: int = 20,
    min_dist_umap: float = 0.1,
    view_palette: Optional[Sequence[str]] = None,   # optional custom view colors
    node_alpha: float = 0.9,
    edge_alpha: float = 0.6,
    max_edges_draw: int = 20000, # safety cap for dense graphs
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 200,
    title: Optional[str] = None,
):
    """
    Build a fused similarity graph, embed with UMAP, draw edges colored by
    the view that contributes most to each pair.

    Returns
    -------
    fig, ax, result
    where result is a dict with:
      - 'pos': np.ndarray (n, 2) coordinates
      - 'area_codes': list[str]
      - 'edges': pd.DataFrame (i,j,w,dominant_view,w_dominant,view_maxes)
      - 'communities': np.ndarray of ints per node
      - 'community_summary': pd.DataFrame with dominant view per community
    """
    sims_by_view, area_codes = _extract_mvsf_views(method_out)
    views = list(sims_by_view.keys())

    # Colors for views
    if view_palette is None:
        view_palette = ["#4B0082", "#E55526", "#008080", "#708090", "#2253C3", "#FF99CC", "#1D0E2A", "#AFF9A2", "#FFFF00"]
    view_colors = {v: view_palette[i % len(view_palette)] for i, v in enumerate(views)}

    # Fused similarity + embedding
    S_fused = _fuse_similarity(sims_by_view)
    X = embed_umap_from_similarity(S_fused, n_neighbors=n_neighbors_umap, min_dist=min_dist_umap)

    # Build edges
    if mode == "ego":
        if ego_area_code is None:
            raise ValueError("mode='ego' requires ego_area_code.")
        if str(ego_area_code) not in area_codes:
            raise KeyError(f"{ego_area_code} not found in area_codes.")
        ego_idx = area_codes.index(str(ego_area_code))
        edges = build_similarity_edges(S_fused, sims_by_view, mode="ego", k=int(top_k_edges), ego_center=ego_idx)
        node_mask = np.zeros(len(area_codes), dtype=bool)
        node_mask[[ego_idx] + list(edges["i"].values) + list(edges["j"].values)] = True
        show_idx = np.where(node_mask)[0].tolist()
    else:
        if threshold is not None:
            edges = build_similarity_edges(S_fused, sims_by_view, mode="threshold", min_sim=float(threshold))
        else:
            edges = build_similarity_edges(S_fused, sims_by_view, mode="knn", k=int(top_k_edges))
        show_idx = list(range(len(area_codes)))

    # Communities on the shown subgraph
    comm = detect_communities_from_edges(len(area_codes), edges)
    comm_summary = summarize_community_dominant_view(edges, comm)

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis("off")

    # nodes
    X_show = X[show_idx]
    ax.scatter(X_show[:, 0], X_show[:, 1], s=18, c="#2b2b2b", alpha=node_alpha, linewidths=0)

    # edges colored by dominant view
    to_draw = edges.head(max_edges_draw) if len(edges) > max_edges_draw else edges
    for _, r in to_draw.iterrows():
        i, j = int(r["i"]), int(r["j"])
        if i not in show_idx or j not in show_idx:
            continue
        c = view_colors.get(r["dominant_view"], "#999999")
        xi, yi = X[i]; xj, yj = X[j]
        ax.plot([xi, xj], [yi, yj], color=c, alpha=edge_alpha, linewidth=0.7)

    # legend for views
    patches = [mpatches.Patch(color=view_colors[v], label=v) for v in views]
    ax.legend(handles=patches, title="Dominant view (edge)", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    if title is None:
        if mode == "ego":
            title = f"UMAP graph — ego of {ego_area_code} (top {top_k_edges})"
        elif threshold is not None:
            title = f"UMAP graph — fused similarity (threshold ≥ {threshold:.3g})"
        else:
            title = f"UMAP graph — fused similarity (kNN, k={top_k_edges})"
    ax.set_title(title, fontsize=12, pad=6, fontweight="bold")

    result = {
        "pos": X, "area_codes": area_codes,
        "edges": edges, "communities": comm,
        "community_summary": comm_summary
    }
    return fig, ax, result