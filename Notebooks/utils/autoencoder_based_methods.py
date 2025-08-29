from __future__ import annotations

import os
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from data_preprocessing import (
    get_combined_views_union,  # returns (X_union_df, area_codes_list)
    preprocess_view,
    exclude_las,
    DEFAULT_META_COLS,
)

RANDOM_STATE = 19042022

# Reproducibility & determinism
def set_seed(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic PyTorch math
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

# NaN-aware robust scaling
def robust_scale_nanaware(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Feature-wise median/IQR scaling on observed values only.
    Missing values are filled with the feature median (for scaling only),
    then transformed to (x - median) / IQR. IQR==0 -> scale 1.0.
    """
    X = X.astype(float, copy=False)
    n, p = X.shape
    meds = np.zeros(p, dtype=float)
    iqrs = np.ones(p, dtype=float)
    Xs = np.empty_like(X, dtype=float)

    for j in range(p):
        col = X[:, j]
        obs = col[~np.isnan(col)]
        if obs.size < 2:
            med, iqr = 0.0, 1.0
        else:
            q1, q3 = np.percentile(obs, [25, 75])
            med = float(np.median(obs))
            iqr = float(q3 - q1) or 1.0
        meds[j], iqrs[j] = med, iqr
        filled = np.where(np.isnan(col), med, col)
        Xs[:, j] = (filled - med) / iqr

    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    return Xs, meds, iqrs

# Models
class MaskedAutoencoder(nn.Module):
    """
    Simple masked AE for early integration (single union matrix).
    Loss is masked MSE; input is scaled features (mask provided separately).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

class MultiBranchAutoencoder(nn.Module):
    """
    Per-view encoders → shared latent (mean aggregation) → single decoder to all features.
    Used for intermediate integration.
    """
    def __init__(self, input_dims: List[int], hidden_dim: int = 128, latent_dim: int = 10):
        super().__init__()
        self.input_dims = input_dims
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            ) for d in input_dims
        ])
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, sum(input_dims))
        )

    def forward(self, inputs: List[torch.Tensor]):
        # inputs: list of tensors (n x d_i)
        z_list = [enc(x) for enc, x in zip(self.encoders, inputs)]
        z = torch.stack(z_list, dim=0).mean(dim=0)  # shared bottleneck
        recon = self.decoder(z)  # (n x sum(d_i))
        return recon, z

# Losses
def masked_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    Masked MSE normalized per sample to avoid bias when rows have many NaNs.
    """
    diff2 = (y_pred - y_true) ** 2 * mask
    per_row = diff2.sum(dim=1) / (mask.sum(dim=1) + eps)
    return per_row.mean()

# Builders / preparation
def _prepare_union_for_ae(views: Dict[str, pd.DataFrame]):
    """
    Returns:
      X_t (torch.FloatTensor, scaled), M_t (mask 0/1), area_codes (list), X_scaled (np), mask (np)
    """
    X_union, area_codes = get_combined_views_union(views)  # DataFrame (n x p)
    X = X_union.values.astype(float)
    mask = ~np.isnan(X)
    X_scaled, _, _ = robust_scale_nanaware(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    M_t = torch.tensor(mask.astype(float), dtype=torch.float32)
    return X_t, M_t, list(area_codes), X_scaled, mask

def _prepare_per_view_for_ae(views: Dict[str, pd.DataFrame]):
    """
    Clean & align all views; per-view scaling & masks (NaN-aware).
    Returns:
      X_list: list[np.ndarray], M_list: list[np.ndarray], input_dims: list[int],
      X_concat: np.ndarray, M_concat: np.ndarray, area_codes (list[str]), view_names (list[str])
    """
    cleaned = {}
    all_idx = set()
    for vname, df in views.items():
        dfc = preprocess_view(df.copy())
        dfc = exclude_las(dfc.copy(), vname)
        if "Local_Authority_Code" not in dfc.columns:
            raise KeyError(f"[{vname}] 'Local_Authority_Code' missing after preprocessing.")
        dfc = dfc.set_index("Local_Authority_Code")
        dfc = dfc.drop(columns=[c for c in DEFAULT_META_COLS if c in dfc.columns], errors="ignore")
        dfc = dfc.apply(pd.to_numeric, errors="coerce").sort_index()
        cleaned[vname] = dfc
        all_idx.update(dfc.index)

    area_codes = sorted(all_idx)
    idx = pd.Index(area_codes, name="Local_Authority_Code")

    X_list, M_list, input_dims = [], [], []
    for vname in cleaned:
        A = cleaned[vname].reindex(idx).to_numpy(dtype=float)
        mask = ~np.isnan(A)
        A_scaled, _, _ = robust_scale_nanaware(A)
        A_scaled[np.isnan(A_scaled)] = 0.0
        X_list.append(A_scaled)
        M_list.append(mask.astype(float))
        input_dims.append(A_scaled.shape[1])

    X_concat = np.concatenate(X_list, axis=1) if X_list else np.zeros((len(idx), 0))
    M_concat = np.concatenate(M_list, axis=1) if M_list else np.zeros((len(idx), 0))
    return X_list, M_list, input_dims, X_concat, M_concat, area_codes, list(cleaned.keys())

# Co-association from label arrays (with missing = -1)
def _coassoc_from_labels(label_list: List[np.ndarray], missing_label: int = -1) -> np.ndarray:
    """
    Builds S in [0,1]: S_ij = fraction of views (among those where both i and j are assigned)
    in which i and j share the same cluster. Labels == missing_label are ignored.
    """
    if not label_list:
        raise ValueError("label_list is empty.")
    n = len(label_list[0])
    agree = np.zeros((n, n), dtype=float)
    count = np.zeros((n, n), dtype=float)

    for lab in label_list:
        if lab.shape[0] != n:
            raise ValueError("Label vectors have inconsistent lengths.")
        valid = lab != missing_label
        pos = np.where(valid)[0]
        count[np.ix_(pos, pos)] += 1.0
        same = (lab[pos, None] == lab[None, pos]).astype(float)
        agree[np.ix_(pos, pos)] += same

    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.true_divide(agree, count)
        S[~np.isfinite(S)] = 0.0

    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S

# Early integration - Masked AE (union)
def run_early_masked_ae_pca_kmeans(
    views: Dict[str, pd.DataFrame],
    latent_dim_range: Iterable[int] = range(5, 11, 5),
    n_pca_range: Iterable[int] = range(2, 6),
    k_range: Iterable[int] = range(4, 16),
    hidden_dim: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    n_init: int | str = "auto",
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
):
    """
    Union features → Robust scale (NaN-aware) → Masked AE → PCA → KMeans grid.
    Dedupes grid to one row per (PCA_n_components, K) keeping the best Latent dim.
    """
    set_seed(random_state)

    X_t, M_t, area_codes, Xs, _ = _prepare_union_for_ae(views)
    n, p = X_t.shape

    best_global = {"sil": -1.0}
    best_by_pk: dict[tuple[int, int], dict] = {}  # (pca,k) -> {"Silhouette": float, "Latent": int}

    best_embedding = None
    best_labels = None
    best_assignments_df = None

    for latent in latent_dim_range:
        model = MaskedAutoencoder(input_dim=p, hidden_dim=hidden_dim, latent_dim=int(latent))
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            recon, z = model(X_t)
            loss = masked_mse_loss(recon, X_t, M_t)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            _, Z = model(X_t)
        Z_np = Z.detach().cpu().numpy()
        Z_np = np.nan_to_num(Z_np, nan=0.0, posinf=0.0, neginf=0.0)

        latent_dim = Z_np.shape[1]
        for n_pca in n_pca_range:
            if n_pca > min(latent_dim, n):
                if verbose:
                    print(f"[EARLY-AE] skip PCA={n_pca} > dim={min(latent_dim, n)}")
                continue
            emb = PCA(n_components=int(n_pca), svd_solver="full", random_state=random_state).fit_transform(Z_np)

            for k in k_range:
                if k < 2 or k > n:
                    continue
                km = KMeans(n_clusters=int(k), init="k-means++", n_init=n_init, random_state=random_state)
                labels = km.fit_predict(emb)
                try:
                    sil = silhouette_score(emb, labels)
                except ValueError:
                    sil = -1.0

                if verbose:
                    print(f"[EARLY-AE] n={n}, Latent={latent}, PCA={n_pca}, K={k} → Sil={sil:.4f}")

                key = (int(n_pca), int(k))
                if (key not in best_by_pk) or (sil > best_by_pk[key]["Silhouette"]):
                    best_by_pk[key] = {"Silhouette": float(sil), "Latent": int(latent)}

                if sil > best_global["sil"]:
                    best_global = {"sil": float(sil), "params": (int(n_pca), int(k)), "Latent": int(latent)}
                    best_embedding = emb
                    best_labels = labels
                    best_assignments_df = pd.DataFrame({"Local_Authority_Code": area_codes, "Cluster": labels})

    grid_results = pd.DataFrame(
        [{"PCA_n_components": p, "K": k, "Silhouette": rec["Silhouette"], "Latent": rec["Latent"]}
         for (p, k), rec in best_by_pk.items()]
    ).sort_values(["PCA_n_components", "K"], ignore_index=True)

    if best_labels is None:
        if verbose:
            print("\nNo valid clustering result found (EARLY-AE).")
        return {
            "S_coassoc": None, "area_codes": area_codes, "labels": np.array([]),
            "embedding": np.empty((0, 0)), "assignments_df": pd.DataFrame(columns=["Local_Authority_Code","Cluster"]),
            "grid_results": grid_results, "clusterings": None,
        }

    if verbose:
        n_pca_best, k_best = best_global["params"]
        print(f"\nBest (EARLY-AE): Latent={best_global['Latent']}, PCA={n_pca_best}, K={k_best}, Silhouette={best_global['sil']:.4f}")

    return {
        "area_codes": area_codes,
        "labels": best_labels,
        "embedding": best_embedding,
        "assignments_df": best_assignments_df,
        "grid_results": grid_results,  # deduped by (PCA,K) with winning Latent
    }


# Intermnediate integration - Multi-branch AE with Shared Bottleneck
def run_intermediate_multibranch_ae_pca_kmeans(
    views: Dict[str, pd.DataFrame],
    latent_dim_range: Iterable[int] = range(5, 11, 5),
    n_pca_range: Iterable[int] = range(2, 6),
    k_range: Iterable[int] = range(4, 16),
    hidden_dim: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    n_init: int | str = "auto",
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
):
    """
    Per-view encoders -> shared bottleneck (mean) -> single decoder.
    PCA on latent Z, KMeans on embedding. Deduped grid with winning Latent stored.
    """
    set_seed(random_state)

    X_list, M_list, input_dims, X_concat, M_concat, area_codes, _ = _prepare_per_view_for_ae(views)
    n = len(area_codes)
    # Convert to torch
    X_t_list = [torch.tensor(x, dtype=torch.float32) for x in X_list]
    M_concat_t = torch.tensor(M_concat, dtype=torch.float32)
    X_concat_t = torch.tensor(X_concat, dtype=torch.float32)

    best_global = {"sil": -1.0}
    best_by_pk: dict[tuple[int, int], dict] = {}

    best_embedding = None
    best_labels = None
    best_assignments_df = None

    for latent in latent_dim_range:
        model = MultiBranchAutoencoder(input_dims=input_dims, hidden_dim=hidden_dim, latent_dim=int(latent))
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            recon, z = model(X_t_list)
            # recon is (n x sum d_i)
            loss = masked_mse_loss(recon, X_concat_t, M_concat_t)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            _, Z = model(X_t_list)
        Z_np = Z.detach().cpu().numpy()
        Z_np = np.nan_to_num(Z_np, nan=0.0, posinf=0.0, neginf=0.0)

        latent_dim = Z_np.shape[1]
        for n_pca in n_pca_range:
            if n_pca > min(latent_dim, n):
                if verbose:
                    print(f"[MBAE] skip PCA={n_pca} > dim={min(latent_dim, n)}")
                continue
            emb = PCA(n_components=int(n_pca), svd_solver="full", random_state=random_state).fit_transform(Z_np)

            for k in k_range:
                if k < 2 or k > n:
                    continue
                km = KMeans(n_clusters=int(k), init="k-means++", n_init=n_init, random_state=random_state)
                labels = km.fit_predict(emb)
                try:
                    sil = silhouette_score(emb, labels)
                except ValueError:
                    sil = -1.0

                if verbose:
                    print(f"[MBAE] n={n}, Latent={latent}, PCA={n_pca}, K={k} → Sil={sil:.4f}")

                key = (int(n_pca), int(k))
                if (key not in best_by_pk) or (sil > best_by_pk[key]["Silhouette"]):
                    best_by_pk[key] = {"Silhouette": float(sil), "Latent": int(latent)}

                if sil > best_global["sil"]:
                    best_global = {"sil": float(sil), "params": (int(n_pca), int(k)), "Latent": int(latent)}
                    best_embedding = emb
                    best_labels = labels
                    best_assignments_df = pd.DataFrame({"Local_Authority_Code": area_codes, "Cluster": labels})

    grid_results = pd.DataFrame(
        [{"PCA_n_components": p, "K": k, "Silhouette": rec["Silhouette"], "Latent": rec["Latent"]}
         for (p, k), rec in best_by_pk.items()]
    ).sort_values(["PCA_n_components", "K"], ignore_index=True)

    if best_labels is None:
        if verbose:
            print("\nNo valid clustering result found (MBAE).")
        return {
            "S_coassoc": None, "area_codes": area_codes, "labels": np.array([]),
            "embedding": np.empty((0, 0)), "assignments_df": pd.DataFrame(columns=["Local_Authority_Code","Cluster"]),
            "grid_results": grid_results, "clusterings": None,
        }

    if verbose:
        n_pca_best, k_best = best_global["params"]
        print(f"\nBest (MBAE): Latent={best_global['Latent']}, PCA={n_pca_best}, K={k_best}, Silhouette={best_global['sil']:.4f}")

    return {
        "area_codes": area_codes,
        "labels": best_labels,
        "embedding": best_embedding,
        "assignments_df": best_assignments_df,
        "grid_results": grid_results,  # deduped by (PCA,K) with winning Latent
    }


# Late integration - AE Ensemble with Consensus Clustering (AECC)
def run_late_ae_coassoc_pca_kmeans(
    views: Dict[str, pd.DataFrame],
    latent_dim_range: Iterable[int] = range(5, 11, 5),
    n_pca_range: Iterable[int] = range(2, 6),  # used for both per-view PCA and consensus PCA (tied)
    k_range: Iterable[int] = range(4, 16),     # used for both per-view and final KMeans (tied)
    hidden_dim: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    n_init: int | str = "auto",
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
):
    """
    Train one AE per view on aligned data, get per-view latent Z_v, PCA(Z_v, n_pca) and KMeans(k)
    → build co-association S (ignoring missing rows per view) → PCA(S, n_pca) → KMeans(k).
    Dedupes by (PCA_n_components, K) with the winning Latent stored.
    """
    set_seed(random_state)

    # Prepare per-view arrays and masks
    X_list, M_list, input_dims, X_concat, M_concat, area_codes, view_names = _prepare_per_view_for_ae(views)
    n = len(area_codes)

    # Train per-view AEs and cache their latents
    Z_per_view: List[np.ndarray] = []
    row_has_obs: List[np.ndarray] = []  # which rows have any observed feature in that view
    for A_scaled, M in zip(X_list, M_list):
        # mask per row: any observed?
        row_obs = (M.sum(axis=1) > 0)
        row_has_obs.append(row_obs)

    best_global = {"sil": -1.0}
    best_by_pk: dict[tuple[int, int], dict] = {}

    best_embedding = None
    best_labels = None
    best_assignments_df = None
    best_S = None
    best_clusterings = None

    for latent in latent_dim_range:
        # Train an AE per view for this latent dim and collect Z_v
        Z_per_view = []
        for A_scaled, M in zip(X_list, M_list):
            X_t = torch.tensor(A_scaled, dtype=torch.float32)
            M_t = torch.tensor(M, dtype=torch.float32)
            mae = MaskedAutoencoder(input_dim=A_scaled.shape[1], hidden_dim=hidden_dim, latent_dim=int(latent))
            mae.train()
            opt = torch.optim.Adam(mae.parameters(), lr=lr)
            for _ in range(epochs):
                opt.zero_grad(set_to_none=True)
                recon, z = mae(X_t)
                loss = masked_mse_loss(recon, X_t, M_t)
                loss.backward()
                opt.step()
            mae.eval()
            with torch.no_grad():
                _, Zv = mae(X_t)
            Z_per_view.append(np.nan_to_num(Zv.detach().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0))

        # For each PCA dim & K (tied across stages)
        for n_pca in n_pca_range:
            # Per-view PCA + KMeans → label arrays (with -1 where row had no obs in that view)
            labels_per_view = []
            for Z, valid_rows in zip(Z_per_view, row_has_obs):
                if n_pca > min(Z.shape[1], n):
                    if verbose:
                        print(f"[AE-CFE] skip per-view PCA={n_pca} > dim={min(Z.shape[1], n)}")
                    continue
                Zp = PCA(n_components=int(n_pca), svd_solver="full", random_state=random_state).fit_transform(Z)
                for k in k_range:
                    pass  # we'll cluster per view inside the k-loop below

            # We'll compute per-view labels inside each K loop to avoid holding all variants.
            for k in k_range:
                if k < 2 or k > n:
                    continue

                labels_per_view = []
                for Z, valid_rows in zip(Z_per_view, row_has_obs):
                    if n_pca > min(Z.shape[1], n):
                        continue
                    Zp = PCA(n_components=int(n_pca), svd_solver="full", random_state=random_state).fit_transform(Z)
                    km1 = KMeans(n_clusters=int(k), init="k-means++", n_init=n_init, random_state=random_state)
                    lab = km1.fit_predict(Zp)
                    # rows with no observed features in this view are marked missing
                    lab = np.where(valid_rows, lab, -1)
                    labels_per_view.append(lab)

                if not labels_per_view:
                    continue

                # Co-association
                S = _coassoc_from_labels(labels_per_view, missing_label=-1)
                S = np.clip(S, 0.0, 1.0)
                S = 0.5 * (S + S.T)
                np.fill_diagonal(S, 1.0)

                # Consensus PCA & final KMeans (tie pca dim with n_pca)
                if n_pca > n:
                    continue
                cons_emb = PCA(n_components=int(n_pca), svd_solver="full", random_state=random_state).fit_transform(S)

                km2 = KMeans(n_clusters=int(k), init="k-means++", n_init=n_init, random_state=random_state)
                final_labels = km2.fit_predict(cons_emb)

                try:
                    sil = silhouette_score(cons_emb, final_labels)
                except ValueError:
                    sil = -1.0

                if verbose:
                    print(f"[AE-CFE] n={n}, Latent={latent}, PCA={n_pca}, K={k} → Sil={sil:.4f}")

                key = (int(n_pca), int(k))
                if (key not in best_by_pk) or (sil > best_by_pk[key]["Silhouette"]):
                    best_by_pk[key] = {"Silhouette": float(sil), "Latent": int(latent)}

                if sil > best_global["sil"]:
                    best_global = {"sil": float(sil), "params": (int(n_pca), int(k)), "Latent": int(latent)}
                    best_embedding = cons_emb
                    best_labels = final_labels
                    best_assignments_df = pd.DataFrame({"Local_Authority_Code": area_codes, "Cluster": final_labels})
                    best_S = S
                    # per-view clusterings for the best config (current n_pca,k)
                    best_clusterings = {vn: lab for vn, lab in zip(view_names, labels_per_view)}

    grid_results = pd.DataFrame(
        [{"PCA_n_components": p, "K": k, "Silhouette": rec["Silhouette"], "Latent": rec["Latent"]}
         for (p, k), rec in best_by_pk.items()]
    ).sort_values(["PCA_n_components", "K"], ignore_index=True)

    if best_labels is None:
        if verbose:
            print("\nNo valid clustering result found (AE-CFE).")
        return {
            "S_coassoc": None, "area_codes": area_codes, "labels": np.array([]),
            "embedding": np.empty((0, 0)), "assignments_df": pd.DataFrame(columns=["Local_Authority_Code","Cluster"]),
            "grid_results": grid_results, "clusterings": {},
        }

    if verbose:
        n_pca_best, k_best = best_global["params"]
        print(f"\nBest (AE-CFE): Latent={best_global['Latent']}, PCA={n_pca_best}, K={k_best}, Silhouette={best_global['sil']:.4f}")

    return {
        "S_coassoc": best_S,
        "area_codes": area_codes,
        "labels": best_labels,
        "embedding": best_embedding,
        "assignments_df": best_assignments_df,
        "grid_results": grid_results,  # deduped by (PCA,K) with winning Latent
        "clusterings": best_clusterings,  # per-view label arrays (−1 where missing)
    }