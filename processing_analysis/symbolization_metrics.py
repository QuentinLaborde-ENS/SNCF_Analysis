# -*- coding: utf-8 -*-

import os
import glob
import pickle
from pathlib import Path
 
import matplotlib.pyplot as plt 

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cophenet
from sklearn.manifold import MDS


def offdiag_values(D: np.ndarray) -> np.ndarray:
    """Return upper-triangular (i<j) entries as a 1D vector."""
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    return D[iu]


def shannon_entropy_from_hist(x: np.ndarray, bins: int = 50, eps: float = 1e-12) -> float:
    """Shannon entropy of histogram-based discrete distribution."""
    hist, _ = np.histogram(x, bins=bins, density=True)
    p = hist / (hist.sum() + eps)
    p = p[p > 0]
    return float(-(p * np.log(p + eps)).sum())


def mds_stress(D: np.ndarray, n_components: int = 2, random_state: int = 0) -> float:
    """
    Compute (normalized) raw stress for classical metric MDS on a precomputed distance matrix.
    Note: this is not exactly Kruskal stress-1, but a stable, interpretable proxy.
    """
    n = D.shape[0]
    if n < 3:
        return float("nan")

    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        metric=True,
        random_state=random_state,
        n_init=4,
        max_iter=500,
        normalized_stress="auto" if "normalized_stress" in MDS().get_params() else False,
    )
    Y = mds.fit_transform(D)

    # Reconstruct Euclidean distances in embedding
    diff = Y[:, None, :] - Y[None, :, :]
    Dh = np.sqrt((diff ** 2).sum(axis=-1))

    iu = np.triu_indices(n, k=1)
    num = ((Dh[iu] - D[iu]) ** 2).sum()
    den = (D[iu] ** 2).sum() + 1e-12
    return float(np.sqrt(num / den))


def spectral_metrics_from_distance(D: np.ndarray, sigma: float = None, topk: int = 5) -> dict:
    """
    Convert distance to similarity and compute eigenvalue-based metrics.
    Similarity: S = exp(-D/sigma). sigma default = median off-diagonal distance.
    """
    v = offdiag_values(D)
    if v.size == 0:
        return {"sigma": float("nan"), "eig_sum": float("nan"), "eig_top1": float("nan")}

    if sigma is None:
        sigma = np.median(v) + 1e-12

    S = np.exp(-D / sigma)
    # ensure symmetry
    S = 0.5 * (S + S.T)

    # eigenvalues
    eigvals = np.linalg.eigvalsh(S)
    eigvals = np.sort(eigvals)[::-1]  # descending

    eig_sum = float(np.sum(eigvals))
    top = eigvals[:topk]
    energy_topk = float(np.sum(top) / (eig_sum + 1e-12))

    out = {
        "sigma": float(sigma),
        "eig_sum": eig_sum,
        "eig_top1": float(eigvals[0]),
        "eig_energy_top5": energy_topk if len(eigvals) >= 5 else float(np.sum(eigvals) / (eig_sum + 1e-12)),
    }
    # also expose first few eigvals as columns
    for i in range(min(topk, len(eigvals))):
        out[f"eig_{i+1}"] = float(eigvals[i])
    return out


def cophenetic_corr(D: np.ndarray, method: str = "ward") -> float:
    """
    Cophenetic correlation coefficient of a hierarchical clustering built from distances.
    Works on a distance matrix between items (here: symbols/centers).
    """
    n = D.shape[0]
    if n < 3:
        return float("nan")
    y = squareform(D, checks=False)
    Z = linkage(y, method=method)
    c, _ = cophenet(Z, y)
    return float(c)


def compute_metrics_for_matrix(D: np.ndarray, bins_entropy: int = 50) -> dict:
    """Compute a bundle of metrics for a distance matrix D."""
    v = offdiag_values(D)
    if v.size == 0:
        return {}

    q25, q50, q75 = np.quantile(v, [0.25, 0.50, 0.75])
    out = {
        "n_items": int(D.shape[0]),
        "mean_offdiag": float(v.mean()),
        "std_offdiag": float(v.std(ddof=0)),
        "var_offdiag": float(v.var(ddof=0)),
        "min_offdiag": float(v.min()),
        "max_offdiag": float(v.max()),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "contrast_q75_q25": float((q75 + 1e-12) / (q25 + 1e-12)),
        "entropy_hist": shannon_entropy_from_hist(v, bins=bins_entropy),
        "mds_stress_2d": mds_stress(D, n_components=2),
        "cophenetic_ward": cophenetic_corr(D, method="ward"),
    }
    out.update(spectral_metrics_from_distance(D, sigma=None, topk=5))
    return out


def load_symbolization_pickles(folder: str = "output/symbolization") -> dict:
    """Load all pkl in folder and return dict: modality_name -> content dict."""
    paths = sorted(glob.glob(os.path.join(folder, "*.pkl")))
    if not paths:
        raise FileNotFoundError(f"No .pkl files found in: {folder}")

    data = {}
    for p in paths:
        name = Path(p).stem  # e.g., 'AoI', 'eda', 'scanpath', ...
        with open(p, "rb") as f:
            data[name] = pickle.load(f)
    return data


def process(folder = "output/symbolization", out_csv="output/symbolization/metrics.csv"):
    data = load_symbolization_pickles(folder)

    rows = []
    for modality, d in data.items():
        if "dist_mat" not in d:
            print(f"[WARN] {modality}: no 'dist_mat' key found, skipping.")
            continue

        D = np.array(d["dist_mat"], dtype=float)
        # sanity checks
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            print(f"[WARN] {modality}: dist_mat is not square, skipping.")
            continue
        # enforce symmetry for safety
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)

        metrics = compute_metrics_for_matrix(D)
        metrics["modality"] = modality

        # OPTIONAL: also compute metrics on the ordered matrix (for a sanity check)
        if "ordered_dist_mat" in d:
            D_ord = np.array(d["ordered_dist_mat"], dtype=float)
            D_ord = 0.5 * (D_ord + D_ord.T)
            np.fill_diagonal(D_ord, 0.0)
            metrics_ord = compute_metrics_for_matrix(D_ord)
            # These should be ~identical (permutation invariance); keep a small diagnostic:
            metrics["diag_check_mean_diff_ordered"] = float(
                abs(metrics["mean_offdiag"] - metrics_ord.get("mean_offdiag", metrics["mean_offdiag"]))
            )

        rows.append(metrics)

    df = pd.DataFrame(rows).sort_values("modality")
    os.makedirs(Path(out_csv).parent, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(df[[
        "modality", "n_items",
        "mean_offdiag", "std_offdiag", "entropy_hist",
        "contrast_q75_q25", "mds_stress_2d",
        "cophenetic_ward", "eig_energy_top5"
    ]].to_string(index=False))

    print(f"\nSaved: {out_csv}")


def plot_dist_mat():
    
    modalities = [
    "oculomotorFixation",
    "oculomotorSaccade",
    "scanpath",
    "AoI",
    "ecg",
    "eda"
    ]
    
    fig, axes = plt.subplots(
    2, 3,
    figsize=(13, 9),          # figure un peu plus haute
    gridspec_kw={"hspace": 0.10, "wspace": 0.15}  # espace entre lignes/colonnes
)
    axes = axes.flatten()
    
    vmin, vmax = None, None
    
    # Optionnel : fixer une échelle commune
    all_vals = []
    for m in modalities:
        with open(f"output/symbolization/{m}.pkl", "rb") as f:
            D = pickle.load(f)["ordered_dist_mat"]
            all_vals.append(D[np.triu_indices_from(D, k=1)])
    all_vals = np.concatenate(all_vals)
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    
    for ax, modality in zip(axes, modalities):
        with open(f"output/symbolization/{modality}.pkl", "rb") as f:
            D = pickle.load(f)["ordered_dist_mat"]
    
        im = ax.imshow(D, cmap="viridis", vmin=vmin, vmax=vmax)
        if modality=='oculomotorFixation':
            ax.set_title('Fixations', fontsize=18)
        if modality=='oculomotorSaccade':
            ax.set_title('Saccades', fontsize=18)
        if modality=='scanpath':
            ax.set_title('Scanpaths', fontsize=18)
        if modality=='AoI':
            ax.set_title('AoIs', fontsize=18)
        if modality=='ecg':
            ax.set_title('ECG', fontsize=18)
        if modality=='eda':
            ax.set_title('EDA', fontsize=18)
         
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02) 
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_dir = "output/symbolization"
    png_path = os.path.join(out_dir, "symbol_dictionary_matrices.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    
    plt.show()
     