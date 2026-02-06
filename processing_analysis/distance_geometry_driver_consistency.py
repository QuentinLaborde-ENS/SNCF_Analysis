# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def offdiag(D):
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def entropy_hist(x, bins=60, eps=1e-12):
    h, _ = np.histogram(x, bins=bins, density=True)
    p = h / (h.sum() + eps)
    p = p[p > 0]
    return float(-(p * np.log(p + eps)).sum())


def basic_matrix_metrics(D):
    x = offdiag(D)
    q25, q50, q75 = np.quantile(x, [0.25, 0.50, 0.75])
    return {
        "N": int(D.shape[0]),
        "mean_offdiag": float(x.mean()),
        "std_offdiag": float(x.std(ddof=0)),
        "entropy_hist": entropy_hist(x),
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "contrast_q75_q25": float((q75 + 1e-12) / (q25 + 1e-12)),
        "min_offdiag": float(x.min()),
        "max_offdiag": float(x.max()),
    }


def inter_modality_corr(D1, D2, method="pearson"):
    x1 = offdiag(D1)
    x2 = offdiag(D2)
    if method == "pearson":
        return float(np.corrcoef(x1, x2)[0, 1])
    elif method == "spearman":
        r1 = pd.Series(x1).rank().to_numpy()
        r2 = pd.Series(x2).rank().to_numpy()
        return float(np.corrcoef(r1, r2)[0, 1])
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")


def driver_intra_inter_stats(D, labels):
    """
    labels: array-like length N (driver id per recording).
    Returns:
      intra_mean: mean distance for pairs within same driver
      inter_mean: mean distance for pairs across drivers
      sep_ratio: inter_mean / intra_mean
    """
    N = D.shape[0]
    iu = np.triu_indices(N, k=1)
    a, b = iu
    same = labels[a] == labels[b]
    intra = D[iu][same]
    inter = D[iu][~same]

    out = {}
    out["intra_mean"] = float(np.mean(intra)) if intra.size else float("nan")
    out["inter_mean"] = float(np.mean(inter)) if inter.size else float("nan")
    out["sep_ratio"] = float((out["inter_mean"] + 1e-12) / (out["intra_mean"] + 1e-12))
    out["n_intra_pairs"] = int(intra.size)
    out["n_inter_pairs"] = int(inter.size)
    return out

 

def process():
    dist_path = "output/distance/fused.pkl"
    info_path = "output/info/driver.pkl"

    with open(dist_path, "rb") as f:
        obj = pickle.load(f)

    with open(info_path, "rb") as f:
        driver_dict = pickle.load(f)

    records = obj["records"]
    modalities = obj["modalities"]
    dist_dict = obj["distances"]
    fused = obj["fused"]

    # --- Alignment check ---
    missing = [r for r in records if r not in driver_dict]
    if missing:
        raise KeyError(f"{len(missing)} records missing in driver_dict. Example: {missing[:5]}")

    labels = np.array([driver_dict[r] for r in records])  # driver id per recording
    n_drivers = len(np.unique(labels))
    print(f"N recordings: {len(records)} | N drivers: {n_drivers}")

    # --- Global matrix metrics (per modality + fused) ---
    rows = []
    for m in modalities:
        D = np.asarray(dist_dict[m], dtype=float)
        met = basic_matrix_metrics(D)
        met.update(driver_intra_inter_stats(D, labels)) 
        met["matrix"] = m
        rows.append(met)

    met_f = basic_matrix_metrics(fused)
    met_f.update(driver_intra_inter_stats(fused, labels)) 
    met_f["matrix"] = "FUSED"
    rows.append(met_f)

    df = pd.DataFrame(rows).sort_values("matrix")
    os.makedirs("output/distance", exist_ok=True)
    df.to_csv("output/distance/recording_distance_metrics.csv", index=False)
 
    # --- Print a compact view ---
    show_cols = [
        "matrix", "N",
        "mean_offdiag", "std_offdiag",
        "intra_mean", "inter_mean", "sep_ratio", 
    ]
    print("\n=== Recording-level metrics ===")
    print(df[show_cols].to_string(index=False))
 
    print("\nSaved:\n- output/distance/recording_distance_metrics.csv\n- output/distance/intermodality_correlations.csv")


 
##############################################################################

# -----------------------
# Utils
# -----------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def reorder_by_driver(records, driver_dict):
    """
    Returns:
      order: indices that sort recordings by driver id (then by record id)
      drivers_sorted: driver labels in the new order
    """
    drivers = np.array([driver_dict[r] for r in records])
    # stable ordering: first by driver, then by record string
    order = np.lexsort((records, drivers))
    return order, drivers[order]


def plot_heatmap(D, title, out_png, cmap="viridis", vmin=None, vmax=None,
                 show_driver_separators=None, separator_lw=1.0, tick_labels=None):
    """
    show_driver_separators: list of indices where driver changes (in reordered order)
    tick_labels: optional list of strings for axis ticks
    """
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(D, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title, fontsize=14, pad=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    if show_driver_separators:
        for k in show_driver_separators:
            ax.axhline(k - 0.5, linewidth=separator_lw)
            ax.axvline(k - 0.5, linewidth=separator_lw)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout() 
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def driver_change_indices(drivers_sorted):
    """Return indices where driver label changes (for separators)."""
    changes = []
    for i in range(1, len(drivers_sorted)):
        if drivers_sorted[i] != drivers_sorted[i-1]:
            changes.append(i)
    return changes


 
def process_figures():
    out_dir = "output/distance"
    ensure_dir(out_dir)

    # --- Load distance object (contains per-modality matrices + fused) ---
    with open(os.path.join(out_dir, "fused.pkl"), "rb") as f:
        obj = pickle.load(f)

    records = obj["records"]
    modalities = obj["modalities"]
    dist_dict = obj["distances"]
    fused = obj["fused"]

    # --- Load driver labels ---
    with open("output/info/driver.pkl", "rb") as f:
        driver_dict = pickle.load(f)

    # sanity check
    missing = [r for r in records if r not in driver_dict]
    if missing:
        raise KeyError(f"{len(missing)} records missing in driver.pkl. Example: {missing[:5]}")

    # --- Reorder by driver for visualization ---
    order, drivers_sorted = reorder_by_driver(records, driver_dict)
    sep_idx = driver_change_indices(drivers_sorted)

    # --- Use a common color scale across all recording-level matrices ---
    # (recommended for comparing modalities)
    all_vals = []
    for m in modalities:
        D = np.asarray(dist_dict[m], dtype=float)
        Dr = D[np.ix_(order, order)]
        all_vals.append(Dr[np.triu_indices_from(Dr, k=1)])
    Fr = fused[np.ix_(order, order)]
    all_vals.append(Fr[np.triu_indices_from(Fr, k=1)])

    all_vals = np.concatenate(all_vals)
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))

    # FUSED heatmap (sorted by driver) ---
    plot_heatmap(
        Fr,
        title="Fused recording-level distance matrix (sorted by driver)", 
        out_png=os.path.join(out_dir, "fused_heatmap_by_driver.png"),
        cmap="viridis",
        vmin=vmin, vmax=vmax,
        show_driver_separators=sep_idx,
        separator_lw=1.0
    )

    # A 2x3 grid of modality heatmaps (sorted by driver) ---
    fig, axes = plt.subplots(
        2, 3, figsize=(14, 9),
        gridspec_kw={"hspace": 0.35, "wspace": 0.15}
    )
    axes = axes.flatten()

    last_im = None
    for ax, m in zip(axes, modalities):
        D = np.asarray(dist_dict[m], dtype=float)
        Dr = D[np.ix_(order, order)]
        last_im = ax.imshow(Dr, cmap="viridis", vmin=vmin, vmax=vmax)

        ax.set_title(m, fontsize=14, pad=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        for k in sep_idx:
            ax.axhline(k - 0.5, linewidth=0.8)
            ax.axvline(k - 0.5, linewidth=0.8)

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.03)
    cbar.ax.tick_params(labelsize=11)
    fig.suptitle("Recording-level distance matrices (sorted by driver)", fontsize=16, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    grid_png = os.path.join(out_dir, "modalities_heatmaps_by_driver.png") 
    plt.savefig(grid_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
 
    # Barplot: sep_ratio per modality (plus FUSED) ---
    met_csv = "output/distance/recording_distance_metrics.csv"
    met_df = pd.read_csv(met_csv)

    # keep an order that reads well
    wanted_order = modalities + ["FUSED"]
    met_df["matrix"] = met_df["matrix"].astype(str)
    met_df = met_df.set_index("matrix").loc[wanted_order].reset_index()

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(met_df["matrix"], met_df["sep_ratio"])
    ax.set_title("Intra- vs inter-driver separation (sep\\_ratio)", fontsize=14, pad=12, fontweight="bold")
    ax.set_ylabel("inter_mean / intra_mean", fontsize=11)
    ax.set_xticklabels(met_df["matrix"], rotation=30, ha="right")
    plt.tight_layout()
 
    bar_png = os.path.join(out_dir, "sep_ratio_barplot.png") 
    plt.savefig(bar_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved figures to output/distance/:")
    print("-", "fused_heatmap_by_driver.png")
    print("-", "modalities_heatmaps_by_driver.png")
    print("-", "intermodality_correlation_heatmap.png")
    print("-", "sep_ratio_barplot.png")


 


    