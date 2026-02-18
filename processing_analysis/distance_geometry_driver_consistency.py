# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.patches import FancyArrowPatch

import seaborn as sns


# =========================
# Helpers (metrics)
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def offdiag(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def entropy_hist(x, bins=60, eps=1e-12):
    counts, _ = np.histogram(x, bins=bins, density=False)
    p = counts / (counts.sum() + eps)
    p = p[p > 0]
    return float(-(p * np.log(p + eps)).sum())



def basic_matrix_metrics(D: np.ndarray) -> dict:
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


def pairwise_partition_stats(D: np.ndarray, labels: np.ndarray) -> dict:
    """
    Generic intra/inter stats for any categorical labels (driver, line, etc.).
    labels: array-like length N
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


def driver_stats_within_each_line_weighted(D: np.ndarray, driver_labels: np.ndarray, line_labels: np.ndarray) -> dict:
    """
    Driver separation conditional on line: compute intra/inter within each line
    and aggregate by weighting by number of pairs.
    """
    line_values = pd.unique(line_labels)
    intra_sum = 0.0
    inter_sum = 0.0
    intra_cnt = 0
    inter_cnt = 0

    for lv in line_values:
        idx = np.where(line_labels == lv)[0]
        if idx.size < 2:
            continue

        Dg = D[np.ix_(idx, idx)]
        drv_g = driver_labels[idx]

        N = Dg.shape[0]
        iu = np.triu_indices(N, k=1)
        a, b = iu
        same = drv_g[a] == drv_g[b]
        intra = Dg[iu][same]
        inter = Dg[iu][~same]

        if intra.size:
            intra_sum += float(np.sum(intra))
            intra_cnt += int(intra.size)
        if inter.size:
            inter_sum += float(np.sum(inter))
            inter_cnt += int(inter.size)

    out = {}
    out["driver_intra_within_line_mean"] = float(intra_sum / intra_cnt) if intra_cnt else float("nan")
    out["driver_inter_within_line_mean"] = float(inter_sum / inter_cnt) if inter_cnt else float("nan")
    out["driver_sep_ratio_within_line"] = float(
        (out["driver_inter_within_line_mean"] + 1e-12) / (out["driver_intra_within_line_mean"] + 1e-12)
    )
    out["driver_intra_within_line_pairs"] = int(intra_cnt)
    out["driver_inter_within_line_pairs"] = int(inter_cnt)
    out["n_line_groups"] = int(len(line_values))
    return out


def driver_stats_per_line(D: np.ndarray, driver_labels: np.ndarray, line_labels: np.ndarray) -> pd.DataFrame:
    """
    Driver separation computed separately for each line group (no aggregation).
    Returns a dataframe with columns:
      line, N_line, intra_mean, inter_mean, sep_ratio, n_intra_pairs, n_inter_pairs
    """
    rows = []
    for lv in sorted(pd.unique(line_labels).tolist()):
        idx = np.where(line_labels == lv)[0]
        if idx.size < 2:
            continue
        Dg = D[np.ix_(idx, idx)]
        drv_g = driver_labels[idx]
        st = pairwise_partition_stats(Dg, drv_g)
        rows.append({
            "line": lv,
            "N_line": int(idx.size),
            **st
        })
    return pd.DataFrame(rows)


def map_line_to_coarse(line_name: str) -> str:
    """
    Collapse detailed line labels into coarse classes:
    - h_line -> Transilien
    - paris_brest / paris_hendaye -> TGV
    """
    if line_name == "h_line":
        return "Transilien"
    elif line_name in ("paris_brest", "paris_hendaye"):
        return "TGV"
    else:
        return "Other"


# =========================
# Main processing (CSV outputs)
# =========================

def process():
    dist_path = "output/distance/fused.pkl"
    driver_path = "output/info/driver.pkl"
    line_path = "output/info/line.pkl"
    out_dir = "output/distance"
    ensure_dir(out_dir)

    with open(dist_path, "rb") as f:
        obj = pickle.load(f)
    with open(driver_path, "rb") as f:
        driver_dict = pickle.load(f)
    with open(line_path, "rb") as f:
        line_dict = pickle.load(f)

    records = obj["records"]
    modalities = obj["modalities"]
    dist_dict = obj["distances"]
    fused = obj["fused"]

    # --- Alignment checks ---
    missing_drv = [r for r in records if r not in driver_dict]
    if missing_drv:
        raise KeyError(f"{len(missing_drv)} records missing in driver.pkl. Example: {missing_drv[:5]}")
    missing_line = [r for r in records if r not in line_dict]
    if missing_line:
        raise KeyError(f"{len(missing_line)} records missing in line.pkl. Example: {missing_line[:5]}")

    driver_labels = np.array([driver_dict[r] for r in records])
    line_labels_fine = np.array([line_dict[r] for r in records])
    line_labels_coarse = np.array([map_line_to_coarse(line_dict[r]) for r in records])

    print(f"N recordings: {len(records)} | N drivers: {len(np.unique(driver_labels))}")
    print(f"Line (fine) groups: {sorted(pd.unique(line_labels_fine).tolist())}")
    print(f"Line (coarse) groups: {sorted(pd.unique(line_labels_coarse).tolist())}")

    # ---------------------------------------------------------
    # (1) Global metrics on the full dataset (as before + line)
    # ---------------------------------------------------------
    rows_global = []

    def compute_row_for_matrix(name: str, D: np.ndarray) -> dict:
        D = np.asarray(D, dtype=float)

        met = basic_matrix_metrics(D)

        # Driver (global)
        met_drv = pairwise_partition_stats(D, driver_labels)
        met.update({f"driver_{k}": v for k, v in met_drv.items()})

        # Line (fine)
        met_line_f = pairwise_partition_stats(D, line_labels_fine)
        met.update({f"lineFine_{k}": v for k, v in met_line_f.items()})

        # Line (coarse)
        met_line_c = pairwise_partition_stats(D, line_labels_coarse)
        met.update({f"lineCoarse_{k}": v for k, v in met_line_c.items()})

        # Driver within line (weighted aggregation, coarse by default)
        met_cond = driver_stats_within_each_line_weighted(D, driver_labels, line_labels_coarse)
        met.update(met_cond)

        met["matrix"] = name
        return met

    for m in modalities:
        rows_global.append(compute_row_for_matrix(m, dist_dict[m]))
    rows_global.append(compute_row_for_matrix("FUSED", fused))

    df_global = pd.DataFrame(rows_global).sort_values("matrix")
    global_csv = os.path.join(out_dir, "recording_distance_metrics_with_line.csv")
    df_global.to_csv(global_csv, index=False)

    show_cols = [
        "matrix", "N", "mean_offdiag", "std_offdiag",
        "driver_intra_mean", "driver_inter_mean", "driver_sep_ratio",
        "lineCoarse_intra_mean", "lineCoarse_inter_mean", "lineCoarse_sep_ratio",
        "driver_sep_ratio_within_line",
    ]
    print("\n=== Recording-level metrics (driver + line) ===")
    print(df_global[show_cols].to_string(index=False))

    # ---------------------------------------------------------
    # (2) Per-line metrics: compute *within each line* separately
    #     (driver separation inside each line group)
    # ---------------------------------------------------------
    per_line_rows = []

    def add_per_line_rows(name: str, D: np.ndarray):
        # Per fine line (h_line / paris_brest / paris_hendaye)
        df_f = driver_stats_per_line(np.asarray(D, float), driver_labels, line_labels_fine)
        if len(df_f):
            df_f.insert(0, "matrix", name)
            df_f.insert(1, "line_level", "fine")
            per_line_rows.append(df_f)

        # Per coarse line (TGV / Transilien)
        df_c = driver_stats_per_line(np.asarray(D, float), driver_labels, line_labels_coarse)
        if len(df_c):
            df_c.insert(0, "matrix", name)
            df_c.insert(1, "line_level", "coarse")
            per_line_rows.append(df_c)

    for m in modalities:
        add_per_line_rows(m, dist_dict[m])
    add_per_line_rows("FUSED", fused)

    if per_line_rows:
        df_by_line = pd.concat(per_line_rows, axis=0, ignore_index=True)
    else:
        df_by_line = pd.DataFrame()

    by_line_csv = os.path.join(out_dir, "driver_separation_by_line.csv")
    df_by_line.to_csv(by_line_csv, index=False)

    print("\nSaved:")
    print("-", global_csv)
    print("-", by_line_csv)


# =========================
# Figures
# =========================

def reorder_by_driver(records: np.ndarray, driver_dict: dict):
    drivers = np.array([driver_dict[r] for r in records])
    order = np.lexsort((records, drivers))  # stable: driver then record id
    return order, drivers[order]


def driver_change_indices(drivers_sorted: np.ndarray):
    changes = []
    for i in range(1, len(drivers_sorted)):
        if drivers_sorted[i] != drivers_sorted[i - 1]:
            changes.append(i)
    return changes

 

def plot_heatmap(D, title, out_png, cmap="viridis", vmin=None, vmax=None,
                 show_separators=None, separator_lw=1.0):

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(D, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks([])
    ax.set_yticks([])

    if show_separators:
        for k in show_separators:
            ax.axhline(k - 0.5, linewidth=separator_lw, color='black')
            ax.axvline(k - 0.5, linewidth=separator_lw, color='black')



    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def process_figures():
    out_dir = "output/distance"
    ensure_dir(out_dir)

    # --- Load distance object ---
    with open(os.path.join(out_dir, "fused.pkl"), "rb") as f:
        obj = pickle.load(f)

    records = obj["records"]
    modalities = obj["modalities"]
    dist_dict = obj["distances"]
    fused = np.asarray(obj["fused"], dtype=float)

    # --- Load driver labels ---
    with open("output/info/driver.pkl", "rb") as f:
        driver_dict = pickle.load(f)

    missing_drv = [r for r in records if r not in driver_dict]
    if missing_drv:
        raise KeyError(f"{len(missing_drv)} records missing in driver.pkl. Example: {missing_drv[:5]}")

    # --- Reorder by driver for visualization ---
    order, drivers_sorted = reorder_by_driver(records, driver_dict)
    sep_idx = driver_change_indices(drivers_sorted)

    # --- Common color scale across all matrices (recommended) ---
    all_vals = []
    for m in modalities:
        D = np.asarray(dist_dict[m], dtype=float)
        Dr = D[np.ix_(order, order)]
        all_vals.append(Dr[np.triu_indices_from(Dr, k=1)])
    Fr = fused[np.ix_(order, order)]
    all_vals.append(Fr[np.triu_indices_from(Fr, k=1)])
    all_vals = np.concatenate(all_vals)
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))

    # --- FUSED heatmap (sorted by driver) ---
    # Note: you asked to remove the TGV/Transilien annotation => no sidebars/legend.
    plot_heatmap(
        Fr,
        title="Fused",
        out_png=os.path.join(out_dir, "fused_heatmap_by_driver.png"),
        cmap="viridis",
        vmin=vmin, vmax=vmax,
        show_separators=sep_idx,
        separator_lw=3.0
    )

    # --- 6 individual modality heatmaps (sorted by driver) ---
    for m in modalities:
        D = np.asarray(dist_dict[m], dtype=float)
        Dr = D[np.ix_(order, order)]
        plot_heatmap(
            Dr,
            title=f"{m} recording-level distance matrix (sorted by driver)",
            out_png=os.path.join(out_dir, f"{m}_heatmap_by_driver.png"),
            cmap="viridis",
            vmin=vmin, vmax=vmax,
            show_separators=sep_idx,
            separator_lw=3.0
        )

    # --- Barplot: separation ratios (driver vs within-line vs line) ---
    met_csv = os.path.join(out_dir, "recording_distance_metrics_with_line.csv")
    met_df = pd.read_csv(met_csv)

    wanted_order = modalities + ["FUSED"]
    met_df["matrix"] = met_df["matrix"].astype(str)
    met_df = met_df.set_index("matrix").loc[wanted_order].reset_index()

    sns.set_theme(style='whitegrid', font_scale = 2.2)
    plot_df = pd.DataFrame({
        "matrix": np.tile(met_df["matrix"].values, 3),
        "separation_ratio": np.concatenate([
            met_df["driver_sep_ratio"].values,
            met_df["driver_sep_ratio_within_line"].values,
            met_df["lineCoarse_sep_ratio"].values
        ]),
        "type": (
            ["$R_{drv}$"] * len(met_df) +
            ["$R_{drv|line}$"] * len(met_df) +
            ["$R_{line}$"] * len(met_df)
        )
    })
    
    label_map = {
        "oculomotorFixation": "Fixations",
        "oculomotorSaccade": "Saccades",
        "scanpath": "Scanpaths",
        "AoI": "AoIs",
        "ecg": "ECG",
        "eda": "EDA",
        "FUSED": "FUSED"
    }
    
    plot_df["matrix"] = plot_df["matrix"].map(label_map)
    
    palette = {
        "$R_{drv}$": "darkblue",
        "$R_{drv|line}$": "cornflowerblue",
        "$R_{line}$": "skyblue"
    }
    
    fig, ax = plt.subplots(figsize=(12, 4.8))
    
    sns.barplot(
        data=plot_df,
        x="matrix",
        y="separation_ratio",
        hue="type",
        palette=palette,
        dodge=True,
        ci=None,
        ax=ax
    )
    
    ax.set_ylabel("Separation ratio (inter / intra)", fontsize=18)
    ax.set_xlabel("")
    ax.legend(title="", frameon=False, ncol=3, fontsize=18)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=15)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "sep_ratio_barplot_driver_line.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


    print("Saved figures to output/distance/:")
    print("-", "fused_heatmap_by_driver.png")
    for m in modalities:
        print("-", f"{m}_heatmap_by_driver.png")
    print("-", "sep_ratio_barplot_driver_line.png")


 
