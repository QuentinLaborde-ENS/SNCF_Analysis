# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def _upper_triangle_vector(D):
    idx = np.triu_indices(D.shape[0], k=1)
    return D[idx]


def process(
    dist_path="output/distance/fused.pkl",
    out_dir="output/distance",
    include_fused=True,
    modality_order=None,
):
 
    os.makedirs(out_dir, exist_ok=True)

    # --- Load distances ---
    with open(dist_path, "rb") as f:
        obj = pickle.load(f)

    dist_dict = dict(obj["distances"])
    if include_fused:
        dist_dict["FUSED"] = obj["fused"]

    # --- Enforce modality order ---
    if modality_order is not None:
        modality_order = [m for m in modality_order if m in dist_dict]
        for m in dist_dict:
            if m not in modality_order:
                modality_order.append(m)
        dist_dict = {m: dist_dict[m] for m in modality_order}

    modalities = list(dist_dict.keys())
    vectors = {m: _upper_triangle_vector(dist_dict[m]) for m in modalities}

    # --- Pairwise correlations ---
    rows = []
    for m1, m2 in combinations(modalities, 2):
        v1, v2 = vectors[m1], vectors[m2]
        r_p, _ = pearsonr(v1, v2)
        r_s, _ = spearmanr(v1, v2)
        rows.append({
            "modality_1": m1,
            "modality_2": m2,
            "pearson": r_p,
            "spearman": r_s,
            "n_pairs": len(v1),
        })

    df_pairs = pd.DataFrame(rows).sort_values("pearson", ascending=False)

    # --- Correlation matrices ---
    M = len(modalities)
    corr_p = pd.DataFrame(np.eye(M), index=modalities, columns=modalities)
    corr_s = pd.DataFrame(np.eye(M), index=modalities, columns=modalities)

    for _, r in df_pairs.iterrows():
        corr_p.loc[r["modality_1"], r["modality_2"]] = r["pearson"]
        corr_p.loc[r["modality_2"], r["modality_1"]] = r["pearson"]
        corr_s.loc[r["modality_1"], r["modality_2"]] = r["spearman"]
        corr_s.loc[r["modality_2"], r["modality_1"]] = r["spearman"]

    # --- Save CSVs ---
    df_pairs.to_csv(
        os.path.join(out_dir, "intermodality_correlations_pairs.csv"),
        index=False
    )
    corr_p.to_csv(
        os.path.join(out_dir, "intermodality_corr_matrix_pearson.csv")
    )
    corr_s.to_csv(
        os.path.join(out_dir, "intermodality_corr_matrix_spearman.csv")
    )

    print("Saved:")
    print("- intermodality_correlations_pairs.csv")
    print("- intermodality_corr_matrix_pearson.csv")
    print("- intermodality_corr_matrix_spearman.csv")
    
 

 

def plot_heatmap(
    D,
    title,
    out_png,
    cmap="viridis",
    vmin=None,
    vmax=None,
    show_separators=None,
    separator_lw=1.0,
    annotate=False,
    annotate_fmt="{:.2f}",
    annotate_fontsize=15,
    annotate_color="white",
):
    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # --- Make sure no background grid is shown ---
    ax.grid(False)
    ax.set_axisbelow(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    im = ax.imshow(D, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks([])
    ax.set_yticks([])

    if show_separators:
        for k in show_separators:
            ax.axhline(k - 0.5, linewidth=separator_lw)
            ax.axvline(k - 0.5, linewidth=separator_lw)

    # --- Add numbers inside cells (optional) ---
    if annotate:
        D = np.asarray(D, float)
        n = D.shape[0]
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i,
                    annotate_fmt.format(D[i, j]),
                    ha="center", va="center",
                    fontsize=annotate_fontsize,
                    color=annotate_color,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=11)
    # --- Remove vertical "Correlation" label ---
    cbar.set_label("")  # or comment this line out entirely

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_figures(
    csv_dir="output/distance",
    out_dir="output/distance",
    use_spearman=False,
    dpi=300,
    modality_order=None,
    add_family_separators=True,
    annotate_cells=True,
    annotate_fmt="{:.2f}",
):
    os.makedirs(out_dir, exist_ok=True)

    # --- Load correlation matrix ---
    fname = (
        "intermodality_corr_matrix_spearman.csv"
        if use_spearman
        else "intermodality_corr_matrix_pearson.csv"
    )
    corr = pd.read_csv(os.path.join(csv_dir, fname), index_col=0)

    # --- Enforce an order consistent with the rest of the chapter ---
    if modality_order is None:
        modality_order = [
            "oculomotorFixation",
            "oculomotorSaccade",
            "scanpath",
            "AoI",
            "ecg",
            "eda",
            "FUSED",
        ]

    # Keep only available modalities, and append any unexpected ones at the end
    modality_order = [m for m in modality_order if m in corr.index]
    for m in corr.index:
        if m not in modality_order:
            modality_order.append(m)

    corr = corr.loc[modality_order, modality_order]
    C = corr.values.astype(float)

    # --- Optional separators to mimic the "block" reading style ---
    sep_idx = None
    if add_family_separators:
        groups = [
            ["oculomotorFixation", "oculomotorSaccade"],  # eye micro
            ["scanpath", "AoI"],                          # spatial
            ["ecg", "eda"],                               # physio
            ["FUSED"],                                    # fused
        ]
        boundaries = []
        k = 0
        for g in groups:
            present = [x for x in g if x in modality_order]
            if not present:
                continue
            k += len(present)
            if k < len(modality_order):
                boundaries.append(k)
        sep_idx = boundaries if boundaries else None

    title = "Inter-modality correlation matrix (Spearman)" if use_spearman else "Inter-modality correlation matrix (Pearson)"

    # --- 1) Paper-style heatmap (same style as distance matrices: no ticks) ---
    out_png = os.path.join(
        out_dir,
        "intermodality_corr_heatmap_spearman.png" if use_spearman else "intermodality_corr_heatmap_pearson.png"
    )

    plot_heatmap(
        C,
        title=title,
        out_png=out_png,
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
        show_separators=sep_idx,
        separator_lw=1.0,
        annotate=annotate_cells,
        annotate_fmt=annotate_fmt,
        annotate_fontsize=20,
        annotate_color="white",
    )

    # --- 2) Reader-friendly heatmap with labels (recommended) ---
    out_png_labeled = os.path.join(
        out_dir,
        "intermodality_corr_heatmap_spearman_labeled.png" if use_spearman else "intermodality_corr_heatmap_pearson_labeled.png"
    )

    label_map = {
        "oculomotorFixation": "Fixations",
        "oculomotorSaccade": "Saccades",
        "scanpath": "Scanpaths",
        "AoI": "AoIs",
        "ecg": "ECG",
        "eda": "EDA",
        "FUSED": "FUSED",
    }
    labels_pretty = [label_map.get(m, m) for m in modality_order]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    # remove background grid/spines explicitly
    ax.grid(False)
    ax.set_axisbelow(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    im = ax.imshow(C, cmap="viridis", vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(len(labels_pretty)))
    ax.set_yticks(range(len(labels_pretty)))
    ax.set_xticklabels(labels_pretty, rotation=30, ha="right", fontsize=16)
    ax.set_yticklabels(labels_pretty, fontsize=16)

    if sep_idx:
        for k in sep_idx:
            ax.axhline(k - 0.5, linewidth=1.0)
            ax.axvline(k - 0.5, linewidth=1.0)

    # annotate values
    if annotate_cells:
        n = C.shape[0]
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i,
                    annotate_fmt.format(C[i, j]),
                    ha="center", va="center",
                    fontsize=10,
                    color="black",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=11)
    # --- Remove vertical "Correlation" label ---
    cbar.set_label("")

    
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_png_labeled, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print("Saved figures:")
    print(f"- {out_png}")
    print(f"- {out_png_labeled}")



