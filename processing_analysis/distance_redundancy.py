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
    
 

def process_figures(
    csv_dir="output/distance",
    out_dir="output/distance",
    use_spearman=False,
    dpi=200,
):
    os.makedirs(out_dir, exist_ok=True)

    # --- Load correlation matrix ---
    fname = (
        "intermodality_corr_matrix_spearman.csv"
        if use_spearman
        else "intermodality_corr_matrix_pearson.csv"
    )
    corr = pd.read_csv(os.path.join(csv_dir, fname), index_col=0)

    labels = list(corr.index)
    C = corr.values.astype(float)

    # --- Heatmap ---
    plt.figure(figsize=(7.5, 6.5))
    im = plt.imshow(C, vmin=-1, vmax=1, cmap="viridis")
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=12)
    plt.yticks(range(len(labels)), labels, fontsize=12)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")

    title = "Inter-modality correlation (Spearman)" if use_spearman else "Inter-modality correlation (Pearson)"
    plt.title(title)
    plt.tight_layout()

    heatmap_path = os.path.join(out_dir, "intermodality_corr_heatmap.png")
    plt.savefig(heatmap_path, dpi=dpi)
    plt.close()

    # --- Dendrogram ---
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")

    plt.figure(figsize=(8.5, 4.8))
    dendrogram(Z, labels=labels, leaf_rotation=30)
    plt.ylabel("Distance (1 − correlation)")
    plt.title("Modality clustering based on distance geometry")
    plt.tight_layout()

    dendro_path = os.path.join(out_dir, "intermodality_dendrogram.png")
    plt.savefig(dendro_path, dpi=dpi)
    plt.close()

    print("Saved figures:")
    print(f"- {heatmap_path}")
    print(f"- {dendro_path}")

