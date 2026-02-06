# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import pearsonr
 
import matplotlib.pyplot as plt
import seaborn as sns



def load_data(dist_path, info_path):
    with open(dist_path, "rb") as f:
        dist = pickle.load(f)
    with open(info_path, "rb") as f:
        driver = pickle.load(f)
    return dist, driver


def intra_inter_means(D, drivers):
    """Compute intra- and inter-driver mean distances."""
    drivers = np.array(drivers)
    intra, inter = [], []

    for i in range(len(drivers)):
        for j in range(i + 1, len(drivers)):
            if drivers[i] == drivers[j]:
                intra.append(D[i, j])
            else:
                inter.append(D[i, j])

    return np.mean(intra), np.mean(inter)


def separation_ratio(D, drivers):
    mu_intra, mu_inter = intra_inter_means(D, drivers)
    return mu_inter / mu_intra


def subsample_indices(n, frac=0.8, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    k = int(frac * n)
    return np.sort(rng.choice(n, size=k, replace=False))


def process(n_iter=100, frac=0.8):

    dist_path = "output/distance/fused.pkl"
    info_path = "output/info/driver.pkl"

    rng = np.random.default_rng(0)
    dist, driver_dict = load_data(dist_path, info_path)

    records = dist["records"]
    drivers = np.array([driver_dict[r] for r in records])

    modalities = list(dist["distances"].keys()) + ["FUSED"]

    results = []

    for modality in modalities:
        if modality == "FUSED":
            D_full = dist["fused"]
        else:
            D_full = dist["distances"][modality]

        full_sep = separation_ratio(D_full, drivers)

        sep_values = []
        corr_values = []

        for _ in range(n_iter):
            idx = subsample_indices(len(records), frac=frac, rng=rng)

            D_sub = D_full[np.ix_(idx, idx)]
            drivers_sub = drivers[idx]

            sep = separation_ratio(D_sub, drivers_sub)
            sep_values.append(sep)

            # Correlation with full distance geometry
            full_vec = D_full[np.triu_indices(len(records), k=1)]
            sub_vec = D_sub[np.triu_indices(len(idx), k=1)]

            # Compare only common indices
            corr, _ = pearsonr(
                full_vec[np.isin(
                    np.arange(len(full_vec)),
                    np.arange(len(sub_vec))
                )][:len(sub_vec)],
                sub_vec
            )
            corr_values.append(corr)

        results.append({
            "modality": modality,
            "sep_full": full_sep,
            "sep_mean": np.mean(sep_values),
            "sep_std": np.std(sep_values),
            "geom_corr_mean": np.mean(corr_values),
            "geom_corr_std": np.std(corr_values),
        })

    df_stability = pd.DataFrame(results)
    out_csv = "output/distance/stability_metrics.csv"
    df_stability.to_csv(out_csv, index=False)
    print(f"Saved stability metrics to {out_csv}")
     


def process_figure(csv_path, out_path):
    
    csv_path = "output/distance/stability_metrics.csv"
    df = pd.read_csv(csv_path)
 
    order = [
        "oculomotorFixation",
        "oculomotorSaccade",
        "scanpath",
        "AoI",
        "ecg",
        "eda",
        "FUSED",
    ]
    df["modality"] = pd.Categorical(df["modality"], categories=order, ordered=True)
    df = df.sort_values("modality")

    plt.figure(figsize=(8, 4))
    sns.boxplot(
        x="modality",
        y="sep_mean",
        data=df,
        color="lightgray",
        linewidth=1.2
    )

    plt.ylabel(r"Separation ratio $R=\mu_{\mathrm{inter}}/\mu_{\mathrm{intra}}$")
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_path = "output/distance/sep_ratio_boxplot.png"
    plt.savefig(out_path)
    plt.close()
    
    print(f"Saved boxplot to {out_path}")

 