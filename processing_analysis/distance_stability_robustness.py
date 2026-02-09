
# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd


def load_data(dist_path, info_path):
    with open(dist_path, "rb") as f:
        dist = pickle.load(f)
    with open(info_path, "rb") as f:
        driver = pickle.load(f)
    return dist, driver


def intra_inter_means(D, labels):
    """Compute mean intra- and inter-label distances for a symmetric distance matrix."""
    labels = np.asarray(labels)
    intra, inter = [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            (intra if labels[i] == labels[j] else inter).append(D[i, j])
    return float(np.mean(intra)), float(np.mean(inter))


def separation_ratio(D, labels):
    mu_intra, mu_inter = intra_inter_means(D, labels)
    return float(mu_inter / mu_intra)


def subsample_indices(n, frac, rng):
    k = int(np.floor(frac * n))
    return np.sort(rng.choice(n, size=k, replace=False))


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def process(n_iter=1000, frac=0.8, seed=0):
    dist_path = "output/distance/fused.pkl"
    info_path = "output/info/driver.pkl"

    rng = np.random.default_rng(seed)
    dist, driver_dict = load_data(dist_path, info_path)

    records = dist["records"]
    drivers = np.array([driver_dict[r] for r in records])

    modalities = list(dist["distances"].keys()) + ["FUSED"]

    results = []
    for modality in modalities:
        D_full = dist["fused"] if modality == "FUSED" else dist["distances"][modality]
        sep_full = separation_ratio(D_full, drivers)

        sep_values = []
        for _ in range(n_iter):
            idx = subsample_indices(len(records), frac=frac, rng=rng)
            D_sub = D_full[np.ix_(idx, idx)]
            drivers_sub = drivers[idx]
            sep_values.append(separation_ratio(D_sub, drivers_sub))

        stats = summarize(sep_values)
        results.append({
            "modality": modality,
            "N_full": int(len(records)),
            "frac": float(frac),
            "n_iter": int(n_iter),
            "sep_full": float(sep_full),
            "sep_boot_mean": stats["mean"],
            "sep_boot_std": stats["std"],
            "sep_boot_q05": stats["q05"],
            "sep_boot_q50": stats["q50"],
            "sep_boot_q95": stats["q95"],
        })

    df = pd.DataFrame(results)
    out_csv = "output/distance/stability_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved stability metrics to {out_csv}")


 
