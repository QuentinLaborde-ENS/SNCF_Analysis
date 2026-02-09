import os
import numpy as np
import pickle
from scipy.spatial.distance import cdist
from weighted_levenshtein import lev


def process(config, path, symbolization_results):

    binning = config['symbolization']['binning']

    # Canonical recording order (from AoI)
    symb_aoi = [f for f in symbolization_results if f.split('.')[0] == 'AoI'][0]
    with open(path + symb_aoi, 'rb') as f:
        symb = pickle.load(f)

    records = np.array(sorted(list(symb['recordings'].keys())))
    record_to_index = {r: i for i, r in enumerate(records)}  # GLOBAL mapping

    modalities = [
        'oculomotorFixation',
        'oculomotorSaccade',
        'scanpath',
        'AoI',
        'ecg',
        'eda'
    ]

    dist_dict = {}

    for type_ in modalities:
        print(f'Processing {type_} distances...')

        symb_file = [f for f in symbolization_results if f.split('.')[0] == type_][0]
        with open(path + symb_file, 'rb') as f:
            symb = pickle.load(f)

        centers = symb['centers']

        # Build symbol sequences per recording
        record_dict = {}
        for record in records:
            seq = symb['recordings'][record]['sequence']
            l_ = symb['recordings'][record]['lengths']
            seq_ = []

            if binning:
                for g in range(len(seq)):
                    seq_.extend([chr(int(seq[g]) + 65)] * int(l_[g]))
            else:
                seq_.extend([chr(int(seq[g]) + 65) for g in range(len(seq))])

            record_dict[record] = seq_

        # Build substitution costs
        centers_dict = {chr(i + 65): centers[i] for i in range(len(centers))}
        d_m, _ = aoi_dict_dist_mat(centers_dict, normalize=True)

        dist_mat = np.zeros((len(records), len(records)), dtype=np.float64)
        for j in range(1, len(records)):
            for i in range(j):
                s_1 = record_dict[records[i]]
                s_2 = record_dict[records[j]]
                dist_mat[i, j] = dist_mat[j, i] = editDistance(s_1, s_2, d_m)
        tri = dist_mat[np.triu_indices(len(records), 1)]
        med = np.median(tri)
        if med > 0:
            dist_mat /= med
        dist_dict[type_] = dist_mat

    # Fused RMS distance
    fused = np.zeros((len(records), len(records)), dtype=np.float64)
    for m in modalities:
        fused += dist_dict[m] ** 2
    fused = np.sqrt(fused / len(modalities))
    print(records)
    out = {
        "records": records,
        "record_to_index": record_to_index,
        "modalities": modalities,
        "distances": dist_dict,
        "fused": fused,
        "binning": binning,
    }

    out_path = 'output/distance/fused.pkl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)


def editDistance(s_1, s_2, d_m):

    n_1 = len(s_1)
    n_2 = len(s_2)
    if n_1 == 0 and n_2 == 0:
        return 0.0

    insert_costs = np.ones(128, dtype=np.float64)
    delete_costs = np.ones(128, dtype=np.float64)

    substitute_costs = np.ones((128, 128), dtype=np.float64)
    K = d_m.shape[0]
    substitute_costs[65:65+K, 65:65+K] = d_m
    # Ensure self-substitution is 0
    np.fill_diagonal(substitute_costs, 0.0)

    s_1 = ''.join(s_1)
    s_2 = ''.join(s_2)

    dist_ = lev(
        s_1, s_2,
        insert_costs=insert_costs,
        delete_costs=delete_costs,
        substitute_costs=substitute_costs
    )
    return float(dist_) / float(max(n_1, n_2))


def aoi_dict_dist_mat(centers, normalize=True):
    c_ = sorted(centers.keys())
    d_ = np.array([centers[k_] for k_ in c_])
    d_m = cdist(d_, d_, metric="euclidean")
    if normalize and np.max(d_m) > 0:
        d_m = d_m / np.max(d_m)
    return d_m, {k_: i for i, k_ in enumerate(c_)}
