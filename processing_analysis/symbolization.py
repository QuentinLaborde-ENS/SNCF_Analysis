# -*- coding: utf-8 -*-


import os 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.figure_factory as ff

import seaborn as sns
import pickle 

from sklearn.decomposition import KernelPCA 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, squareform
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster




def process(config, path, 
            feature_records):
 
    if True:
        ## Process symbolization for fixation features
        oculomotor_features = config['data']['oculomotor_features'] 
        oculomotor_feature_records = [feature_record for feature_record in feature_records
                                      if feature_record.split('.')[0].split('_')[-1] == 'oculomotor']
        fix_feature_set = [feature for feature in oculomotor_features if feature[:3]=='fix']
        process_subset(config, path, 
                       oculomotor_feature_records,
                       fix_feature_set, 
                       'oculomotorFixation')
    if True:
        ## Process symbolization for saccade features
        oculomotor_features = config['data']['oculomotor_features'] 
        oculomotor_feature_records = [feature_record for feature_record in feature_records
                                      if feature_record.split('.')[0].split('_')[-1] == 'oculomotor']
        sac_feature_set = [feature for feature in oculomotor_features if feature[:3]=='sac']
        process_subset(config, path, 
                       oculomotor_feature_records,
                       sac_feature_set, 
                       'oculomotorSaccade')
    if True:
        ## Process symbolization for scanpath features
        scanpath_features = config['data']['scanpath_features'] 
        scanpath_feature_records = [feature_record for feature_record in feature_records
                                    if feature_record.split('.')[0].split('_')[-1] == 'scanpath']
        sp_feature_set = [feature for feature in scanpath_features if feature[:2]=='Sp']
        process_subset(config, path, 
                       scanpath_feature_records,
                       sp_feature_set, 
                       'scanpath')
    if True:
        ## Process symbolization for aoi sequence features 
        aoi_features = config['data']['aoi_features']
        aoi_features_records = [feature_record for feature_record in feature_records
                                if feature_record.split('.')[0].split('_')[-1] == 'AoI']
        aoi_feature_set = [feature for feature in aoi_features if feature[:3] == 'AoI']
        process_subset(config, path, 
                       aoi_features_records, 
                       aoi_feature_set, 
                       'AoI')
    if True:
        ## Process symbolization for aoi sequence features 
        eda_features = config['data']['eda_features']
        eda_features_records = [feature_record for feature_record in feature_records
                                if feature_record.split('.')[0].split('_')[-1] == 'eda']
        eda_feature_set = [feature for feature in eda_features if feature[:3] == 'eda']
        process_subset(config, path, 
                       eda_features_records, 
                       eda_feature_set, 
                       'eda')
        
    if True:
        ## Process symbolization for aoi sequence features 
        ecg_features = config['data']['ecg_features']
        ecg_features_records = [feature_record for feature_record in feature_records
                                if feature_record.split('.')[0].split('_')[-1] == 'ecg']
        ecg_feature_set = [feature for feature in ecg_features if feature[:3] == 'ecg']
        process_subset(config, path, 
                       ecg_features_records, 
                       ecg_feature_set, 
                       'ecg')


def process_subset(config, path, 
                   feature_records, 
                   feature_set, 
                   type_, 
                   display=True):
 
    bkpt_path = 'output/segmentation/'
 
    if type_=='scanpath':
        n_centers = config['symbolization']['nb_clusters']['scanpath']     
    elif type_=='AoI':
        n_centers = config['symbolization']['nb_clusters']['aoi'] 
    elif type_=='eda':
        n_centers = config['symbolization']['nb_clusters']['eda'] 
    elif type_=='ecg':
        n_centers = config['symbolization']['nb_clusters']['ecg'] 
    else:
        n_centers = config['symbolization']['nb_clusters']['oculomotor'] 
 
     
    ## Initialize concatenated data for all subjects
    sub_data = []
 
    for record in feature_records:  
        ## For each subject get feature set  
        df = pd.read_csv(path+record)[feature_set]  
        df = df.to_numpy() 
        name = record.split('.')[0].rsplit("_", 1)[0]
      
        bkpt_name = '{name_}_{type_}.npy'.format(name_=name, 
                                                  type_=type_)
     
        bkpts = np.load(bkpt_path+bkpt_name) 
        for i in range(1, len(bkpts)):
            l_data = df[bkpts[i-1]: bkpts[i]] 
            l_means = np.mean(l_data, axis=0) 
            sub_data.append(l_means)
 
    sub_data=np.array(sub_data) 
    kpca=KernelPCA(n_components=10,
                    kernel="rbf", 
                    n_jobs=-1)
    kpca.fit(sub_data)
    transformed_subdata = kpca.transform(sub_data)
     
    kmeans = KMeans(n_clusters=n_centers, 
                    n_init=100, 
                    random_state=0).fit(transformed_subdata)
    centers = kmeans.cluster_centers_ 
    dist_mat = cdist(centers, centers)

    ## Re-order clusters according to pairwise distances 
    ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, 'ward')
    ## Compute inv_res_order to change cluster labels and center labels
    inv_res_order = np.zeros(len(res_order))
    for k_ in range(len(res_order)):
        inv_res_order[res_order[k_]] = int(k_)
        
    if display:
        plt.style.use("seaborn-v0_8")  
        plt.imshow(ordered_dist_mat, cmap="viridis")
        plt.grid(None)
        plt.title(type_)
        plt.colorbar()
        plt.show()
        plt.clf()
  
    re_ordering = lambda x: [int(inv_res_order[x[i]]) for i in range(len(x))]
    ordered_centers = []
    for k_ in range(len(centers)):
        ordered_centers.append(centers[int(res_order[k_])])
   
  
    result_dict = dict({})
    result_dict.update({'centers': np.array(ordered_centers), 
                        'dist_mat': np.array(dist_mat),
                        'ordered_dist_mat': np.array(ordered_dist_mat),
                        'recordings': dict({})}) 
    for record in feature_records:
         
        sub_data = []
        lengths = []
        df = pd.read_csv(path+record)[feature_set]  
        df = df.to_numpy() 
        name = record.split('.')[0].rsplit("_", 1)[0]
      
        bkpt_name = '{name_}_{type_}.npy'.format(name_=name, 
                                                  type_=type_)
        bkpts = np.load(bkpt_path+bkpt_name)     
        for i in range(1, len(bkpts)): 
            l_data = df[bkpts[i-1]: bkpts[i]] 
            l_means = np.mean(l_data, axis=0)  
            sub_data.append(l_means)
            lengths.append(bkpts[i]-bkpts[i-1])
          
        sub_data=np.array(sub_data)
        transformed_subdata = kpca.transform(sub_data)
        labs_ = kmeans.predict(transformed_subdata)
    
        ordered_labs_ = re_ordering(labs_) 
        lengths = list(np.array(lengths)) 
  
        result_dict['recordings'].update({name: dict()})
        result_dict['recordings'][name].update({'sequence': ordered_labs_, 
                                                'lengths': lengths}) 
   
    filename = '{outpath}/{type_}.pkl'.format(outpath='output/symbolization',  
                                              type_=type_)
    with open(filename, 'wb') as fp:
        pickle.dump(result_dict, fp)   
  
        
  
def seriation(Z,N,cur_index):
 
    ## Computes the order implied by a hierarchical tree (dendrogram) 
    if cur_index < N:
        return [cur_index]
    
    else: 
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1]) 
        return (seriation(Z,N,left) + seriation(Z,N,right))
   
    
    
def compute_serial_matrix(dist_mat,method="ward"):
 
    ## Transformsa distance matrix into a sorted distance matrix according to 
    ## the order implied by the hierarchical tree 
    N = len(dist_mat)
    
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
    ## Compute re-ordering
    res_order = seriation(res_linkage, N, N + N-2)
    
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
  
    return seriated_dist, res_order, res_linkage     
            
    
             

############################################################################## 



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_heatmap_centers(
    D,
    out_png,
    title="",
    cmap="viridis",
    vmin=None,
    vmax=None,
    show_separators=None,
    separator_lw=1.0,
    cbar_ticksize=11,
    figsize=(7.8, 6.8),
):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(D, cmap=cmap, vmin=vmin, vmax=vmax)

    #ax.set_title(title, fontsize=16, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])

    if show_separators:
        for k in show_separators:
            ax.axhline(k - 0.5, linewidth=separator_lw, color="white")
            ax.axvline(k - 0.5, linewidth=separator_lw, color="white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=cbar_ticksize)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _compute_cluster_separators_from_linkage(Z, n_centers, n_groups=6):
    """
    Given a linkage matrix Z over n_centers items, compute separator indices
    after sorting centers by cluster labels obtained from cutting dendrogram
    into n_groups clusters.

    Returns:
      order (np.ndarray): indices permutation of centers
      sep_idx (list[int]): indices where cluster label changes (for separators)
    """
    # cluster labels in {1..n_groups}
    cl = fcluster(Z, t=n_groups, criterion="maxclust")

    # stable order by cluster then by index
    order = np.lexsort((np.arange(n_centers), cl))
    cl_sorted = cl[order]

    sep_idx = []
    for i in range(1, len(cl_sorted)):
        if cl_sorted[i] != cl_sorted[i - 1]:
            sep_idx.append(i)

    return order, sep_idx


def process_symbol_dictionary_figures(
    symbolization_dir="output/symbolization",
    out_dir="output/symbolization",
    modalities=("oculomotorFixation", "oculomotorSaccade", "scanpath", "AoI", "ecg", "eda"),
    label_map=None,
    cmap="viridis",
    common_scale=True,
    n_groups=None,          # e.g. 5 or 6 to show separators; None disables
    separator_lw=1.2,
):
    """
    Plot ordered distance matrices between centers (symbol dictionary geometry),
    one heatmap per modality.

    Expects each output/symbolization/<modality>.pkl to contain:
      - 'ordered_dist_mat'  (NxN)
      - 'dist_mat'          (NxN) optional
      - 'ordered_dist_mat' is produced by compute_serial_matrix(dist_mat,'ward')
      - 'res_linkage' is NOT saved currently; so if you want separators based on Z,
        you must either:
          (A) save 'res_linkage' in your symbolization pickle, OR
          (B) compute clusters from 'ordered_dist_mat' via a new linkage (not identical).

    Here we implement (B): we rebuild a linkage from the *un-ordered* dist_mat if present,
    else from ordered_dist_mat. This is sufficient for visualization separators.
    """

    if label_map is None:
        label_map = {
            "oculomotorFixation": "Fixations",
            "oculomotorSaccade": "Saccades",
            "scanpath": "Scanpaths",
            "AoI": "AoIs",
            "ecg": "ECG",
            "eda": "EDA",
        }

    ensure_dir(out_dir)

    mats = {}
    # --- Load matrices
    for m in modalities:
        pkl_path = os.path.join(symbolization_dir, f"{m}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing file: {pkl_path}")

        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        if "ordered_dist_mat" not in obj:
            raise KeyError(f"{m}.pkl missing 'ordered_dist_mat'")

        D_ord = np.asarray(obj["ordered_dist_mat"], dtype=float)
        mats[m] = {
            "ordered": D_ord,
            "dist_mat": np.asarray(obj["dist_mat"], dtype=float) if "dist_mat" in obj else None,
        }

    # --- Common color scale (recommended)
    vmin = vmax = None
    if common_scale:
        all_vals = []
        for m in modalities:
            D = mats[m]["ordered"]
            all_vals.append(D[np.triu_indices_from(D, k=1)])
        all_vals = np.concatenate(all_vals)
        vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))

    # --- Plot per modality
    for m in modalities:
        D_plot = mats[m]["ordered"]
        title = f"{label_map.get(m, m)}: ordered center distance matrix"

        sep_idx = None
        if n_groups is not None:
            # Build linkage from base dist_mat if available, else from ordered
            base = mats[m]["dist_mat"] if mats[m]["dist_mat"] is not None else mats[m]["ordered"]
            base = 0.5 * (base + base.T)
            np.fill_diagonal(base, 0.0)

            # Recompute linkage on the fly
            # (same as in compute_serial_matrix)
            from scipy.spatial.distance import squareform
            from fastcluster import linkage as fast_linkage

            flat = squareform(base, checks=False)
            Z = fast_linkage(flat, method="ward", preserve_input=True)

            # separators based on cluster cut; then reorder D accordingly
            order, sep_idx = _compute_cluster_separators_from_linkage(Z, base.shape[0], n_groups=n_groups)
            D_plot = D_plot[np.ix_(order, order)]  # apply visualization order

        out_png = os.path.join(out_dir, f"{m}_symbol_centers_ordered.png")
        plot_heatmap_centers(
            D_plot,
            out_png=out_png,
            title=title,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            show_separators=sep_idx,
            separator_lw=separator_lw,
            figsize=(7.8, 6.8),
        )

    print(f"[OK] Saved center-distance heatmaps to: {out_dir}")
    for m in modalities:
        print("-", f"{m}_symbol_centers_ordered.png")


# Example usage:
# process_symbol_dictionary_figures(
#     symbolization_dir="output/symbolization",
#     out_dir="output/symbolization",
#     common_scale=True,
#     n_groups=6,    # or None
# )

 
 
############################################################################## 


def _draw_band(ax, labels, lengths, y, height, cmap, edge=False):
    """Draw one modality band as colored rectangles."""
    x0 = 0.0
    for lab, L in zip(labels, lengths):
        rect = patches.Rectangle(
            (x0, y),
            float(L),
            float(height),
            facecolor=cmap(int(lab)),
            edgecolor="black" if edge else "none",
            linewidth=0.2 if edge else 0.0
        )
        ax.add_patch(rect)
        x0 += float(L)
    return x0


def _recording_duration_minutes(data_root, recording_name):
    """
    Read timestamps (ns) from parsed_data/<recording_name>.pkl and return duration in minutes.
    Expects: pickle file with key 'gaze' containing a DataFrame with column 'timestamp[ns]'.
    """
    pkl_path = os.path.join(data_root, recording_name + ".pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing PKL file for {recording_name}: {pkl_path}")

    with open(pkl_path, "rb"):
        pass  # just to ensure pickle is imported in your file

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if "gaze" not in obj:
        raise KeyError(f"'gaze' key not found in {pkl_path}. Available keys: {list(obj.keys())}")

    df = obj["gaze"]
    if "timestamp[ns]" not in df.columns:
        raise KeyError(f"'timestamp[ns]' not found in gaze columns: {list(df.columns)}")

    t_ns = df["timestamp[ns]"].to_numpy(dtype=np.float64)
    duration_min = (np.nanmax(t_ns) - np.nanmin(t_ns)) / (60.0 * 1e9)
    return float(duration_min)


def generate_all_multimodal_scarfplots(
    symbolization_dir="output/symbolization",
    data_root="parsed_data",
    out_base_dir="output/symbolization",
    modalities=("oculomotorFixation", "oculomotorSaccade", "scanpath", "AoI", "ecg", "eda"),
    modality_labels=("Fixations", "Saccades", "Scanpaths", "AoIs", "ECG", "EDA"),
    window_lengths=None,
    align_mode="max",   # "max" or "min"
    figsize=(14, 7),
    dpi=300,
    edge=False,
    n_ticks=6,
    title=False,
):
    """
    Create ONE aligned multimodal scarf plot per recording, with x-axis labeled in minutes.

    Pipeline per recording:
      (i)  load symbolic sequences for each modality
      (ii) convert segment lengths to a common temporal unit via window_lengths
      (iii) align all modalities to the same total duration T_star by rescaling lengths
      (iv) set x-ticks in minutes using true recording duration from parsed_data/<rec>.pkl

    Output:
      output/symbolization/<recording_name>/multimodal_scarf_aligned_time.png
    """
    assert len(modalities) == len(modality_labels), "modalities and modality_labels must have same length"

    if window_lengths is None:
        window_lengths = {
            "oculomotorFixation": 10,
            "oculomotorSaccade": 10,
            "scanpath": 20,
            "AoI": 30,
            "eda": 20,
            "ecg": 20,
        }

    # ---- Load symbolization pickles
    symb_by_mod = {}
    for mod in modalities:
        pkl_path = os.path.join(symbolization_dir, f"{mod}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing symbolization file: {pkl_path}")
        with open(pkl_path, "rb") as f:
            symb_by_mod[mod] = pickle.load(f)

    # ---- Common recordings (intersection across modalities)
    rec_sets = [set(symb_by_mod[m].get("recordings", {}).keys()) for m in modalities]
    recordings = sorted(set.intersection(*rec_sets))
    if not recordings:
        raise RuntimeError("No common recordings across modalities.")

    os.makedirs(out_base_dir, exist_ok=True)

    for rec_name in recordings:
        # --- True duration from timestamps (minutes)
        T_real_min = _recording_duration_minutes(data_root, rec_name)

        # --- Cache sequences and compute modality durations (after window scaling)
        cache = {}
        durations = {}

        for mod in modalities:
            rec = symb_by_mod[mod]["recordings"][rec_name]
            labels = np.asarray(rec["sequence"], dtype=int)
            lengths = np.asarray(rec["lengths"], dtype=float)

            if labels.size == 0 or lengths.size == 0 or labels.size != lengths.size:
                raise ValueError(f"{rec_name} / {mod}: invalid sequence/lengths")

            if mod not in window_lengths:
                raise KeyError(f"window_lengths missing for modality '{mod}'")

            lengths = lengths * float(window_lengths[mod])

            cache[mod] = (labels, lengths)
            durations[mod] = float(np.sum(lengths))

        # --- Alignment target duration (abstract units)
        if align_mode == "max":
            T_star = max(durations.values())
        elif align_mode == "min":
            T_star = min(durations.values())
        else:
            raise ValueError("align_mode must be 'max' or 'min'.")

        # --- Plot
        fig, ax = plt.subplots(figsize=figsize)

        band_h = 0.85
        gap = 0.25
        y_positions = (np.arange(len(modalities))[::-1]) * (band_h + gap)

        for y, mod in zip(y_positions, modalities):
            labels, lengths = cache[mod]

            # Rescale to force sum(lengths) = T_star (alignment)
            scale = T_star / (float(np.sum(lengths)) + 1e-12)
            lengths_scaled = lengths * scale

            # Numerical fix: enforce exact final time (prevents tiny drift)
            lengths_scaled[-1] += (T_star - float(np.sum(lengths_scaled)))

            centers = symb_by_mod[mod].get("centers", None)
            n_colors = int(len(centers)) if centers is not None else int(labels.max() + 1)
            cmap = plt.cm.get_cmap("viridis", max(n_colors, 1))

            _draw_band(ax, labels, lengths_scaled, y, band_h, cmap, edge=edge)

        # --- Enforce consistent vertical framing (prevents top band looking thinner)
        ax.set_ylim(-gap, y_positions[0] + band_h + gap)

        # --- X-axis in minutes (map abstract [0, T_star] -> real [0, T_real_min])
        ax.set_xlim(0, T_star if T_star > 0 else 1.0)
        tick_pos = np.linspace(0, T_star, int(n_ticks))
        tick_lab = np.linspace(0, T_real_min, int(n_ticks))

        ax.set_xticks(tick_pos)
        ax.set_xticklabels([f"{t:.1f}" for t in tick_lab], fontsize=14)
        ax.set_xlabel("Time (minutes)", fontsize=18)

        # --- Y-axis
        ax.set_yticks(y_positions + band_h / 2.0)
        ax.set_yticklabels(modality_labels, fontsize=18)

        # --- Title (optional)
        if title:
            ax.set_title(f"Multimodal symbolic scarf plot — {rec_name}", fontsize=16, pad=10)

        # --- Styling
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        plt.tight_layout()

        rec_out = os.path.join(out_base_dir, rec_name)
        os.makedirs(rec_out, exist_ok=True)
        out_png = os.path.join(rec_out, "multimodal_scarf_aligned_time.png")

        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] Saved aligned multimodal scarf plots with real-time axis for {len(recordings)} recordings.")
