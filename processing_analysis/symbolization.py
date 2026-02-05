# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

import seaborn as sns
import pickle 

from sklearn.decomposition import KernelPCA 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, squareform
from fastcluster import linkage




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
            
         
           






def plot_scarf(labels, lenghts):
    """


    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    m_len : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    s_lengths = np.zeros((len(lenghts)+1)) 
    s_lengths[1:] = np.cumsum(lenghts)
    
    
    starts = s_lengths[:-1]
    ends = s_lengths[1:]
    ressources = labels
    
    df = pd.DataFrame(
        [
            dict(Task="0", Start=starts[i], Finish=ends[i], Resource=ressources[i])
            for i in range(len(starts))
        ]
    )
  
    colors_sns = sns.color_palette("viridis", n_colors=20)
    d_c = dict()
   
    for idx in (sorted(list(set(labels)))):
        d_c.update({idx: colors_sns[idx]})
 
    colors = dict({})
    for i, c in enumerate(labels):
        colors.update({c: d_c[c]})

    ## Create a scarf plot
    fig = ff.create_gantt(
        df,
        index_col="Resource",
        bar_width=0.4,
        show_colorbar=True,
        group_tasks=True,
        colors=colors,
    )
    ## Update the layout
    fig.update_layout(
        xaxis_type="linear",
        height=400,
        width=max(300, len(starts) * 80),
        xaxis_title="Time (s)",
        yaxis_title="AoI sequence index", 
        legend=dict(title=dict(text="Clusters")),
    )

    fig.show()
    
    return fig       
            
            