# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

import re
from collections import defaultdict



def process(config, path, feature_records):
    
    complete_records, incomplete_records = group_feature_records(feature_records)

    print("Complete records:", len(complete_records))
    print("Incomplete records:", len(incomplete_records))
    
    thr = config['general']['available_segment_prop']
    to_keep = []
    
    for record in complete_records.keys():
        d_local = complete_records[record]
        
        df_o = pd.read_csv(path + d_local['oculomotor'])
        l_o = np.count_nonzero(~np.isnan(df_o.iloc[:,1].to_numpy())) / len(df_o)
         
        df_s = pd.read_csv(path + d_local['scanpath'])
        l_s = np.count_nonzero(~np.isnan(df_s.iloc[:,1].to_numpy())) / len(df_s)
        
        df_a = pd.read_csv(path + d_local['AoI'])
        l_a = np.count_nonzero(~np.isnan(df_a.iloc[:,1].to_numpy())) / len(df_a)
        
        df_ed = pd.read_csv(path + d_local['eda'])
        l_ed = np.count_nonzero(~np.isnan(df_ed.iloc[:,1].to_numpy())) / len(df_a)
        
        df_ec = pd.read_csv(path + d_local['ecg'])
        l_ec = np.count_nonzero(~np.isnan(df_ec.iloc[:,1].to_numpy())) / len(df_a)
        
        if l_o >= thr and l_s >= thr and l_a >= thr and l_ed>=thr and l_ec>=thr:
            to_keep.append(record)
            print('Accepted recording: {s}'.format(s=record)) 
        else:
            print('Rejected recording: {s}'.format(s=record)) 
    
    
    if False:
        oculomotor_feature_records = [r + '_oculomotor.csv' for r in to_keep] 
        process_normalization(config, path, 
                              oculomotor_feature_records, 
                              type_='oculomotor')
    if False:
        scanpath_feature_records = [r + '_scanpath.csv' for r in to_keep] 
        process_normalization(config, path, 
                               scanpath_feature_records, 
                               type_='scanpath')
    if True:
        aoi_feature_records = [r + '_AoI.csv' for r in to_keep] 
        process_normalization(config, path, 
                              aoi_feature_records, 
                              type_='AoI')    
    if True:
        eda_feature_records = [r + '_eda.csv' for r in to_keep] 
        process_normalization(config, path, 
                              eda_feature_records, 
                              type_='eda') 
        
    if True:
        ecg_feature_records = [r + '_ecg.csv' for r in to_keep] 
        process_normalization(config, path, 
                              ecg_feature_records, 
                              type_='ecg') 
    
    
def process_normalization(config, path, 
                          feature_records, 
                          type_):
     
    ## Compute data dict and parameter dict 
    data = dict()
    dict_norm = dict()
    if type_=='oculomotor':
        features = config['data']['oculomotor_features']
    elif type_=='scanpath':
        features = config['data']['scanpath_features']
    elif type_=='AoI':
        features = config['data']['aoi_features']
    elif type_=='eda':
        features = config['data']['eda_features']
    elif type_=='ecg':
        features = config['data']['ecg_features']
    
    for record in feature_records:    
        df = pd.read_csv(path+record)  
        df=df.interpolate(axis=0).ffill().bfill() 
        data.update({record.split('.')[0]: df})
 
    #print('\nProcessing normalization for {type_} features'.format(type_=type_))
    for feature in features:  
        if feature != 'startTime(s)':  
            ## Concatenate data for all subjects
            ts = []
            for file in data.keys(): 
                l_data = data[file]
                vals = pd.to_numeric(l_data[feature], errors="coerce").to_numpy(dtype=float)

                # optionnel : interpolation uniquement sur cette feature
                if np.isnan(vals).any():
                    vals = pd.Series(vals).interpolate(method="linear", limit_direction="both").to_numpy()
                
                # enlever NaN et inf
                vals = vals[np.isfinite(vals)]
                ts += vals.tolist()
       
            feat_params = empirical_cdf(ts)
            dict_norm.update({feature: feat_params})
 
    ## Re-iterate to normalize according to n_params
    for file in data.keys(): 
        #print('\nAnalyzing file {rec_}'.format(rec_=file)) 
        l_data = data[file]
        new_data = dict({})
        
        for feature in features: 
            ts = l_data[feature].values
            if feature != 'startTime(s)':   
                ecdf = dict_norm[feature]
                ## To uniform distribution 
                ts_n = ecdf.evaluate(ts)
                ts_n = np.clip(ts_n, 1e-6, 1 - 1e-6)
                new_data.update({feature: ts_n}) 
            else:
                new_data.update({feature: ts})
        print(file)       
        new_df = pd.DataFrame.from_dict(new_data)    
        filename = 'output/normalized_features/{f_}.csv'.format(f_=file) 
        new_df.to_csv(filename, index=False)
                
                
 
    

def empirical_cdf(time_series, 
                  name=None, 
                  display=False):
 
    res = stats.ecdf(time_series)
    ecdf = res.cdf
 
    if display:
        plt.style.use("seaborn-v0_8") 
        plt.hist(time_series,bins=50,alpha=.3, density=True) 
         
        if name is not None:
            plt.title(name.split('_')[-1])
            fig = plt.gcf()
            path= 'output/CLDrive/figures/normalization/'
            fig.savefig(path+name)
            
        plt.show() 
        plt.clf()
 
      
    return ecdf
    
    
def group_feature_records(feature_records):
    # modalities you expect
    modalities = {"oculomotor", "scanpath", "AoI", "eda", "ecg"}

    # groups[record_id][modality] = filename
    groups = defaultdict(dict)

    # match: "<record_id>_<modality>.csv" where record_id can contain "_" and "-"
    pattern = re.compile(r"^(?P<record>.+)_(?P<mod>oculomotor|scanpath|AoI|eda|ecg)\.csv$")

    for fname in feature_records:
        m = pattern.match(fname)
        if not m:
            # skip or raise, depending on what you want
            # raise ValueError(f"Unrecognized filename format: {fname}")
            continue

        record_id = m.group("record")
        mod = m.group("mod")
        groups[record_id][mod] = fname

    # keep only complete records (have all 5 modalities)
    complete = {rid: mods for rid, mods in groups.items()
                if modalities.issubset(mods.keys())}

    # optionally also return incomplete ones for debugging
    incomplete = {rid: mods for rid, mods in groups.items()
                  if rid not in complete}

    return complete, incomplete


 
