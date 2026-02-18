# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import pandas as pd



def process(feature_records, config):
 
    if False: 
        process_subset(config, 
                       feature_records, 
                       'oculomotorFixation')
    if True: 
        process_subset(config, 
                       feature_records, 
                       'oculomotorSaccade')
    if False: 
        process_subset(config, 
                       feature_records, 
                       'scanpath')
    if False: 
        process_subset(config, 
                       feature_records, 
                       'AoI')
    if False: 
        process_subset(config, 
                       feature_records, 
                       'ecg')
    if False: 
        process_subset(config, 
                       feature_records, 
                       'eda')
        
        
def process_subset(config, 
                   feature_records,  
                   type_, 
                   display=True):
    
    features_path = 'output/features/' 
    bkpt_path = 'output/segmentation/'
    symbolization_path = 'output/symbolization/'
    
    symbolization_results = [f for f in os.listdir(symbolization_path) if f[-4:] == '.pkl']
   
    if type_=='scanpath':
        n_centers = config['symbolization']['nb_clusters']['scanpath'] 
        feature_records = [f for f in feature_records if f.endswith("scanpath.csv")]
        features = config['data']['scanpath_features']
        
        symb_file = [f for f in symbolization_results if f.split('.')[0] == 'scanpath'][0]
        with open(symbolization_path + symb_file, 'rb') as f:
            symb = pickle.load(f)
            
    elif type_=='AoI':
        n_centers = config['symbolization']['nb_clusters']['aoi'] 
        feature_records = [f for f in feature_records if f.endswith("AoI.csv")]
        features = config['data']['aoi_features']
        
        symb_file = [f for f in symbolization_results if f.split('.')[0] == 'AoI'][0]
        with open(symbolization_path + symb_file, 'rb') as f:
            symb = pickle.load(f)
            
    elif type_=='eda':
        n_centers = config['symbolization']['nb_clusters']['eda'] 
        feature_records = [f for f in feature_records if f.endswith("eda.csv")]
        features = config['data']['eda_features']
        
        symb_file = [f for f in symbolization_results if f.split('.')[0] == 'eda'][0]
        with open(symbolization_path + symb_file, 'rb') as f:
            symb = pickle.load(f)
            
    elif type_=='ecg':
        n_centers = config['symbolization']['nb_clusters']['ecg'] 
        feature_records = [f for f in feature_records if f.endswith("ecg.csv")]
        features = config['data']['ecg_features']
        
        symb_file = [f for f in symbolization_results if f.split('.')[0] == 'ecg'][0]
        with open(symbolization_path + symb_file, 'rb') as f:
            symb = pickle.load(f)
            
    elif type_=='oculomotorFixation':
        n_centers = config['symbolization']['nb_clusters']['oculomotor'] 
        feature_records = [f for f in feature_records if f.endswith("oculomotor.csv")]
        features = config['data']['oculomotor_features']
        features = [feature for feature in features if feature[:3]=='fix']
        
        symb_file = [f for f in symbolization_results if f.split('.')[0] == 'oculomotorFixation'][0]
        with open(symbolization_path + symb_file, 'rb') as f:
            symb = pickle.load(f)
            
    elif type_=='oculomotorSaccade':
        n_centers = config['symbolization']['nb_clusters']['oculomotor'] 
        feature_records = [f for f in feature_records if f.endswith("oculomotor.csv")]
        features = config['data']['oculomotor_features']
        features = [feature for feature in features if feature[:3]=='sac']
        
        symb_file = [f for f in symbolization_results if f.split('.')[0] == 'oculomotorSaccade'][0]
        with open(symbolization_path + symb_file, 'rb') as f:
            symb = pickle.load(f)
   
    symb_values = {i: [] for i in range(n_centers)}
    for record in feature_records:    

        df = pd.read_csv(features_path+record)  
        df=df.interpolate(axis=0).ffill().bfill() 
        df = df[features].to_numpy()
      
        name = record.split('.')[0].rsplit("_", 1)[0]
        symb_l = symb['recordings'][name]['sequence'] 
        bkpt_name = '{name_}_{type_}.npy'.format(name_=name, 
                                                  type_=type_)
        bkpts = np.load(bkpt_path+bkpt_name) 
        
        
        if name=='2023-10-26_13-40-27':
            print(symb_l)
            print(bkpts*config['general']['oculomotor_partition_length']/60)
        
        for i in range(0, len(bkpts)-1):
            l_data = df[bkpts[i]: bkpts[i+1]] 
            l_means = np.mean(l_data, axis=0) 
            
            symb_l_l = symb_l[i]
            symb_values[symb_l_l].append(l_means)
            
    for n in symb_values.keys():
        l_symb_values = symb_values[n] 
        l_symb_values = np.array(l_symb_values) 
        l_symb_values = np.median(l_symb_values, axis=0) 
        symb_values.update({n: l_symb_values})
        
    print(symb_values)
        
            
        
        
        
 