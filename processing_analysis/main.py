# -*- coding: utf-8 -*-
import yaml
import pickle 
import os 
import glob


from processing_analysis.feature_extraction import process as fe_process
from processing_analysis.feature_normalization import process as fn_process
from processing_analysis.segmentation import process as se_process



def feature_extraction():
 
    with open('configurations/analysis.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    pkl_files = glob.glob("parsed_data/*.pkl")
    for pkl in pkl_files:
      #if pkl =='parsed_data/2024-04-23_10-46-11.pkl':
        with open(pkl, 'rb') as handle:
            df = pickle.load(handle) 
   
        config.update({'sampling_frequencies': df['info']['sampling_frequencies']})
        record = pkl.split('/')[1].split('.')[0]
 
        fe_process(df['gaze'], df['mapped_gaze'], df['reference_image'],
                   df['ecg'], df['eda'], config, record)
       


def feature_normalization():

    with open('configurations/analysis.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    path = 'output/features/'
    feature_records = [f for f in os.listdir(path) if f[-4:] == '.csv']

    fn_process(config, path, feature_records)
    
    
    
def segmentation():
    
    with open('configurations/analysis.yaml', 'r') as file:
        config = yaml.safe_load(file)

    path = 'output/normalized_features/'
    feature_records = [f for f in os.listdir(path) if f[-4:] == '.csv']

    se_process(config, path, feature_records)



def symbolization():
    
    with open('configurations/analysis.yaml', 'r') as file:
        config = yaml.safe_load(file)
    



if __name__ == '__main__':
    
    
    if False:
        feature_extraction()
        
    if False:
        feature_normalization()

    if True:
        segmentation()
        
    if False:
        symbolization()
        
        
        
        
        
        
        
        
        
        
        
        
        