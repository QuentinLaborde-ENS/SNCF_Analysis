# -*- coding: utf-8 -*-
import yaml
import pickle 
import os 
import glob


from processing_analysis.feature_extraction import process as fe_process
from processing_analysis.feature_normalization import process as fn_process
from processing_analysis.segmentation import process as se_process
from processing_analysis.symbolization import process as sy_process
from processing_analysis.symbolization_metrics import process as sm_process
from processing_analysis.symbolization_metrics import plot_dist_mat
from processing_analysis.distance import process as di_process
from processing_analysis.distance_geometry_driver_consistency import process as gd_process
from processing_analysis.distance_geometry_driver_consistency import process_figures as gd_process_figures
from processing_analysis.distance_stability_robustness import process as sr_process
from processing_analysis.distance_stability_robustness import process_figure as sr_process_figure
from processing_analysis.distance_redundancy import process as rd_process
from processing_analysis.distance_redundancy import process_figures as rd_process_figures



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
        
    path = 'output/normalized_features/'
    feature_records = [f for f in os.listdir(path) if f[-4:] == '.csv']

    sy_process(config, path, feature_records)
    
    
def symbolization_metrics():
    
    #sm_process()
    plot_dist_mat()
    

def compute_distance():
    
    with open('configurations/analysis.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    path = 'output/symbolization/'
    symb_results = [f for f in os.listdir(path) if f[-4:] == '.pkl']
   
    di_process(config, path, symb_results)
    
    
def compute_distance_geometry_driver_consistency():
     
    gd_process()
    gd_process_figures()


def compute_distance_stability_robustness():
    
    sr_process()
    sr_process_figure()
    
    
def compute_distance_redundancy():
    
    rd_process()
    
    

if __name__ == '__main__':
    
    
    if False:
        feature_extraction()
        
    if False:
        feature_normalization()

    if False:
        segmentation()
        
    if False:
        symbolization()
        
    if False:
        symbolization_metrics()
        
    if False:
        compute_distance()
        
    if True:
        compute_distance_geometry_driver_consistency()
        
        
        
        
        
        
    if False:
        compute_distance_stability_robustness()
        
    if False:
        compute_distance_redundancy()
        
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        