# -*- coding: utf-8 -*-
import yaml
import pickle 
import os 
import glob
import numpy as np 
import copy 
import pandas as pd

import vision_toolkit as v


def process(gaze, mapped_gaze, ref_image, 
            ecg, eda, config, record):
    
    process_oculomotor=True 
    process_scanpath=False
    process_aoi=True 
    process_eda=True
    process_ecg=True
    
    if (process_oculomotor or process_scanpath or process_aoi):
        segmentation = v.BinarySegmentation(gaze, 
                                            sampling_frequency = config['sampling_frequencies']['pupil_labs'], 
                                            segmentation_method = config['general']['segmentation_method'], 
                                            size_plan_x = config['general']['size_plan_x'],
                                            size_plan_y = config['general']['size_plan_y'],    
                                            verbose=True, 
                                            display_segmentation=True)
        
        if process_oculomotor:
            process_oculomotor_features_(copy.deepcopy(segmentation), 
                                         config, record)
        if process_scanpath:
            process_scanpath_features_(copy.deepcopy(segmentation), copy.deepcopy(mapped_gaze), 
                                       config, record)
        if process_aoi:
            process_aoi_features_(copy.deepcopy(segmentation), copy.deepcopy(mapped_gaze), 
                                       config, record)
        
    return 0


def process_oculomotor_features_(segmentation, config, record):
  
    n_s = segmentation.config['nb_samples']
    s_f = config['sampling_frequencies']['pupil_labs']
    part_length = config['general']['oculomotor_partition_length']
    part_length = int(np.ceil(part_length * s_f)) 
    
    nb_segments = n_s // part_length
 
    x_ = segmentation.data_set['x_array']
    y_ = segmentation.data_set['y_array']
    
    features = config['data']['oculomotor_features']
    result_df = pd.DataFrame(columns=features)
 
    for n_ in range(nb_segments): 
        try: 
            start=n_*part_length
            end=start+part_length
            
            l_segmentation = copy.deepcopy(segmentation)
            l_segmentation.new_segmentation_results(update_segmentation_results(segmentation.segmentation_results, 
                                                                                start, end, 
                                                                                x_, y_))
            l_segmentation.new_dataset(update_dataset(segmentation.data_set, 
                                                      start, end))
            l_segmentation.new_config(update_config(segmentation.config, 
                                                    start, end)) 
            fix_f = fixation_features(l_segmentation)
            sac_f = saccade_features(l_segmentation) 
            line = [start/s_f] + fix_f + sac_f 
            result_df.loc[len(result_df), :] = line
            
        except: 
            print('Segment: {n_} rejected'.format(n_=n_))
            line = [start/s_f] + [np.nan] * (len(features)-1) 
            result_df.loc[len(result_df), :] = line
            pass 
 
    filename='output/features/{r_}_oculomotor.csv'.format(r_=record.split('.')[0])
    result_df.to_csv(filename, index=False)
    
def fixation_features(segmentation):
    
    fix_a = v.FixationAnalysis(segmentation, 
                               verbose=False) 
    fix_features = []
    fix_features.append(fix_a.fixation_durations(get_raw=False)['duration_mean']) 
    fix_features.append(fix_a.fixation_average_velocity_means(weighted=False, 
                                                              get_raw=False)['average_velocity_means']) 
    fix_features.append(fix_a.fixation_drift_displacements(get_raw=False)['drift_displacement_mean']) 
    fix_features.append(fix_a.fixation_drift_distances(get_raw=False)['drift_cumul_distance_mean']) 
    fix_features.append(fix_a.fixation_drift_velocities(get_raw=False)['drift_velocity_mean']) 
    fix_features.append(fix_a.fixation_BCEA(BCEA_probability=.68, 
                                            get_raw=False)['average_BCEA']) 
     
    return fix_features

 
def saccade_features(segmentation):
 
    sac_a = v.SaccadeAnalysis(segmentation, 
                              verbose=False) 
    sac_features = []
    
    sac_features.append(sac_a.saccade_frequency_wrt_labels()['frequency'])   
    sac_features.append(sac_a.saccade_amplitudes(get_raw=False)['amplitude_mean']) 
    sac_features.append(sac_a.saccade_efficiencies(get_raw=False)['efficiency_mean']) 
    sac_features.append(sac_a.saccade_peak_velocities(get_raw=False)['velocity_peak_mean'])  
    sac_features.append(sac_a.saccade_peak_accelerations(get_raw=False)['peak_acceleration_mean']) 
    sac_features.append(sac_a.saccade_peak_velocity_amplitude_ratios(get_raw=False)['ratio_mean'])
     
    return sac_features
    
def process_scanpath_features_(segmentation, mapped_gaze, config, record):
  
    n_s = segmentation.config['nb_samples']
    s_f = config['sampling_frequencies']['pupil_labs']
    part_length = config['general']['oculomotor_partition_length']
    part_length = int(np.ceil(part_length * s_f)) 
    
    nb_segments = n_s // part_length
 
    x_ = segmentation.data_set['x_array']
    y_ = segmentation.data_set['y_array']
    
    features = config['data']['oculomotor_features']
    result_df = pd.DataFrame(columns=features)
    
    size_plan_y_gaze, size_plan_x_gaze, _ = ref_im.shape
  
    for n_ in range(nb_segments): 
        try: 
            start=n_*part_length
            end=start+part_length
            
            l_segmentation = copy.deepcopy(segmentation)
            l_segmentation.new_segmentation_results(update_segmentation_results(segmentation.segmentation_results, 
                                                                                start, end, 
                                                                                x_, y_))
            l_segmentation.new_dataset(update_dataset(segmentation.data_set, 
                                                      start, end))
            l_segmentation.new_config(update_config(segmentation.config, 
                                                    start, end)) 
            l_mapped_gaze = mapped_gaze.iloc[start: end].copy()
            scanpath = v.Scanpath(l_segmentation,
                                  gaze_df = l_mapped_gaze,
                                  ref_image = ref_im,
                                  size_plan_x_gaze = size_plan_x_gaze,
                                  size_plan_y_gaze = size_plan_y_gaze,
                                  display_scanpath=True,  
                                  verbose=False)
        except:
            print('Segment: {n_} rejected'.format(n_=n_))
            line = [start/s_f] + [np.nan] * (len(features)-1) 
            result_df.loc[len(result_df), :] = line
            pass 
        
        
def process_aoi_features_(segmentation, mapped_gaze, config, record):
  
    n_s = segmentation.config['nb_samples']
    s_f = config['sampling_frequencies']['pupil_labs']
    part_length = config['general']['oculomotor_partition_length']
    part_length = int(np.ceil(part_length * s_f)) 
    
    nb_segments = n_s // part_length
 
    x_ = segmentation.data_set['x_array']
    y_ = segmentation.data_set['y_array']
    
    features = config['data']['oculomotor_features']
    result_df = pd.DataFrame(columns=features)
    
    size_plan_y_gaze, size_plan_x_gaze, _ = ref_im.shape
  
    for n_ in range(nb_segments): 
        #try: 
        start=n_*part_length
        end=start+part_length
        
        l_segmentation = copy.deepcopy(segmentation)
        l_segmentation.new_segmentation_results(update_segmentation_results(segmentation.segmentation_results, 
                                                                            start, end, 
                                                                            x_, y_))
        l_segmentation.new_dataset(update_dataset(segmentation.data_set, 
                                                  start, end))
        l_segmentation.new_config(update_config(segmentation.config, 
                                                start, end)) 
        l_mapped_gaze = mapped_gaze.iloc[start: end].copy()
        aoi = v.AoISequence(l_segmentation,
                              gaze_df = l_mapped_gaze,
                              ref_image = ref_im,
                              AoI_identification_method = 'I_HMM',
                              size_plan_x_gaze = size_plan_x_gaze,
                              size_plan_y_gaze = size_plan_y_gaze,
                              display_AoI_identification=True,  
                              verbose=False)
        # except:
        #     print('Segment: {n_} rejected'.format(n_=n_))
        #     line = [start/s_f] + [np.nan] * (len(features)-1) 
        #     result_df.loc[len(result_df), :] = line
        #     pass   
     
def update_config(config, 
                  start, end): 
 
    config['nb_samples'] = end-start
    return config
 

def update_dataset(data_set, 
                   start, end): 
 
    return dict({
        'x_array': data_set['x_array'][start:end] ,
        'y_array': data_set['y_array'][start:end], 
        'z_array': data_set['z_array'][start:end],
        'status': data_set['status'][start:end] , 
        'absolute_speed': data_set['absolute_speed'][start:end]
            })
  
    
def update_segmentation_results(segmentation_results, 
                                start, end, 
                                x_, y_):
 
    from vision_toolkit.utils.segmentation_utils import centroids_from_ints
 
    f_ints = segmentation_results['fixation_intervals']
    s_ints = segmentation_results['saccade_intervals']
    n_f_ints = [f_int for f_int in f_ints if (f_int[0]>=start and f_int[1]<end)]
    n_s_ints = [s_int for s_int in s_ints if (s_int[0]>=start and s_int[1]<end)]
    n_ctrds = centroids_from_ints(n_f_ints,
                                  x_, y_) 
    n_f_ints = [list(np.array(n_f_int)-start) for n_f_int in n_f_ints]
    n_s_ints = [list(np.array(n_s_int)-start) for n_s_int in n_s_ints]
    i_lab = segmentation_results['is_labeled']
    n_i_lab = i_lab[start: end]
 
    return dict({
        'is_labeled': n_i_lab,
        'fixation_intervals': n_f_ints, 
        'saccade_intervals': n_s_ints,
        'centroids': n_ctrds, 
            })
 


if __name__ == '__main__':
    
    pkl_files = glob.glob("parsed_data/*.pkl")
   
    with open('configurations/analysis.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    for pkl in pkl_files:
      if pkl =='parsed_data/2023-12-06-09-38-25.pkl':
        with open(pkl, 'rb') as handle:
            df = pickle.load(handle) 
   
        config.update({'sampling_frequencies': df['info']['sampling_frequencies']})
 
        gaze = df['gaze']
        mapped_gaze = df['mapped_gaze']
        ref_im = df['reference_image']
        ecg = df['ecg']
        eda = df['eda']
        
        record = pkl.split('/')[1].split('.')[0]
        
        process(gaze, mapped_gaze, ref_im,
                ecg, eda, config, record)
       
        
        
        
        
        
        
        