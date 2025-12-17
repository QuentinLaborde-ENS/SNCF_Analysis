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
    
    print('Processing record {r_}...'.format(r_=record))
    
    process_oculomotor=False 
    process_scanpath=False
    process_aoi=True 
    process_eda=False
    process_ecg=False
    
    if (process_oculomotor or process_scanpath or process_aoi):
        segmentation = v.BinarySegmentation(gaze, 
                                            sampling_frequency = config['sampling_frequencies']['pupil_labs'], 
                                            segmentation_method = config['general']['segmentation_method'], 
                                            size_plan_x = config['general']['size_plan_x'],
                                            size_plan_y = config['general']['size_plan_y'],    
                                            verbose=False, 
                                            display_segmentation=False)
        
        if process_oculomotor:
            process_oculomotor_features_(copy.deepcopy(segmentation), 
                                         config, record)
        if process_scanpath:
            process_scanpath_features_(copy.deepcopy(segmentation), copy.deepcopy(mapped_gaze), 
                                       config, ref_image, record)
        if process_aoi:
            process_aoi_features_(copy.deepcopy(segmentation), copy.deepcopy(mapped_gaze), 
                                       config, ref_image, record)
            
    print('...done \n')
 
    
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
 
    filename='output/features/{r_}_oculomotor.csv'.format(r_=record)
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
    
def process_scanpath_features_(segmentation, mapped_gaze, 
                               config, ref_im, record):
  
    n_s = segmentation.config['nb_samples']
    s_f = config['sampling_frequencies']['pupil_labs']
    part_length = config['general']['scanpath_partition_length']
    part_length = int(np.ceil(part_length * s_f)) 
    
    nb_segments = n_s // part_length
 
    x_ = segmentation.data_set['x_array']
    y_ = segmentation.data_set['y_array']
    
    features = config['data']['scanpath_features']
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
                                  display_scanpath=False,  
                                  verbose=False)
            sp_f = scanpath_features(scanpath) 
            line = [start/s_f] + sp_f 
            result_df.loc[len(result_df), :] = line
        
        except:
            print('Segment: {n_} rejected'.format(n_=n_))
            line = [start/s_f] + [np.nan] * (len(features)-1) 
            result_df.loc[len(result_df), :] = line
            pass 
        
    filename='output/features/{r_}_scanpath.csv'.format(r_=record)
    result_df.to_csv(filename, index=False)
  
def scanpath_features(scanpath):
 
    scan_features = []
    
    ## Compute geometrical descriptors
    geo_a = v.GeometricalAnalysis(scanpath, 
                                  verbose=False) 
    scan_features.append(geo_a.scanpath_length()['length'])
    scan_features.append(geo_a.scanpath_BCEA(BCEA_probability=.68, 
                                             display_results=False, 
                                             display_path=None)['BCEA'])
    scan_features.append(geo_a.scanpath_convex_hull(display_results=False, 
                                                    get_raw=False, 
                                                    display_path=None)['hull_area'])
    scan_features.append(geo_a.scanpath_HFD(HFD_hilbert_iterations=4, 
                                            HFD_k_max=10, 
                                            display_results=False, 
                                            get_raw=False, 
                                            display_path=None)['fractal_dimension']) 
    # Compute RQA descriptors
    rqa_a = v.RQAAnalysis(scanpath,  
                          verbose=False, 
                          display_results=False, 
                          scanpath_RQA_distance_threshold = 100)
    scan_features.append(rqa_a.scanpath_RQA_recurrence_rate()['RQA_recurrence_rate'])
    scan_features.append(rqa_a.scanpath_RQA_determinism(display_results=False, 
                                                        display_path=None)['RQA_determinism'])
    scan_features.append(rqa_a.scanpath_RQA_laminarity(display_results=False,
                                                       display_path=None)['RQA_laminarity']) 
    
    return scan_features
        
def process_aoi_features_(segmentation, mapped_gaze, 
                          config, ref_im, record):
  
    n_s = segmentation.config['nb_samples']
    s_f = config['sampling_frequencies']['pupil_labs']
    part_length = config['general']['aoi_partition_length']
    part_length = int(np.ceil(part_length * s_f)) 
    
    nb_segments = n_s // part_length
 
    x_ = segmentation.data_set['x_array']
    y_ = segmentation.data_set['y_array']
    
    features = config['data']['aoi_features']
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
            aoi = v.AoISequence(l_segmentation,
                                  gaze_df = l_mapped_gaze,
                                  ref_image = ref_im,
                                  AoI_identification_method = 'I_MS',
                                  AoI_IMS_bandwidth = 150,
                                  size_plan_x_gaze = size_plan_x_gaze,
                                  size_plan_y_gaze = size_plan_y_gaze,
                                  display_AoI_identification=True,  
                                  verbose=False)
            assert aoi.nb_aoi > 1
            aoi_f = aoi_features(aoi) 
            line = [start/s_f] + aoi_f 
            result_df.loc[len(result_df), :] = line
         
        except:
            print('Segment: {n_} rejected'.format(n_=n_))
            line = [start/s_f] + [np.nan] * (len(features)-1) 
            result_df.loc[len(result_df), :] = line
            pass   
        
    filename='output/features/{r_}_AoI.csv'.format(r_=record)
    result_df.to_csv(filename, index=False)

def aoi_features(aoi):
 
    aoi_features = []
 
    ## Add basic descriptors
    basic_a = v.AoIBasicAnalysis(aoi, 
                                 verbose=False)
    aoi_features.append(basic_a.AoI_count()['count'])
    aoi_features.append(basic_a.AoI_BCEA(BCEA_probability=.68, 
                                         get_raw=False)['average_BCEA'])
    aoi_features.append(basic_a.AoI_BCEA(BCEA_probability=.68, 
                                         get_raw=False)['disp_BCEA'])
    ## Add lempl ziv complexity 
    lz = v.AoI_lempel_ziv(aoi)
    aoi_features.append(lz['AoI_lempel_ziv_complexity'])
    
    markov_a = v.MarkovBasedAnalysis(aoi, 
                                      verbose=False, 
                                      display_results=False, 
                                      display_AoI_identification=False)  
    
    entropies = markov_a.AoI_transition_entropy(display_results=False) 
    
    aoi_features.append(np.exp(entropies['AoI_transition_stationary_entropy']))
    aoi_features.append(np.exp(entropies['AoI_transition_joint_entropy'])) 
   
    return aoi_features

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
      if pkl =='parsed_data/2024-04-23_10-46-11.pkl':
        with open(pkl, 'rb') as handle:
            df = pickle.load(handle) 
   
        config.update({'sampling_frequencies': df['info']['sampling_frequencies']})
        record = pkl.split('/')[1].split('.')[0]
    
        process(df['gaze'], df['mapped_gaze'], df['reference_image'],
                df['ecg'], df['eda'], config, record)
       
        
        
        
        
        
        
        