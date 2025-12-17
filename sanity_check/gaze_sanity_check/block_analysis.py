# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd  
import copy 
import matplotlib.pyplot as plt 

import vision_toolkit as v


def light_process(gaze, name):
    
    if name is not None:
        print(f"\nAnalyzing record {name}")
   
     
    l_ = len(gaze)
    p_df = gaze.iloc[int(.1 * l_): int(.9 * l_)]
    segmentation = v.BinarySegmentation(p_df, 
                                        sampling_frequency = 200, 
                                        segmentation_method = 'I_HMM',
                                        distance_type = 'euclidean',
                                        size_plan_x = 1920,
                                        size_plan_y = 1080,  
                                        verbose=False, 
                                        display_segmentation=True)
    fix_f = fixation_features(segmentation)
    sac_f = saccade_features(segmentation)
    
    return fix_f, sac_f

    
    
    
def fixation_features(segmentation):
   
    fix_a = v.FixationAnalysis(segmentation, 
                               verbose=False) 
    fix_features = dict()
    f_d = fix_a.fixation_durations(get_raw=True)
    fix_features.update({'duration_mean': f_d['duration_mean']})
    fix_features.update({'durations': f_d['raw']})
    
    f_vel = fix_a.fixation_average_velocity_means(weighted=False, 
                                                 get_raw=True)
    fix_features.update({'average_velocity_means': f_vel['average_velocity_means']})
    fix_features.update({'velocity_means': f_vel['raw']})
   

    f_drift = fix_a.fixation_drift_displacements(get_raw=True) 
    fix_features.update({'drift_displacement_mean': f_drift['drift_displacement_mean']})
    fix_features.update({'drift_displacements': f_drift['raw']})
    
    f_drift_dist = fix_a.fixation_drift_distances(get_raw=True) 
    fix_features.update({'drift_distance_mean': f_drift_dist['drift_cumul_distance_mean']})
    fix_features.update({'drift_distances': f_drift_dist['raw']})
    
    f_drift_vel = fix_a.fixation_drift_velocities(get_raw=True)
    fix_features.update({'drift_velocity_mean': f_drift_vel['drift_velocity_mean']})
    fix_features.update({'drift_velocities': f_drift_vel['raw']})
    
    f_bcea = fix_a.fixation_BCEA(BCEA_probability=.68, 
                                 get_raw=True)
    fix_features.update({'average_BCEA': f_bcea['average_BCEA']})
    fix_features.update({'BCEA': f_bcea['raw']})
   
    return fix_features

 
def saccade_features(segmentation):
   
    sac_a = v.SaccadeAnalysis(segmentation, 
                              verbose=False) 
    sac_features = dict()
    
    s_f = sac_a.saccade_frequency_wrt_labels()
    sac_features.update({'frequency': s_f['frequency']})
    
    
    s_d = sac_a.saccade_durations(get_raw=True)
    sac_features.update({'duration_mean': s_d['duration_mean']})
    sac_features.update({'durations': s_d['raw']})
    
    s_amp = sac_a.saccade_amplitudes(get_raw=True)
    sac_features.update({'amplitude_mean': s_amp['amplitude_mean']})
    sac_features.update({'amplitudes': s_amp['raw']})
    
    s_eff = sac_a.saccade_efficiencies(get_raw=True)
    sac_features.update({'efficiency_mean': s_eff['efficiency_mean']})
    sac_features.update({'efficiencies': s_eff['raw']})
    
    s_dev = sac_a.saccade_successive_deviations(get_raw=True)
    sac_features.update({'successive_deviation_mean': s_dev['successive_deviation_mean']})
    sac_features.update({'successive_deviations': s_dev['raw']})
    
    s_curv = sac_a.saccade_area_curvatures(get_raw=True)
    sac_features.update({'curvature_area_mean': s_curv['curvature_area_mean']})
    sac_features.update({'curvature_areas': s_curv['raw']})
    
    s_pvel = sac_a.saccade_peak_velocities(get_raw=True)
    sac_features.update({'velocity_peak_mean': s_pvel['velocity_peak_mean']})
    sac_features.update({'velocity_peaks': s_pvel['raw']})
   
    s_pacc = sac_a.saccade_peak_accelerations(get_raw=True)
    sac_features.update({'peak_acceleration_mean': s_pacc['peak_acceleration_mean']})
    sac_features.update({'peak_accelerations': s_pacc['raw']})
 
    s_skew = sac_a.saccade_skewness_exponents(get_raw=True)
    sac_features.update({'skewness_exponent_mean': s_skew['skewness_exponent_mean']})
    sac_features.update({'skewness_exponents': s_skew['raw']})
   
    return sac_features
    