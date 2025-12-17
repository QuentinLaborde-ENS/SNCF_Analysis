# -*- coding: utf-8 -*-


import os 
import glob 
import pickle 

import pandas as pd
import numpy as np  

import matplotlib.pyplot as plt
import seaborn as sns 
from matplotlib import rcParams

 
from itertools import groupby
from operator import itemgetter

from gaze_sanity_check import block_analysis as ba


   
def interval_merging(w_i):
    
    intervals = list() 
    lengths = list()
    
    for k, g in groupby(enumerate(w_i), lambda ix : ix[0] - ix[1]): 
        i_l = list(map(itemgetter(1), g)) 
         
        ends_local = [i_l[0], i_l[-1]] 
        length = i_l[-1] - i_l[0] + 1
        intervals.append(ends_local) 
        lengths.append(length)
        
    if len(intervals)>0:
        intervals.pop()
        lengths.pop()
        
    return intervals, lengths   

    
def analyze(to_compute):
    
    data_props = []
    invalid_lengths = []
    blink_frequences = []
    blink_lengths_ = []
    exp_deltas_ = []
   
    
    sampling_frequency = 200
    delta_t = (1/sampling_frequency)*1e9
  
    for record in to_compute:
        print(f"\nProcessing record {record}...")
  
        with open(record, 'rb') as handle:
            results_ = pickle.load(handle) 
            
        raw_gaze = results_['raw_gaze']
        ts_ = raw_gaze['timestamp[ns]'].to_numpy()
    
        diff_ts = ts_[1:]-ts_[:-1]
        valid_diff = (diff_ts>0.5*delta_t) & (diff_ts<1.5*delta_t)
        data_prop = np.sum(valid_diff)/len(valid_diff)
        data_props.append(data_prop)
        
        missing = diff_ts[diff_ts>1.5*delta_t]/1e9
        invalid_length = np.mean(missing)
        invalid_lengths.append(invalid_length)
        
        exp_deltas_+=list(diff_ts)
        
        blinks = raw_gaze['status']
        wi_b = np.where(blinks==0.)[0]
      
        intervals, lengths = interval_merging(wi_b)
        lengths = np.array(lengths)/sampling_frequency
        blink_lengths_ += list(lengths)
        
        blink_frequence = len(lengths)/(len(blinks)/sampling_frequency)
        blink_frequences.append(blink_frequence)

    results = dict({'data_proportions': data_props,
                    'missing_lengths': invalid_lengths,
                    'experimental_deltas': exp_deltas_,
                    'blink_lengths': blink_lengths_,
                    'blink_frequencies': blink_frequences})   
    with open("gaze_sanity_check/output/sanity.pkl", 'wb') as h_:
        pickle.dump(results, h_, protocol=pickle.HIGHEST_PROTOCOL)
 


def feature_extraction(to_compute):
  
    global_dict = dict() 
    for record in to_compute:
         
        with open(record, 'rb') as handle:
            results_ = pickle.load(handle) 
            
        driver = results_['info']['driver']
        gaze = results_['gaze']
        name = record.split('/')[1].split('.')[0]
        fix_f, sac_f = ba.light_process(gaze, 
                                        name=name)
        dict_ = dict({'fixations': fix_f, 
                      'saccades': sac_f,
                      'driver': driver})
        global_dict.update({name: dict_})
            
    with open(f"gaze_sanity_check/output/features.pkl", 'wb') as handle:
        pickle.dump(global_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def parse_sanity():
    
    path = 'gaze_sanity_check/output'
    pkl_ = os.path.join(path, 'sanity.pkl')
    with open(pkl_, 'rb') as handle:
        results_ = pickle.load(handle) 
    
    data_props = results_['data_proportions']
    invalid_lengths = results_['missing_lengths']
    blink_frequences = results_['blink_frequencies']
    blink_lengths_ = results_['blink_lengths']
    exp_deltas_ = results_['experimental_deltas']

    m_data_props = np.mean(data_props)
    sd_data_props = np.std(data_props) 
    print(f"Data proporion (0.5): {m_data_props} ({sd_data_props})")
    
    m_invalid_lengths = np.mean(invalid_lengths)
    sd_invalid_lengths = np.std(invalid_lengths)
    print(f"Invalid lengths (0.5): {m_invalid_lengths} ({sd_invalid_lengths})")

    m_blink_frequences = np.mean(blink_frequences)
    sd_blink_frequences = np.std(blink_frequences)
    print(f"Blink frequence: {m_blink_frequences} ({sd_blink_frequences})")
    
    exp_deltas_ = np.array(exp_deltas_)/1e9
    exp_deltas_ = exp_deltas_[exp_deltas_<.008]
    exp_deltas_ = exp_deltas_[exp_deltas_>0.002]
    exp_deltas_df = pd.DataFrame({"Time delta (s)": exp_deltas_  })
    
    sns.set_theme(style='whitegrid', font_scale = 2.2)
    
    fig = plt.figure(figsize=(16, 6))
    sns.histplot(
    x=exp_deltas_df['Time delta (s)'], 
    bins=31, 
    color='darkblue', 
    stat='proportion'
    )
    rcParams.update({'figure.autolayout': True})
    plt.show()
    fig.savefig('gaze_sanity_check/figures/sanity/time_delta_distribution.png', dpi=150)
    plt.clf()
    
    blink_lengths_ = np.array(blink_lengths_)
    blink_lengths_ = blink_lengths_[blink_lengths_<.6]
    blink_lengths_ = blink_lengths_[blink_lengths_>.05]
    blink_lengths_df = pd.DataFrame({"Blinking interval length (s)": blink_lengths_  })
    fig = plt.figure(figsize=(16, 6))
    sns.histplot(
    x=blink_lengths_df['Blinking interval length (s)'], 
    bins=30, 
    color='darkblue', 
    stat='proportion'
    )
    rcParams.update({'figure.autolayout': True})
    plt.show()
    fig.savefig('gaze_sanity_check/figures/sanity/blink_distribution.png', dpi=150)
    plt.clf()
    
    
def parse_features():
    
    path = 'gaze_sanity_check/output'
    pkl_ = os.path.join(path, 'features.pkl')
    with open(pkl_, 'rb') as handle:
        results_ = pickle.load(handle) 
        
    records = results_.keys()
    print(records)
    global_dict = dict()
   
    for type_ in ['fixations', 'saccades']:
        global_dict.update({type_: dict()})
        for record in records:
            l_dict_results = results_[record][type_]
            for k_ in l_dict_results.keys(): 
                if k_ not in global_dict[type_].keys(): 
                    if isinstance(l_dict_results[k_], np.ndarray):
                        global_dict[type_].update({k_: list(l_dict_results[k_])})
                    else:
                        global_dict[type_].update({k_: [l_dict_results[k_]]})
                else:
                    if isinstance(l_dict_results[k_], np.ndarray):
                        global_dict[type_][k_] += list(l_dict_results[k_])
                    else:
                        global_dict[type_][k_].append(l_dict_results[k_])
                        
    to_plot = dict({'fixations': 
                        dict({
                            'durations': [0, .7, r'Fixation duration ($s$)' ],  
                            'drift_displacements': [0, 90, r'Fixation drift displacement ($px$)' ], 
                            'drift_distances': [0, 160, r'Fixation drift distance ($px$)' ], 
                            'velocity_means': [0, 600, r'Fixation velocity ($px\cdot s^{-1}$)' ], 
                            'drift_velocities': [0, 280, r'Fixation drift velocity ($px\cdot s^{-1}$)'  ], 
                            'BCEA': [0, 450, r'Fixation BCEA ($px^2$)' ], 
                            }),
                    'saccades': 
                        dict({
                            'durations': [0, .150, 'Saccade duration (s)'], 
                            'amplitudes': [0, 250, r'Saccade amplitudes ($px$)'], 
                            'efficiencies': [.9, np.inf, 'Saccade efficiency' ], 
                            'successive_deviations': [0, np.inf, r'Saccade absolute deviation ($deg$)' ], 
                            'curvature_areas': [0, 200, r'Saccade curvature area ($px^2$)' ], 
                            'velocity_peaks': [0, 2500, r'Saccade velocity peak ($px\cdot s^{-1}$)' ], 
                            'peak_accelerations': [0, 200000, r'Saccade peak acceleration ($px\cdot s^{-2}$)'], 
                            'skewness_exponents': [0, .9, 'Saccade skewness exponent']
                            })
                        })       
            
    for type_ in ['fixations', 'saccades']:
        for k_ in global_dict[type_].keys():
            if k_ in to_plot[type_].keys():
                print(k_)
                
                sns.set_theme(style='whitegrid', font_scale = 1.8)
                
                datas_ = np.array(global_dict[type_][k_])
                datas_ = datas_[datas_<to_plot[type_][k_][1]]
                datas_ = datas_[datas_>to_plot[type_][k_][0]]
                
                datas_df = pd.DataFrame({to_plot[type_][k_][2]: datas_  })
              
                if type_=='fixations':
                    c_ = 'skyblue'
                else:
                    c_='cornflowerblue'
                fig = plt.figure(figsize=(5, 6))
                ax = sns.violinplot(y=to_plot[type_][k_][2],

                    data=datas_df, color=c_, linestyle=':', linewidth=2,
                    scale="count", inner="quartile")
                ax.yaxis.get_major_formatter().set_useOffset(False) 
                rcParams.update({'figure.autolayout': True})
                plt.show()
                fig.savefig('gaze_sanity_check/figures/features/'+type_+'_'+k_+'.png', dpi=150)
                plt.clf()
    
    
    
    
    
if __name__=="__main__":  
     
    if False:
        folder = "parsed_data"
        to_compute = glob.glob(os.path.join(folder, "*.pkl"))
        analyze(to_compute)
        
    if True:
        folder = "parsed_data"
        to_compute = glob.glob(os.path.join(folder, "*.pkl"))
        feature_extraction(to_compute)
        
    if False:
        parse_sanity()
        
    if False:
        parse_features()
            
        
        
 
        
        
    
    