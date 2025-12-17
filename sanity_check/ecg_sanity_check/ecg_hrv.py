# -*- coding: utf-8 -*-

"""
Created on Sat Sep  7 19:59:49 2024

@author: quentinlaborde
"""

import numpy as np 
import pandas as pd
import math
 
import pyhrv 
import EntropyHub as EH

from ecg_sanity_check.ecg_algorithms import Pan_Tompkins_QRS, HeartRate


def process(ecg_df, sampling_frequency):
    
    seq_t = ecg_df['ecg'].values 
    status_t = ecg_df['status'].values 
    
    l_seg = 5*60*sampling_frequency 
    nb_seg = len(seq_t)//l_seg
    
    sdnn_, rmssd_, tri_index_ = [], [], []
    hf_pow_, lf_pow_, LF_HF_ = [], [], []
    apEn_, dfa_1_, dfa_2_ = [], [], [] 
    
    for i in range(nb_seg):
        idx_s = i*l_seg
        idx_e = (i+1)*l_seg
    
        seq_ = seq_t[idx_s:idx_e]
        status_ = status_t[idx_s:idx_e]
    
        QRS_detector = Pan_Tompkins_QRS()
        bpass, _, _, mwin = QRS_detector.solve(seq_, sampling_frequency)
        
        hr = HeartRate(seq_,sampling_frequency, 
                       mwin, bpass)
        r_peaks = hr.find_r_peaks() 
        
        nni = []
        no_status = np.argwhere(status_==0.).flatten()
       
        for i in range(1, len(r_peaks)):
            idx = r_peaks[i]
            idx_ = r_peaks[i-1]
            if not(idx_ in no_status or idx in no_status): 
                nni.append((idx-idx_)/sampling_frequency)
    
        nni = np.array(nni)*1000 
        nni = nni[nni<2500] #240 bpm
        nni = nni[nni>250] #24 bpm
      
        if len(nni>0):
            try:
                ## Time domain features
                sdnn = np.float32(pyhrv.time_domain.sdnn(nni)['sdnn'])
                if not np.isnan(sdnn):
                    sdnn_.append(sdnn)
             
                rmssd = np.float32(pyhrv.time_domain.rmssd(nni)['rmssd'])
                if not np.isnan(rmssd):
                    rmssd_.append(rmssd)
                
                tri_index = np.float32(pyhrv.time_domain.triangular_index(nni, plot=False)['tri_index'])
                if not np.isnan(tri_index):
                    tri_index_.append(tri_index)
                
                ## Frequency domain features
                freq_analysis = pyhrv.frequency_domain.welch_psd(nni, show=False) 
                hf_pow, lf_pow = freq_analysis['fft_norm'] 
                hf_pow, lf_pow = np.float32(hf_pow), np.float32(lf_pow)
                if not np.isnan(hf_pow):
                    hf_pow_.append(hf_pow)
                if not np.isnan(lf_pow):
                    lf_pow_.append(lf_pow)
                
                     
                LF_HF = np.float32(freq_analysis['fft_ratio'])
                if not np.isnan(LF_HF):
                    LF_HF_.append(LF_HF)
                 
                ## Non-linear features 
                apEn, _ = EH.ApEn(nni) 
                apEn = np.float32(apEn[-1])
                if not np.isnan(apEn):
                    apEn_.append(apEn)
                 
                dfa = pyhrv.nonlinear.dfa(nni, show=False)
                dfa_1 = np.float32(dfa['dfa_alpha1'])
                dfa_2 = np.float32(dfa['dfa_alpha2'])
              
                if not np.isnan(dfa_1):
                    dfa_1_.append(dfa_1)
                if not np.isnan(dfa_2):
                    dfa_2_.append(dfa_2)
                  
            except:
                pass
           
    dict_results = dict({
        'time':       dict({'sdnn': sdnn_,  
                            'rmssd': rmssd_, 
                            'tri_index': tri_index_}),
        'frequency':  dict({'hf_power': hf_pow_, 
                            'lf_power': lf_pow_,  
                            'LF_HF_ratio': LF_HF_}),
        'non_linear': dict({  
                            'app_entropy': apEn_, 
                            'dfa_alpha_1': dfa_1_, 
                            'dfa_alpha_2': dfa_2_}) 
        })  
    
    return dict_results

 