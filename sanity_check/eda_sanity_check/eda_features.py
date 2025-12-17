# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
import math

from scipy.stats import skew, kurtosis

import matplotlib.pyplot as plt
import seaborn as sns 
from matplotlib import rcParams
 
import antropy as ant

from eda_sanity_check.sparsEDA import sparsEDA
 


def process(eda_df, sampling_frequency):
    
    seq_t = eda_df['eda'].values 
    status_t = eda_df['eda_valid'].values 
    x_t = eda_df['timestamp[ns]'].values/1e9
    SCR_t = eda_df['eda_scr'].values 
    
    l_seg = 2*60*sampling_frequency 
    nb_seg = len(seq_t)//l_seg
    
    kurtosis_s, skewness_s, mean_s = [], [], []
    perimeter_s, integral_s, potency_s, rms_s = [], [], [], []
    activity_s, mobility_s, complexity_s = [], [], [] 
    
    for i in range(nb_seg):
        idx_s = i*l_seg
        idx_e = (i+1)*l_seg
    
        seq_ = seq_t[idx_s:idx_e]
        status_ = status_t[idx_s:idx_e]
        x_ = x_t[idx_s:idx_e]
        SCR = SCR_t[idx_s:idx_e]
         
        ## Statistical domain
        kurtosis_ = kurtosis(SCR)
        if not np.isnan(kurtosis_):
            kurtosis_s.append(kurtosis_)
        skewness_ = skew(SCR)
        if not np.isnan(skewness_):
            skewness_s.append(skewness_)
        mean_ = np.mean(SCR)
        if not np.isnan(mean_):
            mean_s.append(mean_)
     
        ## Morphological domain
        perimeter_ = perimeter(SCR) 
        if not np.isnan(perimeter_):
            perimeter_s.append(perimeter_)
        integral_ = integral(SCR) 
        if not np.isnan(integral_):
            integral_s.append(integral_)
        potency_ = potency(SCR)
        if not np.isnan(potency_):
            potency_s.append(potency_)
        rms_ = rms(SCR)
        if not np.isnan(rms_):
            rms_s.append(rms_)
        
        ## Hjorth domain
        activity_ = activity(SCR) 
        if not np.isnan(activity_):
            activity_s.append(activity_)
        mobility_, complexity_ = ant.hjorth_params(SCR)
        if not np.isnan(mobility_):
            mobility_s.append(mobility_)
        if not np.isnan(complexity_):
            complexity_s.append(complexity_)
            
    dict_results = dict({
        'statistical':    dict({'kurtosis': kurtosis_s,  
                                'skewness': skewness_s, 
                                'mean': mean_s}),
        'morphological':  dict({'rms': rms_s, 
                                'integral': integral_s,  
                                'potency': potency_s}),
        'hjorth':         dict({'activity': activity_s, 
                                'mobility': mobility_s, 
                                'complexity': complexity_s}) 
        })  
 
    return dict_results
    
 
def mean_diff(seq_, x_):
    
    diff_ = np.gradient(seq_, x_)
    m_diff_ = np.mean(diff_)
    
    return m_diff_


def perimeter(seq_):
    
    perimeter_ = np.sqrt(1 + (np.diff(seq_))**2)
    perimeter_ = np.mean(perimeter_)
    
    return perimeter_


def integral(seq_):
    
    integral_ = np.abs(seq_)
    integral_ = np.mean(integral_)
    
    return integral_


def potency(seq_):
    
    potency_ = seq_**2 
    potency_ = np.mean(potency_)
    
    return potency_


def rms(seq_):
    
    rms_ = np.mean(seq_**2)
    rms_ = np.sqrt(rms_)
    
    return rms_
    
    
def activity(seq_):
    
    mu_ = np.mean(seq_)
    activity_ = np.mean((seq_-mu_)**2)
    
    return activity_


 