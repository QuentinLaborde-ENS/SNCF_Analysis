# -*- coding: utf-8 -*-


import os 
import glob 
import pickle 

import pandas as pd
import numpy as np  

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

import itertools 
from itertools import groupby
from operator import itemgetter

from ecg_sanity_check.ecg_algorithms import Pan_Tompkins_QRS, HeartRate
from ecg_sanity_check.ecg_hrv import process as hrv_p

sns.set_theme(style='whitegrid', font_scale = 2.2)


class ECGSanityCheck():
    
    def __init__(self, 
                 ecg_sequence, sampling_frequency, 
                 pathological = False):
   
        self.seq_ = ecg_sequence
        self.n_ = len(ecg_sequence)
        self.s_f = sampling_frequency
        self.pathological = pathological
        
        ## Parameters of the algorithm
        self.flat_thrs = 1.5
        self.flat_prop = .8 
        self.nzc_thrs = 200
        self.amp_thrs = 1500
        self.iscore_thrs_1 = .50
        self.iscore_thrs_2 = 0.90
        ## For event score updating 
        self.a_g = 1.7 
        self.b_g = .5 
        self.c_g = .005 
        self.a_n = 1.6
        self.b_n = .4
        self.c_n = .01
        
        self.acceptable = True
        
        self.heart_rate = None
        self.r_amplitudes = None
        self.r_indexes = None
        
        self.M_x = None
        self.G_x = None
        self.i_score = None
        self.a_score = None
        
        self.process()
        
        
    def process(self): 
    
        self.feasability()
        
        if self.acceptable:
            self.pan_tompkins_QRS() 
        if self.acceptable:
            self.beats_average_correlation()
        if self.acceptable and self.pathological:
            self.beats_clustering()
            
        #print(f"Data sample quality is acceptable: {self.acceptable}")
    
       
    def beats_average_correlation(self):
       
        seq_ = self.seq_
        rr_intervals = np.diff(self.r_indexes)
       
        rr_mean = np.mean(rr_intervals)
        rr_med = np.median(rr_intervals)
        
        beta = min(rr_mean, rr_med)
        half_win = int(np.round(beta/2))
      
        QRS_complexes = []
        for r_index in self.r_indexes: 
            qrs_= seq_[r_index-half_win: r_index+half_win+1]
            if len(qrs_) == 2*half_win +1:
                QRS_complexes.append(qrs_)
          
        QRS_complexes = np.array(QRS_complexes) 
       
        self.M_x = np.corrcoef(QRS_complexes.astype(float))
        self.G_x = np.mean(self.M_x, axis=1)
        self.i_score = np.mean(self.G_x)
        #print(f"I Score: {self.i_score}")
        if self.i_score<self.iscore_thrs_1:
            self.acceptable=False 
     
    
    def beats_clustering(self):
     
        if self.i_score>self.iscore_thrs_2:
            return
        elif self.i_score<self.iscore_thrs_1:
            self.acceptable=False
            return
        
        else: 
            G_x = self.G_x
            M_x = self.M_x
            ## Get the main beat peak
            i_a = np.argmax(G_x) 
            M_ia = M_x[i_a]
            ## Get indexes of suspect and good peaks
            S_g = np.argwhere(M_ia<self.iscore_thrs_1).flatten()
            G_g = np.argwhere(M_ia>=self.iscore_thrs_1).flatten()         
            ## Get corresponding correlation matrices
            M_S_g = M_x[S_g][:,S_g]
            M_G_g = M_x[G_g][:,G_g]
          
            event_scores = []
            noise_count, group_count = 0, 0
            while len(M_S_g)>0:
                
                l_idx = np.argwhere(M_S_g[0]>(self.iscore_thrs_2+0.0051)).flatten()
                
                ## If noise
                if len(l_idx) == 1:
                    noise_count += 1 
                    l_score = self.a_n *np.exp(-self.b_n * noise_count) - self.c_n
                    event_scores.append(l_score)
                ## If group of peaks 
                else:
                    group_count += 1
                    l_score = (np.sum(M_S_g[0][l_idx])-1)/(len(l_idx)-1)  
                    l_score *= self.a_g*np.exp(-self.b_g * group_count) - self.c_g
                    event_scores.append(l_score)
                     
                M_S_g = np.delete(M_S_g, l_idx, 0)
                M_S_g = np.delete(M_S_g, l_idx, 1)
            
            score_S = np.mean(event_scores)
            score_G = np.mean(np.mean(M_G_g, axis=1)) 
            self.a_score = (score_S + score_G)/2
           
    
    def feasability(self):
   
        self.flat_line()
        self.noise_detection()
  
        
    def flat_line(self):
      
        seq_ = self.seq_
        n_ = self.n_
        prop_ = self.flat_prop

        n_seq_ = 1000*(seq_ - np.min(seq_)) / (np.max(seq_) - np.min(seq_)) 
        flat_idx_l = int(self.flat_thrs*self.s_f) 
        current_i = int(flat_idx_l)
        
        while current_i<n_ and self.acceptable:  
            l_seq = n_seq_[current_i-flat_idx_l: current_i]
            diff_ = np.abs(l_seq[1:]-l_seq[:-1]) 
            prop = np.sum(diff_<1)/flat_idx_l
            
            if prop>prop_:
                self.acceptable=False 
            current_i += 50
    
    
    def noise_detection(self):
    
        seq_ = self.seq_
        n_ = self.n_
        
        n_seq_ = 1*(seq_ - np.min(seq_)) / (np.max(seq_) - np.min(seq_))- .5 
        nzc = ((n_seq_[:-1] * n_seq_[1:]) < 0).sum()
        #print(nzc)
        if nzc>self.nzc_thrs:
            self.acceptable=False 
    
    
    def pan_tompkins_QRS(self):
    
        seq_ = self.seq_
        s_f = self.s_f
        
        QRS_detector = Pan_Tompkins_QRS()
        bpass, _, _, mwin = QRS_detector.solve(seq_, s_f)
        
        x_ = np.arange(0, len(seq_), 500)/250
        
        
        # fig =plt.figure(figsize = (20,6), dpi = 100,) 
        # plt.plot(seq_, color = 'darkblue', linewidth=2)         
        # plt.xticks(np.arange(0, len(seq_), 500), x_)
        # plt.xlabel("Time (s)")
        # plt.ylabel('Amplitude (mV)')
        # rcParams.update({'figure.autolayout': True})
        # plt.show()
        # #fig.savefig('ecg_sanity_check/raw_ecg.png', dpi=100)
        # plt.clf()
        
        
        # fig =plt.figure(figsize = (20,6), dpi = 100,) 
        # plt.plot(mwin, color = 'darkblue', linewidth=2)         
        # plt.xticks(np.arange(0, len(seq_), 500), x_)
        # plt.xlabel("Time (s)")
        # plt.ylabel('Amplitude')
        # rcParams.update({'figure.autolayout': True})
        # plt.show()
        # fig.savefig('ecg_sanity_check/mwin_ecg.png', dpi=100)
        # plt.clf()
        
        
        hr = HeartRate(seq_,s_f, 
                       mwin, bpass)
        result = hr.find_r_peaks() 
        ## Clip the x locations less than 0 (Learning Phase)
        result = result[result > 0]
        self.RR_intervals = np.diff(result)/s_f
        self.heart_rate = (60*s_f)/np.average(np.diff(result))
        
        self.r_amplitudes = seq_[result]
        self.r_indexes = result
         
        ## Plotting the R peak locations in ECG signal
        
        # fig =plt.figure(figsize = (20,6), dpi = 100,) 
        # plt.plot(seq_, color = 'darkblue', linewidth=2)        
        # plt.scatter(result, seq_[result], color = 'red', s = 120, marker= 'o') 
        # plt.xticks(np.arange(0, len(seq_), 500), x_)
        # plt.xlabel("Time (s)")
        # plt.ylabel('Amplitude (mV)')
        # rcParams.update({'figure.autolayout': True})
        # plt.show()
        # fig.savefig('ecg_sanity_check/peak_ecg.png', dpi=100)
        # plt.clf()
        
        if any(self.r_amplitudes > self.amp_thrs):
            self.acceptable=False 
            
        if self.heart_rate > 240 or self.heart_rate < 24:
            self.acceptable=False
       

def interval_merging(w_i):
    
    intervals = list() 
    lengths = list()
    
    for k, g in groupby(enumerate(w_i), lambda ix : ix[0] - ix[1]): 
        i_l = list(map(itemgetter(1), g)) 
         
        ends_local = [i_l[0], i_l[-1]] 
        length = i_l[-1] - i_l[0]
        intervals.append(ends_local) 
        lengths.append(length)
        
    if len(intervals)>0:
        intervals.pop()
        lengths.pop()
        
    return intervals, lengths   
         


def analyze(to_compute):
   
    interval_length = 10 
    sampling_frequency = 200
    global_results = dict()
    
    for record in to_compute:
      #if record == 'parsed_data/2024-05-16-08-37-40.pkl':
        print(f"\nComputing {record}...")
        name = record.split('/')[1].split('.')[0]
      
        with open(record, 'rb') as handle:
            raw = pickle.load(handle) 
        
        df = raw['ecg']
        driver = raw['info']['driver']
         
        ## Interval length in samples 
        interval_idx_l = interval_length*sampling_frequency
        #print(len(df))
        i=0
        sanity_1 = np.zeros(len(df))
        sanity_2 = np.zeros(len(df))
        hr_dict = dict()
        hr_1, hr_2 = [], []
        rr_1, rr_2 = [], []
        while (i+interval_idx_l)<len(df):
           
            try:
                sequence_1 = df['ecg_lead_1'].values[i:i+interval_idx_l]
              
                sc1 = ECGSanityCheck(sequence_1, sampling_frequency)
                sanity_1[i:i+interval_idx_l] = sc1.acceptable
                if sc1.acceptable:
                    hr_1.append(sc1.heart_rate)
                    rr_1 += list(sc1.RR_intervals)
            except:
                sanity_1[i:i+interval_idx_l] = 0.
                
            try:
                sequence_2 = df['ecg_lead_2'].values[i:i+interval_idx_l]
                sc2 = ECGSanityCheck(sequence_2, sampling_frequency)
                sanity_2[i:i+interval_idx_l] = sc2.acceptable
               
                if sc2.acceptable:
                    hr_2.append(sc2.heart_rate)
                    rr_2 += list(sc2.RR_intervals)
            except: 
                sanity_2[i:i+interval_idx_l] = 0. 
            
          
            i+=interval_idx_l
        sanity_1[i:] = sanity_1[i]
        sanity_2[i:] = sanity_2[i]
        df['ecg_lead_1_valid'] = sanity_1
        df['ecg_lead_2_valid'] = sanity_2
        #print(df.head())
        print(f"Sanity ECG 1: {np.sum(sanity_1)/len(df)}")
        print(f"Sanity ECG 2: {np.sum(sanity_2)/len(df)}")
        
        hr_dict.update({'hr_1': hr_1, 
                        'hr_2': hr_2,
                        'rr_1': rr_1, 
                        'rr_2': rr_2}) 
        results_ = dict({'df': df, 
                         'hr': hr_dict,
                         'driver': driver})
        global_results.update({name: results_})
    
        
    with open("ecg_sanity_check/output/sanity.pkl", 'wb') as h_:
        pickle.dump(global_results, h_, protocol=pickle.HIGHEST_PROTOCOL)
   
            
def feature_extraction():
    
    global_dict = dict()
    path = 'ecg_sanity_check/output'
    pkl_ = os.path.join(path, "sanity.pkl")
    with open(pkl_, 'rb') as handle:
        sanity = pickle.load(handle) 
    
    records = sanity.keys()
    
    for record in records:
        print(f"\nProcessing record {record}...")
        
        driver = sanity[record]['driver']
        df = sanity[record]['df']
        sanity_1 = df['ecg_lead_1_valid'].values
        sanity_2 = df['ecg_lead_2_valid'].values
        
        prop_1 = np.sum(sanity_1)/len(sanity_1)
        prop_2 = np.sum(sanity_2)/len(sanity_2)
      
        if prop_1>prop_2:
            data = dict({
                    'timestamp[s]': df['timestamp[ns]'].values/1e9,
                    'ecg': df['ecg_lead_1'].values,
                    'status': df['ecg_lead_1_valid'].values,  
                        })  
        else:
            data = dict({
                    'timestamp[s]': df['timestamp[ns]'].values/1e9,
                    'ecg': df['ecg_lead_2'].values,
                    'status': df['ecg_lead_2_valid'].values,  
                        }) 
        df_clean = pd.DataFrame(data=data) 
       
        dict_ = hrv_p(df_clean, 250)
        dict_.update({'driver': driver})
        global_dict.update({record: dict_})
    
    with open("ecg_sanity_check/output/features.pkl", 'wb') as h_:
        pickle.dump(global_dict, h_, protocol=pickle.HIGHEST_PROTOCOL)
      
        
def parse_sanity():
    
    data_props_ = []
    invalid_lengths_ = []   
    
    hr_s = []
    rr_s = []
    
    path = 'ecg_sanity_check/output'
    sanity_pkl = os.path.join(path, "sanity.pkl")
    with open(sanity_pkl, 'rb') as handle:
        results_ = pickle.load(handle) 
        
    records = results_.keys()    
    for record in records:
        df = results_[record]['df']
        hr_dict = results_[record]['hr']
        
        sanity_1 = df['ecg_lead_1_valid'].values
        sanity_2 = df['ecg_lead_2_valid'].values
        
        prop_1 = np.sum(sanity_1)/len(sanity_1)
        prop_2 = np.sum(sanity_2)/len(sanity_2)
       
        if prop_1>prop_2:
            data_props_.append(prop_1)
            wi_i = np.where(sanity_1 == 0.)[0]  
            hr_ = hr_dict['hr_1']
            rr_ = hr_dict['rr_1']
        else:
            data_props_.append(prop_2)
            wi_i = np.where(sanity_2 == 0.)[0]
            hr_ = hr_dict['hr_2']
            rr_ = hr_dict['rr_2']
            
        intervals, lengths = interval_merging(wi_i)
        hr_s+=list(hr_)
        rr_s+=list(rr_)
        
        if len(lengths)>0: 
            lengths = np.array(lengths)/250
            invalid_lengths_ += list(lengths)
            
    data_props_m = np.mean(data_props_)
    print(f"\nFinal data proportion: {data_props_m}")
    data_props_sd = np.std(data_props_)
    print(f"\nFinal data proportion SD: {data_props_sd}")
    
    sns.set_theme(style='whitegrid', font_scale = 2.2)
    
    hr_s = np.array(hr_s)
    hr_s = hr_s[hr_s<120]
    hr_s_df = pd.DataFrame({r'Heart rate ($s^{-1}$)': hr_s  })
    fig = plt.figure(figsize=(16, 6))
    sns.histplot(
    x=hr_s_df[r'Heart rate ($s^{-1}$)'], 
    bins=30, 
    color='darkblue', 
    stat='proportion'
    )
    rcParams.update({'figure.autolayout': True})
    plt.show()
    fig.savefig('ecg_sanity_check/figures/sanity/heart_rate_distribution.png', dpi=150)
    plt.clf()
    
    m_invalid_lengths_ = np.mean(invalid_lengths_)
    sd_invalid_lengths_ = np.std(invalid_lengths_)
    print(f"\nFinal length: {m_invalid_lengths_}, SD: {sd_invalid_lengths_}")
    
    invalid_lengths_ = np.array(invalid_lengths_)
    invalid_lengths_ = invalid_lengths_[invalid_lengths_<80]
    invalid_lengths_df = pd.DataFrame({r"Missing interval length ($s$)": invalid_lengths_  })
    fig = plt.figure(figsize=(16, 6))
    sns.histplot(
    x=invalid_lengths_df[r'Missing interval length ($s$)'], 
    bins=30, 
    color='darkblue', 
    stat='proportion'
    )
    rcParams.update({'figure.autolayout': True})
    plt.show()
    fig.savefig('ecg_sanity_check/figures/sanity/ecg_length_distribution.png', dpi=150)
    plt.clf()
    
    rr_s = np.array(rr_s)*1000
    rr_s = rr_s[rr_s<1600]
    rr_s = rr_s[rr_s>250]
    rr_s_df = pd.DataFrame({r'RR interval ($ms$)': rr_s  })
    fig = plt.figure(figsize=(16, 6))
    sns.histplot(
    x=rr_s_df[r'RR interval ($ms$)'], 
    bins=30, 
    color='darkblue', 
    stat='proportion'
    )
    rcParams.update({'figure.autolayout': True})
    plt.show()
    fig.savefig('ecg_sanity_check/figures/sanity/rr_length_distribution.png', dpi=150)
    plt.clf()
    
 
            
def parse_features():
    
    path = 'ecg_sanity_check/output'
    pkl_ = os.path.join(path, 'features.pkl')
    with open(pkl_, 'rb') as handle:
        results_ = pickle.load(handle) 
        
    records = results_.keys() 
    global_dict = dict()
    
    for type_ in ['time', 'frequency', 'non_linear']:
        global_dict.update({type_: dict()})
        for record in records:
            l_dict_results = results_[record][type_]
            for k_ in l_dict_results.keys(): 
                if k_ not in global_dict[type_].keys():  
                    global_dict[type_].update({k_: l_dict_results[k_]})
                   
                else: 
                    global_dict[type_][k_] += l_dict_results[k_]
               
 
    to_plot = dict({'time': 
                        dict({
                            'sdnn': [0, 140, r'SDNN' ],   
                            'rmssd': [0, 150, r'RMSSD' ], 
                            'tri_index': [0, 20, r'Triangular index' ], 
                            }),
                    'frequency': 
                        dict({
                            'hf_power': [0, 7500, r'High frequency normalized power'], 
                            'lf_power': [0, 4000, r'Low frequency normalized power'],  
                            'LF_HF_ratio': [0, 20, r'LF/HF ratio' ], 
                            }),
                    'non_linear': 
                        dict({  
                            'app_entropy': [0, np.inf, r'Approximate entropy'], 
                            'dfa_alpha_1': [0, np.inf, r'DFA $\alpha_1$' ], 
                            'dfa_alpha_2': [0, np.inf, r'DFA $\alpha_2$'  ],  
                            })
                        }) 
        
 
    for type_ in ['time', 'frequency', 'non_linear']:
        for k_ in global_dict[type_].keys():
            if k_ in to_plot[type_].keys():
              
                sns.set_theme(style='whitegrid', font_scale = 1.8)
                
                datas_ = np.array(global_dict[type_][k_])
                datas_ = datas_[datas_<to_plot[type_][k_][1]]
                datas_ = datas_[datas_>to_plot[type_][k_][0]]
                
                datas_df = pd.DataFrame({to_plot[type_][k_][2]: datas_  })
            
                if type_=='time':
                    c_ = 'skyblue'
                elif type_=='frequency':
                    c_ = 'cornflowerblue'
                else:
                    c_='plum'
                fig = plt.figure(figsize=(5, 6))
                ax = sns.violinplot(y=to_plot[type_][k_][2],

                    data=datas_df, color=c_, linestyle=':', linewidth=2,
                    scale="count", inner="quartile")
                ax.yaxis.get_major_formatter().set_useOffset(False) 
                rcParams.update({'figure.autolayout': True})
                plt.show()
                fig.savefig('ecg_sanity_check/figures/features/'+type_+'_'+k_+'.png', dpi=150)
                plt.clf()
 

if __name__=="__main__":  
    
 
     
    if False:
        folder = "parsed_data"
        to_compute = glob.glob(os.path.join(folder, "*.pkl"))
        analyze(to_compute) 
    if False:
        feature_extraction()
    if False:
        parse_sanity()
    if True:
        parse_features()

        
        
        
        
            