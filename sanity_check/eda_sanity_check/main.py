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

from eda_sanity_check.sparsEDA import sparsEDA
from eda_sanity_check.eda_features import process as eda_feat


class EDASanityCheck():
    
    def __init__(self, 
                 df, sampling_frequency):
      
        self.timestamps = df['timestamp[ns]'].values/1e9
        self.seq_eda = df['eda'].values
        self.seq_temp = df['temperature'].values
        
        self.n_ = len(self.timestamps)
        self.s_f = sampling_frequency
        
        ## Parameters of the algorithm
        self.range_thrs_1 = .02
        self.range_thrs_2 = 60
        self.slope_thrs = 10
        self.temp_thrs_1 = 25
        self.temp_thrs_2 = 40
        self.win_size = 5
        
        self.invalid_range_idx = None
        self.invalid_slope_idx = None
        self.invalid_temp_idx = None
        self.status = None
        
        self.SCR = None
        self.list_MSE = None
        self.list_dr = None
        
        
    def process(self):
        
        self.check_range()
        self.check_slope()
        self.check_temp()
        self.expand()
        
        self.decomposition()
        
        
    def check_range(self):
        
        seq_ = self.seq_eda 
        low_idx = np.argwhere(seq_<self.range_thrs_1).flatten()
        high_idx = np.argwhere(seq_>self.range_thrs_2).flatten()
         
        self.invalid_range_idx = list(set(list(low_idx)+list(high_idx)))
       
    
    def check_slope(self):
         
        seq_ = self.seq_eda
        x_ = self.timestamps 
        slope = np.abs(np.gradient(seq_, x_)) 
        
        self.invalid_slope_idx = list(np.argwhere(slope>self.slope_thrs).flatten())
    
    
    def check_temp(self):
        
        seq_ = self.seq_temp 
        low_idx = np.argwhere(seq_<self.temp_thrs_1).flatten()
        high_idx = np.argwhere(seq_>self.temp_thrs_2).flatten()
        
        self.invalid_temp_idx = list(set(list(low_idx)+list(high_idx)))
     
        #plt.figure(figsize = (20,4), dpi = 100) 
        #plt.plot(seq_[:], color = 'blue')  
        #plt.plot(self.seq_eda[:], color = 'black', linewidth=.2)
        #plt.title('Temperature')
        #plt.show()
        #plt.clf()
       
    
    def expand(self):
       
        invalids_ = list(set(self.invalid_range_idx+self.invalid_slope_idx+self.invalid_temp_idx)) 
        l_idx = 5*self.s_f
        
        n_ = self.n_
        status = np.ones(n_)
        for idx in invalids_:
            status[max(0, idx-l_idx): min(n_, idx+l_idx+1)] = 0
         
        self.status = status
         
        
    def decomposition(self):
        
        driver, SCL, SCR, list_MSE, list_dr = sparsEDA(self.seq_eda, 4)
        SCR = SCR/np.quantile(SCR, 0.995)
        self.SCR = SCR
        self.list_MSE = list_MSE
        self.list_dr = list_dr
         
        sns.set_theme(style='whitegrid', font_scale = 2.2)
        
        x_ = np.arange(len(self.seq_eda))/4
        # fig =plt.figure(figsize = (20,6), dpi = 100,) 
        
        # plt.plot(x_, SCL, color = 'darkblue', linewidth=4)  
        # plt.plot(x_, self.seq_eda, color = 'black', linewidth=2)
        # plt.xlabel("Time (s)")
        # plt.ylabel(r'EDA ($\mu$S)')
        # #plt.title('Tonic component')
        # rcParams.update({'figure.autolayout': True})
        # plt.show()
        # fig.savefig('eda_sanity_check/eda_tonic.png', dpi=100)
        # plt.clf()
         
      
        
        # fig =plt.figure(figsize = (20,6), dpi = 100)  
        # plt.plot(x_, driver, color = 'darkblue', linewidth=2)  
        # plt.xlabel("Time (s)")
        # plt.ylabel('Driver')
        # #plt.title('Phasic component')
        # rcParams.update({'figure.autolayout': True})
        # plt.show()
        # fig.savefig('eda_sanity_check/eda_phasic.png', dpi=100)
        # plt.clf()
      
     
         
        # fig =plt.figure(figsize = (20,6), dpi = 100)  
        
        # plt.plot(x_, SCR, color = 'darkblue', linewidth=2)  
        # plt.xlabel("Time (s)")
        # plt.ylabel('SCR')
        # #plt.title('Phasic component')
        # plt.ylim(0, 1)
        # rcParams.update({'figure.autolayout': True})
        # plt.show()
        # plt.clf()
        
        
         
def analyze(to_compute):
    
    sampling_frequency = 4
    global_results = dict()
    
    for record in to_compute:
        print(f"\nComputing {record}...")
        name = record.split('/')[1].split('.')[0]
       
        with open(record, 'rb') as handle:
            raw = pickle.load(handle) 
        
        df = raw['eda']
        driver = raw['info']['driver']
        
        #plt.figure(figsize = (20,4), dpi = 100)  
        #plt.plot(df['eda'].values[5000: 10000], color = 'blue')   
        #plt.title(record)
        #plt.show()
        #plt.clf()
        
        sc = EDASanityCheck(df, sampling_frequency)
        sc.process()
        
        sanity = sc.status
        SCR = sc.SCR
        
        n_ = sc.n_
        data_prop_ = np.sum(sanity)/n_
        print(f"Sanity EDA: {data_prop_}")
        #plt.figure(figsize = (20,4), dpi = 100)  
        #plt.plot(sanity, color = 'blue')  
        #plt.title(record)
        #plt.show()
        #plt.clf()
        
        df['eda_valid'] = sanity  
        df['eda_scr'] = SCR  
        
        results_ = dict({'df': df, 
                         'mse': sc.list_MSE, 
                         'driver_frequency': sc.list_dr,
                         'driver': driver})
        global_results.update({name: results_})
        
    with open("eda_sanity_check/output/sanity.pkl", 'wb') as h_:
        pickle.dump(global_results, h_, protocol=pickle.HIGHEST_PROTOCOL)
   
         
 
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
            


def feature_extraction():
    
    global_dict = dict()
    path = 'eda_sanity_check/output'
    pkl_ = os.path.join(path, "sanity.pkl")
    with open(pkl_, 'rb') as handle:
        sanity = pickle.load(handle) 
    
    records = sanity.keys()
     
    for record in records:
        print(f"\nProcessing record {record}...")
        df = sanity[record]['df']
        driver = sanity[record]['driver']
        
        dict_ = eda_feat(df, 4)
        dict_.update({'driver': driver})
        global_dict.update({record: dict_})
        
    with open(f"eda_sanity_check/output/features.pkl", 'wb') as h_:
        pickle.dump(global_dict, h_, protocol=pickle.HIGHEST_PROTOCOL)
         
            
def parse_sanity():
    
    data_props_ = []
    invalid_lengths_ = []
    
    driver_frequency_ = []
    mse_ = []
    
    path = 'eda_sanity_check/output'
    sanity_pkl = os.path.join(path, "sanity.pkl")
    with open(sanity_pkl, 'rb') as handle:
        results_ = pickle.load(handle) 
    
    records = results_.keys()    
    for record in records:
        MSE = results_[record]['mse']
        mse_ += MSE
        
        driver_frequency = results_[record]['driver_frequency']
        driver_frequency_+= list(driver_frequency)
        
        df = results_[record]['df']
        sanity = df['eda_valid'].values
        prop = np.sum(sanity)/len(sanity)
        data_props_.append(prop)
        
        wi_i = np.where(sanity == 0.)[0] 
        intervals, lengths = interval_merging(wi_i)
        if len(lengths)>0: 
            lengths = np.array(lengths)/4
            invalid_lengths_ += list(lengths)
     
    m_data_props_ = np.mean(data_props_)
    sd_data_props_ = np.std(data_props_)
    print(f"\nFinal data proportion: {m_data_props_} ({sd_data_props_})\n")
    
    driver_frequency_m = np.mean(driver_frequency_)
    driver_frequency_sd = np.std(driver_frequency_)
    print(f"\nFinal driver frequency: {driver_frequency_m} ({driver_frequency_sd})\n")
    
    sns.set_theme(style='whitegrid', font_scale = 2.2)
    
    m_invalid_lengths_ = np.mean(invalid_lengths_)
    sd_invalid_lengths_ = np.std(invalid_lengths_)
    print(f"\nFinal invalid lengths: {m_invalid_lengths_} ({sd_invalid_lengths_})\n")
    
    
    invalid_lengths_ = np.array(invalid_lengths_)
    invalid_lengths_ = invalid_lengths_[invalid_lengths_<200]
    invalid_lengths_df = pd.DataFrame({"Missing interval length (s)": invalid_lengths_  })
    fig = plt.figure(figsize=(16, 6))
    sns.histplot(
    x=invalid_lengths_df['Missing interval length (s)'], 
    bins=30, 
    color='darkblue', 
    stat='proportion'
    )
    rcParams.update({'figure.autolayout': True})
    plt.show()
    fig.savefig('eda_sanity_check/figures/sanity/eda_length_distribution.png', dpi=150)
    plt.clf()
    
    mse_ = np.array(mse_)
    mse_ = mse_[mse_<0.0003]
    
    m_mse_ = np.mean(mse_)
    sd_mse_ = np.std(mse_)
    print(f"\nFinal mse: {m_mse_} ({sd_mse_})\n")
    
    
    
    mse_df = pd.DataFrame({"MSE of reconstruction": mse_  })
    fig = plt.figure(figsize=(16, 6))
    sns.histplot(
    x=mse_df['MSE of reconstruction'], 
    bins=30, 
    color='darkblue', 
    stat='proportion'
    )
    rcParams.update({'figure.autolayout': True})
    plt.show()
    fig.savefig('eda_sanity_check/figures/sanity/eda_mse.png', dpi=150)
    plt.clf()
    
    driver_frequency_ = np.array(driver_frequency_)
    
    m_driver_frequency_ = np.mean(driver_frequency_)
    sd_driver_frequency_ = np.std(driver_frequency_)
    print(f"\nFinal driver frequency: {m_driver_frequency_} ({sd_driver_frequency_})\n")
    
    
    driver_frequency_ = driver_frequency_[driver_frequency_<2000]
    driver_frequency_df = pd.DataFrame({r'Phasic component driver frequency ($s^{-1}$)': driver_frequency_  })
    fig = plt.figure(figsize=(16, 6))
    sns.histplot(
    x=driver_frequency_df[r'Phasic component driver frequency ($s^{-1}$)'], 
    bins=22, 
    color='darkblue', 
    stat='proportion'
    )
    rcParams.update({'figure.autolayout': True})
    plt.show()
    fig.savefig('eda_sanity_check/figures/sanity/eda_driver_frequecy.png', dpi=150)
    plt.clf()

            
  
    
  
    
  
    
  
    
  
    
  
                
def parse_features():
    
    path = 'eda_sanity_check/output'
    pkl_ = os.path.join(path, 'features.pkl')
    with open(pkl_, 'rb') as handle:
        results_ = pickle.load(handle) 
        
    records = results_.keys()
    
    global_dict = dict()
    
    for type_ in ['statistical', 
                  'morphological', 
                  'hjorth']:
        global_dict.update({type_: dict()})
        
        for record in records:
            l_dict_results = results_[record][type_]
            for k_ in l_dict_results.keys(): 
                if k_ not in global_dict[type_].keys():  
                    global_dict[type_].update({k_: l_dict_results[k_]})
                   
                else: 
                    global_dict[type_][k_] += l_dict_results[k_]
  
    
    to_plot = dict({'statistical': 
                        dict({
                            'kurtosis': [0, 100, r'Kurtosis' ],   
                            'skewness': [0, 10, r'Skewness' ], 
                            'mean': [0, 0.15, r'Mean activity ($\mu S$)' ], 
                            }),
                    'morphological': 
                        dict({
                            'rms': [0, 0.5, r'RMS'], 
                            'integral': [0, 0.2, r'Integral'],  
                            'potency': [0.003, 0.06, r'Potency' ], 
                            }),
                    'hjorth': 
                        dict({  
                            'activity': [0, 0.04, r'Activity'], 
                            'mobility': [0, 0.6, r'Mobility' ], 
                            'complexity': [0, 8, r'Complexity'  ],  
                            })
                        }) 
    
    for type_ in ['statistical', 'morphological', 'hjorth']:
        for k_ in global_dict[type_].keys():
            if k_ in to_plot[type_].keys():
                
                sns.set_theme(style='whitegrid', font_scale = 1.8)
                
                datas_ = np.array(global_dict[type_][k_])
                datas_ = datas_[datas_<to_plot[type_][k_][1]]
                datas_ = datas_[datas_>to_plot[type_][k_][0]]
                
                datas_df = pd.DataFrame({to_plot[type_][k_][2]: datas_  })
                
                if type_=='statistical':
                    c_ = 'skyblue'
                elif type_=='morphological':
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
                fig.savefig('eda_sanity_check/figures/features/'+type_+'_'+k_+'.png', dpi=150)
                plt.clf()
               
   
                
if __name__=="__main__":  
    
     
    if False:
        folder = "parsed_data"
        to_compute = glob.glob(os.path.join(folder, "*.pkl"))
        analyze(to_compute)
    if True:
        feature_extraction()
    if False: 
        parse_sanity() 
    if False:
        parse_features()
        
    
    
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        