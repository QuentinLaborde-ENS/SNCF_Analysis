# -*- coding: utf-8 -*-

import numpy as np
import ruptures as rpt 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter




def process(config, path, feature_records):
 
    outpath = 'output/segmentation/'
    ## For oculomotor features only
    if True:
        print('Segmenting oculomotor features...')
        oculomotor_feature_records = [feature_record for feature_record in feature_records
                              if feature_record.split('.')[0].split('_')[-1] == 'oculomotor']
        
        for record in oculomotor_feature_records:
            df= pd.read_csv(path+record) 
            name = record.split('.')[0]
             
            df_fix = df[[col for col in df.columns if col[:3]=='fix']] 
            signal_fix = df_fix.to_numpy()
            bkps=signal_segmentation(signal_fix,
                                     None,
                                     )
            bkps.insert(0, 0)
            filename = '{out_}{name_}Fixation.npy'.format(out_=outpath, 
                                                          name_=name)
            np.save(filename, np.array(bkps))
            
            df_sac = df[[col for col in df.columns if col[:3]=='sac']] 
            signal_sac = df_sac.to_numpy()
            bkps=signal_segmentation(signal_sac,
                                     None, 
                                     )
            bkps.insert(0, 0)
            filename = '{out_}{name_}Saccade.npy'.format(out_=outpath, 
                                                         name_=name)
            np.save(filename, np.array(bkps)) 
            #display_segmentation(signal_sac, bkps, config['general']['oculomotor_partition_length'])
        print('...done \n')
        
    ## For scanpath features only
    if True:
        print('Segmenting scanpath features...')
        scanpath_feature_records = [feature_record for feature_record in feature_records
                              if feature_record.split('.')[0].split('_')[-1] == 'scanpath']
        for record in scanpath_feature_records:
            df= pd.read_csv(path+record) 
            name = record.split('.')[0]
            df_sp = df[[col for col in df.columns if col[:2]=='Sp']] 
            signal = df_sp.to_numpy()
     
            bkps=signal_segmentation(signal,
                                     None,
                                     )
            bkps.insert(0, 0)
            filename = '{out_}{name_}.npy'.format(out_=outpath, 
                                                  name_=name)
            np.save(filename, np.array(bkps))
            #display_segmentation(signal, bkps, config['general']['scanpath_partition_length'])
        print('...done \n')
        
    ## For aoi sequence features 
    if True: 
        print('Segmenting aoi features...')
        aoi_feature_records = [feature_record for feature_record in feature_records
                              if feature_record.split('.')[0].split('_')[-1] == 'AoI']
        for record in aoi_feature_records:
            df= pd.read_csv(path+record) 
            name = record.split('.')[0]
            df_aoi = df[[col for col in df.columns if col[:3]=='AoI']] 
            signal = df_aoi.to_numpy()
     
            bkps=signal_segmentation(signal,
                                     None,
                                     )
            bkps.insert(0, 0)
            filename = '{out_}{name_}.npy'.format(out_=outpath, 
                                                  name_=name)
            np.save(filename, np.array(bkps))
            #display_segmentation(signal, bkps, config['general']['aoi_partition_length'])
        print('...done \n')
        
    ## For eda sequence features 
    if True: 
        print('Segmenting eda features...')
        eda_feature_records = [feature_record for feature_record in feature_records
                              if feature_record.split('.')[0].split('_')[-1] == 'eda']
        for record in eda_feature_records:
            df= pd.read_csv(path+record) 
            name = record.split('.')[0]
            df_eda = df[[col for col in df.columns if col[:3]=='eda']]  
            signal = df_eda.to_numpy()
            
            bkps=signal_segmentation(signal,
                                     None,
                                     )
            bkps.insert(0, 0)
            filename = '{out_}{name_}.npy'.format(out_=outpath, 
                                                  name_=name)
            np.save(filename, np.array(bkps))  
            #display_segmentation(signal, bkps, config['general']['eda_partition_length'])
        print('...done \n')
   
    ## For ecg sequence features 
    if True: 
        print('Segmenting ecg features...')
        ecg_feature_records = [feature_record for feature_record in feature_records
                              if feature_record.split('.')[0].split('_')[-1] == 'ecg']
        for record in ecg_feature_records:
            df= pd.read_csv(path+record) 
            name = record.split('.')[0]
            df_ecg = df[[col for col in df.columns if col[:3]=='ecg']]   
            signal = df_ecg.to_numpy()
             
            bkps=signal_segmentation(signal,
                                     None,
                                     )
            bkps.insert(0, 0)
            filename = '{out_}{name_}.npy'.format(out_=outpath, 
                                                  name_=name)
            np.save(filename, np.array(bkps))             
            #display_segmentation(signal, bkps, config['general']['ecg_partition_length'])
        print('...done \n')
        
        
        
def signal_segmentation(signal, 
                        nb_bkps = None):
 
    if nb_bkps is not None: 
        algo = rpt.KernelCPD(kernel="linear", jump=1).fit(signal)
        my_bkps = algo.predict(n_bkps=nb_bkps)
        
    else:
        pen = np.log(signal.shape[0])/10
        model='l2' # "l1", "rbf"
        algo = rpt.Pelt(model=model, jump=1).fit(signal)
        my_bkps = algo.predict(pen=pen)
    
    return my_bkps
    
 

def display_segmentation(signal, my_bkps, partition_length, name=None):
    
    plt.style.use("seaborn-v0_8")  

    # ------------------------------------------------------------------
    # 1) Raw signal with breakpoints
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    for x in my_bkps[:-1]:
        plt.axvline(x=x - 1, color='red', linewidth=5, linestyle='dashed')
    if name is not None:
        plt.title(name)
    plt.xlabel("Time (minutes)")
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{(x * partition_length) / 60:.1f}")
    )
    plt.show()
    plt.clf()

    # ------------------------------------------------------------------
    # 2) Heatmap without segmentation
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.imshow(signal.T, aspect=4, cmap='viridis', vmin=0, vmax=1)
    ax.grid(False)

    ax.set_xlabel("Time (minutes)", fontsize=15)
    ax.set_ylabel("Features", fontsize=15)

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{(x * partition_length) / 60:.1f}")
    )

    plt.yticks([])
    plt.tight_layout()
    plt.show()
    plt.clf()

    # ------------------------------------------------------------------
    # 3) Heatmap with segmentation
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.imshow(signal.T, aspect=4, cmap='viridis', vmin=0, vmax=1)
    ax.grid(False)

    for x in my_bkps[:-1]:
        ax.axvline(x=x - 0.5, color='red', linewidth=3, linestyle='dashed')

    ax.set_xlabel("Time (minutes)", fontsize=22)
    ax.set_ylabel("Features", fontsize=22)

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{(x * partition_length) / 60:.1f}")
    )

    plt.yticks([])
    plt.xticks(fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.clf()
    
            
            
            
            
            
            