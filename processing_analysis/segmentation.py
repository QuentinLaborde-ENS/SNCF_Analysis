# -*- coding: utf-8 -*-

import numpy as np
import ruptures as rpt 
import pandas as pd
import matplotlib.pyplot as plt 




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
            
            
        print('...done \n')
        
    ## For scanpath features only
    if False:
        print('Segmenting scanpath features...')
        scanpath_feature_records = [feature_record for feature_record in feature_records
                              if feature_record.split('.')[0].split('_')[-1] == 'scanpath']
        process_scanpath(config, path, scanpath_feature_records)
        print('...done \n')
        
    ## For aoi sequence features 
    if False: 
        print('Segmenting aoi features...')
        aoi_feature_records = [feature_record for feature_record in feature_records
                              if feature_record.split('.')[0].split('_')[-1] == 'AoI']
        process_aoi(config, path, aoi_feature_records)
        print('...done \n')
        
    ## For eda sequence features 
    if False: 
        print('Segmenting eda features...')
        eda_feature_records = [feature_record for feature_record in feature_records
                              if feature_record.split('.')[0].split('_')[-1] == 'eda']
        process_eda(config, path, eda_feature_records)
        print('...done \n')
   
    ## For ecg sequence features 
    if False: 
        print('Segmenting ecg features...')
        ecg_feature_records = [feature_record for feature_record in feature_records
                              if feature_record.split('.')[0].split('_')[-1] == 'ecg']
        process_ecg(config, path, ecg_feature_records)
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
    
 
def display_segmentation(signal, my_bkps, name=None,
                         ):
 
    plt.style.use("seaborn-v0_8")  
  
    plt.plot(signal)
    for x in my_bkps[:-1]:
        plt.axvline(x = x-1, color = 'red', 
                    linewidth=5, linestyle='dashed')
    if name is not None:
        plt.title(name)
    plt.show()
    plt.clf()
  
    fig = plt.figure()
    ax = fig.add_subplot(111)  
    ax.imshow(signal.T, aspect=4, cmap='viridis', vmin=0, vmax=1)
    ax.grid(None)
     
    ax.set_xlabel("Time windows", fontsize = 15)
    ax.set_ylabel("Features", fontsize = 15)
    
    plt.yticks([])
    #plt.savefig("output/GazeBase/figures/segmentation/{name}.png".format(name=name), dpi=150)
    
    plt.show()
    plt.clf()
    
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    
    
    ax.imshow(signal.T, aspect=4, cmap='viridis', vmin=0, vmax=1)
    ax.grid(None)
    
    for x in my_bkps[:-1]:
        ax.axvline(x = x-.5, color = 'red', 
                    linewidth=3, linestyle='dashed') 
        
    ax.set_xlabel("Time windows", fontsize = 22)
    ax.set_ylabel("Features", fontsize = 22)
    
    #plt.yticks(np.arange(7), ['sacFreq', 'sacAmp',   'sacEfficiency', 'sacPeakVel' , 'sacPeakAcc', 'sacSkewnessExponent', 'sacPeakVelAmpRatio' ])
    plt.yticks([])
    plt.xticks(fontsize=16)
    plt.tight_layout()
    #plt.savefig("output/GazeBase/figures/segmentation/{name}_segmented.png".format(name=name), dpi=150)
    plt.show()
    plt.clf()
    
       
            
            
            
            
            
            
            