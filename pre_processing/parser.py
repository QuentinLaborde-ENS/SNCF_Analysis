#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 13:53:48 2025

@author: quentinlaborde
"""

import os
import yaml 
import pickle

import pandas as pd
import numpy as np
import cv2

from bisect import bisect
from datetime import datetime

from matplotlib import pyplot as plt
import glob  

 
def delta_t_ns_int64(x: str, y: str) -> np.int64:
    
    minutes = int(x)
    secondes = int(y)
    total_secondes = minutes * 60 + secondes
    delta_ns = total_secondes * 1_000_000_000
    
    return np.int64(delta_ns)



class DataParser():
    
    def __init__(self, 
                 record, line, driver, 
                 start_offset = 0, end_offset = 0):
        
        self.record = record
        self.line=line
        self.info = dict({
            'line': line,
            'date': record.split('_')[0], 
            'driver': driver
                          })  
          
        self.start_offset = start_offset
        self.end_offset = end_offset
        
        self.pupil_labs_sf = 200 
        self.equivital_sf = 250
        self.empatica_sf = 4 
        self.xsens_sf = 25
        
        self.size_plan_x = 1600
        self.size_plan_y = 1200
        
        self.match_threshold = 10
        self.output_mapping_x = 600
        self.output_mapping_y = 450
        self.tolerance = 0
        
        self.info.update({'sampling_frequencies': dict({
            'pupil_labs': self.pupil_labs_sf, 
            'equivital': self.equivital_sf,   
            'empatica': self.empatica_sf 
                })})
        #print(self.info)
        self.start_ts = None
        self.end_ts = None
   
    
   
    
    def process(self):
        
        
        path = os.path.join('data', self.record)
      
       
        ## For eye-trDacking
        raw_gaze_file = os.path.join(path,'gaze.csv')
        mapped_gaze_file = glob.glob(os.path.join(path, "mappedGaze*"))[0] 
        blink_file = os.path.join(path, 'blinks.csv')
        ref_im = cv2.imread(os.path.join(path, 'image_ref.jpg'))
         
        ## For cardio
        equivital_file = glob.glob(os.path.join(path, "*ECGmV.csv"))[0]
      
        ## For EDA
        eda_file = os.path.join(path, 'eda.csv')
        temp_file = os.path.join(path, 'tem.csv')
        
        ## For XSens
        # md_file = glob.glob(os.path.join(path, "md_*"))[0]
        # mg_file = glob.glob(os.path.join(path, "mg_*"))[0]
        # pd_file = glob.glob(os.path.join(path, "pd_*"))[0]
        # pg_file = glob.glob(os.path.join(path, "pg_*"))[0]
    
        
        ## Compute gaze df and mapped gaze df
        raw_gaze, gaze, mapped_gaze = self.compute_pupil_labs(raw_gaze_file, 
                                                              mapped_gaze_file, 
                                                              blink_file,
                                                              ref_im) 
        
        
        st_ = mapped_gaze['number_point_matches'].to_numpy()
        stt_ = st_>50
        print(np.sum(stt_)/len(stt_))
        
        plt.figure()
        plt.hist(st_, bins=30)  # tu peux changer le nombre de bins
        plt.xlabel("Valeurs")
        plt.ylabel("Fréquence")
        plt.title("Histogramme des points de mapping")
        plt.grid(True)
        
        plt.show()
        plt.clf()
                
    
        #mapped_gaze.to_csv('mapped_gaze.csv')
        ecg = self.compute_equivital(equivital_file)
        
        dt_ = ecg['ecg_lead_2'][200000:210000]
        plt.figure()
        plt.plot(dt_) 
        plt.grid(True)
        
        plt.show()
        plt.clf()
      
        eda = self.compute_empatica([eda_file, 
                                      temp_file])
        
        dt_ = eda['eda']
        plt.figure()
        plt.plot(dt_) 
        plt.grid(True)
        
        plt.show()
        plt.clf()
     
        # xsens = self.compute_xsens([md_file,
        #                             mg_file,
        #                             pd_file,
        #                             pg_file])
        
        dict_data_out = dict({'info': self.info, 
                              'raw_gaze': raw_gaze,
                              'gaze': gaze, 
                              'mapped_gaze': mapped_gaze, 
                              'reference_image': ref_im,
                              'ecg':ecg, 
                              'eda': eda, 
                              })
        with open(f"parsed_data/{self.record}.pkl", 'wb') as handle:
            pickle.dump(dict_data_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
        
       
        
   
    def compute_pupil_labs(self, 
                           raw_gaze_file, mapped_gaze_file, blink_file,
                           ref_im):
        
        ref_size_y, ref_size_x, _ = ref_im.shape 
        
        ## Compute raw gaze df
        raw_gaze = pd.read_csv(raw_gaze_file)
      
        new_raw_gaze = raw_gaze[[
            'timestamp [ns]', 
            'gaze x [px]', 
            'gaze y [px]'
            ]] 
        
        #new_raw_gaze.loc[:,'timestamp [ns]'] = new_raw_gaze['timestamp [ns]']/1e9
        new_raw_gaze = new_raw_gaze.rename(columns={
            'timestamp [ns]': 'timestamp[ns]', 
            'gaze x [px]': 'gazeX', 
            'gaze y [px]': 'gazeY'
            })
       
        blink_status = self.blink_status(new_raw_gaze['timestamp[ns]'].values, blink_file)
        status = blink_status.copy()
        x_, y_ = new_raw_gaze['gazeX'], new_raw_gaze['gazeY']
        x_ = x_.copy()
        y_ = y_.copy()
        x_[x_>self.size_plan_x] = np.nan
        y_[y_>self.size_plan_y] = np.nan
        
        nan_idx_x = (np.argwhere(np.isnan(x_)).flatten())
        status[nan_idx_x] = 0 
        nan_idx_y = (np.argwhere(np.isnan(y_)).flatten()) 
        status[nan_idx_y] = 0
        new_raw_gaze['status'] =  status
        
        ## Compute mapped gaze df
        mapped_gaze = pd.read_csv(mapped_gaze_file, 
                                  low_memory=False)  
        
        raw_ts = raw_gaze['timestamp [ns]'].values 
        new_mapped_gaze = pd.DataFrame(data=dict({
            'gaze_ts': list(raw_ts), 
            'worldFrame': [np.nan]*len(raw_ts), 
            'world_gazeX': x_,
            'world_gazeY': y_,
            'ref_gazeX': [np.nan]*len(raw_ts), 
            'ref_gazeY': [np.nan]*len(raw_ts), 
            'number_point_matches': [np.nan]*len(raw_ts)
            }))
         
        new_mapped_gaze = (
            new_mapped_gaze.set_index('gaze_ts').combine_first(
                mapped_gaze[mapped_gaze['gaze_ts'].isin(new_mapped_gaze['gaze_ts'])]  
                   .set_index('gaze_ts')[['worldFrame','ref_gazeX','ref_gazeY','number_point_matches']]
            ).reset_index()
        )
       
       
        nb_nan = new_mapped_gaze["number_point_matches"].isna().sum()  
        assert nb_nan < .01*len(new_mapped_gaze), 'RIM not completed, too many NaN'
          
        mapped = np.ones(len(raw_ts))
        x_ = new_mapped_gaze['ref_gazeX'].values
        y_ = new_mapped_gaze['ref_gazeY'].values
        
        ## Mark nan as non mapped
        nan_idx_x = (np.argwhere(np.isnan(x_)).flatten())
        mapped[nan_idx_x] = 0 
        nan_idx_y = (np.argwhere(np.isnan(y_)).flatten()) 
        mapped[nan_idx_y] = 0
     
        ## Mark low number of matches as non mapped
        lim_matches = self.match_threshold
        n_matches = new_mapped_gaze['number_point_matches'].values
        nan_idx_matches = (np.argwhere(np.isnan(n_matches)).flatten())
         
        n_matches[nan_idx_matches] = 0 
        new_mapped_gaze['number_point_matches'] = n_matches 
        non_mapped_idx = np.argwhere(n_matches<lim_matches).flatten()
        mapped[non_mapped_idx] = 0
        
        new_mapped_gaze['mapped'] = mapped 
        new_mapped_gaze['no_blink'] = blink_status
    
        ## Compute out of bounds entry: looking inside the reference image ?
        lim_x = self.output_mapping_x
        lim_y = self.output_mapping_y
        
        oob = np.ones(len(raw_ts)) 
        oob[np.argwhere(x_>(lim_x+self.tolerance)).flatten()]=0
        oob[np.argwhere(x_<(0-self.tolerance)).flatten()]=0
        oob[np.argwhere(y_>(lim_y+self.tolerance)).flatten()]=0
        oob[np.argwhere(y_<(0-self.tolerance)).flatten()]=0
        
        new_mapped_gaze['in_bounds'] = oob
      
        ## Change coordinates according to ref image dimensions
        new_mapped_gaze['ref_gazeX'] /= lim_x
        new_mapped_gaze['ref_gazeX'] *= ref_size_x
        new_mapped_gaze['ref_gazeY'] /= lim_y
        new_mapped_gaze['ref_gazeY'] *= ref_size_y
        
        new_mapped_gaze = new_mapped_gaze.rename(columns={'gaze_ts': 'timestamp[ns]'})
        
        status = new_mapped_gaze['mapped'] * new_mapped_gaze['in_bounds'] * new_mapped_gaze['no_blink']
        new_mapped_gaze['status'] = status
        
        final_mapped_gaze = new_mapped_gaze[['timestamp[ns]',
                                             'worldFrame',
                                             'world_gazeX',
                                             'world_gazeY',
                                             'ref_gazeX',
                                             'ref_gazeY',
                                             'number_point_matches',
                                             'status']]
      
        ## Ressample raw gaze
        s_f = self.pupil_labs_sf 
        g = new_raw_gaze[['timestamp[ns]', 'gazeX', 'gazeY', 'status']].dropna().copy()
        g = g.groupby('timestamp[ns]', as_index=False).last()
      
        old_ts = g['timestamp[ns]'].to_numpy(dtype=np.int64)  
        old_x  = g['gazeX'].to_numpy(dtype=np.float64)
        old_y  = g['gazeY'].to_numpy(dtype=np.float64)
        old_s  = g['status'].to_numpy(dtype=np.int8)
       
        start   = np.int64(old_ts[0])
        stop    = np.int64(old_ts[-1])
        step_ns = np.int64(round(1_000_000_000 / float(s_f)))  
         
        last_aligned = start + ((stop - start) // step_ns) * step_ns  
        new_ts = np.arange(start, last_aligned + step_ns, step_ns, dtype=np.int64)
        
        new_x = np.interp(new_ts, old_ts, old_x)  
        new_y = np.interp(new_ts, old_ts, old_y)
     
        idx = np.searchsorted(old_ts, new_ts, side='left') 
        idx = np.clip(idx, 1, len(old_ts)-1)
        new_status = (old_s[idx-1] & old_s[idx]).astype(np.int8)   
     
        resampled_raw_gaze = pd.DataFrame({
            'timestamp[ns]': new_ts.astype(np.int64), 
            'gazeX': new_x,
            'gazeY': new_y,
            'status': new_status
        })
       
        ## Ressample mapped gaze  
        g = final_mapped_gaze.dropna().copy()
        g = g.sort_values('timestamp[ns]', kind='mergesort')  
        g = g.groupby('timestamp[ns]', as_index=False).last()
    
        old_ts = g['timestamp[ns]'].to_numpy(dtype=np.int64) 
        old_fr = g['worldFrame'].to_numpy(dtype=np.int32)
        old_wx  = g['world_gazeX'].to_numpy(dtype=np.float64)
        old_wy  = g['world_gazeY'].to_numpy(dtype=np.float64)
        old_x  = g['ref_gazeX'].to_numpy(dtype=np.float64)
        old_y  = g['ref_gazeY'].to_numpy(dtype=np.float64)
        old_s  = g['status'].to_numpy(dtype=np.int8)
        old_nb_points = g['number_point_matches'].to_numpy()
        
        new_x = np.interp(new_ts, old_ts, old_x)  
        new_y = np.interp(new_ts, old_ts, old_y)
        new_wx = np.interp(new_ts, old_ts, old_wx)  
        new_wy = np.interp(new_ts, old_ts, old_wy)
     
        idx = np.searchsorted(old_ts, new_ts, side='left') 
        idx = np.clip(idx, 1, len(old_ts)-1)
        new_status = (old_s[idx-1] & old_s[idx]).astype(np.int8)    
        new_world_fr = old_fr[idx] 
        new_nb_points = old_nb_points[idx]
     
        resampled_mapped_gaze = pd.DataFrame({
            'timestamp[ns]': new_ts.astype(np.int64),  
            'worldFrame': new_world_fr,
            'world_gazeX': new_wx,
            'world_gazeY': new_wy,
            'ref_gazeX': new_x,
            'ref_gazeY': new_y,
            'number_point_matches': new_nb_points,
            'status': new_status
        })
        
        self.start_ts = resampled_raw_gaze['timestamp[ns]'].values[0] + self.start_offset
        self.end_ts = resampled_raw_gaze['timestamp[ns]'].values[-1] - self.end_offset
        
        resampled_raw_gaze = resampled_raw_gaze[resampled_raw_gaze['timestamp[ns]']>=self.start_ts]
        resampled_raw_gaze = resampled_raw_gaze[resampled_raw_gaze['timestamp[ns]']<=self.end_ts]
  
        resampled_mapped_gaze = resampled_mapped_gaze[resampled_mapped_gaze['timestamp[ns]']>=self.start_ts]
        resampled_mapped_gaze = resampled_mapped_gaze[resampled_mapped_gaze['timestamp[ns]']<=self.end_ts]
       
        return new_raw_gaze, resampled_raw_gaze, resampled_mapped_gaze
        
        
    def blink_status(self, ts, blink_file):
    
        blink_df = pd.read_csv(blink_file)
         
        nb_blinks = len(blink_df)
        status = np.ones(len(ts))
       
        for i in range(nb_blinks): 
            start = blink_df['start timestamp [ns]'].iloc[i]
            end = blink_df['end timestamp [ns]'].iloc[i]
      
            start_idx = bisect(ts, start) 
            end_idx = bisect(ts, end)   
            ## To the left, since we used bissect
            status[start_idx-1: end_idx+1] = 0
            
        return status 
   
    
    def l_inter(self,
                xa,xb,
                ya,yb,xc):
       
        m = (ya - yb) / (xa - xb)
        yc = (xc - xb) * m + yb
        
        return yc
   
    
   
    def compute_equivital(self, 
                          equivital_file):
   
        raw_ecg = pd.read_csv(
                    equivital_file,
                    usecols=[0, 1, 2, 3],  
                    names=["Date (dd/MM/yyyy)", "Time (HH:mm:ss.000)", "ECG Lead 1", "ECG Lead 2"], 
                    header=0                
                )
     
        raw_ecg["ECG Lead 1"] = raw_ecg["ECG Lead 1"].astype(np.float32)
        raw_ecg["ECG Lead 2"] = raw_ecg["ECG Lead 2"].astype(np.float32)
          
        date_str = raw_ecg['Date (dd/MM/yyyy)'].iloc[0]
        time_col = raw_ecg['Time (HH:mm:ss.000)']  # ex: "12:34:56.000"
      
        base_day = pd.to_datetime(date_str, dayfirst=True).normalize() 
        dt_series = base_day + pd.to_timedelta(time_col.astype(str)) 
        equivital_dt = dt_series.dt.tz_localize('Europe/Paris').dt.tz_convert('UTC') 
        timestamps = equivital_dt.astype('int64').to_numpy()  
      
        assert timestamps[0] == datetime.strptime('{date_} {time_}'.format(date_ = date_str,
                                                                           time_ = time_col.values[0]), 
                                                  '%d/%m/%Y %H:%M:%S.%f').timestamp()*1e9, "Verify time zone"
        
        ecg_1, ecg_2 = raw_ecg['ECG Lead 1'].values, raw_ecg['ECG Lead 2'].values 
         
        start   = np.int64(timestamps[0])
        stop    = np.int64(timestamps[-1])
        s_f = self.equivital_sf
        step_ns = np.int64(round(1_000_000_000 / float(s_f)))  
        
        last_aligned = start + ((stop - start) // step_ns) * step_ns 
        
        new_ts = np.arange(start, last_aligned + step_ns, step_ns, dtype=np.int64)
        new_ecg_1 = np.interp(new_ts, timestamps, ecg_1)  
        new_ecg_2 = np.interp(new_ts, timestamps, ecg_2)
        
        if self.end_ts is not None and self.start_ts is not None:
            assert new_ts[0] < self.end_ts and new_ts[-1] > self.start_ts, 'Equivital recording is not compatible with eye-tracking recording'
            start_idx = bisect(new_ts, self.start_ts)  
            end_idx = bisect(new_ts, self.end_ts)   
            new_ecg = pd.DataFrame(data=dict({
                'timestamp[ns]': new_ts[start_idx: end_idx], 
                'ecg_lead_1': new_ecg_1[start_idx: end_idx],
                'ecg_lead_2': new_ecg_2[start_idx: end_idx]
                }))
        else:
            new_ecg = pd.DataFrame(data=dict({
                'timestamp[ns]': new_ts, 
                'ecg_lead_1': new_ecg_1,
                'ecg_lead_2': new_ecg_2,
                }))
       
        return new_ecg
    
    
    def compute_empatica(self, 
                         empatica_files):
        
        raw_eda = pd.read_csv(empatica_files[0])
        raw_temp = pd.read_csv(empatica_files[1])
        
        timestamps = raw_eda['unix_timestamp'].to_numpy(dtype=np.int64) * np.int64(1000)
        temp_ts    = raw_temp['unix_timestamp'].to_numpy(dtype=np.int64) * np.int64(1000)
        temp_ = raw_temp['temperature'].values  
        eda_ = raw_eda['eda'].values
        
        app_temp = np.zeros_like(eda_)
        for i in range(len(app_temp)-1):
            idx = bisect(temp_ts, timestamps[i])
            try:
                app_temp[i] = temp_[idx]
            except:
                app_temp[i] = temp_[idx-1]
        app_temp[-1] = temp_[-1]
        
        start   = np.int64(timestamps[0]) 
        stop    = np.int64(timestamps[-1])
        s_f = self.empatica_sf
        step_ns = np.int64(round(1_000_000_000 / float(s_f)))  
 
        last_aligned = start + ((stop - start) // step_ns) * step_ns  
        new_ts = np.arange(start, last_aligned + step_ns, step_ns, dtype=np.int64)
        new_eda = np.interp(new_ts, timestamps, eda_)  
        new_temp = np.interp(new_ts, timestamps, app_temp)
         
        if self.end_ts is not None and self.start_ts is not None: 
            assert new_ts[0] < self.end_ts and new_ts[-1] > self.start_ts, 'Empatica recording is not compatible with eye-tracking recording' 
            start_idx = bisect(new_ts, self.start_ts)  
            end_idx = bisect(new_ts, self.end_ts)  
            new_eda = pd.DataFrame(data=dict({
                'timestamp[ns]': new_ts[start_idx: end_idx],
                'eda': new_eda[start_idx: end_idx],
                'temperature': new_temp[start_idx: end_idx],
                }))
        else:
            new_eda = pd.DataFrame(data=dict({
                'timestamp[ns]': new_ts,
                'eda': new_eda,
                'temperature': new_temp,
                })) 
     
       
        return new_eda
        

    def compute_xsens(self,
                      xsens_files):
        
        m_ = dict({
            'md': 'right_hand',
            'mg': 'left_hand',
            'pd': 'right_foot',
            'pg': 'left_foot'
            })
        
        dict_data = dict({})
        for xsens_file in xsens_files: 
            try:
                file = xsens_file.split('/')[-1]
                
                raw_accelero = pd.read_csv(xsens_file, header=1, low_memory=False)
                
                ts = raw_accelero['SampleTimeFine'].to_numpy(dtype=np.int64) * np.int64(1000)
                ts -= ts[0]
             
                date = file.split('.')[0][-15:-7]       
                time_str = file.split('.')[0][-6:]      
                
                xsens_dt = datetime(
                    year=int(date[:4]),
                    month=int(date[4:6]),
                    day=int(date[6:]),
                    hour=int(time_str[:2]),
                    minute=int(time_str[2:4]),
                    second=int(time_str[4:])
                )
                 
                xsens_time_ns = int(xsens_dt.timestamp() * 1e9) 
                timestamps = ts + xsens_time_ns
           
                start   = np.int64(timestamps[0]) 
                stop    = np.int64(timestamps[-1])
                s_f = self.xsens_sf
                step_ns = np.int64(round(1_000_000_000 / float(s_f)))  
         
                last_aligned = start + ((stop - start) // step_ns) * step_ns  
                new_ts = np.arange(start, last_aligned + step_ns, step_ns, dtype=np.int64)
                
                new_eulerX = np.interp(new_ts, timestamps, raw_accelero['Euler_X'].values)  
                new_eulerY = np.interp(new_ts, timestamps, raw_accelero['Euler_Y'].values)  
                new_eulerZ = np.interp(new_ts, timestamps, raw_accelero['Euler_Z'].values)  
                new_accX = np.interp(new_ts, timestamps, raw_accelero['Acc_X'].values)
                new_accY = np.interp(new_ts, timestamps, raw_accelero['Acc_Y'].values)
                new_accZ = np.interp(new_ts, timestamps, raw_accelero['Acc_Z'].values)
                new_gyrX = np.interp(new_ts, timestamps, raw_accelero['Gyr_X'].values)  
                new_gyrY = np.interp(new_ts, timestamps, raw_accelero['Gyr_Y'].values)  
                new_gyrZ = np.interp(new_ts, timestamps, raw_accelero['Gyr_Z'].values)  
                
                start_idx = bisect(timestamps, self.start_ts)  
                end_idx = bisect(timestamps, self.end_ts) 
                print(timestamps)
                new_acc = pd.DataFrame(data=dict({
                    'timestamp[ns]': timestamps[start_idx: end_idx],
                    'eulerX': new_eulerX[start_idx: end_idx],
                    'eulerY': new_eulerY[start_idx: end_idx],
                    'eulerZ': new_eulerZ[start_idx: end_idx],
                    'accX': new_accX[start_idx: end_idx],
                    'accY': new_accY[start_idx: end_idx],
                    'accZ': new_accZ[start_idx: end_idx],
                    'gyrX': new_gyrX[start_idx: end_idx],
                    'gyrY': new_gyrY[start_idx: end_idx],
                    'gyrZ': new_gyrZ[start_idx: end_idx],
                    }))
                print(new_acc)
                dict_data.update({m_[file[:2]]: new_acc})
            except:
                pass
  
        return dict_data
    
 
if __name__=="__main__":  
    
    to_compute = [
        name for name in os.listdir('data')
        if os.path.isdir(os.path.join('data', name))
    ]
    
   
    
    for record in to_compute:
      #if record=='2024-06-27-16-25-10':
          
          print(f"\nComputing record {record}...")
          
          with open(f'data/{record}/info.yaml', 'r') as file:
              info = yaml.safe_load(file)
              
          line = info['railway_line']
          driver = info['driver']
          
          start_offset_m, start_offset_s = info['start_offset'].split(':')
          end_offset_m, end_offset_s = info['end_offset'].split(':')
          
          start_offset = delta_t_ns_int64(start_offset_m, start_offset_s)
          end_offset = delta_t_ns_int64(end_offset_m, end_offset_s)
          
          parser_ = DataParser(record, line, driver,
                               start_offset, end_offset)
          parser_.process()
          
           
          
          
      
              
   
    
   
   