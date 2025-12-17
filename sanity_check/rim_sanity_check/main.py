# -*- coding: utf-8 -*-
  
import os 
import glob 
import re

import pandas as pd
import numpy as np 

import cv2 
import pickle 

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from matplotlib import rcParams

import itertools 
from itertools import groupby
from operator import itemgetter
 

np.random.seed(1)
 
         
def test_séquence(mappedGaze, worldCamera, reference_image, frames_to_compute):
 
    def click_event(event, x, y, flags, params): 
      
        ## Checking for left mouse clicks 
        if event == cv2.EVENT_LBUTTONDOWN:   
            x_scaled = int(x*x_combined_scale)
            y_scaled = int(y*y_combined_scale)
            
            coords.append([x_scaled, 
                           y_scaled])
            ## Displaying the coordinates on the image window 
            font = cv2.FONT_HERSHEY_SIMPLEX 
            
            cv2.putText(combined_frame, '_' + str(x_scaled) + ', ' +
                        str(y_scaled), (x,y), font, 
                        1, (0, 0, 255), 2) 
            cv2.imshow('image', combined_frame) 
    
    df = pd.read_csv(mappedGaze, low_memory=False)  
    cap = cv2.VideoCapture(worldCamera)

    ## Load the reference image 
    reference_image = cv2.imread(reference_image)
    ref_image_height, ref_image_width, _ = reference_image.shape
 
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    max_world_x, max_world_y = 600, 450 
    
    x_world_scale = frame_width / max_world_x  
    y_world_scale = frame_height / max_world_y  

    ## Scale for the reference image  
    x_ref_scale = ref_image_width / max_world_x  
    y_ref_scale = ref_image_height / max_world_y  
    
    new_ref_width = int(ref_image_width * (frame_height / ref_image_height))
    resized_ref_image = cv2.resize(reference_image, (new_ref_width, frame_height))
 
    combined_height = frame_height
    combined_width = frame_width+new_ref_width
 
    x_combined_scale = combined_width/(1500) * (max_world_x / new_ref_width)
    y_combined_scale = combined_height/(600) * (max_world_y / frame_height)
 
    output_dict = dict()
    cv2.destroyAllWindows() 
    
    current_frame = 0 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
       
        if current_frame in frames_to_compute:
               
            print(f"Processing frame {current_frame}/{total_frames}") 
            gaze_points = df[df['worldFrame'] == current_frame]
            ref_frame = resized_ref_image.copy() 
            coords = [] 
            ## Scale the gaze coordinates to the video frame size
            x_world = int(gaze_points['world_gazeX'].values[0] * x_world_scale)
            y_world = int(gaze_points['world_gazeY'].values[0] * y_world_scale) 
            cv2.circle(frame, (x_world, y_world), 20, (0, 0, 255), -1) 
            ## Scale the gaze coordinates to the resized reference image size
            x_ref = int(gaze_points['ref_gazeX'].values[0]* x_ref_scale * (frame_height / ref_image_height))
            y_ref = int(gaze_points['ref_gazeY'].values[0]* y_ref_scale * (frame_height / ref_image_height))
            
            combined_frame = cv2.resize(np.hstack((ref_frame, frame)), (1500, 600))  
            cv2.imshow('image', combined_frame)  
            cv2.setMouseCallback('image', click_event)  
            cv2.waitKey(0)  
            output_dict.update({current_frame: dict({'observer': coords[-1],
                                                     'RIM': [int(gaze_points['ref_gazeX'].values[0]), 
                                                             int(gaze_points['ref_gazeY'].values[0])]
                                }) }) 
            cv2.circle(ref_frame, (x_ref, y_ref), 8, (0, 0, 255), -1)
            combined_frame = cv2.resize(np.hstack((ref_frame, frame)), (1500, 600))
            cv2.imshow('image', combined_frame) 
            cv2.waitKey(2000)  
            
            print(f"True values: {int(gaze_points['ref_gazeX'].values[0])}, {int(gaze_points['ref_gazeY'].values[0])}")
            print(f"Inferred values: {coords[-1][0]}, {coords[-1][1]}")
           
        current_frame += 1
    
    print(output_dict)
    return output_dict
     
 
def choose_frames(mappedGaze, match_threshold, frame_number):
    
    df = pd.read_csv(mappedGaze, low_memory=False)  
    tot_l = len(df)
    
    retained = df[df['number_matches'] >= match_threshold] 
    ret_l = len(retained)
    
    prop = (ret_l/tot_l)*100
    print(f"Valid proportion: {prop}%")
     
    retained_frames = np.unique(retained['worldFrame'].values) 
    frames_to_compute = np.sort(np.random.choice(retained_frames, frame_number, 
                                          replace=False)) 
    return frames_to_compute

    
def sep_issue(csv_file):
 
    with open(csv_file, 'r') as f:
        my_csv_text = f.read()
    ## Replace ',,' by ','
    new_csv_str = re.sub(',,', ',', my_csv_text)
    ## Save it 
    with open(csv_file, 'w') as f:
        f.write(new_csv_str)
  
        
def annotate(to_compute):
     
    for record in to_compute: 
        print(f"\nProcessing {record}...")
        path = f"input/{record}/"
        mappedGaze = glob.glob(os.path.join(path, 'mappedGaze*'))[0]
        worldCamera = glob.glob(os.path.join(path, '*.mp4'))[0]
        reference_image = glob.glob(os.path.join(path, '*.jpg'))[0]
     
        match_threshold = 50
        frame_number = 40
        
        sep_issue(mappedGaze)
        #frames_to_compute = choose_frames(mappedGaze, match_threshold, 
        #                                  frame_number)
        
        ## Be sure of using same frames as Deyo
        pkl_deyo = f"output/deyo/{record}.pkl" 
        with open(pkl_deyo, 'rb') as handle:
            d_r = pickle.load(handle) 
      
        frames_to_compute = sorted(list(d_r.keys()))
        #frames_to_compute = [28968] [5822, 12638, 14954, 28968, 31546, 35636, 48136]
        print(frames_to_compute)
        
        output_dict = test_séquence(mappedGaze, worldCamera, 
                                    reference_image, frames_to_compute) 
        #with open(f"output/quentin/{record}.pkl", 'wb') as f:
        #    pickle.dump(output_dict, f)
        
        
def parse_annotation(to_compute):
    
    max_world_x, max_world_y = 600, 450 
    min_world_x, min_world_y = 0, 0
   
    dist_diag = np.linalg.norm(np.array([max_world_x-min_world_x, 
                                         max_world_y-min_world_y]))
    dists_= []
   
    for record in to_compute:
        print(f"\nProcessing {record}...") 
        pkl_deyo = f"rim_sanity_check/output/deyo/{record}.pkl" 
        pkl_quentin = f"rim_sanity_check/output/quentin/{record}.pkl" 
        with open(pkl_deyo, 'rb') as handle:
            record_deyo = pickle.load(handle) 
        with open(pkl_quentin, 'rb') as handle:
            record_quentin = pickle.load(handle) 
          
        dists_l = []
        try:
            assert record_quentin.keys() == record_deyo.keys() 
            keys = sorted(list(record_quentin.keys()))
           
            for k_ in keys: 
                rim_ = np.array(record_quentin[k_]['RIM']) 
                if rim_[0]>min_world_x and rim_[0]<max_world_x:
                    if rim_[1]>min_world_y and rim_[1]<max_world_y:
                        d_ = np.array(record_deyo[k_]['observer'])
                        d_d = np.linalg.norm(d_-rim_)
                        
                        q_ = np.array(record_quentin[k_]['observer'])
                        q_d = np.linalg.norm(q_-rim_)
                     
                        dists_l.append(np.mean([d_d, q_d]))
        except:
            print('Different frames computed: alternative method')
            keys = sorted(list(record_quentin.keys()))
            for k_ in keys: 
                rim_ = np.array(record_quentin[k_]['RIM'])
                if rim_[0]>min_world_x and rim_[0]<max_world_x:
                    if rim_[1]>min_world_y and rim_[1]<max_world_y:
                        q_ = np.array(record_quentin[k_]['observer'])
                        q_d = np.linalg.norm(q_-rim_)
                        dists_l.append(q_d)
        dists_+= dists_l
     
    dists_ = np.array(dists_)/dist_diag 
    dists_ = dists_[dists_<.06]
    dists_df = pd.DataFrame({"RIM normlized distance to annotation": dists_  })
    
    m_dists_ = np.mean(dists_)
    sd_dists_ = np.std(dists_)
    print(f"RIM error: {m_dists_} ({sd_dists_})")
    
    sns.set_theme(style='whitegrid', font_scale = 2.2)
    
    fig = plt.figure(figsize=(16, 6))
    sns.histplot(
    x=dists_df["RIM normlized distance to annotation"], 
    bins=30,  
    color='darkblue',
    stat='proportion' ) 
    rcParams.update({'figure.autolayout': True})
    plt.show()
    fig.savefig('rim_sanity_check/figures/annotation_distance_distribution.png', dpi=150)
    plt.clf()
        
   
def parse_rim(to_compute):
    
    match_threshold = 50
    matches = []
    valid_props_ = []
    for record in to_compute:
        name = record.split('/')[1].split('.')[0]
      
        with open(record, 'rb') as handle:
            pkl = pickle.load(handle) 
        mappedGaze = pkl['mapped_gaze']
       
        matches_l = list(mappedGaze['number_point_matches'].values)
        matches += matches_l
        prop = np.sum(np.array(matches)>=match_threshold)/len(matches)
        valid_props_.append(prop)
        
    m_valid_props_ = np.mean(valid_props_)
    sd_valid_props_ = np.std(valid_props_)
    print(f"RIM proportion: {m_valid_props_} ({sd_valid_props_})")
    
    matches = np.array(matches)
    retained = matches >= match_threshold
    wi_f = np.where(retained == False)[0]
    intervals, lengths = interval_merging(wi_f)
    i_length = np.mean(lengths)/200
    i_length_sd = np.std(np.array(lengths)/200)
    print(f"Invalid lengths: {i_length}, sd: {i_length_sd}")
        
    invalid_lengths_ = np.array(lengths)/200
    invalid_lengths_ = invalid_lengths_[invalid_lengths_<2]
    
    invalid_lengths_df = pd.DataFrame({"RIM missing interval length (s)": invalid_lengths_  })
    sns.set_theme(style='whitegrid', font_scale = 2.2)
    fig = plt.figure(figsize=(16, 6))
    sns.histplot(
    x=invalid_lengths_df['RIM missing interval length (s)'], 
    bins=30,  
    color='darkblue',
    stat='proportion' ) 
    rcParams.update({'figure.autolayout': True})
    plt.show()
    fig.savefig('rim_sanity_check/figures/rim_length_distribution.png', dpi=100)
    plt.clf()
    
    matches_ = np.array(matches)  
    m_matches = np.mean(matches_)
    sd_matches = np.std(matches_)
    print(f"RIM matches: {m_matches} ({sd_matches})")
    
    matches_df = pd.DataFrame({"RIM matching keyponts": matches_  })
    sns.set_theme(style='whitegrid', font_scale = 2.2)
    fig = plt.figure(figsize=(16, 6))
    sns.histplot(
    x=matches_df['RIM matching keyponts'], 
    bins=30,  
    color='darkblue',
    stat='proportion' ) 
    plt.axvline(x=50, color='red',linewidth= 2.)
    rcParams.update({'figure.autolayout': True})
    plt.show()
    fig.savefig('rim_sanity_check/figures/rim_match_distribution.png', dpi=100)
    plt.clf()
        
   
    
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
        
        
    
    
if __name__=="__main__":  
    
    to_compute = [
        ## Paris-Hendaye
        '2024-02-19_12-56-12', 
        '2024-02-27_14-05-54', 
        '2024-04-16_14-13-11', 
        
        ## Paris-Brest
        '2023-11-06_07-27-43', 
        '2023-11-06_14-29-09', 
        '2023-11-27_13-22-16', 
        '2023-12-08_13-19-41', 
        '2023-12-20_09-29-56', 
          
        ## Ligne H
        '2023-11-28_10-35-41', 
        '2023-11-28_11-39-17', 
        '2023-11-28_12-33-50', 
        '2023-11-28_13-25-53', 
        '2023-12-06_09-38-25', 
        '2023-12-06_10-32-12', 
        '2023-12-06_12-33-49',  
        '2023-12-13_10-04-43', 
        '2023-12-13_11-06-24', 
        '2023-12-13_12-05-46', 
        '2023-12-13_12-54-52', 
        '2024-04-22_11-26-13', 
        '2024-04-22_12-45-58', 
        '2024-04-22_14-25-13', 
        '2024-04-22_15-45-23' 
                  ] 
    
     
    if False:
        annotate(to_compute)
    if False: 
        parse_annotation(to_compute)
    if True:
        folder = "parsed_data"
        to_compute = glob.glob(os.path.join(folder, "*.pkl"))
        parse_rim(to_compute)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 