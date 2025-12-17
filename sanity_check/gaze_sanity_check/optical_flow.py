#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 22:06:52 2024

@author: quentinlaborde
"""

import cv2  
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


def lucas_kanade_method(video_path):
    # Read the video 
    cap = cv2.VideoCapture(video_path)
 
    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
 
    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.00001),
    )
  
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
  
   
    grid_2D = np.mgrid[0:old_gray.shape[0]-1:20, 0:old_gray.shape[1]-1:20]
    p0 = np.expand_dims(grid_2D.reshape(2, -1).T, axis=1).astype(dtype = np.float32)
   
   
    
  
    #plt.imshow(old_gray)
    #plt.show()
    #plt.clf()
 
    points = []
    frames_gray = []
    i =0
    while True:
        # Read new frame
        ret, frame = cap.read()
        if not ret:
            break
        print(i)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_gray.append(frame_gray)
     
        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
       
        
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        vect_ = np.mean(good_new - good_old, axis=0) * np.array([1, -1,])
        points.append(vect_)
    
        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        #p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        i+=1 
       
    
    points = np.array(points)
    frames_gray = np.array(frames_gray)
    
   
    plt.plot(points[:,0], points[:,1])
    plt.show()
    plt.clf()
    
    r_aspect = frames_gray[0].shape[1]/frames_gray[0].shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2) 
    
    def animate(i):
        ax2.clear()
        ax2.set_aspect('equal')
        ax2.set_axis_off()
        ax2.set(xlim=[-50, 50], ylim=[-50, 50])
        # Get the point from the points list at index i
        point = points[i]
        # Plot that point using the x and y coordinates
        ax2.scatter(-point[0], -point[1], s=50) 
        
        ax1.clear()
        ax1.imshow(frames_gray[i], aspect=r_aspect) 
        ax1.set_axis_off()
        
    ani = FuncAnimation(fig=fig, func=animate, frames=280, interval=100)
    ani.save(filename="opticalflow_example.mp4", writer="ffmpeg")


    
if __name__ == '__main__':
    
    lucas_kanade_method('data/test_worldCamera.mp4')