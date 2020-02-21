# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:03:40 2020

@author: Joao
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import math

import auxiliar as aux
from ipywidgets import widgets, interact, interactive, FloatSlider, IntSlider

# If you want to open a video, just change this path
#cap = cv2.VideoCapture('hall_box_battery.mp4')

# Parameters to use when opening the webcam.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)






while(True):
    # Capture frame-by-frame
    #print("New frame")
    ret, frame = cap.read()

    frame_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cor_menor = np.array([172, 50, 50])
    cor_maior = np.array([180, 255, 255])
    cor_menor2 = np.array([80, 50, 50])
    cor_maior2 = np.array([110, 255, 255])
    mask_frame = cv2.inRange(frame_hsv, cor_menor, cor_maior)
    mask_frame2 = cv2.inRange(frame_hsv, cor_menor2, cor_maior2)
    segmentado_frame = cv2.morphologyEx(mask_frame,cv2.MORPH_CLOSE,np.ones((4, 4)))
    segmentado_frame2 = cv2.morphologyEx(mask_frame2,cv2.MORPH_CLOSE,np.ones((4, 4)))
    contornos, arvore = cv2.findContours(segmentado_frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contornos2, arvore2 = cv2.findContours(segmentado_frame2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contornos_img = frame_rgb.copy()
    contornos_img2 = frame_rgb.copy()
    img1 = cv2.drawContours(contornos_img, contornos, -1, [0, 0, 255], 3);
    img2 = cv2.drawContours(contornos_img2, contornos2, -1, [0, 0, 255], 3);
    foreground_img = cv2.bitwise_or(img1, img2)



    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    # Display the resulting frame
    cv2.imshow('Detector de circulos',foreground_img)
    #print("No circles were found")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture

cap.release()
cv2.destroyAllWindows()