# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:55:02 2020

@author: Joao
"""

import cv2
import numpy as np
import math


# If you want to open a video, just change this path

# Parameters to use when opening the webcam.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1
# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged



while(True):
    # Capture frame-by-frame
    #print("New frame")
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    # Detect the edges present in the image
    bordas = auto_canny(blur)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    circles = []


    # Obtains a version of the edges image where we can draw in color
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)

    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    circles = None
    circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=60)
    centro= []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(bordas_color,(i[0],i[1]),i[2],(int(frame[i[1],i[0]][0]),int(frame[i[1],i[0]][1]),int(frame[i[1],i[0]][2])),7)
            # draw the center of the circle
            cv2.circle(bordas_color,(i[0],i[1]),2,(int(frame[i[1],i[0]][0]),int(frame[i[1],i[0]][1]),int(frame[i[1],i[0]][2])),3)
            centro.append((i[0],i[1]))
        # draws only if 2 or more circles detected
        if len(centro)>= 2:
            cv2.line(bordas_color, centro[0], centro[1], (255,255,255), 2)
            # distance calculated using pitagoras theorem
            length = np.sqrt(int((centro[0][0])- int(centro[1][0]))**2 + (int(centro[0][1])- int(centro[1][1])) ** 2)
            # transforms pixels in centimeters
            dist = 6650/length
            dy = abs(int(centro[0][0])- int(centro[1][0]))
            dx = abs(int(centro[0][1])- int(centro[1][1]))
            # avoids tg(0)
            if dy != 0:
                # calculates inclination angle
                angle = np.arctan(dx/dy)
                angle_degree = angle * (180/math.pi)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(bordas_color,str(dist),(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(bordas_color,str(angle_degree),(0,100), font, 1,(255,255,255),2,cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Detector de circulos',bordas_color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture

cap.release()
cv2.destroyAllWindows()