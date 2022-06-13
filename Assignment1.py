# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:38:10 2022

@author: Dell
"""

import cv2
import numpy as np


img = cv2.imread('9.jpg')

#original size
h = img.shape[0]
w = img.shape[1]

#new size
new_h = int(h/2)
new_w = int(w/2)

#resize image
resized_image = cv2.resize(img,(new_w,new_h))
cv2.imshow('img',resized_image)


# convert the image from rgb to lab 
lab =  cv2.cvtColor(resized_image ,cv2.COLOR_BGR2LAB)
#normalize it to make it from(0:255) to (-127:127)
norm_image = cv2.normalize(lab, None, alpha = -127, beta = 127, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_64F)

# separate the lab image to it's three components L A B
L,A,B=cv2.split(norm_image)

rows = lab.shape[0]  #540
cols = lab.shape[1]  #960
threshold = 0  # This is our threshold here (above is +ve (yellow)) (below is -ve (blue))

# Now we ar gonna loop over the image and check if there is any blue-yellow edge
# if there is any edge we will whiten it so we make sure there is no direct contact between
# blue and yellow colors

for i in range(1,rows-1):
    for j in range(1,cols-1):
        
        if B[i][j-1] >= threshold and B[i][j+1]<threshold: #Horizontal edge detection
            lab[i][j] = np.array([255,127,127])   # make it white pixel
            # Here the white is [255,127,127] not [127,0,0] because we work here in lab 
            # which is the originalimage not the normalized one
            
        elif B[i][j-1] < threshold and B[i][j+1]>=threshold: #Horizontal edge detection
            lab[i][j] = np.array([255,127,127])
            
        elif B[i-1][j] >= threshold and B[i+1][j]<threshold: #Vertical edge detection
            lab[i][j] = np.array([255,127,127])
        
        elif B[i-1][j] >= threshold and B[i+1][j]<threshold: #Vertical edge detection
            lab[i][j] = np.array([255,127,127])
            
        elif B[i-1][j-1] >= threshold and B[i+1][j+1]<threshold: #Main Diagonal
            lab[i][j] = np.array([255,127,127])
            
        elif B[i-1][j-1] < threshold and B[i+1][j+1]>=threshold: #Main Diagonal
            lab[i][j] = np.array([255,127,127])
            
        elif B[i+1][j-1] >= threshold and B[i-1][j+1]<threshold: #Reverse Diagonal
            lab[i][j] = np.array([255,127,127])
            
        elif B[i+1][j-1] < threshold and B[i-1][j+1]>=threshold: #Reverse Diagonal
            lab[i][j] = np.array([255,127,127])
            
        else:
            lab[i][j] = lab[i][j]
            
final_image = cv2.cvtColor(lab ,cv2.COLOR_LAB2BGR)
cv2.imshow('finalimg',final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()