### Function : read and grade the bubble sheet scanner
### Option : 
###     -i/--image : path of input image

import numpy as np
import argparse
import cv2
import imutils
from display import *
import math

def blur_detection (frame, thresh=1000): 
    shape = frame.shape
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_value = 0.01*math.sqrt(shape[0]*shape[1])*cv2.Laplacian(gray_img, cv2.CV_64F).var()   
    text = "Not Blurry"
    if lap_value < thresh : 
        text = "Blurry"
    return (text, lap_value)

def run (image, thresh):
    frame = cv2.imread(image)
    (text, lap_value) = blur_detection(frame, thresh)
    cv2.putText(frame, "{}: {:.2f}".format(text, lap_value), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    display([[frame]])

if __name__ == "__main__":
    #create input option
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='path to image')
    ap.add_argument('-t', '--thresh', type=float, default=1000.0, help='the threshold')
    args = vars(ap.parse_args())

    run(args['image'], args['thresh'])
