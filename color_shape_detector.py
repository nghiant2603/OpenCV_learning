### Function : detect the shape of an specific color
### Option : 
###     -i/--image : input image
import cv2
import numpy as np
import argparse
import imutils
from display import *

def color_shape_detector (image): 
    o_img = image.copy()

    alpha = 1.3         # constrast : 1.0 -> 3.0
    beta = 0            # brightness : 0 -> 100
    aj_img = cv2.convertScaleAbs(image, alpha = alpha, beta = beta)
    blur_img = cv2.GaussianBlur(aj_img, (5, 5), 0)

    #lo_color_range = np.array([150, 0, 0])    #BGR, detect blue shape
    #hi_color_range = np.array([255, 150, 150])     
    lo_color_range = np.array([0, 150, 150])    #BGR, detect yellow shape
    hi_color_range = np.array([150, 255, 255])     

    mask = cv2.inRange(image, lo_color_range, hi_color_range)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        cv2.drawContours(o_img, [c], -1, (0,255,0), 2)

    return o_img

def run (image):
    frame = cv2.imread(image)
    o_frame = color_shape_detector(frame)
    display([[frame, o_frame]])

if __name__ == "__main__":
    #create input option
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='path to image')
    args = vars(ap.parse_args())

    run(args['image'])