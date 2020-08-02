### Function : detect contour and center of object
### Option : 
###     -i/--image : input image
import cv2
import numpy as np
import argparse
import imutils
from display import *


def shape_center_detector (image): 

    alpha = 1.3         # constrast : 1.0 -> 3.0
    beta = 0            # brightness : 0 -> 100
    aj_img = cv2.convertScaleAbs(image, alpha = alpha, beta = beta)
    gray_img = cv2.cvtColor(aj_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    thresh_img = cv2.threshold(blur_img, 100, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        M = cv2.moments(c)
        if (M["m00"] != 0) : 
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
            cv2.drawContours(image, [c], -1, (0,255,0), 2)
            cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image

def run (image):
    frame = cv2.imread(image)
    frame = shape_center_detector(frame) 
    display([[frame]])

if __name__ == "__main__":
    #create input option
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='path to image')
    args = vars(ap.parse_args())
    shape_center_detector(args['image'])