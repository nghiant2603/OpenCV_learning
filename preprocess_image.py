### Function : improve the quality of input image
### Option : 
###     -i/--image : input image
import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt
from display import *

#create input option
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='path to image')
args = vars(ap.parse_args())
def preprocess_image(frame):

    # Increase constrast
    #alpha = 1.3         # constrast : 1.0 -> 3.0
    #beta = 10            # brightness : 0 -> 100
    #frame = cv2.convertScaleAbs(frame, alpha = alpha)

    # reduce highlight 
    #hsvImg = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #hsvImg[...,2] = hsvImg[...,2]*0.8
    #frame = cv2.cvtColor(hsvImg,cv2.COLOR_HSV2BGR)

    #kernel = np.ones((5,5),np.uint8)
    #erode_img = cv2.erode(thresh_img, kernel, 1)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype('float32')
    (l,a,b) = cv2.split(lab)
    l = l - l.mean() + 120
    l = np.clip(l, 0, 255)
    frame = cv2.cvtColor(cv2.merge([l, a, b]).astype('uint8'), cv2.COLOR_LAB2BGR)

    #print(l.mean())
    #print(a.mean())
    #print(b.mean())
    #print(l.std())
    #print(a.std())
    #print(b.std())

    return frame

def run (image):
    frame = cv2.imread(image)
    frame = preprocess_image(frame)
    display(frame)

if __name__ == "__main__":
    run(args['image'])
