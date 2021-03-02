### Function : seam carving image
### Option : 
###     -i/--image : path of input image
import cv2
import numpy as np
from skimage import transform
from skimage import filters
import argparse
from display import *

def seam_carving (image, direction): 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mag = filters.sobel(gray.astype("float"))
    for num_seam in range(20, 140, 20):
        carved = transform.seam_carve(image, mag, direction, num_seam)
    return carved

def run (image, direction):
    frame = cv2.imread(image)
    o_frame = seam_carving(frame, direction)
    print (frame.shape)
    print (o_frame.shape)
    display([[o_frame]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='path to image')
    ap.add_argument('-d', '--direction', type=str, default="vertical", help='seam carving direction')
    args = vars(ap.parse_args())
    run(args['image'], args['direction'])