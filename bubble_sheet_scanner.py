### Function : read and grade the bubble sheet scanner
### Option : 
###     -i/--image : path of input image

import numpy as np
import argparse
import cv2
import imutils
from shape_position_corrector import *
from sort_contour import *
from display import *

#create input option
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='path to image')
args = vars(ap.parse_args())

def bubble_sheet_scanner (frame) : 
    o_frame = frame.copy()
    o_frame = shape_position_corrector(frame)
    o_frame = sort_contour(o_frame, method = 'YA', dark_mode=False, min_area=800, max_area=1200)
    return o_frame

def run (image):
    frame = cv2.imread(image)
    frame = bubble_sheet_scanner(frame)
    display(frame)

if __name__ == "__main__":
    run(args['image'])
