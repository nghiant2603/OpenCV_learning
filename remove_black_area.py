### Function : read and grade the bubble sheet scanner
### Option : 
###     -i/--image : path of input image

import numpy as np
import argparse
import cv2
import imutils
from display import *
from preprocess_image import *
from shape_position_corrector import *

def remove_black_area (frame) : 

    #frame = preprocess_image(frame)
    thresh_b_img = cv2.threshold(frame[:,:,0], 10, 255, cv2.THRESH_BINARY)[1]
    thresh_g_img = cv2.threshold(frame[:,:,1], 10, 255, cv2.THRESH_BINARY)[1]
    thresh_r_img = cv2.threshold(frame[:,:,2], 10, 255, cv2.THRESH_BINARY)[1]
    thresh_bg_img = cv2.bitwise_or(thresh_b_img, thresh_g_img)
    thresh_img = cv2.bitwise_or(thresh_bg_img, thresh_r_img)
    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = imutils.grab_contours(cnts)[0]
    left_edge = c[abs(c[:, 0, 0] - min(c[:, 0, 0])) < 3]
    max_row = max(left_edge[:, 0, 1])
    min_row = min(left_edge[:, 0, 1])
    bot_edge = c[abs(c[:, 0, 1] - max_row) < 3]
    top_edge = c[abs(c[:, 0, 1] - min_row) < 3]
    max_col = min([max(bot_edge[:, 0, 0]), max(top_edge[:, 0, 0])]) 
    o_frame = np.delete(frame, np.s_[max_col:frame.shape[1]:1], axis=1)
    return o_frame

def run (image):
    frame = cv2.imread(image)
    frame = remove_black_area(frame)
    display([[frame]])

if __name__ == "__main__":
    #create input option
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='path to image')
    args = vars(ap.parse_args())

    run(args['image'])
