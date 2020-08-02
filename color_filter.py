### Function : filter a specific color
### Option : 
###     -i/--image : path of input image
import cv2
import numpy as np
import argparse
from display import *

# BGR
select_range = ([0, 0, 0], [55, 255, 255])

def color_filter (image, select_range): 
    o_img = np.zeros(image.shape)

    mask = cv2.bitwise_and(cv2.inRange(image[:, :, 0], select_range[0][0], select_range[1][0]), cv2.inRange(image[:, :, 1], select_range[0][1], select_range[1][1]))
    mask = cv2.bitwise_and(mask, cv2.inRange(image[:, :, 2], select_range[0][2], select_range[1][2]))

    o_img = cv2.bitwise_and(image, image, mask=mask)
    return o_img

def run (image):
    frame = cv2.imread(image)
    frame = color_filter(image, select_range)
    display([[frame]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='path to image')
    args = vars(ap.parse_args())
    run(args['image'])