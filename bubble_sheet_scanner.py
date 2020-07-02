### Function : read and grade the bubble sheet scanner
### Option : 
###     -i/--image : path of input image

import numpy as np
import argparse
import cv2
import imutils

#create input option
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='path to image')
args = vars(ap.parse_args())

def bubble_sheet_scanner (image) : 
    i_img = cv2.imread(image)

    cv2.imshow('window', i_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    bubble_sheet_scanner(args['image'])