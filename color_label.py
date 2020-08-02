### Function : label the color of shape 
### Option : 
###     -i/--image : input image
import cv2
import numpy as np
import argparse
import imutils
from scipy.spatial import distance as dist
from collections import OrderedDict
from display import *

def color_label (image): 
    # create color dictionary
    colors = OrderedDict({'red':(255, 0, 0), 'green':(0, 255, 0), 'blue':(0, 0, 255), 'yellow':(255, 255, 0), 'orange':(255, 150, 0)}) 
    lab = np.zeros((len(colors), 1, 3), dtype="uint8")
    colorNames = []
    for (i, (name, rgb)) in enumerate(colors.items()):
    	# update the L*a*b* array and the color names list
        lab[i] = rgb
        colorNames.append(name)
    lab = cv2.cvtColor(lab, cv2.COLOR_RGB2LAB)

    alpha = 1.3         # constrast : 1.0 -> 3.0
    beta = 0            # brightness : 0 -> 100
    aj_img = cv2.convertScaleAbs(image, alpha = alpha, beta = beta)
    gray_img = cv2.cvtColor(aj_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    thresh_img = cv2.threshold(blur_img, 100, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    o_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    for c in cnts:
        mask = np.zeros(o_img.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(o_img, mask=mask)[:3]
        minDist = (np.inf, None)
        # loop over the known L*a*b* color values
        for (i, row) in enumerate(lab):
            d = dist.euclidean(row[0], mean)
            if d < minDist[0]:
            	minDist = (d, i)


        M = cv2.moments(c)
        if (M["m00"] != 0) : 
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
            #cv2.drawContours(o_img, [c], -1, (0,255,0), 2)
            cv2.circle(o_img, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(o_img, str(colorNames[minDist[1]]), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    o_img = cv2.cvtColor(o_img, cv2.COLOR_LAB2BGR)
    return o_img

def run (image):
    frame = cv2.imread(image)
    o_frame = color_label(frame)
    display([[frame, o_frame]])

if __name__ == "__main__":
    #create input option
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='path to image')
    args = vars(ap.parse_args())

    run(args['image'])