### Function : sort contour according to their size/area 
### Option : 
###     -i/--image : input image
import cv2
import numpy as np
import argparse
import imutils
from display import *
from preprocess_image import *

#create input option
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='path to image')
args = vars(ap.parse_args())

def sort_contour (frame, method = 'XA', dark_mode=True, min_area=10, max_area = 1000000):    # XA : sort X_axis Ascending , XD : sort x_axis Descending, YA, YD    if (method == 'LT') : 
    if (method == 'XA') :
        y_axis = 0          # 0 : sort x_axis - 1 : sort y_axis
        reverse = False     # False : ascending - True : descending
    elif (method == 'XD') : 
        y_axis = 0          # 0 : sort x_axis - 1 : sort y_axis
        reverse = True     # False : ascending - True : descending
    elif (method == 'YA') : 
        y_axis = 1          # 0 : sort x_axis - 1 : sort y_axis
        reverse = False     # False : ascending - True : descending
    else : # YD 
        y_axis = 1          # 0 : sort x_axis - 1 : sort y_axis
        reverse = True     # False : ascending - True : descending

    o_frame = frame.copy()

    frame = preprocess_image(frame)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    if dark_mode :
        thresh_img = cv2.threshold(blur_img, 150, 255, cv2.THRESH_BINARY)[1]
    else : 
        thresh_img = ~cv2.threshold(blur_img, 100, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][y_axis], reverse=reverse))

    i = 0
    for c in cnts : 
        area = cv2.contourArea(c) 
        if (area > min_area) and (area < max_area) : 
            mask = np.zeros(thresh_img.shape,np.uint8)
            cv2.drawContours(mask,[c],0,255,-1)
            mean = cv2.mean(thresh_img, mask = mask)
            if (mean[0] > 150) : 
                cv2.drawContours(o_frame, [c], -1, (0,255,0), 2)
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(o_frame, (x, y), (x+w, y + h), (255, 255, 255), 1)
                M = cv2.moments(c)
                if (M["m00"] != 0) : 
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    #draw the countour number on the image
                    cv2.putText(o_frame, "#{}".format(i + 1), (cX - 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                i = i + 1
    return o_frame

def run (image):
    frame = cv2.imread(image)
    frame = sort_contour(frame, method = 'XA') 
    display([[frame]])

if __name__ == "__main__":
    run(args['image'])