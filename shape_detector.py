### Function : detect the shape of contour base on its conner
### Option : 
###     -i/--image : input image
import cv2
import numpy as np
import argparse
import imutils
from preprocess_image import *
from display import *
from scipy.spatial import distance as dist

def shape_detector (frame, dark_mode=True, n_points=(0, 1000), areas=(0, 100), color_point=[255, 0, 0], color_range=60): 
    o_frame = frame.copy()
    lab_frame = cv2.cvtColor(o_frame, cv2.COLOR_BGR2LAB)

    color = np.array(color_point).reshape(1, 1, 3).astype("uint8")
    color_lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)

    frame = preprocess_image(frame) 

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    if dark_mode : 
        thresh_img = cv2.threshold(blur_img, 150, 255, cv2.THRESH_BINARY)[1]
    else : 
        thresh_img = ~cv2.threshold(blur_img, 100, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    size = frame.shape
    full_area = size[0]*size[1]
    i = 0
    for c in cnts:
        mask = np.zeros(lab_frame.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(lab_frame, mask=mask)[:3]
        d = dist.euclidean(color_lab, mean)
        print(i, " -- ", d)
        
        peri = cv2.arcLength(c, True)
        points = cv2.approxPolyDP(c, 0.04 * peri, True)
        n_point = len(points)
        area = 100*cv2.contourArea(c)/(full_area) 

        if ((n_point >= n_points[0]) and (n_point < n_points[1])) : 
            if ((area >= areas[0]) and (area < areas[1])) : 
                M = cv2.moments(c)
                if (M["m00"] != 0) : 
                    cX = int(M["m10"]/M["m00"])
                    cY = int(M["m01"]/M["m00"])
                    cv2.drawContours(o_frame, [c], -1, (0,255,0), 2)
                    cv2.circle(o_frame, (cX, cY), 3, (255, 255, 255), -1)
                    cv2.putText(o_frame, str(i), (cX - 5, cY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        i += 1
    return o_frame

def run (image):
    #create input option
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='path to image')
    args = vars(ap.parse_args())

    frame = cv2.imread(image)
    frame = shape_detector(frame, dark_mode=True, n_points=(0,100), areas=(0, 5), color_point=[255, 0, 0]) 
    display(frame)

if __name__ == "__main__":
    run(args['image'])