### Function : detect the shape base on contour, its angle, area and color
### Option : 
###     -i/--image : input image
import cv2
import numpy as np
import argparse
import imutils
from preprocess_image import *
from display import *
from scipy.spatial import distance as dist

def shape_detector (frame, dark_mode=True, n_points=(0, 1000), areas=(0, 100),color_point=[0, 0, 0], color_range=1000): 
    """Function : detect the shape base on contour, its angle, area and color\n\
        \t\t- dark_mode : True : True when the image background is darker than object\n\
        \t\t- n_points : (0, 1000) : min/max corner of detecting object\n\
        \t\t- areas : (0, 100) : the area of detecting object, in percentage of full image\n\
        \t\t- color_point : [B, G, R] : the center color of detecting object\n\
        \t\t- color_range : 255 : tolerance in euclidean distance, use with color_point """
    o_frame = frame.copy()
    lab_frame = cv2.cvtColor(o_frame, cv2.COLOR_BGR2LAB)

    color = np.array(color_point).reshape(1, 1, 3).astype("uint8")
    color_lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)

    frame = preprocess_image(frame) 

    #gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blur_img = cv2.GaussianBlur(frame, (5, 5), 0)
    if dark_mode : 
        thresh_b_img = cv2.threshold(frame[:,:,0], 150, 255, cv2.THRESH_BINARY)[1]
        thresh_g_img = cv2.threshold(frame[:,:,1], 150, 255, cv2.THRESH_BINARY)[1]
        thresh_r_img = cv2.threshold(frame[:,:,2], 150, 255, cv2.THRESH_BINARY)[1]
        thresh_bg_img = cv2.bitwise_or(thresh_b_img, thresh_g_img)
        thresh_img = cv2.bitwise_or(thresh_bg_img, thresh_r_img)
    else : 
        thresh_b_img = ~cv2.threshold(frame[:,:,0], 100, 255, cv2.THRESH_BINARY)[1]
        thresh_g_img = ~cv2.threshold(frame[:,:,1], 100, 255, cv2.THRESH_BINARY)[1]
        thresh_r_img = ~cv2.threshold(frame[:,:,2], 100, 255, cv2.THRESH_BINARY)[1]
        thresh_bg_img = cv2.bitwise_or(thresh_b_img, thresh_g_img)
        thresh_img = cv2.bitwise_or(thresh_bg_img, thresh_r_img)

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
        
        peri = cv2.arcLength(c, True)
        points = cv2.approxPolyDP(c, 0.04 * peri, True)
        n_point = len(points)
        area = 100*cv2.contourArea(c)/(full_area) 
        
        if (d <= color_range) : 
            if ((n_point >= n_points[0]) and (n_point < n_points[1])) : 
                if ((area >= areas[0]) and (area < areas[1])) : 
                    M = cv2.moments(c)
                    if (M["m00"] != 0) : 
                        cX = int(M["m10"]/M["m00"])
                        cY = int(M["m01"]/M["m00"])
                        cv2.drawContours(o_frame, [c], -1, (0,255,0), 2)
                        cv2.circle(o_frame, (cX, cY), 5, (0, 0, 0), -1)
                        cv2.putText(o_frame, str(i), (cX - 5, cY - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        i += 1
    return o_frame

def run (image):
    #create input option
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='path to image')
    args = vars(ap.parse_args())

    frame = cv2.imread(image)
    frame = shape_detector(frame, dark_mode=True, areas=(1,100)) 
    display(frame)

if __name__ == "__main__":
    run(args['image'])