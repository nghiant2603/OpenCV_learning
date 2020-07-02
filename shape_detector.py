### Function : detect the shape of contour base on its conner
### Option : 
###     -i/--image : input image
import cv2
import numpy as np
import argparse
import imutils

#create input option
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='path to image')
args = vars(ap.parse_args())

def shape_detector (image): 
    i_img = cv2.imread(image)

    hsvImg = cv2.cvtColor(i_img,cv2.COLOR_BGR2HSV)
    # decreasing the V channel by a factor from the original
    hsvImg[...,2] = hsvImg[...,2]*0.6
    i_img = cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB)

    alpha = 2         # constrast : 1.0 -> 3.0
    beta = 0            # brightness : 0 -> 100
    aj_img = cv2.convertScaleAbs(i_img, alpha = alpha, beta = beta)
    
    gray_img = cv2.cvtColor(i_img, cv2.COLOR_BGR2GRAY)
    #Blur image to reduce mistake
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    thresh_img = cv2.threshold(blur_img, 100, 255, cv2.THRESH_BINARY)[1]

    o_img = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        M = cv2.moments(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if (M["m00"] != 0) : 
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
            cv2.drawContours(i_img, [c], -1, (0,255,0), 2)
            cv2.circle(i_img, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(i_img, str(len(approx)), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.imshow('window', np.hstack([i_img, o_img]))
    #cv2.imshow('window', i_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    shape_detector(args['image'])