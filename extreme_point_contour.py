### Function : detect extreme points in contours
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


def extreme_point_contour (image): 
    i_img = cv2.imread(image)
    o_img = i_img.copy()

    alpha = 1.3         # constrast : 1.0 -> 3.0
    beta = 0            # brightness : 0 -> 100
    aj_img = cv2.convertScaleAbs(i_img, alpha = alpha, beta = beta)
    gray_img = cv2.cvtColor(i_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    thresh_img = cv2.threshold(blur_img, 70, 255, cv2.THRESH_BINARY)[1]
    #thresh_img = cv2.erode(thresh_img, None, iterations=2)
    #thresh_img = cv2.dilate(thresh_img, None, iterations=2)

    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    cv2.drawContours(o_img, [c], -1, (0,255,0), 2)
    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    cv2.circle(o_img, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(o_img, extRight, 8, (0, 255, 255), -1)
    cv2.circle(o_img, extTop, 8, (255, 0, 0), -1)
    cv2.circle(o_img, extBot, 8, (255, 255, 0), -1)
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.imshow('window', np.hstack([i_img, o_img]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    extreme_point_contour(args['image'])