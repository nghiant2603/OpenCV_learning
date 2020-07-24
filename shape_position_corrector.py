### Function : correct the position of shape in image
### Option : 
###     -i/--image : input image
import numpy as np
import argparse
import cv2
import imutils
from display import *
from preprocess_image import *

#create input option
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='path to image')
args = vars(ap.parse_args())

def order_points(pts): 
    rect = np.zeros((4,2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
    	[0, 0],
    	[maxWidth - 1, 0],
    	[maxWidth - 1, maxHeight - 1],
    	[0, maxHeight - 1]], dtype = "float32")
    # compute the perspective shape_position_corrector matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def shape_position_corrector(frame, dark_mode = True): 
    o_frame = frame.copy()

    # reduce highlight 
    #hsvImg = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #hsvImg[...,2] = hsvImg[...,2]*0.6
    #frame = cv2.cvtColor(hsvImg,cv2.COLOR_HSV2RGB)
    frame = preprocess_image(frame)

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    if dark_mode : 
        thresh_img = cv2.threshold(blur_img, 150, 255, cv2.THRESH_BINARY)[1]
    else : 
        thresh_img = cv2.threshold(blur_img, 100, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if (len(approx) == 4) : 
            # apply the four point tranform to obtain a "birds eye view" of
            # the image
            o_frame = four_point_transform(o_frame, approx.reshape(4,2))
    return o_frame

def run (image):
    frame = cv2.imread(image)
    frame = shape_position_corrector(frame) 
    display([[frame]])

if __name__ == "__main__":
    run(args['image'])