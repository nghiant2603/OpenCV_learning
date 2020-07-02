### Function : sort contour according to their size/area 
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


def sort_contour (image, method = 'XA'):    # XA : sort X_axis Ascending , XD : sort x_axis Descending, YA, YD    if (method == 'LT') : 
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
    i_img = cv2.imread(image)
    o_img = i_img.copy()

    #alpha = 1.3         # constrast : 1.0 -> 3.0
    #beta = 0            # brightness : 0 -> 100
    #aj_img = cv2.convertScaleAbs(i_img, alpha = alpha, beta = beta)
    gray_img = cv2.cvtColor(i_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    thresh_img = cv2.threshold(blur_img, 70, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][y_axis], reverse=reverse))

    i = 0
    for c in cnts : 
        cv2.drawContours(o_img, [c], -1, (0,255,0), 2)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(o_img, (x, y), (x+w, y + h), (255, 255, 255), 1)
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the countour number on the image
        cv2.putText(o_img, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        i = i + 1

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.imshow('window', np.hstack([i_img, o_img]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sort_contour(args['image'], method='XD')