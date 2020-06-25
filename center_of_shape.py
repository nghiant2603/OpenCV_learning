import cv2
import numpy as np
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='path to image')
args = vars(ap.parse_args())


def shape_center (image): 
    i_img = cv2.imread(image)
    gray_img = cv2.cvtColor(i_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    thresh_img = cv2.threshold(blur_img, 150, 255, cv2.THRESH_BINARY)[1]
    o_img = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
    #cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = imutils.grab_contours(cnts)

    #for c in cnts:
    #    M = cv2.moments(c)
    #    if (M["m00"] != 0) : 
    #        cX = int(M["m10"]/M["m00"])
    #        cY = int(M["m01"]/M["m00"])
    #        cv2.drawContours(i_img, [c], -1, (0,255,0), 2)
    #        cv2.circle(i_img, (cX, cY), 7, (255, 255, 255), -1)
    #        cv2.putText(i_img, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.imshow('window', np.hstack([i_img, o_img]))
    #cv2.imshow('window', i_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    shape_center(args['image'])