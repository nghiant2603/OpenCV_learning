import cv2
import numpy as np
import argparse

select_range = ([0, 150, 0], [55, 255, 255])
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help='path to image')
args = vars(ap.parse_args())

def color_detection (image, select_range): 
    i_img = cv2.imread(image)

    o_img = np.zeros(i_img.shape)

    mask = cv2.bitwise_and(cv2.inRange(i_img[:, :, 0], select_range[0][0], select_range[1][0]), cv2.inRange(i_img[:, :, 1], select_range[0][1], select_range[1][1]))
    mask = cv2.bitwise_and(mask, cv2.inRange(i_img[:, :, 2], select_range[0][2], select_range[1][2]))

    o_img = cv2.bitwise_and(i_img, i_img, mask=mask)

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.imshow('window', np.hstack([i_img, o_img]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    color_detection(args['image'], select_range)