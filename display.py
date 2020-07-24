import cv2
import imutils
import numpy as np

def wait () :
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display (frame, name='Window') :
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    vframe_list = []
    for i, row_img in enumerate(frame) : 
        hframe_list = []
        for j in row_img : 
            hframe_list.append(imutils.resize(j, height=400))
        vframe_list.append(cv2.hconcat(hframe_list))
    cv2.imshow('window', cv2.vconcat(vframe_list))
    wait()

