import cv2
import imutils
import numpy as np


def wait():
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display(frame, name='Window'):
    vframe_list = []
    for i, row_img in enumerate(frame):
        hframe_list = []
        for j in row_img:
            j_shape = j.shape
            hframe_list.append(imutils.resize(j, height=300))
        vframe_list.append(imutils.resize(cv2.hconcat(hframe_list), width=1200))
    o_frame = cv2.vconcat(vframe_list)
    o_frame_shape = o_frame.shape 
    if o_frame_shape[0] > o_frame_shape[1] :            # Portrait
        o_frame = imutils.resize(o_frame, height=1200)
    else :                                              # Landscape
        o_frame = imutils.resize(o_frame, width=1600)
    cv2.imshow(name, o_frame)
    wait()
