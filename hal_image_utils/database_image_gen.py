import cv2
import numpy as np
import imutils
import color_filter as h_color_filter
import argparse
import glob
import os

def image_rotate(i_frame, rot_angle = [10, 20, 30]) : 
    '''
        Rotate input image frame, return 2 list of rotated frame (right_rotate, left_rotate)    \n
        i_frame : input image frame
        rot_angle : angle rotated list
    '''
    o_frame_r = []
    o_frame_l = []
    kernel = np.ones((3, 3), 'uint8')
    for i in rot_angle : 
        tmp_img = imutils.rotate(i_frame, i)
        mask = h_color_filter.color_filter(tmp_img, ([0, 0, 0], [5, 5, 5]))     # get black background
        mask = cv2.dilate(mask, kernel)
        rot_img = cv2.bitwise_and(tmp_img, tmp_img, mask=~mask)
        fillup = cv2.bitwise_and(i_frame, i_frame, mask=mask)
        fillup = cv2.bitwise_or(fillup, rot_img)
        for i in range(5) :
            fillup = cv2.blur(fillup, (3,3))
        fillup = cv2.bitwise_and(fillup, fillup, mask=mask)
        result_img = cv2.bitwise_or(fillup, rot_img)
        o_frame_l.append(result_img)
    for i in rot_angle : 
        tmp_img = imutils.rotate(i_frame, 360 - i)
        mask = h_color_filter.color_filter(tmp_img, ([0, 0, 0], [5, 5, 5]))     # get black background
        mask = cv2.dilate(mask, kernel)
        rot_img = cv2.bitwise_and(tmp_img, tmp_img, mask=~mask)
        fillup = cv2.bitwise_and(i_frame, i_frame, mask=mask)
        fillup = cv2.bitwise_or(fillup, rot_img)
        for i in range(5) :
            fillup = cv2.blur(fillup, (3,3))
        fillup = cv2.bitwise_and(fillup, fillup, mask=mask)
        result_img = cv2.bitwise_or(fillup, rot_img)
        o_frame_r.append(result_img)
    return o_frame_r + o_frame_l

def database_gen(actions = ["rotation"], input_dir=None, output_dir = None):
    '''
    Enlarge image database by : rotation. \n
    - image_dir : original image directory\n
    - output_dir : result directory path
    - action : action to enlarge database such as rotation...
    '''

    if (os.path.exists(output_dir)== False):
        os.mkdir(output_dir)

    images = glob.glob(input_dir + "/*.jpg")
    for image in images : 
        print("Process {0}".format(image))
        frame = cv2.imread(image)
        for action in actions : 
            if action == "rotation" : 
                if (os.path.exists(output_dir + os.sep + action)== False):
                    os.mkdir(output_dir + os.sep + action)
                o_frames = image_rotate(frame)
                for o_indx, o_frame in enumerate(o_frames) : 
                    o_file = output_dir + os.sep + action + os.sep + "rotation__" + str(o_indx) + "__" + image.split(os.sep)[-1]
                    cv2.imwrite(o_file, o_frame)

if __name__ == "__main__":
    #create input option
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', help='original image dir')
    ap.add_argument('-o', '--output', help='output image dir')
    args = vars(ap.parse_args())
    database_gen(input_dir = args["input"], output_dir=args["output"])
