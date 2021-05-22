### Function : filter a specific color
### Option : 
###     -i/--src : path of input src
import cv2
import numpy as np
import argparse

def color_filter (src, select_range, dst = None): 
    '''
        Filter color of input image frame (remove color out of range), return mask (when dst = None) or output frame  \n
        src : input frame   \n
        select_range : ([start_B, start_G, start_R], [end_B, end_G, end_R]) : start/end color value of specific color, (0 -> 256) \n
        dst : the output frame which color filter will be applied. Otherwise, mask will be return   \n
        Note : input/output are frame (not image file)  \n
    '''
    o_img = np.zeros(src.shape)

    mask = cv2.bitwise_and(cv2.inRange(src[:, :, 0], select_range[0][0], select_range[1][0]), cv2.inRange(src[:, :, 1], select_range[0][1], select_range[1][1]))
    mask = cv2.bitwise_and(mask, cv2.inRange(src[:, :, 2], select_range[0][2], select_range[1][2]))

    if dst is not None : 
        o_img = cv2.bitwise_and(dst, dst, mask = mask)
        return o_img
    else :
        return mask

if __name__ == "__main__":
    #create input option
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='path to image')
    ap.add_argument('-o', '--output', help='path to output image')
    args = vars(ap.parse_args())
    # BGR
    select_range = ([0, 0, 0], [255, 255, 55])    # BGR - remove red color
    i_frame = cv2.imread(args["image"])
    o_frame = color_filter (i_frame, select_range, dst = i_frame) 
    cv2.imwrite(args["output"], o_frame)

