import cv2
import numpy as np
import imutils
import hal_image_process_utils.color_filter as h_color_filter

def image_rotate(i_frame, rot_angle, o_frame) : 

    kernel = np.ones((3, 3), 'uint8')
    rot_ang = [10, 20, 30]
    hresult = np.hstack((frame, frame))
    vresult = np.hstack((frame, frame))
    for i in rot_ang : 
        tmp_img = imutils.rotate(frame, i)
        mask = h_color_filter.color_filter(tmp_img, ([0, 0, 0], [5, 5, 5]))
        mask = cv2.dilate(mask, kernel)
        fillup = cv2.bitwise_and(frame, frame, mask=mask)
        tmp_img = cv2.bitwise_and(tmp_img, tmp_img, mask=~mask)
        rot_img = cv2.bitwise_or(fillup, tmp_img)
        hresult = np.hstack((hresult, rot_img))
    for i in rot_ang : 
        tmp_img = imutils.rotate(frame, 360 - i)
        mask = h_color_filter.color_filter(tmp_img, ([0, 0, 0], [5, 5, 5]))
        mask = cv2.dilate(mask, kernel)
        fillup = cv2.bitwise_and(frame, frame, mask=mask)
        tmp_img = cv2.bitwise_and(tmp_img, tmp_img, mask=~mask)
        rot_img = cv2.bitwise_or(fillup, tmp_img)
        vresult = np.hstack((vresult, rot_img))
    result = np.vstack((hresult, vresult))
    cv2.imshow("Image", result)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()