# Function : create panorama image
# Option :
# -i1/--image1 : path of input image1
# -i2/--image2 : path of input image2

import numpy as np
import argparse
import cv2
import imutils
from display import *
from shape_position_corrector import *
from shape_center_detector import *
from remove_black_area import *

def stitch(images, ratio=0.75, reprojThresh=4.0):
    # unpack the images, then detect keypoints and extract
    # local invariant descriptors from them
    n_img = len(images)
    if n_img > 1 : 
        result = images[0]
        for i in range(n_img-1) : 
            (imageB, imageA) = (result, images[i + 1])
            (kpsA, featuresA) = detectAndDescribe(imageA)
            (kpsB, featuresB) = detectAndDescribe(imageB)
            # match features between the two images
            M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
            # keypoints to create a panorama
            if M != None:
                (matches, H, status) = M
                result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
                result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
                result = remove_black_area(result)
        return result
    else : 
        print("ERROR : At least 2 Image as Input")
        exit()

def detectAndDescribe(image):

    # detect and extract features from the image
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)

    # convert the keypoints from KeyPoint objects to NumPy
    # arrays
    kps = np.float32([kp.pt for kp in kps])
    # return a tuple of keypoints and features
    return (kps, features)


def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                         reprojThresh)
        # return the matches along with the homograpy matrix
        # and status of each matched point
        return (matches, H, status)
    # otherwise, no homograpy could be computed
    return None


def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    # return the visualization
    return vis


def panorama(image):
    frame = []
    for i, img in enumerate(image) : 
        f_img = cv2.imread(img)
        frame.append(imutils.resize(f_img, height=600))
    o_frame = stitch(frame)
    cv2.imwrite("test.jpg", o_frame)
    #(A_frame, B_frame) = stitch([frame1, frame2], showMatches=True)
    return o_frame


def run(image):
    frame = panorama(image)
    display([[frame]])


if __name__ == "__main__":
    # create input option
    ap = argparse.ArgumentParser()
    ap.add_argument('-i1', '--image1', help='Path to image 1')
    ap.add_argument('-i2', '--image2', help='Path to image 2')
    ap.add_argument('-i3', '--image3', help='Path to image 2')
    ap.add_argument('-i4', '--image4', help='Path to image 2')
    args = vars(ap.parse_args())
    in_args = [i for i in list(args.values()) if i != None]

    run(in_args)
