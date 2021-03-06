### Function : update target image by color style in source image
### Option : 
###     -s/--source : path of source image (style image)
###     -t/--target : path of target image 
import numpy as np 
import cv2
import argparse
from display import *

def image_transfer (s_img, t_img) : 
    source = cv2.cvtColor(s_img, cv2.COLOR_BGR2LAB).astype('float32')
    target = cv2.cvtColor(t_img, cv2.COLOR_BGR2LAB).astype('float32')
    
    (tl, ta, tb) = cv2.split(target)
    (sl, sa, sb) = cv2.split(source)

    l = tl
    a = ta
    b = tb

    l -= tl.mean()
    a -= ta.mean()
    b -= tb.mean()

    l = l*tl.std()/sl.std()
    a = a*ta.std()/sa.std()
    b = b*tb.std()/sb.std()

    l += sl.mean()
    a += sa.mean()
    b += sb.mean()

    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    o_img = cv2.cvtColor(cv2.merge([l, a, b]).astype('uint8'), cv2.COLOR_LAB2BGR)

    return o_img

def run (s_image, t_image):
    s_frame = cv2.imread(s_image)
    t_frame = cv2.imread(t_image)
    o_frame = image_transfer(s_frame, t_frame) 
    display([[s_frame, t_frame], [o_frame]])

if __name__ == "__main__":
    #create input option
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--source', help='path to source image')
    ap.add_argument('-t', '--target', help='path to target image')
    args = vars(ap.parse_args())

    run(args['source'], args['target']) 