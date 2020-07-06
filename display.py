import cv2

def display (frame, name='Window') :
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
    cv2.imshow(name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

