import cv2
import numpy as np
import sys

if __name__ == '__main__':
    try:
        im_template = cv2.imread("templates/template.png")
        vidFile = cv2.VideoCapture("sample_videos/VID_20161024_165559.mp4")
    except:
        print "Problem opening input stream"
        sys.exit(1)

    if not vidFile.isOpened():
        print "Capture stream not open"
        sys.exit(1)
        
    while ret:
        cv2.imshow("frameWindow",gray)
        cv2.waitKey(int(1/fps*1000))
        ret, frame = vidFile.read()
