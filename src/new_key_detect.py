import cv2
import numpy as np
import sys
from extract_key import get_keymaps
from extract_calibration_frame import *

def detect_white_keys(frame,points):
    kernel_horizontal = np.array([[-1,-3,-3,-1],[0,0,0,0],[1,3,3,1]])
    diff = cv2.filter2D(frame,cv2.CV_8U,kernel_horizontal)

    if len(points) > 3:
        med = np.mean(points,0)
        diff = np.abs(np.float64(diff) - np.float64(med))
        diff = np.uint8(diff)

        counts = np.bincount(diff.flatten())
        max = counts[-1]
        for i in xrange(len(counts) -2 ,0,-1):
            if (max - counts[i]) < 30:
                print (counts[i],max)
                break

        diff[diff < i] = 0
        diff[keymap == 0] = 0
        del points[0]
    points.append(diff)
    return diff

if __name__ == '__main__':
    try:
        vidFile = cv2.VideoCapture("../sample_videos/Piano/VID_20161102_203548.mp4")
    except:
        print "Problem opening input stream"
        sys.exit(1)

    if not vidFile.isOpened():
        print "Capture stream not open"
        sys.exit(1)

    calibration_frame = extract_calibration_frame(vidFile)
    keymap = get_keymaps(calibration_frame)
    # cv2.imshow("Er",keymap)
    nFrames = int(vidFile.get(cv2.CAP_PROP_FRAME_COUNT))
    print "frame number: %s" %nFrames
    fps = vidFile.get(cv2.CAP_PROP_FPS)
    print "FPS value: %s" %fps
    ret, frame = vidFile.read()
    points = []

    while ret:
        blur = cv2.GaussianBlur(frame,(0,0),3)
        gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

        diff = detect_white_keys(gray,points)

        gray = cv2.resize(diff,(500,500))
        cv2.imshow("frameWindow",gray)
        cv2.waitKey(int(1/fps*1000))
        ret, frame = vidFile.read()

    # Release the VideoCapture object, wait for user to press a key and then close all windows
    vidFile.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
