import cv2
import numpy as np
import sys
import math
from extract_key import get_keymaps
from extract_calibration_frame import *

def detect_white_keys(frame,points, keymap):
    kernel_horizontal = np.array([[-1,-3,-3,-1],[0,0,0,0],[1,3,3,1]])
    diff = cv2.filter2D(frame,cv2.CV_8U,kernel_horizontal)

    if len(points) > 10:
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

def detect_black_keys(frame, keymap):
    pts = []
    if len(points) > 10:
        med = np.mean(points,0)
        diff = np.abs(np.float64(frame) - np.float64(med))
        diff = np.uint8(diff)

        kernel_horizontal = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        d = cv2.filter2D(diff,cv2.CV_8U,kernel_horizontal)
        d[keymap > 100] = 0
        d[keymap == 0] = 0
        d[d < 50] = 0
        _,contours,_ = cv2.findContours(d.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if h < 1.2*w:
                pts.append((x,y,w,h))
        del points[0]
    points.append(frame)
    return pts

def smart_threshold(hsv):
    hue = hsv[:,:,0]

    #selecting Largest connected component :- Ground
    counts = np.bincount(hue.flatten())
    for i in xrange(len(counts)):
        if counts[i] == 0:
            break
    hue[hue > i] = 0
    return hue

if __name__ == '__main__':
    try:
        vidFile = cv2.VideoCapture("../sample_videos/Piano/VID_20161102_204909.mp4")
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
        blur = np.uint8((np.float64(blur) + 10)*245/265)
        gray = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        pts = detect_black_keys(gray[:,:,1],keymap)
        if len(pts) > 0:
            for (x,y,w,h) in pts:
                key = keymap[y,x]
                if key != 0:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),-1)
        # w2d(gray,keymap)
        # gray = cv2.resize(diff,(500,500))
        frame = cv2.resize(frame,(500,500))
        # cv2.imshow("frameWindow",gray)
        cv2.imshow("frameWindow2",frame)
        cv2.waitKey(int(1/fps*1000))
        ret, frame = vidFile.read()

    # Release the VideoCapture object, wait for user to press a key and then close all windows
    vidFile.release()
    cv2.destroyAllWindows()
