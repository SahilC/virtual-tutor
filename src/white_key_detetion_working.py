import cv2
import numpy as np
import sys
import math
from extract_key import *
from extract_calibration_frame import *

def detect_keypress(frame, points, keymap):
    cur_key_presses = set()
    for (x,y,w,h) in points:
        key = keymap[y,x]
        if key != 0:
            cur_key_presses.add(key)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),-1)
        if len(prev_key_presses) < time_slice:
            prev_key_presses.add(key)
        else:
            prev_key_presses  = set()

def smart_threshold(im):
    counts =  np.histogram(im,bins=range(0,256),range=(0,255),density=False)
    counts = counts[0]
    max = counts[-1]
    count = 0
    for i in xrange(len(counts)-2 ,0,-1):
        if counts[i] - max > 100:
            count += 1
            if count > 1:
                break
            else:
                im[im > i] = 0
        max = counts[i]
    im[im < i] = 0
    return i

if __name__ == '__main__':
    try:
        vidFile = cv2.VideoCapture("../sample_videos/Piano/VID_20161113_171215.mp4")
    except:
        print "Problem opening input stream"
        sys.exit(1)

    if not vidFile.isOpened():
        print "Capture stream not open"
        sys.exit(1)

    calibration_frame = extract_calibration_frame(vidFile)
    keymap = get_white_keymap(cv2.cvtColor(calibration_frame,cv2.COLOR_BGR2HSV))
    nFrames = int(vidFile.get(cv2.CAP_PROP_FRAME_COUNT))
    print "frame number: %s" %nFrames
    fps = vidFile.get(cv2.CAP_PROP_FPS)
    print "FPS value: %s" %fps
    ret, frame = vidFile.read()
    time_slice = 5
    frames = []
    prev_key_presses =  set()
    while ret:
        blur = cv2.GaussianBlur(frame,(0,0),3)

        blur = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        kernel_horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        diff = cv2.filter2D(blur,cv2.CV_8U,kernel_horizontal)

        if len(frames) > 10:
            med = np.mean(frames,0)
            diff = np.abs(np.float64(diff) - np.float64(med))
            diff = np.uint8(diff)

            smart_threshold(diff)

            cv2.imshow("Keymap",diff)

            points = []
            _,contours,_ = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                 x,y,w,h = cv2.boundingRect(cnt)
                 if w*h > 50 and w > h:
                     points.append((x,y,w,h))

            detect_keypress(frame, points, keymap)

            del frames[0]
        frames.append(diff)

        frame = cv2.resize(frame,(500,500))
        cv2.imshow("frameWindow",frame)
        cv2.waitKey(int(1/fps*1000))
        ret, frame = vidFile.read()

    # Release the VideoCapture object, wait for user to press a key and then close all windows
    vidFile.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
