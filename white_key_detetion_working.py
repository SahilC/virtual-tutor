import cv2
import numpy as np
import sys
import math
from extract_key import *
from extract_calibration_frame import *

def detect_white_keys(frame):
    points = []
    kernel_horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    temp = cv2.filter2D(frame,cv2.CV_8U,kernel_horizontal)
    temp[temp > 80] = 0
    temp[temp < 50] = 0
    _,contours,_ = cv2.findContours(temp.copy(), 1, 2)
    for cnt in contours:
         x,y,w,h = cv2.boundingRect(cnt)
         if w*h > 100 and w*h < 300 and w > h:
             points.append((x,y,w,h))
    return points

if __name__ == '__main__':
    try:
        vidFile = cv2.VideoCapture("sample_videos/VID_20161113_171215.mp4")
    except:
        print "Problem opening input stream"
        sys.exit(1)

    if not vidFile.isOpened():
        print "Capture stream not open"
        sys.exit(1)

    calibration_frame = extract_calibration_frame(vidFile)
    # keymap = get_keymaps(calibration_frame)
    keymap = get_white_keymap(cv2.cvtColor(calibration_frame,cv2.COLOR_BGR2HSV))
    cv2.imshow("white",keymap)
    nFrames = int(vidFile.get(cv2.CAP_PROP_FRAME_COUNT))
    print "frame number: %s" %nFrames
    fps = vidFile.get(cv2.CAP_PROP_FPS)
    print "FPS value: %s" %fps
    ret, frame = vidFile.read()
    counter = 0
    time_slice = 5
    frames = []
    lower_white = np.array([0,20,30])
    upper_white = np.array([20,255,255])
    prev_key_presses =  set()
    while ret:
        blur = cv2.GaussianBlur(frame,(0,0),3)
        blur_hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

        blur = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(blur_hsv,lower_white,upper_white)
        kernel_horizontal = np.array([[-1,-3,-3,-1],[0,0,0,0],[1,3,3,1]])
        diff = cv2.filter2D(blur,cv2.CV_8U,kernel_horizontal)

        if len(frames) > 10:
            med = np.mean(frames,0)
            diff = np.abs(np.float64(diff) - np.float64(med))
            #diff = diff -1*np.min(diff)
            # for i in diff:
            #     for j in i:
            #         print j,
            #     print ''
            diff = np.uint8(diff)
            #kernel_horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

            # diff[keymap == 0] = 0
            # hist = cv2.calcHist([diff],[0],None,[256],[0,256])
            # plt.plot(hist)
            # diff[mask == 255] = 0
            counts = np.bincount(diff.flatten())
            max = counts[-1]
            print counts
            for i in xrange(len(counts) -2 ,0,-1):
                if math.fabs(max - counts[i]) < 30:
                    break
            print(i)
            diff[diff < i] = 0
            diff[keymap == 0] = 0
            # diff[diff > 0] = 255
            points = []
            _,contours,_ = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                 x,y,w,h = cv2.boundingRect(cnt)
                 # if w*h > 100 and w*h < 500 and w > h:
                 if w*h > 50 and w > h:
                     points.append((x,y,w,h))

            diff1 = cv2.resize(diff,(500,500))
            cv2.imshow("DIFF",diff1)

            cur_key_presses = set()
            for (x,y,w,h) in points:
                key = keymap[y,x]
                if key != 0:
                    cur_key_presses.add(key)
                    print(key)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),-1)
                    # cv2.rectangle(gray,(x,y),(x+w,y+h),255,-1)
                    # cv2.circle(frame, (x,y), 5, (0,0,255), 3)
                if len(prev_key_presses) < time_slice:
                    prev_key_presses.add(key)
                else:
                    prev_key_presses  = set()


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
