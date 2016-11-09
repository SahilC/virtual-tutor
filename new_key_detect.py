import cv2
import numpy as np
import sys
from extract_key import get_keymaps
from extract_calibration_frame import *
if __name__ == '__main__':
    try:
        vidFile = cv2.VideoCapture("sample_videos/VID_20161106_194815.mp4")
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
    counter = 0
    time_slice = 5
    points = []
    lower_white = np.array([0,20,30])
    upper_white = np.array([20,255,255])

    while ret:
        blur = cv2.GaussianBlur(frame,(0,0),3)
        blur_hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

        blur = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(blur_hsv,lower_white,upper_white)
        if len(points) > 10:
            med = np.mean(points,0)
            diff = np.abs(np.float64(blur) - np.float64(med))
            #diff = diff -1*np.min(diff)
            # for i in diff:
            #     for j in i:
            #         print j,
            #     print ''
            diff = np.uint8(diff)
            #kernel_horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
            kernel_horizontal = np.array([[-1,-3,-3,-1],[0,0,0,0],[1,3,3,1]])
            diff = cv2.filter2D(diff,cv2.CV_8U,kernel_horizontal)
            # diff[keymap == 0] = 0
            # hist = cv2.calcHist([diff],[0],None,[256],[0,256])
            # plt.plot(hist)
            diff[mask == 255] = 0
            diff[diff < 50] = 0
            # diff[diff > 0] = 255
            diff = cv2.resize(diff,(500,500))
            cv2.imshow("DIFF",diff)
            #plt.show()
            # kernel_horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
            # temp = cv2.filter2D(diff,cv2.CV_8U,kernel_horizontal)
            #cv2.imshow("DIFF",cv2.resize(hist,(500,500)))
            del points[0]
        points.append(blur)

        gray = cv2.resize(blur,(500,500))
        cv2.imshow("frameWindow",gray)
        cv2.waitKey(int(1/fps*1000))
        ret, frame = vidFile.read()

    # Release the VideoCapture object, wait for user to press a key and then close all windows
    vidFile.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
