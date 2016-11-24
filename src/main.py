import cv2
import numpy as np
import sys
from play_note import *
import thread
from extract_key import get_keymaps
from extract_calibration_frame import *

def detect_keypress(frame, points, keymap, prev_key_presses, time_slice = 5):
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

def detect_white_keys(frame, points, prev_key_presses):
    blur = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    kernel_horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    diff = cv2.filter2D(blur,cv2.CV_8U,kernel_horizontal)

    pts = []
    if len(points) > 10:
        med = np.mean(points,0)
        diff = np.abs(np.float64(diff) - np.float64(med))
        diff = np.uint8(diff)

        smart_threshold(diff)

        _,contours,_ = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
             x,y,w,h = cv2.boundingRect(cnt)
             if w*h > 50 and w > h:
                 pts.append((x,y,w,h))

        del points[0]
    points.append(diff)
    return pts


element_big = cv2.getStructuringElement(cv2.MORPH_RECT,( 10,10 ),( 0, 0))
def detect_black_keys(frame, keymap, points):
    blur = np.uint8((np.float64(frame) + 10)*245/265)
    frame = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    frame = frame[:,:,1]
    pts = []
    if len(points) > 10:
        med = np.mean(points,0)
        diff = np.abs(np.float64(frame) - np.float64(med))
        diff = np.uint8(diff)

        kernel_horizontal = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        d = cv2.filter2D(diff,cv2.CV_8U,kernel_horizontal)
        d = cv2.erode(d,element_big)
        d[keymap > 100] = 0
        d[keymap == 0] = 0
        # cv2.imshow("Er",d)
        # cv2.imshow("km",keymap)
        # blur = cv2.GaussianBlur(d,(0,0),3)
        _,contours,_ = cv2.findContours(d.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            # if h < 1.2*w:
            pts.append((x,y,w,h))
        del points[0]
    points.append(frame)
    return pts

if __name__ == '__main__':
    try:
        vidFile = cv2.VideoCapture("../sample_videos/Piano/VID_20161113_171032.mp4")
    except:
        print "Problem opening input stream"
        sys.exit(1)

    if not vidFile.isOpened():
        print "Capture stream not open"
        sys.exit(1)

    white_points = []
    black_points = []
    b_key_presses =  set()
    w_key_presses =  set()
    calibration_frame = extract_calibration_frame(vidFile)
    keymap = get_keymaps(calibration_frame)

    nFrames = int(vidFile.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidFile.get(cv2.CAP_PROP_FPS)
    print "frame number: %s" %nFrames
    print "FPS value: %s" %fps
    ret, frame = vidFile.read()

    while ret:
        blur = cv2.GaussianBlur(frame,(0,0),3)

        pts = detect_black_keys(blur,keymap, black_points)
        detect_keypress(frame, pts, keymap, b_key_presses)

        pts = detect_white_keys(blur,white_points, w_key_presses)
        detect_keypress(frame, pts, keymap, w_key_presses)
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
