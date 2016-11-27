import cv2
import numpy as np
import sys
from play_note import *
import thread
from extract_key import get_keymaps
from extract_calibration_frame import *

element_big = cv2.getStructuringElement(cv2.MORPH_RECT,( 10,10 ),( 0, 0))
element_small = cv2.getStructuringElement(cv2.MORPH_RECT,( 5,5 ),( 0, 0))
def detect_keypress(frame, points, keymap, prev_key_presses, time_slice = 10, key_id_press =[]):
    cur_key_presses = []
    for (x,y,w,h) in points:
        key = keymap[y,x]
        if key != 0 and key not in prev_key_presses:
            cur_key_presses.append(key)
# Play the sound asynchronously
            thread.start_new_thread(play_key, (key, key_id_map))
            if key < 100:
                print key
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),-1)
            prev_key_presses.append(key)
        # if len(prev_key_presses) > time_slice:
        #     del prev_key_presses[0]

def smart_threshold(im):
    counts =  np.histogram(im,bins=range(0,256),range=(0,255),density=False)
    counts = counts[0]
    max = counts[-1]
    count = 0
    for i in xrange(len(counts)-2 ,0,-1):
        if counts[i] > 300:
            count += 1
            if count > 1:
                break
            else:
                im[im > i] = 0
        max = counts[i]
    im[im < i] = 0
    return i

def detect_white_keys(frame, keymap, points, prev_key_presses):
    blur = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    kernel_horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    diff = cv2.filter2D(blur,cv2.CV_8U,kernel_horizontal)

    pts = []
    if len(points) > 10:
        med = np.mean(points,0)
        diff = np.abs(np.float64(diff) - np.float64(med))
        diff = np.uint8(diff)
        smart_threshold(diff)
        diff[keymap < 100] = 0

        diff = cv2.dilate(diff,element_big)
        diff = cv2.erode(diff,element_big)

        # diss = cv2.resize(diff,(500,500))
        # cv2.imshow("diff",diss)

        _,contours,_ = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
             x,y,w,h = cv2.boundingRect(cnt)
             area = cv2.contourArea(cnt)
             if area > 50 and w > h:
                 pts.append((x,y,w,h))

        del points[0]
    points.append(diff)
    return pts


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
        d[keymap > 100] = 0
        d[keymap == 0] = 0

        # d = cv2.GaussianBlur(d,(0,0),3)
        d = cv2.erode(d,element_big)
        d = cv2.dilate(d,element_big)
        diss = cv2.resize(d,(500,500))
        cv2.imshow("Er",diss)
        # cv2.imshow("km",keymap)
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
        vidFile = cv2.VideoCapture("../sample_videos/Piano/VID_20161106_194815.mp4")
    except:
        print "Problem opening input stream"
        sys.exit(1)

    if not vidFile.isOpened():
        print "Capture stream not open"
        sys.exit(1)

    white_points = []
    black_points = []
    b_key_presses =  []
    w_key_presses =  []
    calibration_frame = extract_calibration_frame(vidFile)
    keymap = get_keymaps(calibration_frame)
    km = cv2.resize(keymap,(500,500))
    cv2.imshow("KM",km)
    nFrames = int(vidFile.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidFile.get(cv2.CAP_PROP_FPS)
    print "frame number: %s" %nFrames
    print "FPS value: %s" %fps
    ret, frame = vidFile.read()

    count = 0
    key_id_map = get_key_id_map(np.unique(keymap))
    while ret:
        blur = cv2.GaussianBlur(frame,(0,0),3)

        pts = detect_black_keys(blur,keymap , black_points)
        detect_keypress(frame, pts, keymap, b_key_presses,key_id_map)
        # print b_key_presses

        pts = detect_white_keys(blur,keymap , white_points, w_key_presses)
        detect_keypress(frame, pts, keymap, w_key_presses,key_id_map)
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
        if len(b_key_presses) > 5:
            del b_key_presses[0]
        if len(w_key_presses) > 5:
            del w_key_presses[0]

    # Release the VideoCapture object, wait for user to press a key and then close all windows
    vidFile.release()
    cv2.destroyAllWindows()
