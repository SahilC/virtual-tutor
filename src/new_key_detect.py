import cv2
import numpy as np
import sys
import pickle
import pywt
from extract_key import get_keymaps
from extract_calibration_frame import *

def w2d(im,keymap, mode='haar'):
    # compute coefficients
    # coeffs = pywt.dwt2(im, mode)
    #
    # #Process Coefficients
    # coeffs_H = list(coeffs)
    # # coeffs_H[0] *= 0
    # cH,cV,cD = coeffs_H[1]
    # # cV *= 0
    # # cD *= 0
    # coeffs_H[1] = (cH,cV,cD)
    # # reconstruction
    # imArray_H = pywt.waverec2(coeffs_H, mode);
    # # imArray_H *= 255;
    # imArray_H =  np.uint8(imArray_H)
    #Display result
    kernel_horizontal = np.array([[-1,-3,-3,-1],[0,0,0,0],[1,3,3,1]])

    imArray_H = 255 - im
    if len(points) > 10:
        med = np.mean(points,0)
        diff = np.abs(np.float64(imArray_H) - np.float64(med))
        diff = np.uint8(diff)
        diff[keymap > 100] = 0
        diff[keymap == 0] = 0

        # counts = np.bincount(diff.flatten())
        # max = counts[-1]
        # for i in xrange(len(counts) -2 ,0,-1):
        #     if (max - counts[i]) < 30:
        #         print (counts[i],max)
        #         break
        #
        # diff[diff < i] = 0
        # diff[keymap == 0] = 0
        mask = cv2.filter2D(diff ,cv2.CV_8U,kernel_horizontal)
        cv2.imshow('image',mask)
        del points[0]
    points.append(imArray_H)

def ApplyToImage(img, clf):
    data = np.reshape(img,(img.shape[0]*img.shape[1],3))
    predictedLabels = clf.predict(data)

    imgLabels = np.reshape(predictedLabels,(img.shape[0],img.shape[1]))
    imgLabels = ((-(imgLabels-1)+1)*100)
    return np.uint8(imgLabels)

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

def detect_black_keys(frame,keymap):
    if len(points) > 10:
        med = np.mean(points,0)
        diff = np.abs(np.float64(frame) - np.float64(med))
        diff = np.uint8(diff)

        kernel_horizontal = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        d = cv2.filter2D(diff,cv2.CV_8U,kernel_horizontal)
        d[keymap > 100] = 0
        d[keymap == 0] = 0
        cv2.imshow("Eh",d)
        del points[0]
    points.append(frame)
    return frame

def smart_threshold(hsv):
    hue = hsv[:,:,0]

    #selecting Largest connected component :- Ground
    counts = np.bincount(hue.flatten())
    for i in xrange(len(counts)):
        if counts[i] == 0:
            break
    hue[hue > i] = 0
    return hue

# def detect_black_keys(frame, points, keymap):
#     hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#     hue = smart_threshold(hsv)
#     hue[keymap == 0] = 0
#     #hue[keymap > 100] = 0
#     hue[hue != 0] = 255
#     return hue

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
    with open('../models/tree3.pkl', 'rb') as f:
        clf = pickle.load(f)

    while ret:
        blur = cv2.GaussianBlur(frame,(0,0),3)
        blur = np.uint8((np.float64(blur) + 10)*245/265)
        gray = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        diff = detect_black_keys(gray[:,:,1],keymap, clf)
        # w2d(gray,keymap)
        # gray = cv2.resize(diff,(500,500))
        frame = cv2.resize(frame,(500,500))
        # cv2.imshow("frameWindow",gray)
        cv2.imshow("frameWindow2",blur)
        cv2.waitKey(int(1/fps*1000))
        ret, frame = vidFile.read()

    # Release the VideoCapture object, wait for user to press a key and then close all windows
    vidFile.release()
    cv2.destroyAllWindows()
