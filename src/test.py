import sys
import cv2
import pickle
import numpy as np
from extract_key import get_keymaps
from extract_calibration_frame import *

lower_white = np.array([0,20,30])
upper_white = np.array([20,255,255])

def ApplyToImage(img, clf):
    data = np.reshape(img,(img.shape[0]*img.shape[1],3))
    predictedLabels = clf.predict(data)

    imgLabels = np.reshape(predictedLabels,(img.shape[0],img.shape[1]))
    imgLabels = ((-(imgLabels-1)+1)*255)
    return imgLabels

def activation_map(hsv,points):
    blur = cv2.cvtColor(blur,cv2.COLOR_HSV2GRAY)
    mask = cv2.inRange(blur_hsv,lower_white,upper_white)
    if len(points) > 5:
        med = np.mean(points,0)
        diff = np.abs(np.float64(blur) - np.float64(med))
        diff = np.uint8(diff)
        kernel_horizontal = np.array([[-1,-3,-3,-1],[0,0,0,0],[1,3,3,1]])
        diff = cv2.filter2D(diff,cv2.CV_8U,kernel_horizontal)
        diff[mask == 255] = 0
        diff[diff < 50] = 0
        diff[keymap == 0] = 0
        del points[0]
    points.append(blur)
    return diff

def smart_threshold(hsv):
    hue = hsv[:,:,0]

    #selecting Largest connected component :- Ground
    counts = np.bincount(hue.flatten())
    for i in xrange(len(counts)):
        if counts[i] == 0:
            break
    print i
    hue[hue > i] = 0
    return hue

try:
    #vidFile = cv2.VideoCapture("../sample_videos/VID_20161106_194815.mp4")
    vidFile = cv2.VideoCapture("../sample_videos/Piano/VID_20161024_165559.mp4")
except:
    print "problem opening input stream"
    sys.exit(1)


if not vidFile.isOpened():
    print "capture stream not open"
    sys.exit(1)

calibration_frame = extract_calibration_frame(vidFile)
keymap = get_keymaps(calibration_frame)
nFrames = int(vidFile.get(cv2.CAP_PROP_FRAME_COUNT)) # one good way of namespacing legacy openCV: cv2.cv.*
print "frame number: %s" %nFrames
fps = vidFile.get(cv2.CAP_PROP_FPS)
print "FPS value: %s" %fps
#selecting image within HSV-Range
ret, frame = vidFile.read() # read first frame, and the return code of the function.
cv2.imshow("frameWindow2", keymap)
while ret:  # note that we don't have to use frame number here, we could read from a live written file.
    frame = cv2.GaussianBlur(frame,(0,0),3)
    with open('../models/tree2.pkl', 'rb') as f:
        clf = pickle.load(f)

    hue = np.uint8(ApplyToImage(frame, clf))
    # hsv =  cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #
    # hue = smart_threshold(hsv)
    # hue[keymap == 0] = 0
    # hue[hue != 0] = 255
    # max_label = np.argmax(counts)

    # output[output != max_label] = 0
    # output[output == max_label] = 255
    # plt.hist(gray.ravel(),256,[0,256])
    # plt.show()

    # hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    # blur = cv2.bilateralFilter(gray, 15, 70,25)
    # blur[blur > 73] = 255
    #ret,thresh1 = cv2.threshold(gray.copy(),126,255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    #thresh1 = cv2.adaptiveThreshold(gray.copy(),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,0)
    #thresh1 = cv2.medianBlur(thresh1,7)
    # #edges = cv2.Canny(frame.copy(),90,100,3)
    # #retval, labels = cv2.connectedComponents(gray)
    # #output = np.zeros_like(labels, dtype=np.uint8)
    # #cv2.normalize(labels, output, 0 , 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #
    # #counts = np.bincount(output.flatten())
    # #max_label = np.argmax(counts)
    # #output[output != max_label] = 0
    # #output[output == max_label] = 255
    # _, contours, _ = cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # max_area = 0
    # if len(contours) != 0:
    #     for i in range(len(contours)):
    #         cnt = contours[i]
    #         area = cv2.contourArea(cnt)
    #         #x,y,w,h = cv2.boundingRect(cnt)
    #         if(area>max_area):
    #             max_area = area
    #             ci=i
    #
    #     cnt=contours[ci]
    #
    #     hull = cv2.convexHull(cnt)
    #     drawing = np.zeros(frame.shape,np.uint8)
    #     cv2.drawContours(frame,[cnt],0,(0,255,0),-1)
    #     #cv2.drawContours(frame,[hull],0,(0,0,255),2)
    hue = cv2.resize(hue,(500,500))
    frame = cv2.resize(frame,(500,500))
    cv2.imshow("frameWindow", hue)
    cv2.imshow("origWindow", frame)
    cv2.waitKey(int(1/fps*1000)) # time to wait between frames, in mSec
    # cv2.waitKey(0)
    ret, frame = vidFile.read() #
