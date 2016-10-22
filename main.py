import cv2
import numpy as np
import sys

element_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,( 10,10 ),( 0, 0))

try:
    vidFile = cv2.VideoCapture("piano.mkv")
except:
    print "problem opening input stream"
    sys.exit(1)

if not vidFile.isOpened():
    print "capture stream not open"
    sys.exit(1)

nFrames = int(vidFile.get(cv2.CAP_PROP_FRAME_COUNT)) # one good way of namespacing legacy openCV: cv2.cv.*
print "frame number: %s" %nFrames
fps = vidFile.get(cv2.CAP_PROP_FPS)
print "FPS value: %s" %fps

ret, frame = vidFile.read() # read first frame, and the return code of the function.
previous = None
i = 0
while ret:  # note that we don't have to use frame number here, we could read from a live written file.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([0,50,50])
    upper_green = np.array([15,256,256])

    #selecting image within HSV-Range
    mask = cv2.inRange(hsv, lower_green, upper_green)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray,(5,5))
    #gray[gray < 100] = 0
    #show = frame
    _,show = cv2.threshold(gray.copy(),150,255,cv2.THRESH_BINARY_INV)
    #show = cv2.adaptiveThreshold(gray.copy(),256,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,0)
    if previous != None:
        disp = show - previous
    else:
        disp = show
    #mask = cv2.dilate(mask,element_dilate)
    #disp = cv2.dilate(cv2.erode(disp,element_dilate),element_dilate)
    #disp[mask == 255] = 0
    cv2.imshow("frameWindow",disp)
    if i%50 == 0:
        previous = show
    i += 1
    cv2.waitKey(int(1/fps*1000)) # time to wait between frames, in mSec
    ret, frame = vidFile.read()
