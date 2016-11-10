import cv2
import numpy as np
import sys
# from matplotlib import pyplot as plt
try:
    vidFile = cv2.VideoCapture("../sample_videos/VID_20161024_165559.mp4")
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
while ret:  # note that we don't have to use frame number here, we could read from a live written file.

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # plt.hist(gray.ravel(),256,[0,256])
    # plt.show()

    # hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    # blur = cv2.bilateralFilter(gray, 15, 70,25)
    # blur[blur > 73] = 255
    #ret,thresh1 = cv2.threshold(gray.copy(),126,255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    thresh1 = cv2.adaptiveThreshold(gray.copy(),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,0)
    thresh1 = cv2.medianBlur(thresh1,7)
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
    cv2.imshow("frameWindow", thresh1)
    cv2.waitKey(int(1/fps*1000)) # time to wait between frames, in mSec
    ret, frame = vidFile.read() #
