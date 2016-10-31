import cv2
import numpy as np
import sys
try:
    vidFile = cv2.VideoCapture("sample_videos/VID_20161024_165559.mp4")
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

element_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(100,100 ),( 0, 0))
ret, frame = vidFile.read()
frames = []
while ret:
    temp = np.zeros((frame.shape[0],frame.shape[1]),np.uint8)
    temp2 = np.zeros((frame.shape[0]+2,frame.shape[1]+2),np.uint8)
    blur = cv2.GaussianBlur(frame,(0,0),3)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    kernel_horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    temp = cv2.filter2D(gray,cv2.CV_8U,kernel_horizontal)
    temp[temp > 80] = 0
    temp[temp < 50] = 0
    # print len(frames)
    # if len(frames) > 3:
    #    #im = np.median(frames,0)
    #    #temp[temp <= im] = 0
    #    im =  np.float32(temp) - np.float32(frames[0])
    #    im[im < 0] = 0
    #    del frames[0]
    # #    for i in im:
    # #        for j in i:
    # #            print j,
    # #        print ''
    # #    cv2.imshow("mean image",np.uint8(im))
    # frames.append(np.float32(temp))
    _,contours,_ = cv2.findContours(temp.copy(), 1, 2)
    for cnt in contours:
         x,y,w,h = cv2.boundingRect(cnt)
         if w*h > 100 and w*h < 300 and w > h:
             cv2.rectangle(gray,(x,y),(x+w,y+h),255,-1)
            #  cv2.floodFill(edges,temp2,(x,y),255)
    gray = cv2.resize(gray,(500,0500))
    cv2.imshow("HELLO",gray)
    temp = cv2.resize(temp,(500,500))
    cv2.imshow("frameWindow",temp)
    cv2.waitKey(int(1/fps*1000))
    ret, frame = vidFile.read()
