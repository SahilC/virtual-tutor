import cv2
import numpy as np
import sys

element_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,( 2,2 ),( 0, 0))

try:
    vidFile = cv2.VideoCapture("sample_videos/piano.mkv")
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
#kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
while ret:  # note that we don't have to use frame number here, we could read from a live written file.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([0,50,50])
    upper_green = np.array([15,256,256])

    #selecting image within HSV-Range
    mask = cv2.inRange(hsv, lower_green, upper_green)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray,(5,5))
    _,show = cv2.threshold(gray.copy(),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #temp = cv2.filter2D(gray,-1,kernel)
    #lines = cv2.HoughLines(temp,1,np.pi/180,200)
    #if lines != None:
    #    count = 0
    #    avg_theta = 0
    #    for rho,theta in lines[0]:
    #        count += 1
    #        avg_theta += theta
    #    avg_theta = avg_theta/count
#
 #       rows,cols = gray.shape
  #      M = cv2.getRotationMatrix2D((cols/2,rows/2),90 - avg_theta*57.2958,1)
   #     dst = cv2.warpAffine(gray,M,(cols,rows))
    #gray[gray < 100] = 0
    #show = frame
    #_,show = cv2.threshold(gray.copy(),150,255,cv2.THRESH_BINARY_INV)
    #show = cv2.adaptiveThreshold(gray.copy(),256,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,0)
    temp = cv2.erode(show,element_dilate)
    #if previous != None:
    #    disp = show - previous
    #else:
    #    disp = show
    #disp = cv2.medianBlur(disp,5)
    #mask = cv2.dilate(mask,element_dilate)
    #disp = cv2.dilate(cv2.erode(disp,element_dilate),element_dilate)
    #disp[mask == 255] = 0
    #val = frame.copy()
    #_,contour,_ = cv2.findContours(disp.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #for cnt in contour:
    #    x,y,w,h = cv2.boundingRect(cnt)
    #	if h*1.0/w > 5:
#		cv2.rectangle(val,(x,y),(x+30,y+150),(255,0,0),2)
    #val[:,:,0][disp == 255] = 255
    #val[:,:,1][disp == 255] = 0
    #val[:,:,2][disp == 255] = 0
    temp = show - temp
    cv2.imshow("frameWindow",temp)
    #if i%24 == 0:
    #    previous = show
    i += 1
    cv2.waitKey(int(1/fps*1000)) # time to wait between frames, in mSec
    ret, frame = vidFile.read()
