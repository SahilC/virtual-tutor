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

def find_directional_gradient(im, kernel, axis,bias):
     temp = cv2.filter2D(im,-1,kernel)
     if axis == 0:
         temp = np.cumsum(temp,axis=0)[-1]
     else:
         temp = np.cumsum(temp.T,axis=0)[-1]

     x = (np.sum(temp)*1.0/len(temp) - bias)
     x = 3200
     temp2 = np.zeros(len(temp))
     temp2[temp > x] = 1
     temp2[temp < x] = 0
     # print len(temp2)
     return temp2

nFrames = int(vidFile.get(cv2.CAP_PROP_FRAME_COUNT)) # one good way of namespacing legacy openCV: cv2.cv.*
print "frame number: %s" %nFrames
fps = vidFile.get(cv2.CAP_PROP_FPS)
print "FPS value: %s" %fps

element_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(100,100 ),( 0, 0))
ret, frame = vidFile.read() # read first frame, and the return code of the function.
while ret:  # note that we don't have to use frame number here, we could read from a live written file.
    temp = np.zeros((frame.shape[0],frame.shape[1]),np.uint8)
    #temp[frame > 200] = 0
    #temp[frame < 50] = 255
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(frame, (0,0),3)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    #gray[gray < 50] = 50
    kernel_horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    #kernel_vertical = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    temp = cv2.filter2D(gray,cv2.CV_8U,kernel_horizontal)
    #temp = find_directional_gradient(gray,kernel_vertical,0,0)
    #kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    #vertical_gradient  = find_directional_gradient(gray,kernel,0,0)
    #temp2 = np.zeros((gray.shape[0],gray.shape[1]),dtype=np.uint8)
    #for i in xrange(len(gray)):
    #     temp2[i] = np.multiply(gray[i],vertical_gradient)

    #horizontal_gradient  = find_directional_gradient(gray,kernel.T,1,0)
    #temp4  = np.zeros((gray.shape[1],gray.shape[0]),dtype=np.uint8)
    #im2 = gray.copy()
    #im2 = im2.T
    #for i in xrange(len(im2)):
    #     temp4[i] = np.multiply(im2[i],horizontal_gradient)
    #temp4 = temp4.T
    #temp4 = cv2.bitwise_and(temp2,temp4)
    #temp4 = cv2.bitwise_and(gray,temp4)

    #cv2.imshow("XYYZZ",temp)
    #temp4 = cv2.bitwise_and(im,im,mask = temp4)
    #temp[temp > 70] = 0
    #ret,thresh1 = cv2.threshold(temp.copy(),100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #temp[temp < 50] = 0
    #temp = cv2.dilate(temp,element_dilate)
    #fast = cv2.FastFeatureDetector_create(threshold=45)
    #edges = cv2.Canny(blur, 1, 10)
    #lower_green = np.array([0,0,180])
    #upper_green = np.array([180,256,256])
    #hsv =  cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
            #selecting image within HSV-Range

    #mask = cv2.inRange(hsv,lower_green,upper_green)
    #mask = cv2.erode(cv2.dilate(mask,element_dilate),element_dilate)
    #temp[ mask == 0 ] = 0
    #i = 0
    #_,contours,_ = cv2.findContours(mask.copy(), 1, 2)
    #x,y,w,h = cv2.boundingRect(contours[0])
    #cv2.rectangle(mask,(x,y),(x+w,y+h),255,-1)
    #temp[mask == 0] = 0
    #for cnt in contours:
    #    hull = cv2.convexHull(cnt)
   #     cv2.drawContours(temp,[hull],0,255,8)
   #     i += 1
    #frame[mask == 255] = 10
    #plines = cv2.HoughLinesP(edges, 1, np.pi/180, 10,None, 50, 100)
    #i = 0
    #for l in plines:
    #    (x,y,a,b) = l[0]
#	if a != x and y != b:
#		angle = np.arctan2(b - y, a - x) * 180. / np.pi
 #               #if angle < 15 and angle > -15:
  #              cv2.line(temp, (x,y), (a,b), 255, 5)
#	i += 1

    #kp = fast.detect(gray,None)
    #img2 = cv2.drawKeypoints(temp, kp,temp, color=(255,0,0))
    #lines = cv2.HoughLines(temp, 1, np.pi/180, 10)
    #blur = cv2.bilateralFilter(gray, 15, 70,25)
    #blur[blur > 73] = 255
    #_, contours, _ = cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #max_area = 0
    #if len(contours) != 0:
    #    for i in range(len(contours)):
    #        cnt = contours[i]
    #        area = cv2.contourArea(cnt)
    #        #x,y,w,h = cv2.boundingRect(cnt)
    #        if(area>max_area):
    #            max_area = area
    #            ci=i

     #   cnt=contours[ci]

      #  hull = cv2.convexHull(cnt)
      #  drawing = np.zeros(frame.shape,np.uint8)
      #  cv2.drawContours(frame,[cnt],0,(0,255,0),-1)
        #cv2.drawContours(frame,[hull],0,(0,0,255),2)
    cv2.imshow("frameWindow",temp)
    cv2.waitKey(int(1/fps*1000)) # time to wait between frames, in mSec
    ret, frame = vidFile.read() #
