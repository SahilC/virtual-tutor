import cv2
import numpy as np
import sys
def detect_white_keys(frame):
    points = []
    kernel_horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    temp = cv2.filter2D(frame,cv2.CV_8U,kernel_horizontal)
    temp[temp > 80] = 0
    temp[temp < 50] = 0
    _,contours,_ = cv2.findContours(temp.copy(), 1, 2)
    for cnt in contours:
         x,y,w,h = cv2.boundingRect(cnt)
         if w*h > 100 and w*h < 300 and w > h:
             points.append((x,y,w,h))
    return points

def detect_black_keys(frame):
    points = []
    lower_green = np.array([0,0,0])
    upper_green = np.array([180,255,80])
    #selecting image within HSV-Range

    hsv =  cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lower_green,upper_green)
    hsv[mask == 0] = 0

    sat = hsv[:,:,1]
    sat[sat < 200] = 0
    _,contours,_ = cv2.findContours(sat.copy(), 1, 2)
    for cnt in contours:
         x,y,w,h = cv2.boundingRect(cnt)
         if w*h > 100 and w*h < 300 and w > h:
             points.append((x,y,w,h))
    return points

if __name__ == '__main__':
    try:
        vidFile = cv2.VideoCapture("sample_videos/VID_20161024_165559.mp4")
    except:
        print "Problem opening input stream"
        sys.exit(1)

    if not vidFile.isOpened():
        print "Capture stream not open"
        sys.exit(1)

    nFrames = int(vidFile.get(cv2.CAP_PROP_FRAME_COUNT))
    print "frame number: %s" %nFrames
    fps = vidFile.get(cv2.CAP_PROP_FPS)
    print "FPS value: %s" %fps
    ret, frame = vidFile.read()
    while ret:
        blur = cv2.GaussianBlur(frame,(0,0),3)
        points  = detect_black_keys(blur)

        gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        points += detect_white_keys(gray)
        for (x,y,w,h) in points:
            cv2.rectangle(gray,(x,y),(x+w,y+h),255,-1)

        gray = cv2.resize(gray,(500,500))
        cv2.imshow("frameWindow",gray)
        cv2.waitKey(int(1/fps*1000))
        ret, frame = vidFile.read()
