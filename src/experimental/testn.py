import cv2
import numpy as np
import sys
from extract_key import get_keymaps
from extract_calibration_frame import *
if __name__ == '__main__':
    try:
        vidFile = cv2.VideoCapture("../sample_videos/Piano/VID_20161102_203548.mp4")
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
    f = open('myfile.txt','w')
    while ret:
        lower_green = np.array([0,20,60])
        upper_green = np.array([20,150,255])
        #selecting image within HSV-Range

        hsv =  cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,lower_green,upper_green)

        cont_pos = 0
        cont_neg = 0
        for x in xrange(frame.shape[0]):
            for y in xrange(frame.shape[1]):
                if mask[x,y] == 255:
                    cont_pos += 1
                    if cont_pos < 200000:
                        f.write(str(frame[x,y,0])+" "+str(frame[x,y,1])+" "+str(frame[x,y,2])+" 1\n")
                    else:
                        break
                else:
                    cont_neg += 1
                    if cont_neg < 200000:
                        f.write(str(frame[x,y,0])+" "+str(frame[x,y,1])+" "+str(frame[x,y,2])+" 2\n")
                    else:
                        break
            if cont_neg > 200000 or cont_pos > 200000:
                break

        # cv2.imshow("Eh",mask)
        cv2.waitKey(int(1/fps*1000))
        ret, frame = vidFile.read()
