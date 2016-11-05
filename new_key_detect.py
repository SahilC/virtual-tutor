import cv2
import numpy as np
import sys
if __name__ == '__main__':
    try:
        vidFile = cv2.VideoCapture("sample_videos/VID_20161025_122058.mp4")
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
    counter = 0
    time_slice = 5
    points = []
    while ret:
        blur = cv2.GaussianBlur(frame,(0,0),3)
        blur = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        if len(points) > 5:
            med = np.median(points,0)
            diff = blur - med
            diff = diff -1*np.min(diff)
            # for i in diff:
            #     for j in i:
            #         print j,
            #     print ''
            diff = np.uint8(diff)
            # hist = cv2.calcHist([diff],[0],None,[256],[0,256])
            # plt.plot(hist)
            diff[diff < 50] = 0
            diff[diff > 50] = 255
            cv2.imshow("DIFF",diff)
            #plt.show()
            # kernel_horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
            # temp = cv2.filter2D(diff,cv2.CV_8U,kernel_horizontal)
            #cv2.imshow("DIFF",cv2.resize(hist,(500,500)))
            del points[0]
        points.append(blur)

        gray = cv2.resize(blur,(500,500))
        cv2.imshow("frameWindow",gray)
        cv2.waitKey(int(1/fps*1000))
        ret, frame = vidFile.read()

    # Release the VideoCapture object, wait for user to press a key and then close all windows
    vidFile.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
