import cv2
import numpy as np

element_erode = cv2.getStructuringElement(cv2.MORPH_RECT,( 10,10 ),( 0, 0))
element_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,( 5,5 ),( 0, 0))

def get_white_keymap(frame):
    con = np.zeros((frame.shape[0],frame.shape[1]),np.uint8)
    lower_white = np.array([0,0,180])
    upper_white = np.array([180,255,255])

    hsv =  cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lower_white,upper_white)

    _,contours,_ = cv2.findContours(mask.copy(), 1, 2)

    for cnt in contours:
       hull = cv2.convexHull(cnt)
       cv2.drawContours(con,[hull],0,255,-1)

    mask = cv2.erode(mask,element_erode)
    mask = cv2.dilate(mask,element_dilate)
    mask = cv2.erode(mask,element_erode)
    mask = cv2.dilate(mask,element_dilate)
    mask = cv2.erode(mask,element_erode)
    mask = cv2.dilate(mask,element_dilate)

    mask2 = cv2.bitwise_not(mask)
    mask = cv2.bitwise_and(mask,con)
    mask2 = cv2.bitwise_and(mask2,con)
    cv2.imshow("CON",mask2)
    retval, labels = cv2.connectedComponents(mask)
    output = np.zeros_like(labels, dtype=np.uint8)
    cv2.normalize(labels, output, 0 , 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return output

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
        white_output = get_white_keymap(frame)
        #cv2.imshow("huh",white_output)
        cv2.waitKey(int(1/fps*1000))
        ret, frame = vidFile.read()
