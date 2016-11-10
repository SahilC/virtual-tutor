import cv2
import numpy as np

element_big = cv2.getStructuringElement(cv2.MORPH_RECT,( 10,10 ),( 0, 0))
element_small = cv2.getStructuringElement(cv2.MORPH_RECT,( 5,5 ),( 0, 0))

def get_black_keymap(frame):
    lower_black = np.array([0,0,0])
    upper_black = np.array([180,255,100])
    mask_black = cv2.inRange(frame,lower_black,upper_black)
    mask_black = cv2.erode(mask_black,element_big)
    retval, labels = cv2.connectedComponents(mask_black)
    output = np.zeros_like(labels, dtype=np.uint8)
    cv2.normalize(labels, output, 0 , 100, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    mask = output.copy()
    return output

def get_white_keymap(frame):
    con = np.zeros((frame.shape[0],frame.shape[1]),np.uint8)
    lower_white = np.array([0,0,180])
    upper_white = np.array([180,255,255])

    mask = cv2.inRange(frame,lower_white,upper_white)
    _,contours,_ = cv2.findContours(mask.copy(), 1, 2)

    for cnt in contours:
       hull = cv2.convexHull(cnt)
       cv2.drawContours(con,[hull],0,255,-1)

    mask = cv2.erode(mask,element_big)
    mask = cv2.dilate(mask,element_small)
    mask = cv2.erode(mask,element_big)
    mask = cv2.dilate(mask,element_small)
    mask = cv2.erode(mask,element_big)
    mask = cv2.dilate(mask,element_small)
    mask = cv2.dilate(mask,element_big)

    mask = cv2.bitwise_and(mask,con)
    retval, labels = cv2.connectedComponents(mask)
    output = np.zeros_like(labels, dtype=np.uint8)
    cv2.normalize(labels, output, 100 , 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    counts = np.bincount(output.flatten())
    ind = np.argpartition(counts,-3)[-3:]
    output[ output == ind[0]] = 0
    output[ output == ind[1]] = 0
    output[ output == ind[2]] = 0
    # max_label = np.argmax(counts)
    # output[output == max_label] = 0
    return output

def get_keymaps(calibration_im):
    calibration_im = cv2.imread('../templates/frame5.jpg')
    hsv =  cv2.cvtColor(calibration_im,cv2.COLOR_BGR2HSV)
    white_keymap = get_white_keymap(hsv)
    black_keymap = get_black_keymap(hsv)
    keymap = cv2.bitwise_or(white_keymap,black_keymap)
    return keymap
