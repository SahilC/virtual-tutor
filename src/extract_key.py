import cv2
import numpy as np

element_big = cv2.getStructuringElement(cv2.MORPH_RECT,( 10,10 ),( 0, 0))
element_small = cv2.getStructuringElement(cv2.MORPH_RECT,( 5,5 ),( 0, 0))

def otsuMulti(im):
    N = float(im.shape[0]*im.shape[1])

    histogram =  np.histogram(im,bins=range(0,256),range=(0,255),density=False)
    histogram = histogram[0]

    optimalThresh1 = 0
    optimalThresh2 = 0

    W0K = 0
    W1K = 0

    M0K = 0
    M1K = 0

    MT = 0
    maxBetweenVar = 0
    for k in xrange(0,255):
        MT += k * (histogram[k] /  N)


    for t1 in xrange(0,255):
        W0K += histogram[t1] /  N
        M0K += t1 * (histogram[t1] / N)
        M0 = M0K / (W0K + 0.00001)

        W1K = 0
        M1K = 0

        for t2 in xrange(t1 + 1,255):
            W1K += histogram[t2] / N
            M1K += t2 * (histogram[t2] / N)
            M1 = M1K / (W1K + 0.00001)

            W2K = 1 - (W0K + W1K)
            M2K = MT - (M0K + M1K)

            if (W2K <= 0):
                break

            M2 = M2K / W2K

            currVarB = W0K * (M0 - MT) * (M0 - MT) + W1K * (M1 - MT) * (M1 - MT) + W2K * (M2 - MT) * (M2 - MT)

            if (maxBetweenVar < currVarB):
                maxBetweenVar = currVarB
                optimalThresh1 = t1
                optimalThresh2 = t2

    return (optimalThresh1, optimalThresh2)

def get_black_keymap(frame,idx):
    lower_black = np.array([0,0,0])
    upper_black = np.array([180,255,idx])
    mask_black = cv2.inRange(frame,lower_black,upper_black)
    mask_black = cv2.dilate(mask_black,element_big)
    retval, labels = cv2.connectedComponents(mask_black)
    output = np.zeros_like(labels, dtype=np.uint8)
    cv2.normalize(labels, output, 0 , 100, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    mask = output.copy()
    return output

def get_white_keymap(frame,idx):
    con = np.zeros((frame.shape[0],frame.shape[1]),np.uint8)
    lower_white = np.array([0,0,idx])
    upper_white = np.array([180,255,255])

    mask = cv2.inRange(frame,lower_white,upper_white)
    _,contours,_ = cv2.findContours(mask.copy(), 1, 2)
    # cv2.imshow("Mask",mask)

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
    hsv =  cv2.cvtColor(calibration_im,cv2.COLOR_BGR2HSV)
    idx = otsuMulti(hsv[:,:,2])
    white_keymap = get_white_keymap(hsv,idx[1])
    black_keymap = get_black_keymap(hsv,idx[0])
    keymap = cv2.bitwise_or(white_keymap,black_keymap)
    return keymap
