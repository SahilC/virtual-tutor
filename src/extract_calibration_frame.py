# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 21:52:06 2016

@author: Pranav
"""

import cv2

def extract_calibration_frame(vc):
    if vc.isOpened():
        nFrames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        ret = vc.set(cv2.CAP_PROP_POS_FRAMES, nFrames-2)
        ret, calibration_frame = vc.read()
        print(ret)
        vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if ret:
            return calibration_frame
    return None
