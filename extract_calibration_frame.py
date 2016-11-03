# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 21:52:06 2016

@author: Pranav
"""

import cv2

def extract_calibration_frame(vc):
    if vc.isOpened():
        nFrames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        vc.set(cv2.CAP_PROP_FRAME_COUNT, nFrames-1)
        ret, calibration_frame = vc.read()
        vc.set(cv2.CAP_PROP_FRAME_COUNT, 0)
        if ret:
            return calibration_frame
    return None