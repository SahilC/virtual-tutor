import cv2
import sys
import os
import numpy as np

def flip_image(im,class_val,filename):
    flip = cv2.flip(im,1)
    print(class_val+"/flipped_"+filename)
    cv2.imwrite(class_val+"/flipped_"+filename,flip)

def resize_image(im,class_val,filename):
    resized = cv2.resize(im,(180,100))
    spl = filename.replace('.png','').split("_")

    if "flipped" in spl:
        new_filename = "_".join(spl[:5])
        for i in xrange(5,len(spl)):
            num = int(spl[i])
            if i%2 == 1:
                new_x = int(((1280.0 - num)/1280.0)*180.0)
                new_filename += "_"+str(new_x)
            else:
                new_y = int((1.0*num/720)*100.0)
                new_filename += "_"+str(new_y)
    else:
        new_filename = "_".join(spl[:4])
        for i in xrange(4,len(spl)):
            num = int(spl[i])
            if i%2 == 0:
                new_x = int((1.0*num/1280)*180.0)
                new_filename += "_"+str(new_x)
            else:
                new_y = int((1.0*num/720)*100.0)
                new_filename += "_"+str(new_y)

    print(filename+":"+new_filename)
    cv2.imwrite("resized_train/"+class_val+"/"+new_filename+".png",resized)

def morph_exposures_merten(im,filename):
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process([im])
    res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
    cv2.imshow("HELLO",cv2.resize(res_mertens_8bit,(500,500)))

def morph_exposures_robertson(im,filename):
    exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)
    merge_robertson = cv2.createMergeRobertson()
    for i in exposure_times:
        hdr_robertson = merge_robertson.process([im], times=np.array([i]))
        tonemap2 = cv2.createTonemapDurand(gamma=0.8)
        res_robertson = tonemap2.process(hdr_robertson.copy())
        res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
        cv2.imshow("HELLO"+str(i),cv2.resize(res_robertson_8bit,(500,500)))
        cv2.waitKey(0)

def morph_exposures_dubvec(im,filename):
    exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)
    merge_debvec = cv2.createMergeDebevec()
    hdr_debvec = merge_debvec.process(im, times=exposure_times.copy())
    tonemap1 = cv2.createTonemapDurand(gamma=2.2)
    res_debvec = tonemap1.process(hdr_debvec.copy())
    res_debvec_8bit = np.clip(res_debvec*255, 0, 255).astype('uint8')

if __name__ == '__main__':
    for root, dirs, files in os.walk('PianoTouches/negative/'):
        for basename in files:
            filename = os.path.join(root, basename)
            im = cv2.imread(filename)
            #flip_image(im,"PianoTouches/negative/",basename)
            resize_image(im,"negative",basename)

    # im = cv2.imread('Test_images/VID_20161025_122058_360_749_317_860_332_987_285.png')
    # cv2.imshow("Im",cv2.resize(im,(500,500)))
    # morph_exposures_robertson(im,'Test_images/VID_20161025_122058_360_749_317_860_332_987_285.png')
