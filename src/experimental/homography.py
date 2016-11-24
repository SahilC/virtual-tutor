import cv2
import numpy as np
import sys

def transform_points(transform_matrix,points):
    val, H = cv2.invert(transform_matrix)
    for (x,y,w,h) in points:
        h0 = H[0,0]
        h1 = H[0,1]
        h2 = H[0,2]
        h3 = H[1,0]
        h4 = H[1,1]
        h5 = H[1,2]
        h6 = H[2,0]
        h7 = H[2,1]
        h8 = H[2,2]

        tx = (h0*x + h1*y + h2)
        ty = (h3*x + h4*x + h5)
        tz = (h6*x + h7*y + h8)

        px = int(tx/tz)
        py = int(ty/tz)
        Z = int(1/tz)

        yield (px, py)

def plot_homography(frame,pts_list):
    hessian_threshold = 85
    sift = cv2.xfeatures2d.SURF_create(hessian_threshold)
    MIN_MATCH_COUNT = 10
    im_template = cv2.imread("../templates/template_repainted.png")
    kp_template, des_template = sift.detectAndCompute(im_template,None,useProvidedKeypoints = False)
    kp_frame, des_frame = sift.detectAndCompute(frame,None,useProvidedKeypoints = False)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_template,des_frame,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([ kp_template[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(dst_pts,src_pts, cv2.LMEDS,5.0)
        matchesMask = mask.ravel().tolist()
        if M is not None and len(pts_list) > 0:
            pts = []
            for (x,y,w,h) in pts_list:
                pts.append([x,y])
            pts = np.float32(pts).reshape(-1,1,2)

            dst = np.uint8(cv2.perspectiveTransform(pts,M))
            for point in dst:
                print (point[0][0],point[0][1])
                cv2.rectangle(im_template,(point[0][0],point[0][1]),(point[0][0]+5,point[0][1]+5),255)
            cv2.imshow("Template",im_template)
        # h,w,c = im_template.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # if M != None:
        #     dst = cv2.perspectiveTransform(pts,M)
        #     img2 = cv2.polylines(frame,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
        #     draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
        #     img3 = cv2.drawMatches(im_template,kp_template,img2,kp_frame,good,None,**draw_params)
        #     img3 = cv2.resize(img3,(500,500))
        #     cv2.imshow("frameWindow",img3)
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

def plot_homography_improved(frame,pts_list):
    hessian_threshold = 85
    orb = cv2.ORB_create()
    MIN_MATCH_COUNT = 10
    im_template = cv2.imread("../templates/template_repainted.png")
    kp_template, des_template = orb.detectAndCompute(im_template,None)
    kp_frame, des_frame = orb.detectAndCompute(frame,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)
    matches = sorted(bf.match(des_template,des_frame),key = lambda x:x.distance)
    im3 = cv2.drawMatches(im_template,kp_template,frame,kp_frame,matches[:5],None,flags = 2)
    # cv2.imshow("MATCHED",im3)
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)
    #
    # if len(good) >= MIN_MATCH_COUNT:
    #     src_pts = np.float32([ kp_template[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #     dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #
    #     M, mask = cv2.findHomography(dst_pts,src_pts, cv2.LMEDS,5.0)
    #     matchesMask = mask.ravel().tolist()
    #     if M is not None and len(pts_list) > 0:
    #         pts = []
    #         for (x,y,w,h) in pts_list:
    #             pts.append([x,y])
    #         pts = np.float32(pts).reshape(-1,1,2)
    #
    #         dst = np.uint8(cv2.perspectiveTransform(pts,M))
    #         for point in dst:
    #             print (point[0][0],point[0][1])
    #             cv2.rectangle(im_template,(point[0][0],point[0][1]),(point[0][0]+5,point[0][1]+5),255)
    #         cv2.imshow("Template",im_template)
        # h,w,c = im_template.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # if M != None:
        #     dst = cv2.perspectiveTransform(pts,M)
        #     img2 = cv2.polylines(frame,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
        #     draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
        #     img3 = cv2.drawMatches(im_template,kp_template,img2,kp_frame,good,None,**draw_params)
        #     img3 = cv2.resize(img3,(500,500))
        #     cv2.imshow("frameWindow",img3)
    # else:
    #     print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    #     matchesMask = None
