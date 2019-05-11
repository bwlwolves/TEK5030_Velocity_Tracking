import numpy as np
import cv2
import glob
import os

def getChessBoardImages(cap, cap2):
    if (len(os.listdir('images')) != 0):
        print("Folder is not empty -- Existing images --")
        return

    #cap = cv2.VideoCapture(cam_index1)
    #cap2 = cv2.VideoCapture(cam_index2)
    i = 0
    while (True):
        ret, left_Frame = cap.read()
        ret2, right_Frame = cap2.read()
        left_Frame = cv2.cvtColor(left_Frame, cv2.COLOR_BGR2GRAY)
        right_Frame = cv2.cvtColor(right_Frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Left-camera", left_Frame)
        cv2.imshow("Right-camera", right_Frame)
        k = cv2.waitKey(1)
        if k == 32:
            cv2.imwrite('images/left_'+str(i)+'.png', left_Frame)
            cv2.imwrite('images/right_'+str(i)+'.png', right_Frame)
            i += 1
        elif k == 27:
            break

def camera_stereoCalibrate(c_size_x, c_size_y):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((c_size_x*c_size_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:c_size_x,0:c_size_y].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpointsL = [] # 3d point in real world space
    imgpointsL = [] # 2d points in image plane.
    objpointsR = []
    imgpointsR = []

    images_left = glob.glob('images/left*.png')
    images_left.sort()
    for fname in images_left:
        img = cv2.imread(fname)
        grayL = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, cornersL = cv2.findChessboardCorners(grayL,(c_size_x, c_size_y),None)
        #print(cornersL)
        if ret == True:
            ret = cv2.drawChessboardCorners(img, (c_size_x,c_size_y), cornersL, ret)
            cv2.imshow("Left-camera",ret)
            cv2.waitKey(1)
            objpointsL.append(objp)
            cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
            imgpointsL.append(cornersL)

    images_right = glob.glob('images/right*.png')
    images_right.sort()

    for fname in images_right:
        img = cv2.imread(fname)
        grayR = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, cornersR = cv2.findChessboardCorners(grayR, (c_size_x,c_size_y),None)
        if ret == True:
            ret = cv2.drawChessboardCorners(img, (c_size_x,c_size_y), cornersR, ret)
            cv2.imshow("Right-camera",ret)
            cv2.waitKey(1)
            objpointsR.append(objp)
            cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
            imgpointsR.append(cornersR)

    rt1, m1,d1,r1,t1 = cv2.calibrateCamera(objpointsL, imgpointsL, grayL.shape, None, None)
    rt2, m2,d2,r2,t2 = cv2.calibrateCamera(objpointsR, imgpointsR, grayR.shape, None, None)

    nL = np.array(imgpointsL).shape[0]
    nR = np.array(imgpointsR).shape[0]
    nImg = min(nL, nR)

    retval,cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpointsL[:nImg], imgpointsL[:nImg], imgpointsR[:nImg],
                                                                                                    m1, d1, m2, d2, grayL.shape)
    img_shape = grayL.shape
    return retval,cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, img_shape

def camera_stereoRectify(camMatrix1, camMatrix2, distCoeffs1, distCoeffs2, R, T, image_shape):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(camMatrix1, distCoeffs1, camMatrix2, distCoeffs2, image_shape, R, T,
                                                      cv2.CALIB_ZERO_DISPARITY, -1, image_shape)

    return R1, R2, P1, P2, Q

"DO THIS PROPERLY WHEN THAT WITCH IS NOT HERE!!!"
def undistort_RectifyMap(camMatrix, distCoeffs, R, P1, P2, image_shape):
    map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix=camMatrix, distCoeffs=distCoeffs, R=R, newCameraMatrix=P1, size=image_shape)
    return map1, map2

"""
def main():
    getChessBoardImages(cam_index1=4, cam_index2=6)
    retval,camMatrix1, distCoeffs1, camMatrix2, distCoeffs2, R, T, E, F, img_shape = camera_stereoCalibrate(c_size_x=7, c_size_y=7)
    r1,r2,p1,p2, disp_depth = camera_stereoRectify(camMatrix1, camMatrix2, distCoeffs1, distCoeffs2, R, T, img_shape)

if __name__ == '__main__':
    main()
"""
