import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from random import randint
import stereo_camera_calibration as scc

# Measures the velocity - NO SCALE #
def measure_velocity_no_depth(curr_contours, prev_contours, amount_of_bbox, fps):
    if curr_contours == None or prev_contours == None:
        return

    bbox_1 = []
    bbox_2 = []

    tot_bbox_1 = 0
    for curr_c in curr_contours:
        if tot_bbox_1 == amount_of_bbox:
            break
        x1,y1,w1,h1 = cv2.boundingRect(curr_c)
        bbox_1.append([x1,y1])
        tot_bbox_1 += 1

    tot_bbox_2 = 0
    for prev_c in prev_contours:
        if tot_bbox_2 == amount_of_bbox:
            break
        x2,y2,w2,h2 = cv2.boundingRect(prev_c)
        bbox_2.append([x2,y2])
        tot_bbox_2 += 1

    bbox_1 = np.array(bbox_1)
    bbox_2 = np.array(bbox_2)
    velocity = []
    if(bbox_2.shape[0] == bbox_1.shape[0] and bbox_2.shape[0] != 0):
        for c2,c1 in zip(bbox_2, bbox_1):
            speed = c1-c2/fps
            velocity.append(speed)
        velocity = np.array(velocity)
        velocity = np.abs(np.mean(a=velocity, axis=1))
        velocity = np.where(np.isnan(velocity), 0, velocity)
        velocity = np.round(a=velocity, decimals=1)
    velocity = np.array(velocity)
    return velocity, curr_contours, prev_contours

# "Displays the velocity - NO SCALE"
def display_velocity_no_depth(velocity_boxes, curr_contour, prev_contour, amount_of_bbox, orig_image):
    tot_box = 0
    coordinates = []

    for c in curr_contour:
        if tot_box == amount_of_bbox:
            break
        x,y,w,h = cv2.boundingRect(c)
        coordinates.append([x,y])
        tot_box += 1
    coordinates = np.array(coordinates)

    for i in range(velocity_boxes.shape[0]):
        x = coordinates[i][0]
        y = coordinates[i][1]
        if i == amount_of_bbox-1:
            break
        cv2.putText(orig_image, "Velocity:"+str(velocity_boxes[i]), (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,0,0))
        print("Object: {}, Speed: {}  ".format(i, velocity_boxes[i]))

def bounding_circle(img_binary, orig_image, radius_threshold):
    contours, hierachy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_bin_ret = 0
    for c in contours:
        # Circles inaddDepthstead of rectangular boxes #
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        if radius < radius_threshold:
            img_bin_ret = orig_image
        else:
            img_bin_ret = cv2.circle(orig_image, center, radius, (0, 0, 255), 2)
            cv2.putText(orig_image, "Object detected", (int(x)+radius,int(y)+radius), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))
    return img_bin_ret, contours

def bounding_box(img_binary, orig_image, keypoints, w_bound, h_bound):
    contours, hierachy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_bin_ret = orig_image
    tot_iterations = 0
    for c in contours:
        # NON-rotating box #
        if(tot_iterations == keypoints):
            break

        x,y,w,h = cv2.boundingRect(c)
        if w < w_bound and h < h_bound: # FIND SOME BETTER THRESHOLDING METHOD
            img_bin_ret = orig_image
        else:
            img_bin_ret = cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(orig_image, "Object detected", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))

        tot_iterations += 1
    return img_bin_ret, contours

def extract_features_dynamic(image,ksize,upper_r,upper_g,upper_b, lower_r, lower_g, lower_b):
    lower_color_bounds = np.array((lower_b,lower_b,lower_r))
    upper_color_bounds = np.array((upper_b,upper_g,upper_r))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(image, lower_color_bounds,upper_color_bounds)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    image = image & mask_rgb

    # Different kernel?
    kernel = np.ones((ksize,ksize), np.uint8)
    #image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.erode(image, kernel, 1)
    image = cv2.dilate(image, kernel, 1)
    return image


def extract_features(image,ksize,color_filter):
    colors = ['B', 'G', 'R']
    if color_filter == colors[2]: # RED
        lower_color_bounds = np.array((0,0,80))
        upper_color_bounds = np.array((45,45,255))
    elif color_filter == colors[1]: # GREEN
        lower_color_bounds = np.array((0,40,0))
        upper_color_bounds = np.array((50,255,50))
    elif color_filter == colors[0]: # BLUE
        lower_color_bounds = np.array((40,0,0))
        upper_color_bounds = np.array((255,50,50))
    else:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(image, lower_color_bounds,upper_color_bounds)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    image = image & mask_rgb

    kernel = np.ones((ksize,ksize), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    #image = cv2.erode(image, kernel, 1)
    #image = cv2.dilate(image, kernel, 1)

    return image

def binary_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(src=gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh

def find_ADELE_Depth(Q_matrix, disparity):
    baseline_ = 1.0/Q_matrix[3][2]
    focal_ = Q_matrix[2][3]
    depth = (focal_ * baseline_)/disparity
    return depth

def open_stereo_camera(cam_index1,cam_index2 ,width, height, delta):
    cap = cv2.VideoCapture(cam_index1) #"IF TEST TO CHECK FI THE CAMERA EXISTS"
    cap2 = cv2.VideoCapture(cam_index2) #"IF TEST TO CHECK IF THE CAMERA EXISTS"
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    scc.getChessBoardImages(cap, cap2)
    retval,camMatrix1, distCoeffs1, camMatrix2, distCoeffs2, R, T, E, F, img_shape = scc.camera_stereoCalibrate(c_size_x=7, c_size_y=7)
    r1,r2,p1,p2, Q = scc.camera_stereoRectify(camMatrix1, camMatrix2, distCoeffs1, distCoeffs2, R, T, img_shape)

    def nothing(x):
        pass
    cv2.namedWindow('Color_filtration', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Upper R','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Upper G','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Upper B','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Lower R','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Lower G','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Lower B','Color_filtration',0,255,nothing)
    switch_rect = 'rectangle ON \nrectangle OFF'
    switch_circ = 'circle ON \ncircle OFF'
    cv2.createTrackbar(switch_circ, 'Color_filtration',0,1,nothing)
    cv2.createTrackbar(switch_rect, 'Color_filtration',0,1,nothing)


    while(True):
        ret, left_Frame = cap.read()
        ret2, right_Frame = cap2.read()

        upper_r = cv2.getTrackbarPos('Upper R', 'Color_filtration')
        upper_g = cv2.getTrackbarPos('Upper G', 'Color_filtration')
        upper_b = cv2.getTrackbarPos('Upper B', 'Color_filtration')
        lower_r = cv2.getTrackbarPos('Lower R', 'Color_filtration')
        lower_g = cv2.getTrackbarPos('Lower G', 'Color_filtration')
        lower_b = cv2.getTrackbarPos('Lower B', 'Color_filtration')
        track_circle = cv2.getTrackbarPos(switch_circ, 'Color_filtration')
        track_rectangle = cv2.getTrackbarPos(switch_rect, 'Color_filtration')
        """
        stuff1 = extract_features_dynamic(image=left_Frame,ksize=3,
                                              upper_r=upper_r, upper_g=upper_g, upper_b=upper_b,
                                              lower_r=lower_r, lower_g=lower_g, lower_b=lower_b)
        stuff2 = extract_features_dynamic(image=right_Frame,ksize=3,
                                              upper_r=upper_r, upper_g=upper_g, upper_b=upper_b,
                                              lower_r=lower_r, lower_g=lower_g, lower_b=lower_b)

        """
        stuff1 = extract_features(image=left_Frame, ksize=3, color_filter='R')
        stuff2 = extract_features(image=right_Frame, ksize=3, color_filter='R')

        threshold_bin1 = binary_threshold(stuff1)
        threshold_bin2 = binary_threshold(stuff2)

        #cv2.imshow("Left-camera", left_Frame)
        #cv2.imshow("Right-camera", right_Frame)
        cv2.imshow("Thresh1", threshold_bin1)
        cv2.imshow("Thresh2", threshold_bin2)

        if track_rectangle == 1:
            bbox, contours_curr = bounding_box(img_binary=threshold_bin1, orig_image=left_Frame, keypoints=5, w_bound=70, h_bound=70)
            bbox2, contours_curr2 = bounding_box(img_binary=threshold_bin2, orig_image=right_Frame, keypoints=10, w_bound=30, h_bound=30)
            cv2.imshow('bounding box1',bbox)
            cv2.imshow('bounding box2',bbox2)

        left_Frame_g = cv2.cvtColor(left_Frame, cv2.COLOR_BGR2GRAY)
        right_Frame_g = cv2.cvtColor(right_Frame, cv2.COLOR_BGR2GRAY)


        stereo = cv2.StereoSGBM_create(numDisparities=32, blockSize=5)
        disparity = stereo.compute(left_Frame_g, right_Frame_g)
        disparity_norm = cv2.normalize(src=disparity, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        points_disp = None
        points_disp = cv2.reprojectImageTo3D(disparity=disparity,_3dImage=points_disp,Q=Q, handleMissingValues=False, ddepth=cv2.CV_32F)
        #depths = find_ADELE_Depth(Q_matrix=Q, disparity=disparity)
        #print(depths)
        cv2.imshow("disp", disparity_norm)
        cv2.imshow("points", points_disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cap2.release()
    cv2.destroyAllWindows() # BENJAMIN HACKED!!!!!

def open_one_camera(cam_index, width, height,delta):
    cap = cv2.VideoCapture(cam_index)
    fps = cap.get(cv2.CAP_PROP_FPS)

    def nothing(x):
        pass
    cv2.namedWindow('Color_filtration', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Upper R','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Upper G','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Upper B','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Lower R','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Lower G','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Lower B','Color_filtration',0,255,nothing)
    switch_rect = 'rectangle ON \nrectangle OFF'
    switch_circ = 'circle ON \ncircle OFF'
    cv2.createTrackbar(switch_circ, 'Color_filtration',0,1,nothing)
    cv2.createTrackbar(switch_rect, 'Color_filtration',0,1,nothing)

    delayed_frame = None
    delayed_bbox = None
    frame_skipped = 0

    curr_contour = None
    prev_contour = None
    while(True):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (width+(3*delta),height+(4*delta)))

        upper_r = cv2.getTrackbarPos('Upper R', 'Color_filtration')
        upper_g = cv2.getTrackbarPos('Upper G', 'Color_filtration')
        upper_b = cv2.getTrackbarPos('Upper B', 'Color_filtration')
        lower_r = cv2.getTrackbarPos('Lower R', 'Color_filtration')
        lower_g = cv2.getTrackbarPos('Lower G', 'Color_filtration')
        lower_b = cv2.getTrackbarPos('Lower B', 'Color_filtration')
        track_circle = cv2.getTrackbarPos(switch_circ, 'Color_filtration')
        track_rectangle = cv2.getTrackbarPos(switch_rect, 'Color_filtration')
        red_stuffs = extract_features_dynamic(image=frame,ksize=3,
                                              upper_r=upper_r, upper_g=upper_g, upper_b=upper_b,
                                              lower_r=lower_r, lower_g=lower_g, lower_b=lower_b)

        threshold_bin = binary_threshold(red_stuffs)
        "MAYBE DO THIS IN A METHOD???"
        if frame_skipped%fps == 0:
            delayed_frame = frame
            bbox_delayed, contours_delayed = bounding_box(img_binary=threshold_bin, orig_image=delayed_frame, keypoints=5, w_bound=70, h_bound=70)
            delayed_bbox = bbox_delayed
            prev_contour = contours_delayed

            cv2.imshow('Bounding box delayed', bbox_delayed)
            frame_skipped = 0
        frame_skipped += 1
        "MAYBE DO THIS IN A METHOD"

        if track_circle == 1:
            bbox, contours = bounding_circle(img_binary=threshold_bin, orig_image=frame, radius_threshold=30)
            #velocity, c_curr, c_prev = measure_velocity_no_depth(prev_contours=prev_contour, curr_contours=curr_contour, amount_of_bbox=5, fps=fps)
            #display_velocity_no_depth(velocity_boxes=velocity, curr_contour=c_curr, prev_contour=c_prev, amount_of_bbox=5, orig_image=frame)
            cv2.imshow('bounding box',bbox)
        if track_rectangle == 1:
            bbox, contours_curr = bounding_box(img_binary=threshold_bin, orig_image=frame, keypoints=5, w_bound=70, h_bound=70)
            curr_contour = contours_curr
            velocity, c_curr, c_prev = measure_velocity_no_depth(prev_contours=prev_contour, curr_contours=curr_contour, amount_of_bbox=5, fps=fps)
            display_velocity_no_depth(velocity_boxes=velocity, curr_contour=c_curr, prev_contour=c_prev, amount_of_bbox=5, orig_image=frame)
            cv2.imshow('bounding box',bbox)

        if track_circle == 0 and track_rectangle == 0:
            cv2.imshow('bounding box',frame)

        cv2.imshow('Binary frame', threshold_bin)
        cv2.imshow('Thresholded frame', red_stuffs)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        first_iteration = True
    cap.release()
    cv2.destroyAllWindows()

def main():
    #open_one_camera(cam_index=0, width=640, height=480, delta=0)
    open_stereo_camera(cam_index1=0, cam_index2=6, width=640, height=480, delta=0)

if __name__ == '__main__':
    main()
