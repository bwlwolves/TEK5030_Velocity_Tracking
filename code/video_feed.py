import numpy as np
import cv2 as cv2

# "Displays the velocity "
def display_velocity(curr_velocity, curr_frame):
    """
    param: curr_velocity - scalar
           curr_frame = math.shape --> Current frame
    """
    pass
    return None

# "Bounding-Box"
def bounding_box(object, threshold=0.5):
    """
    param: object - object-detection
           threshold - scalar, determines the strictness of tracking.
    """
    
    pass
    return None

# "Measures the velocity"
def measure_velocity(frame_1, frame_2, method):
    """
    param: frame_1 - math.shape
           frame_2 - math.shape
    """

    pass
    return None

# "Necessary preprocessing of video-frames"
def preprocess_frames(frame_1, frame_2, frame_list):
    """
    param: frame_1 - math.shape
           frame_2 - math.shape
           frame_list - np.array()
    """

    pass
    return None

"Getting camera-feed"
def open_camera(cam_index, width, height,delta):
    cap = cv2.VideoCapture(cam_index)
    curr_frame = None
    ref_frame = None
    while(True):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (width+(3*delta),height+(4*delta)))
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    open_camera(cam_index=0, width=640, height=480, delta=0)

if __name__ == '__main__':
    main()
