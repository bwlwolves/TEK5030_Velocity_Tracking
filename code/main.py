import cv2

def main():
    print("Welcome!")
    vidCap = cv2.VideoCapture(0)
    cam = vidCap.open(0)


if __name__ == '__main__':
    main()
