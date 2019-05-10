import matplotlib as pyplot
import cv2

cap = cv2.VideoCapture(4)
cap2 = cv2.VideoCapture(6)
i = 0
while (True):
    ret, left_Frame = cap.read()
    ret2, right_Frame = cap2.read()
    #print((left_Frame == right_Frame))
    #print((left_Frame[:,:,0] == left_Frame[:,:,2]))
    left_Frame = cv2.cvtColor(left_Frame, cv2.COLOR_BGR2GRAY)
    right_Frame = cv2.cvtColor(right_Frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Left-camera", left_Frame)
    cv2.imshow("Right-camera", right_Frame)
    k = cv2.waitKey(1)
    if k == 32:
        print("WTF")
        cv2.imwrite('images/left_'+str(i)+'.png', left_Frame)
        cv2.imwrite('images/right_'+str(i)+'.png', right_Frame)
        i += 1
    elif k == 27:
        break
