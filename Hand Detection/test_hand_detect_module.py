import cv2
import mediapipe as mp
import time
import hand_detection_module as hdm

capture = cv2.VideoCapture(0)
ctime = 0
ptime = 0
detector = hdm.Hand_Detector()

while True:
    success, img = capture.read()
    
    img = detector.detect_hands(img)
    pos_list = detector.get_pos(img, 1)
    if len(pos_list) != 0:
        print(pos_list[4])  
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    
    cv2.putText(img, str(int(fps)), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
    
    cv2.imshow('Image', img)
    cv2.waitKey(1)


