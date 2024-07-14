import hand_detection_module as hdm
import cv2
import math
import time
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

vol = volume.GetMasterVolumeLevel()

cap = cv2.VideoCapture(0)
ctime = 0
ptime = 0

detector = hdm.Hand_Detector(detect_min=0.7)

while True:
    ret, img = cap.read()
    img = detector.detect_hands(img, draw=False)
    landmarks_list = detector.get_lm(img, draw= False)
    cv2.rectangle(img, (500, 250), (520, 350), (255, 0, 0), 3)

    if len(landmarks_list) != 0:
        # print(landmarks_list)     
        thumb_x, thumb_y= landmarks_list[4][1], landmarks_list[4][2]
        index_x, index_y = landmarks_list[8][1], landmarks_list[8][2]
        
        dist = math.dist((thumb_x, thumb_y), (index_x, index_y))
        cx, cy = abs(thumb_x + index_x)//2, abs(index_y + thumb_y)//2
        print(dist)

        cv2.circle(img, (thumb_x, thumb_y), 8, (255,0,255), cv2.FILLED)
        cv2.circle(img, (index_x, index_y), 8, (255,0,255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 8, (255,0,255), cv2.FILLED)

        cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 5)
        
        if dist < 13:
            cv2.circle(img, (cx, cy), 8, (0,255,0), cv2.FILLED)

        vol = np.interp(dist, [5, 130], [-65.25, 0.0])
        volume.SetMasterVolumeLevel(vol, None)
        # [7, 125] [-65.25, 0.0]
    
    vol_bar = int(np.interp(vol, [-65.25, 0.0], [350, 250]))
    
    cv2.rectangle(img, (500, vol_bar), (520, 350), (255, 0, 0), cv2.FILLED)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 2)
    
    cv2.imshow("Video", img)
    cv2.waitKey(1)
