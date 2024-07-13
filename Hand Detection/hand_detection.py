import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)
my_hands = mp.solutions.hands
hands = my_hands.Hands()
mp_draw = mp.solutions.drawing_utils



ctime = 0
ptime = 0


while True:
    success, img = capture.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_lms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                print(id, cx, cy)
                
                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
                
                
            mp_draw.draw_landmarks(img, hand_lms, my_hands.HAND_CONNECTIONS)
            
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    
    cv2.putText(img, str(int(fps)), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
    
    cv2.imshow('Image', img)
    cv2.waitKey(1)