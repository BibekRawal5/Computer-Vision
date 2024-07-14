import mediapipe as mp
import cv2
import time

class Hand_Detector():
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detect_min=0.5, track_min=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.track_min = track_min
        self.detect_min = detect_min
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity, self.track_min, self.detect_min)
        self.mp_draw = mp.solutions.drawing_utils

    
    def detect_hands(self, img, draw= True):
        """_summary_

        Args:
            img (list/np.array): image array
            draw (bool, optional): to draw hand detected or not. Defaults to True.

        Returns:
            img (list/np.array): image with treacked hand marks drawn.
        """
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgrgb)
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                    if draw:
                        self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def get_lm(self, img, hand_no=0, draw= True):
        """_summary_

        Args:
            img (list/np.array): image array
            hand_no (int, optional): which hand to get position of. 0 -> New hand, 1-> old hand. Defaults to 0.
            draw (bool, optional): to draw on the hand or not. Defaults to True.

        Returns:
            position (list): positions of asked hand points
        """
        
        
        positions = []
        my_hand = self.results.multi_hand_landmarks
        if self.results.multi_hand_landmarks:
            if len(my_hand) >= hand_no + 1:
                for id, lm in enumerate(my_hand[hand_no].landmark):
                    h, w, c = img.shape
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    positions.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        return positions
    
def main():
    capture = cv2.VideoCapture(0)
    ctime = 0
    ptime = 0
    detector = Hand_Detector()

    while True:
        success, img = capture.read()
        
        img = detector.detect_hands(img)
        lm_list = detector.get_lm(img, 1)
        if len(lm_list) != 0:
            print(lm_list[4])  
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        
        cv2.putText(img, str(int(fps)), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
        
        cv2.imshow('Image', img)
        cv2.waitKey(1)
    

if __name__ == '__main__':
    main()