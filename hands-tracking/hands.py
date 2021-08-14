import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mhands = mp.solutions.hands
hands = mhands.Hands()
mdraw = mp.solutions.drawing_utils

ptime = 0


while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:

        for handslm in results.multi_hand_landmarks:
            print(results.multi_hand_landmarks)
            mdraw.draw_landmarks(img, handslm, mhands.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 1, (0, 255, 255), 1)

    cv2.imshow('Hand markings ', img)
    cv2.waitKey(1)
