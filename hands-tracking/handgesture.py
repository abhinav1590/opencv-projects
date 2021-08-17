import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mhands = mp.solutions.hands
hands = mhands.Hands(min_tracking_confidence=0.8, min_detection_confidence=0.8)
mdraw = mp.solutions.drawing_utils
ptime = 0
state = None
while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for lmhands in results.multi_hand_landmarks:
            mdraw.draw_landmarks(img, lmhands, mhands.HAND_CONNECTIONS,
                                 mdraw.DrawingSpec((0, 0, 0), thickness=2, circle_radius=2),
                                 mdraw.DrawingSpec((255, 255, 255), thickness=1))
    lmlist=[]
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[0]
        for id, lm in enumerate(myHand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmlist.append([id, cx, cy])
    if len(lmlist) != 0:
        list1 = [lmlist[12][1], lmlist[12][2]]
        list2 = [lmlist[16][1], lmlist[16][2]]
        list3 = [lmlist[11][1], lmlist[11][2]]
        list4 = [lmlist[15][1], lmlist[15][2]]
        if list1 < list3 and list2 < list4:
            state = "spider man"
        else:
            state = None
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, state, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 1, (0, 255, 255), 1)
    cv2.putText(img, state, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (204, 255, 229), 4)
    cv2.imshow('Hand markings ', img)
    cv2.waitKey(1)
