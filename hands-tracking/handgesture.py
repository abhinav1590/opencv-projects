import cv2
import mediapipe as mp
import time
import numpy as np


cap = cv2.VideoCapture(0)
mhands = mp.solutions.hands
hands = mhands.Hands(min_tracking_confidence=0.8, min_detection_confidence=0.8)
mdraw = mp.solutions.drawing_utils
ptime = 0
state = None


def detection(img, model):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.process(imgRGB)

    return results


def draw_landmarks(img, results):

    if results.multi_hand_landmarks:

        for lmhands in results.multi_hand_landmarks:

            mdraw.draw_landmarks(img, lmhands, mhands.HAND_CONNECTIONS,
                            mdraw.DrawingSpec((0, 0, 0), thickness=2, circle_radius=2),
                            mdraw.DrawingSpec((255, 255, 255), thickness=1))


def find_position(results):

    lmlist = []
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[0]

        for id, lm in enumerate(myHand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmlist.append([id, cx, cy])

    return lmlist


def cal_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)


    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return int(angle)


def fingers_up_check(lmlist):

    finger_check = []
    angle = []
    count = 5
    position =[]
    for i in range(4):
        counter = 1
        if count != 20:
            while counter != 4:
                pos = [lmlist[count][1], lmlist[count][2]]
                position.append(pos)
                count += 1
                counter += 1
            count += 1
            angle.append(cal_angle(position[0],position[1],position[2]))
            position.clear()

    for j in range(len(angle)):
        if angle[j] < 50:
            finger_check.append(1)
        else:
            finger_check.append(0)

    return finger_check


def gesture(finger_up):

    c1 = 0
    c2 = 0
    if len(finger_up) != 0:
        for i in finger_up:
            if i == 1:
                c1 += 1
            elif i == 0:
                c2 += 1

        if  c1!=0 and c1 == c2:
            return 1
        elif c1 == 4:
            return 2


while True:
    success, img = cap.read()
    if success == False:
        state = None
    results = detection(img, hands)
    draw_landmarks(img, results)
    lmlist = find_position(results)

    if len(lmlist) != 0:

        finger_up = fingers_up_check(lmlist)

        if gesture(finger_up) == 1:
            state = 'YOO'
        elif gesture(finger_up) == 2:
            state = 'FIST'
        else:
            state = None

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img,f'Gesture - { state } ', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255 ,0), 3)
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 1, (0, 255, 255), 1)
    cv2.imshow('Hand Gestures', img)
    cv2.waitKey(1)
