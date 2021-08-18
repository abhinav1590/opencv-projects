import cv2
import mediapipe as mp
import time
import numpy as np


cap = cv2.VideoCapture(0)
mhands = mp.solutions.hands
hands = mhands.Hands(min_tracking_confidence=0.7, min_detection_confidence=0.7)
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


def spider_man(lmlist):

    angle = []
    count = 4
    for i in range(5):
        if count != 20 and i != 4:
            count += 1
            a = [lmlist[count][1], lmlist[count][2]]
            count += 1
            b = [lmlist[count][1], lmlist[count][2]]
            count += 1
            c = [lmlist[count][1], lmlist[count][2]]
            count += 1
            angle.append(cal_angle(c, b, a))
    if (angle[0] and angle[1]) < 50 and (angle[2] and angle[3]) > 60:
        return True
    else:
        return False


def fist(lmlist):

    angle = []
    count = 4
    for i in range(5):
        if count != 20 and i != 4:
            count += 1
            a = [lmlist[count][1], lmlist[count][2]]
            count += 1
            b = [lmlist[count][1], lmlist[count][2]]
            count += 1
            c = [lmlist[count][1], lmlist[count][2]]
            count += 1
            angle.append(cal_angle(c, b, a))

    x = 0
    for j in angle:
        if j <= 50:
            x += 1
    if x == 4:
        return True


while True:
    _, img = cap.read()
    results = detection(img, hands)
    draw_landmarks(img, results)
    lmlist = find_position(results)

    if len(lmlist) != 0:

        if spider_man(lmlist):
            state = "SPIDER MAN"

        elif fist(lmlist):
            state = "FIST"

        else:
            state = None

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, state, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 1, (0, 255, 255), 1)
    cv2.imshow('Hand Gestures', img)
    cv2.waitKey(1)