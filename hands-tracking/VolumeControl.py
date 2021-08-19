import cv2
import mediapipe as mp
import time
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math

cap = cv2.VideoCapture(0)
mhands = mp.solutions.hands
hands = mhands.Hands(min_tracking_confidence=0.7, min_detection_confidence=0.7)
mdraw = mp.solutions.drawing_utils
ptime = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volrange = volume.GetVolumeRange()

minvol = volrange[0]
maxvol = volrange[1]
volBar = 400
volPer = 0


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


while True:
    _, img = cap.read()
    results = detection(img, hands)
    draw_landmarks(img, results)
    lmlist = find_position(results)

    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1], lmlist[8][2]
        x2, y2 = lmlist[4][1], lmlist[4][2]

        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        length = math.hypot(x2 - x1, y2 - y1)

        vol = np.interp(length, [50, 250], [minvol, maxvol])
        volBar = np.interp(length, [50, 250], [400, 150])
        volPer = np.interp(length, [50, 250], [0, 100])
        print(int(length), int(vol))
        volume.SetMasterVolumeLevel(vol, None)

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 1, (0, 255, 255), 1)
    cv2.imshow('Volume Control', img)
    cv2.waitKey(1)
