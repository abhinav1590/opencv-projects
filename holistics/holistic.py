import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mp_draw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_tracking_confidence=0.5, min_detection_confidence=0.5)
ptime = 0


def detection(img, model):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    results = model.process(imgRGB)

    return results


def draw_landmarks(img, results):
    # face connections
    mp_draw.draw_landmarks(img, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                           mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))
    # pose connections
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                           mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))
    # left hand connections
    mp_draw.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                           mp_draw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))
    # right hand connections
    mp_draw.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                           mp_draw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))


while True:
    _, img = cap.read()

    results = detection(img, holistic)

    draw_landmarks(img, results)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.imshow('holistic ', img)
    cv2.waitKey(10)

