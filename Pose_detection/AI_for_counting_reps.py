import cv2
import mediapipe as mp
import time
import numpy as np

count = 0
refer = None
cap = cv2.VideoCapture(1)
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
ptime = 0
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)


    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(173,216,230), thickness=2, circle_radius=2),
                            mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))

    try:
        landmarks = results.pose_landmarks.landmark
        a = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        b = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        c = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        angle = calculate_angle(a, b, c)
        cv2.putText(img, str(int(angle)), tuple(np.multiply(b, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if angle > 140:
            refer = 'down'
        if angle < 30 and refer == 'down':
            refer = 'up'
            count += 1
            print(count, end=" ")

    except:
        pass

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(img, 'UP', (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(img, str(int(fps)), (0,475), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    cv2.imshow('POSE', img)
    cv2.waitKey(1)