import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

ptime = 0

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(173,216,230), thickness=2, circle_radius=2),
                            mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (0,475), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
    
    
    cv2.imshow('POSE', img)
    cv2.waitKey(1)