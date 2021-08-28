import cv2
import mediapipe as mp
import numpy as np
import time

cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

ptime = 0
fps = 0
drawingspecs = mp_drawing.DrawingSpec((255, 255, 255),1, 1)


def detection(img, model):

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.process(imgRGB)

    return results


def draw_landmarks(img, result1,result2):
    if result1.detections:
        for id, detection in enumerate(result1.detections):
            mp_drawing.draw_detection(img, detection, drawingspecs)

            bbox = detection.location_data.relative_bounding_box
            h,w,c = img.shape
            boundbox =[ int(bbox.xmin * w),int(bbox.ymin* h), int(bbox.width * w),int(bbox.height * h)]

            cv2.putText(img, f'{int(detection.score[0]*100)}%',(boundbox[0],boundbox[1]-20)
                        ,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    if result2.multi_face_landmarks:
        for face in result2.multi_face_landmarks:
            mp_drawing.draw_landmarks(img, face, mp_face_mesh.FACE_CONNECTIONS, drawingspecs)


with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detections:
    with mp_face_mesh.FaceMesh(min_tracking_confidence=0.8, min_detection_confidence=0.6) as face_mesh:

        while cap.isOpened():
            _, img = cap.read()
            img = cv2.flip(img,1)
            result1 = detection(img, face_detections)
            result2 = detection(img, face_mesh)

            draw_landmarks(img, result1, result2)

            ctime = time.time()
            fps = int(1/(ctime - ptime))
            ptime = ctime

            cv2.putText(img, str(fps),(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2)
            cv2.imshow("FACE ",img)
            cv2.waitKey(1)

cap.release()
cap.destroyAllWindows()
