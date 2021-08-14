import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, 1.1, 3)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h) ,(255, 0, 0), 3)

        r1= gray[y:y+h , x:x+w]
        r2= img[ y:y+h , x:x+w]

        eyes=eye.detectMultiScale(r1)

        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(r2, (ex, ey), (ex+ew, ex+eh), (255, 0, 255), 2)

        cv2.imshow('img', img)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
cap.release()
