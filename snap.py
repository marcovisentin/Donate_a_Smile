'''
Currently this file is not used by the app
'''

import cv2
import mediapipe as mp
import cvzone
import random
def add_filter(cap):
    #cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    overlay1 = cv2.imread('overlays/sunglass.png', cv2.IMREAD_UNCHANGED)
    overlay2 = cv2.imread('overlays/star.png', cv2.IMREAD_UNCHANGED)

    overlays_filters = []
    overlays_filters.append(overlay1)
    overlays_filters.append(overlay2)
    overlay = random.choice(overlays_filters)

    #face_mesh = mp.solutions.face_mesh.FaceMesh()
    while True:
        success, frame = cap.read()
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray_scale)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            overlay_resize = cv2.resize(overlay, (w, h))
            frame = cvzone.overlayPNG(frame, overlay_resize, [x,y])
        cv2.imshow('Snap Dude', frame)
        if cv2.waitKey(10) == ord('q'):
            break
