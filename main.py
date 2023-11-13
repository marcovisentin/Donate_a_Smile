import cv2
import mediapipe as mp
import cvzone

from utils import detect_smile

# Initialise variables and models
smile_time_threshold = 1 # how much time before the filter is shown
count_smile_time = 0 # time spent smiling 
waiting_for_smile = True 
smile_memory = [] # keeps track of the smile/no smile for last three frames.
length_smile_memory = 3 

cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')
overlay = cv2.imread('overlays/sunglass.png', cv2.IMREAD_UNCHANGED)
#face_mesh = mp.solutions.face_mesh.FaceMesh()

# Start camera
cap = cv2.VideoCapture(0) 

while True:
    successful_frame_read, frame = cap.read()
    if not successful_frame_read:
        break
    
    # Preprocess frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect face/faces
    faces = cascade_face.detectMultiScale(gray_frame, minSize=(256,256)) 
    if waiting_for_smile:
        # Check if smiling
        is_smiling = detect_smile(gray_frame, faces, cascade_smile) 
        print(is_smiling)
        smile_memory = (smile_memory + [is_smiling])[-length_smile_memory:]
        # set time counted to zero if in non of the last three frames there was a smile
        count_smile_time = count_smile_time + 1 if any(smile_memory) else 0 
        if count_smile_time == smile_time_threshold * cap.get(cv2.CAP_PROP_FPS):
            waiting_for_smile = False
    else:
        for (x, y, w, h) in faces:
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            overlay_resize = cv2.resize(overlay, (w, h))
            frame = cvzone.overlayPNG(frame, overlay_resize, [x,y])
            frame = cvzone.overlayPNG(frame, )
            
    cv2.imshow('Snap Dude', frame)
    if cv2.waitKey(10) == ord('q'):
        break
    
    