import cv2
import mediapipe as mp
import cvzone

def detect_smile(gray_frame, faces, cascade_smile):
    smiles = []
    for (x, y, w, h) in faces:
        the_face = gray_frame[y:y+h, x:x+w] # get face bounding box
        smiles = cascade_smile.detectMultiScale(the_face,scaleFactor=2, minNeighbors=35) # detect smile
        for (x_, y_, w_, h_) in smiles:
            cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (0,255,0), 2) 
    return len(smiles) > 0

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
cap = cv2.VideoCapture(1) 

while True:
    successful_frame_read, frame = cap.read()
    if not successful_frame_read:
        break
    
    # Preprocess frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect face/faces
    faces = cascade_face.detectMultiScale(gray_frame, minSize=(256,256)) 
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        the_face = frame[y:y+h, x:x+w] # get face bounding box
        smiles = cascade_smile.detectMultiScale(the_face,scaleFactor=2, minNeighbors=30) # detect smile
        for (x_, y_, w_, h_) in smiles:
            cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (0,255,0), 2) 

    cv2.imshow('Snap Dude', frame)
    if cv2.waitKey(10) == ord('q'):
        break
    
    