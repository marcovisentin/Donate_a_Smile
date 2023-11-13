from flask import Flask,request,render_template, Response
import cv2
import cvzone
from utils import detect_smile

app = Flask(__name__)

# TODO: might be not the best postion to initialise variables. Ask Xiaocheng
# Models
cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')
# Filter
overlay = cv2.imread('overlays/rayban2.png', cv2.IMREAD_UNCHANGED)
logo_overlay = cv2.imread('overlays/rayban_logo.png', cv2.IMREAD_UNCHANGED)
reward_overlay = cv2.imread('overlays/you_donated_a_smile.png', cv2.IMREAD_UNCHANGED)

camera = cap = cv2.VideoCapture(1)

def show_reward(reward_overlay, frame, x, y, w, h, relative_y = 0.6):
    h_, w_, d_ = reward_overlay.shape
    reward_overlay_new_shape = (w, int((h_/w_)*int(w)))
    reward_overlay_resize = cv2.resize(reward_overlay, reward_overlay_new_shape)
    frame = cvzone.overlayPNG(frame, reward_overlay_resize, [x, y-int(relative_y*h)])
    
    
def show_rayban_filter(logo_overlay, overlay, frame, x, y, w, h, relative_y = 0.07, logo_width = 256):
    h_, w_, d_ = overlay.shape
    h__, w__, d__ = logo_overlay.shape
    # define shape
    overlay_new_shape = (w, int((h_/w_) * int(w)))
    logo_overlay_new_shape = (logo_width, int((h__/w__) * logo_width))
    #reshape
    overlay_resize = cv2.resize(overlay, overlay_new_shape)
    logo_overlay_resize = cv2.resize(logo_overlay, logo_overlay_new_shape)
    #add
    frame = cvzone.overlayPNG(frame, overlay_resize, [x, y - int(relative_y * h)])
    frame = cvzone.overlayPNG(frame, logo_overlay_resize, [20,20])
    
    
def gen_frames(cascade_face, cascade_smile):  # generate frame by frame from camera
    # Initialise
    smile_time_threshold = 1 # how much time before the filter is shown
    filter_time_threshold = 10
    reward_time_threshold = 3
    count_smile_time = 0 # time spent smiling 
    count_filter_time = 0 # time spent with the filter on
    waiting_for_smile = True 
    smile_memory = [] # keeps track of the smile/no smile for last three frames.
    length_smile_memory = 3 
    fps = 30 #TODO: change 
    
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
            smile_memory = (smile_memory + [is_smiling])[-length_smile_memory:]
            # set time counted to zero if in non of the last three frames there was a smile
            count_smile_time = count_smile_time + 1 if any(smile_memory) else 0 
            if count_smile_time == smile_time_threshold * fps:
                waiting_for_smile = False
                count_smile_time = 0
        else:
            for (x, y, w, h) in faces:
                #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                show_rayban_filter(logo_overlay, overlay, frame, x, y, w, h)
                if count_filter_time > reward_time_threshold * fps:
                    show_reward(reward_overlay, frame, x, y, w, h)
                count_filter_time += 1
                if count_filter_time == filter_time_threshold * fps:
                    waiting_for_smile = True
                    count_filter_time = 0
                
        # display the resulting frame in browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #cv2.imshow('Snap Dude', frame)
        if cv2.waitKey(10) == ord('q'):
            break

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(cascade_face, cascade_smile), mimetype='multipart/x-mixed-replace; boundary=frame')
 

@app.route('/', methods=['GET'])
def home():
    if request.method == 'GET':
        return render_template('/home.html')

 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True,debug=True)
