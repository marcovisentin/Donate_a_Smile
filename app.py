from flask import Flask,request,render_template, Response
import cv2
import cvzone

app = Flask(__name__)

camera = cap = cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            overlay = cv2.imread('overlays/sunglass.png', cv2.IMREAD_UNCHANGED)
            gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray_scale)

            # given the location and width/height of faces, resize the overlay filters
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                overlay_resize = cv2.resize(overlay, (w, h))
                frame = cvzone.overlayPNG(frame, overlay_resize, [x,y])
            
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
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
 

@app.route('/', methods=['GET'])
def home():
    if request.method == 'GET':
        return render_template('/home.html')

 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True,debug=True)
