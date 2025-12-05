import cv2
from ultralytics import YOLO
import threading
from flask import Flask, Response

app = Flask(__name__)
camera = cv2.VideoCapture(0)

#Global variables used for threading
newframe = None
lock = threading.Lock()

#Yolo model (nano)
model = YOLO("yolo11n.pt")
#Image Size for inference (default is 640)
Size = 320


#Multithreaded frame capturing to relieve cpu stress
def capture_frame():
    global newframe
    while True:
        #reads a new frame from the camera
        framecheck, frame = camera.read()
        if framecheck == True:
            with lock:
                newframe = frame

#Begin the 2nd thread
capturethread = threading.Thread(target=capture_frame, daemon=True)
capturethread.start()



def detection(frame):
    #read the frame for everything inside of it
    results = model(frame, imgsz=Size, verbose=False)
    #sort out only the bounding boxes of the detected objects from the results
    for box in results[0].boxes:
        #grabs the ID of the object detected 
        id = int(box.cls[0])
        #nof filter by person, they have an ID of 0
        if id == 0:
            #get coordinates of the boxed detection (human in this case)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            #append the video output with the new bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #Label the Box
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        

    

def generate_frames():
    while True:
        # Read camera and encode to JPG without error checking, The check returns two values so we discard the first one
        with lock:
            #makes sure the program does not crash if the frame is empty
            if newframe is None:
                continue
            frame = newframe.copy()
        #run the detector
        detection(frame)
        #encode the frame to JPG for efficiancy
        _, buffer = cv2.imencode('.jpg', frame)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')



















@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006)