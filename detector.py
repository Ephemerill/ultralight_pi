import cv2
from ultralytics import YOLO
import threading
from datetime import datetime
from flask import Flask, Response

app = Flask(__name__)
camera = cv2.VideoCapture(0)
detected_people = []

#Global variables used for threading
newframe = None
frame = None
lock = threading.Lock()
frame_count = 0
flock = 0


#Frame sizing
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#camera.set(cv2.CAP_PROP_FPS, desired_fps)

#Yolo model (nano)
model = YOLO("yolo11x.pt")
#Image Size for inference (default is 640)
Size = 640
#how many frames in between yolo loop
frame_skip = 15



#Multithreaded frame capturing to relieve cpu stress
def capture_frame():
    global newframe
    global frame_count
    while True:
        #reads a new frame from the camera
        framecheck, frame = camera.read()
        if framecheck == True:
            frame_count += 1
            print(frame_count)
            with lock:
                newframe = frame

#Begin the 2nd thread
capturethread = threading.Thread(target=capture_frame, daemon=True)
capturethread.start()

def draw_frame(frame):
    for (x1, y1, x2, y2) in detected_people:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        


def detection():
    global frame_count
    global flock
    while True:
        #Grabs new frame from threaded frame cap function
        with lock:
            #makes sure the program does not crash if the frame is empty
            if newframe is None:
                continue
            frame = newframe.copy()
        print("Running Detection")
        #read the frame for everything inside of it
        results = model(frame, imgsz=Size, verbose=False)
        #sort out only the bounding boxes of the detected objects from the results
        detected_people.clear()
        for box in results[0].boxes:
            #grabs the ID of the object detected 
            id = int(box.cls[0])
            #nof filter by person, they have an ID of 0
            if id == 0:
                #get coordinates of the boxed detection (human in this case)
                coords = map(int, box.xyxy[0])
                #there is a chance this could crash due to threading issues
                detected_people.append(list(coords))

        
detectionthread = threading.Thread(target=detection, daemon=True)
detectionthread.start()
    

def generate_frames():
    global frame_count
    global flock
    while True:

        #Grabs new frame from threaded frame cap function
        with lock:
            #makes sure the program does not crash if the frame is empty
            if newframe is None:
                continue
            frame = newframe.copy()

        
        #draw boxes on the frame
        draw_frame(frame)

        #encode the frame to JPG for efficiancy
        _, buffer = cv2.imencode('.jpg', frame)
        
        #render the frame to flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')






#Flask routes
@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006)