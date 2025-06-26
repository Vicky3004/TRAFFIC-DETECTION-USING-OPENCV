from flask import Flask, render_template, Response
import cv2
import imutils
import datetime
import numpy as np

app = Flask(__name__)

# Load car detection cascade
cascade_src = r'cars.xml'
car_cascade = cv2.CascadeClassifier(cascade_src)

# Initialize camera
cam = cv2.VideoCapture(0)  # Use 0 for the default webcam

def detect_cars():
    """Generates frames with car detection, time, and vehicle counting"""
    while True:
        ret, img = cam.read()
        if not ret:
            break

        img = imutils.resize(img, width=1000)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

        # Vehicle count
        car_count = len(cars)

        # Traffic condition
        traffic_status = "No Traffic" if car_count < 5 else "Traffic"

        # Create a black background panel for information display
        info_panel = np.zeros((img.shape[0], 400, 3), dtype=np.uint8)
        info_panel[:] = (30, 30, 30)  # Dark grey background

        # Display time, count, and traffic status on the info panel
        time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(info_panel, f"VEHICLE STATUS", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_panel, f"Time: {time_now}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_panel, f"Cars: {car_count}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(info_panel, f"Traffic: {traffic_status}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw rectangles around detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Combine the two images side by side (video feed + info panel)
        combined_img = np.hstack((img, info_panel))

        _, buffer = cv2.imencode('.jpg', combined_img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_cars(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)