import cv2
import numpy as np
from scipy.signal import find_peaks


# Function : Smooth signal

def smooth_signal(signal, window_size=10):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')


# Load Face Detection Model

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Open Webcam

camera = cv2.VideoCapture(0)

pulse_values = []
frame_count = 0
result_displayed = False

print("--------------------------------------------------")
print("CONTACTLESS HEART RATE MONITORING SYSTEM")
print("Please sit still in front of the camera.")
print("Ensure good lighting and minimal movement.")
print("Measuring for 20 seconds...")
print("--------------------------------------------------")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Process each detected face
    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]

        # Extract green channel (rPPG signal)
        green_channel = face_region[:, :, 1]
        pulse_values.append(np.mean(green_channel))

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display instruction on screen
        cv2.putText(frame, "Sit Still - Measuring...",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    frame_count += 1
    cv2.imshow("Face Based Heart Rate Monitor", frame)

    # After 400 frames (~20 seconds)
    if frame_count == 400 and not result_displayed:

        filtered_signal = smooth_signal(pulse_values, 10)
        peaks, _ = find_peaks(filtered_signal, distance=8)

        heart_rate = (len(peaks) / 20) * 60

        print("Estimated Heart Rate :", int(heart_rate), "BPM")
        print("Measurement Completed Successfully.")

        result_displayed = True

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
