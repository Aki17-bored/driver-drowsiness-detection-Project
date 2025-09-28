import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
from datetime import datetime

# Initialize mixer for alarm sound
mixer.init()
alarm_sound = mixer.Sound('alarm.wav')
alarm_playing = False  # flag to check if alarm is already playing

# Haar cascade files for face, left eye, and right eye detection
face_detection = cv2.CascadeClassifier(r'haar cascade files/haarcascade_frontalface_alt.xml')
left_eye_detection = cv2.CascadeClassifier(r'haar cascade files/haarcascade_lefteye_2splits.xml')
right_eye_detection = cv2.CascadeClassifier(r'haar cascade files/haarcascade_righteye_2splits.xml')

# Labels for eye states
labels_text = ['Closed', 'Open']

# Load the trained model
model = load_model('models/custmodel.h5')

path = os.getcwd()

# Open webcam
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    raise IOError("Cannot open webcam")

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Declaring variables
counter = 0
drowsy_time = 0
thick = 2
right_eye_pred = [1]
left_eye_pred = [1]

while True:
    ret, frame = capture.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes
    faces = face_detection.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = left_eye_detection.detectMultiScale(gray)
    right_eye = right_eye_detection.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (100, height), (0, 0, 0), thickness=cv2.FILLED)
    cv2.rectangle(frame, (290, height - 50), (540, height), (0, 0, 0), thickness=cv2.FILLED)

    # Draw bounding boxes on faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # Right eye prediction
    for (x, y, w, h) in right_eye:
        right_one = frame[y:y + h, x:x + w]
        right_one_gray = cv2.cvtColor(right_one, cv2.COLOR_BGR2GRAY)
        right_one_gray = cv2.resize(right_one_gray, (24, 24)) / 255.0
        right_one_gray = right_one_gray.reshape(24, 24, -1)
        right_one_gray = np.expand_dims(right_one_gray, axis=0)
        right_eye_pred = np.argmax(model.predict(right_one_gray), axis=-1)

        label = labels_text[right_eye_pred[0]]
        cv2.putText(frame, label, (x, y - 10), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        break

    # Left eye prediction
    for (x, y, w, h) in left_eye:
        left_one = frame[y:y + h, x:x + w]
        left_one_gray = cv2.cvtColor(left_one, cv2.COLOR_BGR2GRAY)
        left_one_gray = cv2.resize(left_one_gray, (24, 24)) / 255.0
        left_one_gray = left_one_gray.reshape(24, 24, -1)
        left_one_gray = np.expand_dims(left_one_gray, axis=0)
        left_eye_pred = np.argmax(model.predict(left_one_gray), axis=-1)

        label = labels_text[left_eye_pred[0]]
        cv2.putText(frame, label, (x, y - 10), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        break

    # Check drowsiness
    if right_eye_pred[0] == 0 and left_eye_pred[0] == 0:
        drowsy_time += 1
        cv2.putText(frame, "Inactive", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Play alarm continuously
        if not alarm_playing:
            alarm_sound.play(-1)  # -1 loops the sound continuously
            alarm_playing = True

    else:
        drowsy_time -= 1
        cv2.putText(frame, "Active", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Stop alarm if playing
        if alarm_playing:
            alarm_sound.stop()
            alarm_playing = False

    if drowsy_time < 0:
        drowsy_time = 0

    cv2.putText(frame, 'Wake up Time: ' + str(drowsy_time), (300, height - 20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    if drowsy_time > 10:
        # Save frame with timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cv2.imwrite(os.path.join(path, f'drowsy_{timestamp}.jpg'), frame)

        # Draw red rectangle alert
        if thick < 16:
            thick += 2
        else:
            thick -= 2
            if thick < 2:
                thick = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thick)

    cv2.imshow('Driver Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop alarm before exiting
if alarm_playing:
    alarm_sound.stop()

capture.release()
cv2.destroyAllWindows()