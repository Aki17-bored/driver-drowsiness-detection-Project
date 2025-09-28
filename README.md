🚗 Driver Drowsiness Detection System

This project is a real-time driver drowsiness detection system built using Python, OpenCV, Keras, and TensorFlow.
The system monitors the driver’s eyes through a webcam feed and alerts when drowsiness is detected.


---

🔹 Features

Detects driver’s face and eyes using Haar Cascade Classifiers.

Classifies eye state (Open / Closed) using a pre-trained Keras model.

Triggers an alarm sound when both eyes are closed beyond a threshold (sign of drowsiness).

Saves snapshots with timestamps whenever drowsiness is detected.

Lightweight, real-time detection with webcam support.



---

🔹 Tech Stack

Python 3.7+ (tested on Python 3.13)

OpenCV – for image processing and face/eye detection.

TensorFlow / Keras – deep learning model for eye state classification.

NumPy – array and image handling.

Pygame – for playing alarm sound.



---

🔹 Folder Structure

driver-drowsiness-detection-Project/
│
├── haar cascade files/               # Haarcascade XML files
│   ├── haarcascade_frontalface_alt.xml
│   ├── haarcascade_lefteye_2splits.xml
│   └── haarcascade_righteye_2splits.xml
│
├── models/                           # Pre-trained deep learning model
│   └── custmodel.h5
│
├── alarm.wav                         # Alarm sound file
├── my_drowsiness_detection.py        # Main script
└── README.md                         # Project description


---

🔹 How It Works

1. The webcam captures video frames.


2. Haar cascades detect the face and eyes.


3. The Keras model classifies whether eyes are Open or Closed.


4. If both eyes remain Closed for more than 10 frames:

The screen shows a red alert.

An alarm sound plays.

A snapshot is saved with a timestamp.





---

🔹 Usage

1. Clone or download the project.


2. Install required dependencies:

pip install opencv-python pygame numpy keras tensorflow


3. Run the script:

python my_drowsiness_detection.py


4. Press q to quit.




---

🔹 Applications

Prevents road accidents by alerting drowsy drivers.

Can be extended to fleet management and long-distance trucking.

Useful for real-time monitoring systems.



---