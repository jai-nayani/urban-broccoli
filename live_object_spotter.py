# Copyright (c) 2025 Jai Adithya Ram Nayani
# This work is derived from an open-source project by Surya-Murali, originally shared under the MIT License.
# Permission is granted to anyone to use this software freely, including rights to adapt, share, or distribute it,
# as long as this notice remains intact in all copies or significant portions of the code.
# This software comes with no guarantees—use it at your own risk! The original creators and I aren’t liable for any issues.
# For the full MIT License terms, refer to: https://github.com/Surya-Murali/Real-Time-Object-Detection-With-OpenCV

# Bringing in the essentials for video processing and detection
from imutils.video import VideoStream, FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# Setting up a way to grab user inputs from the command line
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prototxt", required=True, help="location of the Caffe prototxt configuration")
parser.add_argument("-m", "--model", required=True, help="path to the pre-trained Caffe model file")
parser.add_argument("-c", "--confidence", type=float, default=0.2, help="threshold for filtering low-confidence detections")
inputs = vars(parser.parse_args())

# Categories the model can recognize, stored for quick access
OBJECTS = ["aeroplane", "background", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", 
           "sheep", "sofa", "train", "tvmonitor"]

# Fun colors for each object type, generated randomly
SHADES = np.random.uniform(0, 255, size=(len(OBJECTS), 3))

# Loading the detection model—let's get this party started!
print("[STARTUP] Initializing detection engine...")
detector = cv2.dnn.readNetFromCaffe(inputs["prototxt"], inputs["model"])

# Constants for image preprocessing, keeping things snappy
TARGET_SIZE = (300, 300)
SCALE = 1 / 127.5
BASE_VALUE = 127.5

# Kicking off the camera feed and performance tracker
print("[STARTUP] Activating video feed...")
camera = VideoStream(src=0).start()
time.sleep(2.0)  # Giving the camera a moment to wake up
speed_tracker = FPS().start()

# Styling for text overlays, pre-set for efficiency
TEXT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SIZE = 0.5
TEXT_WIDTH = 2

# The main event: capturing and analyzing video frames
while True:
    # Snagging a fresh frame and resizing it for consistency
    frame = camera.read()
    frame = imutils.resize(frame, width=400)
    height, width = frame.shape[:2]

    # Transforming the frame into a format the model loves
    processed_frame = cv2.dnn.blobFromImage(frame, SCALE, TARGET_SIZE, BASE_VALUE, swapRB=True, crop=False)
    detector.setInput(processed_frame)
    results = detector.forward()

    # Digging into the detection results
    for i in range(results.shape[2]):
        certainty = results[0, 0, i, 2]
        if certainty > inputs["confidence"]:
            object_id = int(results[0, 0, i, 1])
            bounds = results[0, 0, i, 3:7] * np.array([width, height, width, height])
            (left, top, right, bottom) = bounds.astype("int")

            # Keeping the box inside the frame—no wild escapes!
            left, top = max(0, left), max(0, top)
            right, bottom = min(width - 1, right), min(height - 1, bottom)

            # Crafting a label with the object name and confidence
            tag = f"{OBJECTS[object_id]}: {certainty * 100:.2f}%"
            print(f"[DETECTED] {tag}")

            # Drawing a colorful box and adding the label
            cv2.rectangle(frame, (left, top), (right, bottom), SHADES[object_id], 2)
            label_y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, tag, (left, label_y), TEXT_STYLE, TEXT_SIZE, SHADES[object_id], TEXT_WIDTH, cv2.LINE_AA)

    # Showing off the latest frame with detections
    cv2.imshow("Live Detection", frame)

    # Ready to quit? Just hit 'q'!
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Keeping tabs on how fast we’re going
    speed_tracker.update()

# Wrapping up: shutting down and sharing some stats
speed_tracker.stop()
print(f"[FINISH] Total Time: {speed_tracker.elapsed():.2f} seconds")
print(f"[FINISH] Speed: {speed_tracker.fps():.2f} frames per second")
cv2.destroyAllWindows()
camera.stop()

# All done—hope you enjoyed the show!