# Live Object Spotter

Live Object Spotter is an advanced real-time object detection system leveraging cutting-edge computer vision technologies. This project utilizes OpenCV’s Deep Neural Network (DNN) module, powered by the MobileNet Single Shot Detector (SSD) model, to perform efficient object detection on live video streams from a webcam. Optimized for performance and customized with a unique implementation, this repository offers a robust platform for developers and researchers to explore object recognition in dynamic environments.

## Technical Overview
The system processes video frames at a resolution of 400 pixels wide, rescaled to 300x300 for inference, using the MobileNet SSD architecture—a lightweight convolutional neural network (CNN) designed for mobile and embedded vision applications. The tech stack includes:

- **Python 3.x**: The core programming language, providing a flexible and extensible environment for script execution.
- **OpenCV 4.x**: Employs the DNN module for model inference, alongside image processing utilities for frame manipulation and visualization.
- **NumPy**: Handles efficient array operations for bounding box calculations and color generation.
- **imutils**: Simplifies video stream management and frame resizing, enhancing real-time performance.
- **MobileNet SSD**: A pre-trained Caffe model with 21 object classes (including "person," "car," and "dog"), balancing speed and accuracy via depthwise separable convolutions.

The script preprocesses frames into blobs with a scaling factor of 1/127.5 and mean subtraction of 127.5, feeding them into the SSD model for inference. Detected objects are annotated with bounding boxes and confidence scores, rendered with anti-aliased text overlays for clarity.

## Files in This Repository
- **`live_object_spotter.py`**: The primary script implementing the detection pipeline, featuring optimized preprocessing and a custom visualization layer.
- **`MobileNetSSD_deploy.prototxt`**: Defines the MobileNet SSD network architecture, specifying convolutional layers, prior boxes, and detection outputs.
- **`MobileNetSSD_deploy.caffemodel`**: Pre-trained weights (23MB) for the SSD model, enabling immediate inference without retraining.

## Prerequisites
Ensure the following dependencies are installed:
- **Python 3.x**: Required for script execution (verify with `python3 --version`).
- **OpenCV**: Install via `pip install opencv-python` (ensure DNN support is included).
- **NumPy**: Install via `pip install numpy` for numerical computations.
- **imutils**: Install via `pip install imutils` for video stream utilities.

Install all dependencies in a single command:
```bash
pip install opencv-python numpy imutils