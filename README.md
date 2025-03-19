# Live Object Spotter

Live Object Spotter uses Python 3, OpenCV DNN, and MobileNet SSD to detect 21 object classes in real-time webcam video. Frames are resized to 400px, processed as 300x300 blobs (scale: 1/127.5, mean: 127.5), and annotated with boxes/labels. NumPy and imutils optimize performance. Includes prototxt and caffemodel.

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
```

## Setup and Execution
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jai-nayani/urban-broccoli.git
   cd urban-broccoli
   ```

2. **Verify File Integrity**:
   Confirm that `live_object_spotter.py`, `MobileNetSSD_deploy.prototxt`, and `MobileNetSSD_deploy.caffemodel` are present in the working directory.

3. **Run the Application**:
   - **Windows/Linux**:
     ```bash
     python live_object_spotter.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel
     ```
   - **macOS**: Use `python3` to ensure Python 3.x is invoked:
     ```bash
     python3 live_object_spotter.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel
     ```
   - **Custom Confidence Threshold**: Adjust the default threshold (0.2) for stricter detection:
     ```bash
     python3 live_object_spotter.py --prototxt MobileNetSSD_deploy.prototxt --model MobileNetSSD_deploy.caffemodel --confidence 0.5
     ```

4. **Operation**:
   - The script initializes a VideoStream from the default webcam (index 0).
   - Frames are processed, and detections are overlaid with bounding boxes and labels.
   - Exit by pressing `q` or `Ctrl + C` in the display window.


## Technical Notes
- **Optimization**: Constants like blob size (300x300) and text styling are predefined to minimize runtime overhead.
- **Hardware**: For enhanced performance, compile OpenCV with CUDA support to leverage GPU acceleration.
- **Model**: The included `MobileNetSSD_deploy.caffemodel` is pre-trained, eliminating the need for external downloads.

## License
This project is a derivative of [Surya-Murali’s Real-Time-Object-Detection-With-OpenCV](https://github.com/Surya-Murali/Real-Time-Object-Detection-With-OpenCV), originally released under the MIT License. My enhancements are copyrighted © 2025 Jai Adithya Ram Nayani and distributed under the same MIT License. See `live_object_spotter.py` for full licensing details.

## Acknowledgments
- Credit to Surya-Murali for the foundational MIT-licensed codebase.
- Built with appreciation for the open-source computer vision community.

Explore, modify, and deploy! For issues or contributions, please open a ticket or pull request.
```
