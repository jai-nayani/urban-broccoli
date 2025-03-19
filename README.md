# urban-broccoli
Live Object Spotter uses Python 3, OpenCV DNN, and MobileNet SSD to detect 21 object classes in real-time webcam video. Frames are resized to 400px, processed as 300x300 blobs (scale: 1/127.5, mean: 127.5), and annotated with boxes/labels. NumPy and imutils optimize performance. Includes prototxt and caffemodel.
