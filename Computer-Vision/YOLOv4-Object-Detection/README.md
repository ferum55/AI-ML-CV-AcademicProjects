# YOLOv4 Object Detection with OpenCV

An object detection project using a pretrained YOLOv4 model through the OpenCV DNN module.

The application performs inference on an input image, filters detections by confidence, applies Non-Maximum Suppression and visualizes detected objects with bounding boxes and class labels.

## Detection Pipeline

The program:

1. Loads the pretrained YOLOv4 network using OpenCV DNN.
2. Loads COCO class names.
3. Reads an input image.
4. Creates a normalized `416 x 416` input blob.
5. Performs a forward pass through the YOLO network.
6. Extracts predicted classes, confidence scores and bounding boxes.
7. Filters predictions using a confidence threshold of `0.5`.
8. Applies Non-Maximum Suppression with a threshold of `0.4`.
9. Draws the final bounding boxes and class labels on the image.

## Technologies

- Python
- OpenCV
- OpenCV DNN
- NumPy
- YOLOv4
- COCO dataset classes

## Project Structure

```text
YOLOv4-Object-Detection/
├── detect.py
├── images/
│   └── test.jpg
└── yolo/
    ├── coco.names
    └── yolov4.cfg
```

## Model Weights

The pretrained `yolov4.weights` file is not stored in this repository because of its size.

Download the pretrained YOLOv4 weights separately and place the file at:

```text
yolo/yolov4.weights
```

The final YOLO directory should contain:

```text
yolo/
├── coco.names
├── yolov4.cfg
└── yolov4.weights
```

## Running the Project

Install the required Python packages:

```bash
pip install opencv-python numpy
```

Run the detector from the project directory:

```bash
python detect.py
```

The default input image is:

```text
images/test.jpg
```

## Notes

The project uses a pretrained YOLOv4 model for inference. The neural network is not trained from scratch in this repository.
