# OpenCV Computer Vision Coursework

A collection of practical computer vision exercises implemented in Python using OpenCV and NumPy.

The projects progress from basic image manipulation to segmentation, feature extraction, feature matching and real-time video analysis.

## Projects

| Project | Topics |
|---|---|
| [01 – Image Basics](./01-Image-Basics) | Image loading and saving, image dimensions, grayscale conversion and visualization |
| [02 – Color Spaces and Segmentation](./02-Color-Spaces-and-Segmentation) | Pixel manipulation, BGR to HSV conversion, color masks and color-based segmentation |
| [03 – Filtering, Edges and Morphology](./03-Filtering-Edges-Morphology) | Box, Gaussian and median filtering, Sobel, Laplacian, Canny, contours, thresholding and morphology |
| [04 – Segmentation and Features](./04-Segmentation-and-Features) | Simple, adaptive and Otsu thresholding, K-means segmentation, Harris corners, ORB descriptors and feature matching |
| [05 – Video and Motion Analysis](./05-Video-and-Motion-Analysis) | Webcam capture, frame preprocessing, ROI processing, motion detection, contours and video recording |

## 01 – Image Basics

Introduces basic OpenCV image operations:

- Loading images with `cv2.imread`
- Reading image dimensions and channel count
- Saving images with `cv2.imwrite`
- BGR to grayscale conversion
- Image visualization

## 02 – Color Spaces and Segmentation

Explores pixel-level operations and color-based segmentation:

- Reading and modifying individual pixels and image regions
- BGR to HSV conversion
- HSV range definition
- Binary color masks
- Color-based object extraction

The implementation isolates red regions using two HSV hue ranges.

## 03 – Filtering, Edges and Morphology

Compares several classical image-processing techniques:

- Box filtering
- Gaussian filtering
- Median filtering
- Different Gaussian kernel sizes
- Sobel gradients
- Laplacian edge detection
- Canny edge detection
- Contour extraction
- Otsu thresholding
- Adaptive thresholding
- Erosion and dilation
- Opening and closing
- Morphological gradient
- Binary noise reduction

## 04 – Segmentation and Feature Matching

Covers segmentation and local feature techniques:

- Global thresholding
- Adaptive thresholding
- Otsu thresholding
- K-means color segmentation
- Harris corner detection
- ORB keypoint detection and descriptors
- Brute Force feature matching
- Hamming distance for binary descriptors

The project matches ORB descriptors between two input images and visualizes the strongest correspondences.

## 05 – Video and Motion Analysis

Processes a live webcam stream and detects motion inside a Region of Interest.

The pipeline includes:

- Webcam capture with OpenCV
- Grayscale conversion
- Gaussian preprocessing
- Region of Interest selection
- Frame differencing
- Binary motion masks
- Morphological noise removal
- Contour detection
- Area-based motion filtering
- Bounding-box visualization
- Processed video recording

## Technologies

- Python
- OpenCV
- NumPy

## Running the Projects

Install the dependencies:

```bash
pip install opencv-python numpy
```

Each project contains its own `main.py`.

Run the required project from its directory, for example:

```bash
cd 04-Segmentation-and-Features
python main.py
```

Some projects require the image files stored in their corresponding directories.

`05-Video-and-Motion-Analysis` requires access to a webcam.

## Purpose

These exercises cover fundamental classical computer vision techniques and provide the foundation for the more advanced object detection, Structure-from-Motion and visual localization projects included in this repository.
