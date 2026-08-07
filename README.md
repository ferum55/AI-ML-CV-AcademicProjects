# AI, Machine Learning & Computer Vision Projects

A collection of academic projects focused on artificial intelligence, deep learning, generative AI and computer vision.

The repository includes practical work with RAG pipelines, recurrent neural networks, visual localization, object detection and classical computer vision algorithms.

## Projects

| Project | Description | Technologies |
|---|---|---|
| [RAG Pipeline](./Generative-AI/RAG-Pipeline) | Local Retrieval-Augmented Generation pipeline for question answering over PDF documents using embeddings, semantic retrieval and LLM-based generation. | Python, RAG, LLMs, Embeddings |
| [PyTorch LSTM Music Generation](./Deep-Learning/PyTorch-RNN-Music-Generation) | Recurrent neural network trained to generate music sequences in ABC notation, including experiments with model and training hyperparameters. | Python, PyTorch, RNN |
| [HLOC Structure from Motion](./Computer-Vision/HLOC-Structure-from-Motion) | Structure-from-Motion and visual localization pipeline using a custom indoor image dataset and different local feature configurations. | Python, HLOC, SuperPoint, DISK |
| [YOLOv4 Object Detection](./Computer-Vision/YOLOv4-Object-Detection) | Object detection using a pretrained YOLOv4 model with OpenCV DNN, including preprocessing, inference and Non-Maximum Suppression. | Python, OpenCV, YOLOv4 |
| [OpenCV Coursework](./Computer-Vision/OpenCV) | Practical computer vision tasks covering image processing, segmentation, feature extraction, matching and video analysis. | Python, OpenCV, NumPy |

## Computer Vision Coursework

The OpenCV section contains a progression of practical computer vision tasks:

| Project | Topics |
|---|---|
| [Image Basics](./Computer-Vision/OpenCV/01-Image-Basics) | Image loading, saving, grayscale conversion and pixel operations |
| [Color Spaces and Segmentation](./Computer-Vision/OpenCV/02-Color-Spaces-and-Segmentation) | BGR/HSV conversion, masks and color-based segmentation |
| [Filtering, Edges and Morphology](./Computer-Vision/OpenCV/03-Filtering-Edges-Morphology) | Gaussian and median filtering, Sobel, Laplacian, Canny and morphological operations |
| [Segmentation and Features](./Computer-Vision/OpenCV/04-Segmentation-and-Features) | Thresholding, Otsu, K-means, Harris corners, ORB descriptors and feature matching |
| [Video and Motion Analysis](./Computer-Vision/OpenCV/05-Video-and-Motion-Analysis) | Webcam processing, motion detection, ROI processing and video output |

## Technologies

Python, PyTorch, OpenCV, NumPy, HLOC, YOLOv4, Jupyter Notebook, Google Colab, RAG, LLMs, embeddings and semantic search.

## Repository Structure

```text
AI-ML-CV-AcademicProjects/
├── Generative-AI/
│   └── RAG-Pipeline/
├── Deep-Learning/
│   └── PyTorch-RNN-Music-Generation/
├── Computer-Vision/
│   ├── HLOC-Structure-from-Motion/
│   ├── YOLOv4-Object-Detection/
│   └── OpenCV/
│       ├── 01-Image-Basics/
│       ├── 02-Color-Spaces-and-Segmentation/
│       ├── 03-Filtering-Edges-Morphology/
│       ├── 04-Segmentation-and-Features/
│       └── 05-Video-and-Motion-Analysis/
└── README.md
