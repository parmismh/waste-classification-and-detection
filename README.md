# Waste Classification and Real-Time Detection

A computer vision project for waste detection and classification using YOLO and ResNet18, including a real-time desktop application for detecting waste from a camera stream.

## Overview

This project implements two different deep learning approaches for waste analysis:

- **YOLOv8** for object detection of three waste categories:
  - Bottle
  - Dry waste
  - Wet waste

- **ResNet18** with transfer learning for image classification using the **TrashNet** dataset:
  - Cardboard
  - Glass
  - Metal
  - Paper
  - Plastic
  - Trash

- A **real-time desktop application** using Python, Tkinter, OpenCV, and YOLO for camera-based and image-based waste detection.

---

## YOLO Waste Detection

The YOLO pipeline detects and localizes waste objects belonging to three classes:

`bottle`, `dry`, and `wet`.

### Dataset Preparation

The dataset was balanced using undersampling. The smallest class contained 130 images, so 130 images were selected from each class.

The balanced dataset was divided into:

- 70% Training
- 15% Validation
- 15% Testing

The YOLOv8s model was trained using the Ultralytics framework.

### Training Configuration

- Model: YOLOv8s
- Image size: 640 × 640
- Epochs: 100
- Batch size: 16
- Optimizer: Adam
- Initial learning rate: 0.001
- Weight decay: 0.0005
- Early stopping patience: 10

The complete YOLO workflow is available in:

`notebooks/YOLO_Model.ipynb`

---

## TrashNet Classification

A second approach was implemented using the **TrashNet dataset** and a pretrained **ResNet18** model.

The model performs image classification across six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

### Training Configuration

- Architecture: ResNet18
- Input size: 224 × 224
- Batch size: 32
- Epochs: 30
- Learning rate: 0.0001
- Optimizer: Adam
- Loss function: Cross Entropy Loss
- Learning-rate scheduler: ReduceLROnPlateau
- Transfer learning from ImageNet-pretrained weights

Data augmentation was applied using random resized crops, horizontal flipping, rotation, and color jitter.

The complete implementation is available in:

`notebooks/TrashNet-Model.ipynb`

---

## Real-Time Application

A desktop application was developed to provide an interface for waste detection.

The application supports two modes:

### Real-Time Detection

The application receives a video stream from a camera and performs YOLO inference on the incoming frames.

### Picture Detection

Users can select an image file and run waste detection on the selected image.

The application interface was developed using:

- Python
- Tkinter
- OpenCV
- Pillow
- Ultralytics YOLO

The application code and example images are available in:

`RealTimeApp/`

---

## Project Structure

```text
waste-classification-and-detection/
│
├── notebooks/
│   ├── YOLO_Model.ipynb
│   └── TrashNet-Model.ipynb
│
├── RealTimeApp/
│   ├── trash_detection_version_APP.py
│   ├── bottle.png
│   ├── dry.png
│   └── wet.jpg
│
├── results/
│
├── requirements.txt
├── .gitignore
└── README.md
