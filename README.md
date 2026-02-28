# PyTracking_ONNX-TensorRT
Implement ONNX conversion and TensorRT deployment of OSTrack/AVTrack single-object tracking models based on the PyTracking library, enabling end-to-end inference acceleration for single-object tracking and adapting to the high-performance deployment requirements of NVIDIA GPUs.

## 📌 Project Overview
This project focuses on the industrial-level deployment of Transformer-based single-object tracking models (OSTrack/AVTrack) in the PyTracking library. It solves the problems of low inference efficiency and operator incompatibility of native PyTorch models, and achieves real-time inference acceleration of single-object tracking through TensorRT FP16 precision optimization.

The whole process includes:
- PyTorch (.pth) → ONNX conversion (solving operator compatibility issues such as SDPA replacement)
- ONNX model simplification and validity verification
- TensorRT (.engine) engine construction and CUDA asynchronous inference
- End-to-end pre/post-processing logic reproduction (bbox decoding, Hanning window penalty, coordinate smoothing, etc.)
- Video tracking visualization (support ROI selection, pause, target re-selection)

## 🚀 Key Features
- Decoupled architecture based on PyTracking, compatible with Transformer-style single-object tracking model deployment
- Full-link deployment: PyTorch → ONNX → TensorRT, supports two export schemes (decoupled/wrapped) for OSTrack/AVTrack
- Complete pre/post-processing logic, consistent with the original model's tracking accuracy
- TensorRT FP16 precision acceleration, balancing inference speed and accuracy (adapt to NVIDIA GPU)
- Interactive video tracking: support ROI box selection, confidence threshold filtering, heatmap visualization

## 🛠️ Supported Models
| Model Name | Introduction |
|------------|--------------|
| OSTrack    | Classic One-Stream Transformer single-object tracker, realized decoupled ONNX conversion |
| AVTrack    | Efficient evolution version of OSTrack (Token sparsification/active screening), realized wrapped ONNX conversion without modifying source code |

## ⚙️ Environment Dependencies
```bash
# Basic environment
Python >= 3.8
PyTorch >= 1.10.0
CUDA >= 11.3, cuDNN (matching CUDA version)

# ONNX related
onnx >= 1.14.0
onnx-simplifier >= 0.4.33

# TensorRT related
TensorRT >= 8.0 (Recommend 10.x)
pycuda >= 2022.2.1

# Other dependencies
opencv-python >= 4.8.0
numpy >= 1.24.0
