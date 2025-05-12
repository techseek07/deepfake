# Deepfake Detection Using Hybrid Deep Learning Techniques 🎥🧠

A robust, modular pipeline for detecting deepfake content in videos using a hybrid ensemble of deep learning models: **Xception**, **CNN + LSTM**, and **Vision Transformer (ViT)**. This project includes GPU support, automated face extraction, preprocessing, labeling, model training, and video analysis.

---

## 📌 Features

- 🔍 Automated face detection from videos using RetinaFace
- 🧼 Image preprocessing pipeline (resize, normalization, RGB)
- 🏷️ Label generation using metadata
- 🧠 Hybrid model architecture combining spatial, temporal, and patch-based features
- 📊 Visualization of predictions, confidence, and frame-level analysis
- 🛠️ End-to-end training and testing modules with multiprocessing support

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/deepfake.git
cd deepfake
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Prepare Dataset
Organize your dataset as follows:
```bash
deepdata/
├── train_sample_videos/
├── test_videos/
├── metadeta/
│   └── metadata.json

```
### 4. Module Overview 
| File            | Description                                                     |
| --------------- | --------------------------------------------------------------- |
| `gpu_test.py`   | Verifies if TensorFlow can access a GPU                         |
| `image_crop.py` | Extracts faces from video frames using RetinaFace               |
| `preProcess.py` | Preprocesses cropped faces (resizing, RGB, normalization)       |
| `labelling.py`  | Generates labels (REAL/FAKE) from `metadata.json`               |
| `train.py`      | Trains a hybrid deep learning model (Xception + CNN+LSTM + ViT) |
| `test.py`       | Analyzes test videos and visualizes results                     |

### 5. Model Architecture
The ensemble model consists of:
Xception:Pretrained on ImageNet for deep spatial features.
CNN + LSTM: Extracts spatial and short-term temporal features.
ViT (Vision Transformer): Captures long-range dependencies via image patches.
Feature Fusion: Attention-based mechanism combining all branches.
Loss Function: Custom Weighted Focal Loss for handling class imbalance.
