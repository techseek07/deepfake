# 🎥 Deepfake Detection Pipeline Using Hybrid Deep Learning

This repository contains a complete pipeline for detecting deepfakes using a hybrid ensemble model combining Xception, CNN+LSTM, and Vision Transformer (ViT). The pipeline handles video frame extraction, image preprocessing, labeling, training, and deepfake analysis with visualization support.

---

## 📁 Project Structure

.
├── gpu_test.py # Detects GPU availability
├── image_crop.py # Extracts faces from videos using RetinaFace
├── preProcess.py # Preprocesses extracted faces (resize, RGB)
├── labelling.py # Labels preprocessed images using metadata.json
├── train.py # Trains hybrid deep learning model (CNN+LSTM+ViT)
├── test.py # Analyzes videos using trained model

yaml
Copy
Edit

---

## 📦 Requirements

Install all required dependencies:

```bash
pip install tensorflow opencv-python pandas tqdm seaborn pillow retina-face tensorflow-addons
Also required:

CUDA-compatible GPU (recommended)

RetinaFace pre-installed

Folder structure as described below

🚀 Pipeline Steps
1. ✅ GPU Check
Check if your machine has a compatible GPU:

bash
Copy
Edit
python gpu_test.py
2. 🎞️ Face Extraction
Extracts faces from video frames using RetinaFace and saves them:

bash
Copy
Edit
python image_crop.py
Customize video_dir, cropped_faces_dir, etc., inside the script.

3. 🧼 Image Preprocessing
Converts cropped face images to 128x128 RGB format:

bash
Copy
Edit
python preProcess.py
4. 🏷️ Label Images
Reads metadata.json and labels images (FAKE = 1, REAL = 0):

bash
Copy
Edit
python labelling.py
5. 🧠 Train the Model
Trains a hybrid deep learning model using:

Xception (pretrained on ImageNet)

CNN + LSTM

Vision Transformer (ViT)

Feature fusion with attention mechanism

Weighted focal loss for class imbalance

bash
Copy
Edit
python train.py
Outputs:

deepfake_ensemble_model.h5

Training metrics, ROC curve, confusion matrix

6. 🔬 Test and Analyze Videos
Analyzes new videos for deepfake classification using trained model:

bash
Copy
Edit
python test.py
Displays:

Frame-by-frame probability scores

Temporal consistency

Final classification (Authentic, Deepfake, Uncertain)

Confidence visualization

🧠 Model Architecture
🔗 Hybrid Model: Combines Xception, CNN-LSTM, and Vision Transformer

🎯 Loss Function: Weighted Focal Loss

📉 Optimizer: Adam with learning rate scheduling

🧮 Metrics: Accuracy, AUC, Precision, Recall

📂 Expected Dataset Structure
markdown
Copy
Edit
deepdata/
├── train_sample_videos/
├── test_videos/
├── cropped_faces/
├── preprocessed_faces/
├── metadeta/
│   └── metadata.json
├── output_preprocessed/
│   └── face_labels.csv
└── saved_models/
    └── deepfake_ensemble_model.h5
Make sure paths inside each script are updated to match your local directory.

📊 Visualization
The pipeline provides:

📈 Training accuracy & loss plots

📉 ROC curves

🧩 Confusion matrix

📹 Frame-wise prediction plots with moving average and peak detection
