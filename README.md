🎥 Deepfake Detection Pipeline Using Hybrid Deep Learning
This repository contains a complete pipeline for detecting deepfakes using a hybrid ensemble model combining Xception, CNN+LSTM, and Vision Transformer (ViT). It includes GPU checks, video face extraction, preprocessing, labeling, training, and testing with visualization support.

📁 Project Files
gpu_test.py: Checks GPU availability on your system.

image_crop.py: Extracts and crops faces from videos using RetinaFace.

preProcess.py: Preprocesses cropped face images (resizing, RGB conversion).

labelling.py: Labels images based on metadata.json.

train.py: Trains a hybrid deep learning model (Xception + CNN+LSTM + ViT).

test.py: Analyzes videos using the trained model and shows results.

📦 Requirements
Install all required packages using pip:

bash
Copy
Edit
pip install tensorflow opencv-python pandas tqdm seaborn pillow retina-face tensorflow-addons
📂 Expected Dataset Structure
graphql
Copy
Edit
deepdata/
├── train_sample_videos/         # Training videos
├── test_videos/                 # Test videos
├── cropped_faces/               # Auto-generated: face crops
├── preprocessed_faces/          # Auto-generated: resized RGB faces
├── metadeta/
│   └── metadata.json            # Metadata with labels
├── output_preprocessed/
│   └── face_labels.csv          # Auto-generated labels for images
└── saved_models/
    └── deepfake_ensemble_model.h5  # Trained model file
Make sure the paths in the Python scripts match your folder structure.

🚀 How to Use
1. ✅ Check GPU Availability
bash
Copy
Edit
python gpu_test.py
2. 🎞️ Extract Faces from Videos
bash
Copy
Edit
python image_crop.py
3. 🧼 Preprocess Cropped Faces
bash
Copy
Edit
python preProcess.py
4. 🏷️ Label the Images
bash
Copy
Edit
python labelling.py
5. 🧠 Train the Model
bash
Copy
Edit
python train.py
6. 🔍 Test Videos
bash
Copy
Edit
python test.py
🧠 Model Architecture
This project uses an ensemble deep learning model that combines:

Xception (pretrained on ImageNet)

CNN + LSTM (for spatio-temporal feature extraction)

Vision Transformer (ViT) (for patch-based learning)

All feature branches are fused using a custom attention mechanism, followed by a dense classification head. The model is trained using Weighted Focal Loss to handle class imbalance.

📊 Output
Frame-level prediction plots

Temporal consistency analysis

Final classification: Deepfake, Authentic, or Uncertain

Confidence-based visualization

Confusion matrix and ROC curve

