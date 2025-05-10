import os
import gc
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense,
                                     Dropout, BatchNormalization, LSTM, Reshape, Concatenate, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.data import AUTOTUNE
from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. CHECK GPU AVAILABILITY
# ---------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU is available:", gpus)
else:
    print("❌ No GPU available; running on CPU.")

# ---------------------------
# 2. PATHS & DATA LOADING
# ---------------------------
base_dir = r"C:\Users\himan\Desktop\deepdata"
preprocessed_faces_dir = os.path.join(base_dir, "preprocessed_faces")
labels_csv_path = os.path.join(base_dir, "output_preprocessed", "face_labels.csv")

df = pd.read_csv(labels_csv_path)
print(f"✅ Loaded {len(df)} labeled images.")

# Stratified split (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
print(f"📊 Train Size: {len(train_df)} | Test Size: {len(test_df)}")

# Build label dictionary.
label_dict = dict(zip(df["filename"], df["label"]))

# ---------------------------
# 3. DATA LOADING FUNCTIONS
# ---------------------------
def load_and_preprocess_image(filename):
    """
    Load an image from preprocessed_faces_dir (saved as JPEG in RGB with values in 0-255),
    decode it as a 3-channel image, convert it to float32, and normalize to [-1, 1]
    (as expected by pre-trained models such as Xception).
    """
    path = os.path.join(preprocessed_faces_dir, filename.numpy().decode("utf-8"))
    img = tf.io.read_file(path)
    # Decode as 3-channel (RGB) image.
    img = tf.image.decode_jpeg(img, channels=3)
    # Convert to float32 and scale to [0,1].
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Normalize to [-1, 1] (Xception's default).
    img = (img * 2.0) - 1.0
    return img

def get_label(filename):
    label = label_dict.get(filename.numpy().decode("utf-8"), 0)
    return tf.convert_to_tensor(label, dtype=tf.int32)

def process_path(filename):
    img = tf.py_function(load_and_preprocess_image, [filename], tf.float32)
    label = tf.py_function(get_label, [filename], tf.int32)
    # Explicitly set shape: images are now (128, 128, 3)
    img.set_shape([128, 128, 3])
    label.set_shape([])
    return img, label

batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices(train_df["filename"].values)
train_dataset = train_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=len(train_df), reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(test_df["filename"].values)
test_dataset = test_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(AUTOTUNE)

# ---------------------------
# 4. MODEL BRANCHES (Using the models we had previously used)
# ---------------------------
# Branch 1: Xception (Pre-trained) with gradual fine-tuning.
def build_xception_branch(inputs):
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        pooling='avg'
    )
    # Freeze all layers first.
    base_model.trainable = False
    # Unfreeze last 20 layers for adaptation.
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    out = Dense(128, activation='relu')(base_model.output)
    return out

# Branch 2: CNN + LSTM branch.
def build_cnn_lstm_branch(inputs):
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = GlobalAveragePooling2D()(x)
    x_seq = tf.expand_dims(x, axis=1)  # Simulate time dimension (T=1)
    x_lstm = LSTM(128, dropout=0.3, recurrent_dropout=0.3)(x_seq)
    return x_lstm

# Branch 3: Vision Transformer branch.
def build_vit_branch(inputs):
    patch_size = 16
    num_patches = (128 // patch_size) * (128 // patch_size)
    patches = tf.keras.layers.Conv2D(filters=64, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    patches = tf.keras.layers.Reshape((num_patches, 64))(patches)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=64)(positions)
    x = patches + pos_embedding
    x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x1, x1)
    x2 = tf.keras.layers.Add()([attention_output, x])
    x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = tf.keras.layers.Dense(128, activation='relu')(x3)
    x3 = Dropout(0.3)(x3)
    x3 = tf.keras.layers.Dense(64, activation='relu')(x3)
    vit_output = tf.keras.layers.Add()([x3, x2])
    vit_features = tf.keras.layers.GlobalAveragePooling1D()(vit_output)
    vit_features = Dense(128, activation='relu')(vit_features)
    return vit_features

# ---------------------------
# 5. FUSION & META-LEARNER
# ---------------------------
def feature_fusion(branch_feats, units=128):
    """
    Fuse features from multiple branches using an attention mechanism.
    branch_feats: list of tensors with shape (batch, feature_dim)
    """
    stacked = tf.stack(branch_feats, axis=1)  # shape: (batch, num_branches, feature_dim)
    stacked = BatchNormalization()(stacked)
    attn = Dense(units, activation='tanh')(stacked)
    attn = Dense(1, use_bias=False)(attn)
    attn = tf.nn.softmax(attn, axis=1)
    fused = tf.reduce_sum(attn * stacked, axis=1)
    return fused

def build_ensemble_model():
    # Input shape: images loaded from preprocessed_faces_dir (shape (128,128,3)).
    inp = Input(shape=(128, 128, 3))
    branch1 = build_xception_branch(inp)   # Xception branch.
    branch2 = build_cnn_lstm_branch(inp)     # CNN+LSTM branch.
    branch3 = build_vit_branch(inp)          # ViT branch.
    fused_features = feature_fusion([branch1, branch2, branch3], units=128)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fused_features)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    return model

# ---------------------------
# 6. DEFINE WEIGHTED FOCAL LOSS
# ---------------------------
def weighted_focal_loss(gamma=2., alpha=0.25, weight_0=8.0, weight_1=1.0):
    """
    Weighted focal loss for imbalanced binary classification.
    weight_0: weight for class 0 (Real) is increased to 8.0 from 6.0.
    weight_1: weight for class 1 (Fake) remains 1.0.
    """
    def loss(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha * weight_1, (1 - alpha) * weight_0)
        loss_val = -alpha_factor * tf.pow((1 - pt), gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss_val)
    return loss

# ---------------------------
# 7. CALLBACK: MEMORY CLEAR
# ---------------------------
class MemoryClearCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

# ---------------------------
# 8. SET CLASS WEIGHTS & HYPERPARAMETERS
# ---------------------------
# Updated class weights: Increased weight for class 0 (Real) to 8.0 to better balance the dataset.
class_weight = {0: 8.0, 1: 1.0}
print("Class weights used for training:", class_weight)
LEARNING_RATE =  1e-4


# ---------------------------
# 10. BUILD, COMPILE & TRAIN THE MODEL
# ---------------------------
ensemble_model = build_ensemble_model()
ensemble_model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
    loss=weighted_focal_loss(gamma=2., alpha=0.25, weight_0=8.0, weight_1=1.0),
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
ensemble_model.summary()

callbacks = [
    MemoryClearCallback(),
    TqdmCallback(verbose=1),
    EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint(os.path.join(base_dir, "saved_models", "deepfake_ensemble_model.h5"),
                    monitor='val_auc', mode='max', save_best_only=True, verbose=1)
]

history = ensemble_model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=50,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=0
)

# ---------------------------
# 11. PLOT TRAINING METRICS
# ---------------------------
def plot_metrics(history):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(base_dir, "saved_models", "training_metrics.png")
    plt.savefig(save_path)
    plt.show()

plot_metrics(history)

# ---------------------------
# 12. EVALUATE MODEL PERFORMANCE
# ---------------------------
y_pred_prob = ensemble_model.predict(test_dataset).flatten()
y_pred = (y_pred_prob > 0.5).astype(np.int32)
y_true = test_df["label"].values

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()