import tensorflow as tf
import numpy as np
import cv2
from typing import List, Tuple


# ---------------------------
# 1. CUSTOM LOSS DEFINITION
# ---------------------------
def weighted_focal_loss(gamma=2., alpha=0.25, weight_0=8.0, weight_1=1.0):
    """Custom focal loss implementation"""

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha * weight_1, (1 - alpha) * weight_0)
        loss_val = -alpha_factor * tf.pow(1. - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss_val)

    return loss


# ---------------------------
# 2. MODEL LOADING
# ---------------------------
MODEL_PATH = "outputs/deepfake_ensemble_model.h5"

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'loss': weighted_focal_loss(
                gamma=2.,
                alpha=0.25,
                weight_0=8.0,
                weight_1=1.0
            )
        }
    )
except Exception as e:
    raise RuntimeError(f"Model initialization failed: {str(e)}")


# ---------------------------
# 3. PREDICTION PIPELINE (CRITICAL FIX)
# ---------------------------
def predict_fake_or_real(faces: List[np.ndarray]) -> Tuple[List[int], List[float]]:
    """Returns tuple of (predictions, confidences) even on error"""
    try:
        if not faces:
            return [], []

        processed = [preprocess_face(f) for f in faces]
        batch = np.array(processed)
        confidences = model.predict(batch, verbose=0).flatten()
        return (
            [1 if c > 0.5 else 0 for c in confidences],  # Binary predictions
            confidences.tolist()  # Confidence scores
        )
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Log to console
        return [], []  # Return empty lists instead of None


def preprocess_face(face: np.ndarray) -> np.ndarray:
    """Face preprocessing pipeline"""
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (128, 128))
    return (face_resized / 127.5) - 1.0  # Normalize to [-1, 1]
