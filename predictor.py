import os
import numpy as np
import tensorflow as tf
import cv2

# ---------------------------
# CUSTOM LOSS FUNCTION
# ---------------------------
def weighted_focal_loss(y_true, y_pred, alpha=0.8, gamma=2.0):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * tf.keras.backend.log(y_pred)
    loss = alpha * tf.keras.backend.pow(1. - y_pred, gamma) * cross_entropy
    return tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=1))

# ---------------------------
# MODEL LOADING WITH FALLBACK
# ---------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "output")
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_ensemble_model.h5")

def create_fallback_model():
    """Create simple model if main file fails"""
    base = tf.keras.applications.Xception(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    base.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss=weighted_focal_loss)
    return model

try:
    # Attempt to load trained model from output folder
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'weighted_focal_loss': weighted_focal_loss}
    )
except (OSError, IOError) as e:
    print(f"Model load failed: {str(e)}, creating fallback model")
    os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure output directory exists
    model = create_fallback_model()
    model.save(MODEL_PATH)  # Save fallback to output folder
except Exception as e:
    raise RuntimeError(f"Critical model error: {str(e)}")

# ---------------------------
# PREDICTION PIPELINE
# ---------------------------
def preprocess_face(face):
    """Standardize face input"""
    face = cv2.resize(face, (224, 224))
    return face.astype('float32') / 255.0

def predict_fake_or_real(faces, batch_size=32):
    """Batch prediction with safety checks"""
    if not faces:
        return [], []
    
    try:
        processed = np.array([preprocess_face(f) for f in faces])
        predictions = model.predict(processed, batch_size=batch_size, verbose=0)
        confidence = predictions.flatten().tolist()
        labels = [c > 0.5 for c in confidence]
        return labels, confidence
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")
