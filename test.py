import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import gc
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm
from scipy.signal import find_peaks

# Custom layers & loss from training script
from tensorflow_addons.layers import StochasticDepth
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.models import load_model

# Re‐define your weighted_focal_loss here exactly as in training:
def weighted_focal_loss(gamma=2., alpha=0.25, weight_0=8.0, weight_1=1.0):
    """
    Weighted focal loss for imbalanced binary classification.
    weight_0: weight for class 0 (Real).
    weight_1: weight for class 1 (Fake).
    """
    def loss(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(
            tf.equal(y_true, 1),
            alpha * weight_1,
            (1 - alpha) * weight_0
        )
        return tf.reduce_mean(-alpha_factor * tf.pow((1 - pt), gamma) * tf.math.log(pt))
    return loss

# ---------------------------
# Configuration
# ---------------------------
BASE_DIR         = r"C:\Users\himan\Desktop\deepdata"
TEST_VIDEOS_DIR  = os.path.join(BASE_DIR, "test_videos")
MODEL_PATH       = os.path.join(BASE_DIR, "saved_models", "deepfake_ensemble_model.h5")
ANALYSIS_PARAMS  = {
    'temporal_window': 15,
    'peak_threshold' : 0.5,
    'min_peak_distance': 10,
    'uncertainty_limit': 0.25
}

# ---------------------------
# Video Processing Utilities
# ---------------------------
def extract_video_frames(video_path, frame_interval=20):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    frames, indices = [], []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
            indices.append(count)
        count += 1
    cap.release()
    return frames, indices

def preprocess_video_frame(frame):
    resized = cv2.resize(frame, (128, 128))
    normalized = resized.astype(np.float32) / 255.0
    return normalized

# ---------------------------
# Analysis Core Functions
# ---------------------------
def calculate_temporal_consistency(preds, window_size):
    mov_avg = np.convolve(preds, np.ones(window_size)/window_size, mode='valid')
    deviations = np.abs(preds[window_size-1:] - mov_avg)
    return np.mean(deviations), mov_avg

def detect_anomalous_segments(preds, min_value, min_distance):
    peaks, _ = find_peaks(preds, height=min_value, distance=min_distance)
    return len(peaks)

def compute_adaptive_threshold(preds):
    hist, edges = np.histogram(preds, bins=10, range=(0,1))
    main_peak = np.argmax(hist)
    return max(edges[main_peak+1] * 0.9, 0.5)

# ---------------------------
# Video Analysis Pipeline
# ---------------------------
def analyze_video_deepfake(video_path, model, frame_skip=20):
    print(f"\nAnalyzing: {os.path.basename(video_path)}")
    raw_frames, frame_idxs = extract_video_frames(video_path, frame_skip)
    if not raw_frames:
        return None

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        proc_frames = list(tqdm(ex.map(preprocess_video_frame, raw_frames),
                                total=len(raw_frames), desc="Processing Frames"))
    del raw_frames; gc.collect()

    preds = []
    batch_size = 4
    for i in range(0, len(proc_frames), batch_size):
        batch = np.stack(proc_frames[i:i+batch_size])
        batch_pred = model.predict(batch, verbose=0).flatten()
        preds.extend(batch_pred)
        del batch; gc.collect()
    del proc_frames; gc.collect()

    arr = np.array(preds)
    if arr.size == 0: return None

    # 1) Temporal consistency
    temp_var, mov_avg = calculate_temporal_consistency(arr, ANALYSIS_PARAMS['temporal_window'])
    # 2) Anomaly count (peaks)
    anomaly_count = detect_anomalous_segments(
        arr, ANALYSIS_PARAMS['peak_threshold'], ANALYSIS_PARAMS['min_peak_distance']
    )
    # 3) Confidence‐weighted fusion
    weights      = np.where(arr > 0.5, arr*2, (1 - arr)*2)
    weighted_avg = np.sum(arr * weights) / np.sum(weights)
    # 4) Trimmed mean
    sorted_arr   = np.sort(arr)
    trim_cnt     = int(len(sorted_arr) * 0.25)
    trim_mean    = np.mean(sorted_arr[trim_cnt:-trim_cnt]) if trim_cnt > 0 else np.median(arr)
    # 5) Final score
    final_score  = (weighted_avg * 0.6) + (trim_mean * 0.4)
    # 6) Uncertainty
    uncertainty  = np.std(arr) * (1 - abs(final_score - 0.5))

    # 7) Decision (now require ≥5 peaks)
    video_class = "Authentic"
    if uncertainty < ANALYSIS_PARAMS['uncertainty_limit']:
        criteria = [
            final_score > max(0.5, compute_adaptive_threshold(arr)),
            (np.median(arr) > 0.55 and np.std(arr) < 0.2),
            anomaly_count >= 5
        ]
        if any(criteria):
            video_class = "Deepfake"
    else:
        video_class = "Uncertain"

    return {
        'final_score'      : final_score,
        'median_score'     : np.median(arr),
        'std_deviation'    : np.std(arr),
        'temporal_variance': temp_var,
        'anomaly_count'    : anomaly_count,
        'uncertainty'      : uncertainty,
        'classification'   : video_class,
        'frame_analysis'   : (frame_idxs, arr, mov_avg)
    }

# ---------------------------
# Visualization
# ---------------------------
def visualize_analysis_results(results, video_name):
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,10), gridspec_kw={'height_ratios':[3,1]})
    frames, preds, trend = results['frame_analysis']
    ax1.plot(frames, preds, alpha=0.3, label='Per‐Frame Scores')
    ax1.plot(frames[ANALYSIS_PARAMS['temporal_window']-1:], trend, lw=2, label='Trend')
    ax1.set_title(f"{video_name} → {results['classification']} ({results['final_score']:.2f})")
    ax1.set_ylabel('Probability'); ax1.legend(); ax1.grid(True)

    ax2.barh(0, results['uncertainty'],
             color='green' if results['classification']!='Uncertain' else 'red', height=0.4)
    ax2.set_xlim(0,0.5); ax2.axvline(ANALYSIS_PARAMS['uncertainty_limit'], linestyle='--')
    ax2.set_xlabel('Analysis Confidence'); ax2.set_yticks([])
    ax2.text(0.05,0.2,'High Confidence',transform=ax2.transAxes)
    ax2.text(0.7,0.2,'Low Confidence', transform=ax2.transAxes)

    plt.tight_layout(); plt.show()

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Register custom objects so load_model can find them :contentReference[oaicite:0]{index=0}
    custom_objects = {
        'StochasticDepth'     : StochasticDepth,
        'MultiHeadAttention'  : MultiHeadAttention,
        # The loss function was saved under the generic name 'loss', so map that back:
        'loss'                : weighted_focal_loss()
    }
    model = load_model(MODEL_PATH, custom_objects=custom_objects)  # :contentReference[oaicite:1]{index=1}

    video_files = [f for f in os.listdir(TEST_VIDEOS_DIR)
                   if f.lower().endswith(('.mp4','.avi','.mov'))][:20]

    analysis_results = []
    for vf in video_files:
        full_path = os.path.join(TEST_VIDEOS_DIR, vf)
        try:
            res = analyze_video_deepfake(full_path, model)
            if res:
                analysis_results.append(res)
                print(f"{vf} → {res['classification']} | Score: {res['final_score']:.2f} | Peaks: {res['anomaly_count']}")
                visualize_analysis_results(res, vf)
        except Exception as e:
            print(f"Error on {vf}: {e}")

    # Summary
    tot = len(analysis_results)
    deepf = sum(1 for r in analysis_results if r['classification']=='Deepfake')
    auth  = sum(1 for r in analysis_results if r['classification']=='Authentic')
    unct  = sum(1 for r in analysis_results if r['classification']=='Uncertain')

    print(f"\nProcessed: {tot} videos")
    print(f"Deepfake:   {deepf}")
    print(f"Authentic:  {auth}")
    print(f"Uncertain:  {unct}")