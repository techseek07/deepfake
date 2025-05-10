import os
import cv2
import gc
import numpy as np
import tensorflow as tf
from retinaface import RetinaFace
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# ---------------------------
# Paths - Update as needed
# ---------------------------
video_dir = r"C:\Users\himan\Desktop\deepdata\train_sample_videos"  # Folder with videos for face extraction
cropped_faces_dir = r"C:\Users\himan\Desktop\deepdata\cropped_faces"  # Folder to save cropped faces
preprocessed_faces_dir = r"C:\Users\himan\Desktop\deepdata\preprocessed_faces"  # Folder for final preprocessed images
test_video_path = r"C:\Users\himan\Desktop\deepdata\test_videos\aassnaulhq_converted.mp4"  # Test video for evaluation

# Create directories if they don't exist
os.makedirs(cropped_faces_dir, exist_ok=True)
os.makedirs(preprocessed_faces_dir, exist_ok=True)

# Verify paths
print("Video Directory:", video_dir)
print("Cropped Faces Directory:", cropped_faces_dir)
print("Preprocessed Faces Directory:", preprocessed_faces_dir)
print("Test Video Path:", test_video_path)

if os.path.exists(test_video_path):
    print("✅ Test video file exists.")
else:
    print("❌ Test video file does NOT exist. Please check the path!")


# ---------------------------
# Helper: Get Last Processed Frame Number for a Video
# ---------------------------
def get_last_processed_frame(video_path):
    """
    For the given video, check the cropped_faces_dir for files whose names
    start with the video's basename and extract the highest processed frame number.
    Expected filename format: <basename>_frame{frame_number}_face{i}.jpg
    """
    base = os.path.basename(video_path)
    max_frame = 0
    for f in os.listdir(cropped_faces_dir):
        if f.startswith(base):
            try:
                parts = f.split('_')
                for part in parts:
                    if part.startswith("frame"):
                        frame_str = part.replace("frame", "")
                        frame_num = int(frame_str)
                        if frame_num > max_frame:
                            max_frame = frame_num
            except Exception as e:
                print(f"Error parsing {f}: {e}")
    return max_frame


# ---------------------------
# 1️⃣ Face Extraction Function (with resume support and skip_rate)
# ---------------------------
def extract_faces_from_video(video_path, skip_rate=20):
    """
    Extract faces from the video at 'video_path', processing every 'skip_rate'-th frame.
    If some frames were already processed, resume from the next unprocessed frame.
    """
    try:
        last_frame = get_last_processed_frame(video_path)
        print(f"Resuming {os.path.basename(video_path)} from frame {last_frame + 1}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit if no more frames
            frame_count += 1

            # Skip frames that have already been processed
            if frame_count <= last_frame:
                del frame
                gc.collect()
                continue

            # Process only every 'skip_rate'-th frame
            if frame_count % skip_rate != 0:
                del frame
                gc.collect()
                continue

            # Detect faces using RetinaFace
            faces = RetinaFace.detect_faces(frame)
            if isinstance(faces, dict):
                print(f"Video {os.path.basename(video_path)} Frame {frame_count}: Detected {len(faces)} face(s).")
                for i, key in enumerate(faces.keys()):
                    face_info = faces[key]
                    x, y, w, h = face_info["facial_area"]

                    # Crop & Resize Face
                    cropped_face = frame[y:h, x:w]
                    try:
                        resized_face = cv2.resize(cropped_face, (128, 128))
                    except Exception as e:
                        print(f"Error resizing face on frame {frame_count}: {e}")
                        continue

                    # Save Cropped Face
                    filename = os.path.join(
                        cropped_faces_dir,
                        f"{os.path.basename(video_path)}_frame{frame_count}_face{i}.jpg"
                    )
                    if cv2.imwrite(filename, resized_face):
                        print(f"Saved: {filename}")
                    else:
                        print(f"Failed to save: {filename}")
            else:
                print(f"Video {os.path.basename(video_path)} Frame {frame_count}: No faces detected.")

            # Free memory after processing each frame
            del frame, faces
            gc.collect()

        cap.release()
        gc.collect()
    except Exception as ex:
        print(f"Exception in processing {video_path}: {ex}")


# ---------------------------
# 2️⃣ Build List of Video Files (Full List)
# ---------------------------
def build_full_video_list():
    all_video_files = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ]
    all_video_files.sort()  # Sort alphabetically
    print("Found video files:", all_video_files)

    # Print list with indexes for reference
    print("\nList of video files with indexes:")
    for idx, video in enumerate(all_video_files):
        print(f"{idx}: {os.path.basename(video)}")
    return all_video_files


# ---------------------------
# 3️⃣ Process Videos in Batches Using Multiprocessing (Based on Full List)
# ---------------------------
def process_videos_in_batches(video_files, batch_size=2, skip_rate=20):
    """
    Process a list of video files in batches using multiprocessing.
    Only 'batch_size' videos are processed concurrently.
    """
    total = len(video_files)
    batch_number = 1
    for i in range(0, total, batch_size):
        batch = video_files[i:i + batch_size]
        print(f"\nProcessing batch {batch_number} of {((total - 1) // batch_size) + 1}: {batch}")
        process_func = partial(extract_faces_from_video, skip_rate=skip_rate)
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            list(executor.map(process_func, batch))
        print(f"Finished batch {batch_number}.")
        batch_number += 1
        gc.collect()


# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # Build the full list of videos
    full_video_files = build_full_video_list()

    # Prompt for a starting index based on the full list
    if full_video_files:
        try:
            start_index = int(input("Enter starting index (0 for first video): ") or "0")
        except ValueError:
            start_index = 0
        print(f"Starting processing from index {start_index}: {os.path.basename(full_video_files[start_index])}")
        videos_to_process = full_video_files[start_index:]
    else:
        print("No video files found.")
        videos_to_process = []

    # Process videos sequentially in batches using multiprocessing
    if videos_to_process:
        process_videos_in_batches(videos_to_process, batch_size=2, skip_rate=20)
        print("✅ Face extraction complete with multiprocessing!")
    else:
        print("No videos to run.")
