import os
import json
import pandas as pd
import multiprocessing
import gc  # Import garbage collector

# Define root directory
base_dir = r"C:\Users\himan\Desktop\deepdata"

# Corrected Paths
metadata_path = os.path.join(base_dir, "metadeta", "metadata.json")  # ✅ Correct folder
preprocessed_faces_folder = os.path.join(base_dir, "preprocessed_faces")
output_csv = os.path.join(base_dir, "output_preprocessed", "face_labels.csv")

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# ✅ Check if metadata.json exists before proceeding
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"⚠️ Metadata file not found at: {metadata_path}")

print(f"✅ Metadata file found at: {metadata_path}")
print(f"📂 Preprocessed Faces Folder: {preprocessed_faces_folder}")
print(f"📄 Output CSV File: {output_csv}")

# Load metadata.json
with open(metadata_path, "r") as file:
    metadata = json.load(file)


# Function to process a single image and extract its label
def process_image(filename):
    """Extract video name from image filename and assign a label from metadata."""
    if filename.endswith(".jpg"):  # Ensure it's an image file
        video_name = filename.split("_frame")[0]  # Extract video filename (without .mp4)

        if video_name in metadata:  # ✅ FIXED: No extra ".mp4" added
            label = 1 if metadata[video_name]["label"] == "FAKE" else 0  # FAKE=1, REAL=0

            # ✅ Memory cleanup after processing each image
            del video_name
            gc.collect()

            return (filename, label)

    return None  # If no label found, return None


# Use multiprocessing for faster execution
if __name__ == "__main__":
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(process_image, os.listdir(preprocessed_faces_folder))

    # Filter out None values (for files with no metadata)
    labeled_data = [result for result in results if result is not None]

    # Convert to DataFrame
    df = pd.DataFrame(labeled_data, columns=["filename", "label"])

    # ✅ Check if data was added
    if not df.empty:
        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"✅ Labels saved to: {output_csv}")
    else:
        print("⚠️ No matching labels found. Check metadata.json and filenames.")

    # ✅ Final Memory Cleanup
    del metadata
    del labeled_data
    del df
    gc.collect()
