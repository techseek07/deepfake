import os
import gc
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from PIL import Image

# ---------------------------
# Paths - Update as needed
# ---------------------------
cropped_faces_dir = r"C:\Users\himan\Desktop\deepdata\cropped_faces"  # Folder with cropped faces
preprocessed_faces_dir = r"C:\Users\himan\Desktop\deepdata\preprocessed_faces"  # Folder for final preprocessed images

# Create directories if they don't exist
os.makedirs(preprocessed_faces_dir, exist_ok=True)

def preprocess_image(filename, input_dir, output_dir):
    """
    Reads an image from input_dir, converts it to RGB (if not already),
    resizes it to 128x128, and saves it to output_dir.
    This version uses PIL to ensure images are saved with the proper RGB channel order.
    """
    image_path = os.path.join(input_dir, filename)
    try:
        with Image.open(image_path) as img:
            # Convert to RGB (this works for both grayscale and colored images)
            img = img.convert("RGB")
            # Resize to 128x128
            img = img.resize((128, 128))
            # Save the image in RGB order in standard uint8 format
            output_path = os.path.join(output_dir, filename)
            img.save(output_path)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
    # Explicitly clear memory
    gc.collect()
    return filename

if __name__ == '__main__':
    # Get list of image files from the cropped faces folder.
    image_files = [f for f in os.listdir(cropped_faces_dir) if f.lower().endswith(('.jpg', '.png'))]
    print("Number of images to preprocess:", len(image_files))

    # Process images concurrently using 4 worker processes.
    process_func = partial(preprocess_image, input_dir=cropped_faces_dir, output_dir=preprocessed_faces_dir)
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(process_func, image_files),
                  total=len(image_files), desc="Preprocessing Faces"))

    print("✅ Preprocessing complete!")
