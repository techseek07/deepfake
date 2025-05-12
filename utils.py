import os
import cv2

# Create the 'outputs' directory if it doesn't exist
def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Save each face image to the 'outputs' directory
def save_face_images(faces, output_dir="outputs"):
    create_output_dir(output_dir)

    for idx, face in enumerate(faces):
        face_filename = os.path.join(output_dir, f"face_{idx}.jpg")
        cv2.imwrite(face_filename, face)
