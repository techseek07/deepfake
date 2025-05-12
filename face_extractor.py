import cv2
import numpy as np
from retinaface import RetinaFace


def extract_faces_from_video(video_path):
    # Initialize the video capture
    video_capture = cv2.VideoCapture(video_path)

    faces = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Extract faces using RetinaFace
        detected_faces = RetinaFace.detect_faces(frame)

        # If faces are detected, crop and append them
        for _, face in detected_faces.items():
            x1, y1, x2, y2 = face['facial_area']
            face_image = frame[y1:y2, x1:x2]
            faces.append(face_image)

    video_capture.release()
    return faces
