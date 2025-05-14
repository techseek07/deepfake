import cv2
import imageio.v3 as iio
from mtcnn import MTCNN

def extract_faces_from_video(video_path, target_size=(160, 160)):
    """Extract faces using MTCNN detector"""
    detector = MTCNN()
    faces = []
    
    try:
        for frame in iio.imiter(video_path):
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = detector.detect_faces(rgb_frame)
            
            for detection in detections:
                x, y, w, h = detection['box']
                face = rgb_frame[y:y+h, x:x+w]
                if face.size > 0:
                    resized_face = cv2.resize(face, target_size)
                    faces.append(resized_face)
                    
    except Exception as e:
        print(f"Face extraction error: {str(e)}")
        
    return faces
