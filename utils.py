import cv2
import numpy as np
from keras_facenet import FaceNet

# Initialize FaceNet model globally
embedder = FaceNet()

def process_image(image_bytes):
    """
    Converts image bytes to an OpenCV image, detects the face using 
    FaceNet's built-in detector, and extracts the embedding.
    """
    # Convert uploaded raw bytes into an OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, None

    # Convert BGR to RGB (FaceNet requires RGB format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Use FaceNet's built-in crop utility to find faces
    # it detects boxes in format: [x, y, width, height]
    detections = embedder.extract(img_rgb, threshold=0.95)
    
    # If no face is detected, return empty values
    if len(detections) == 0:
        return [], None
    
    # Get details for the first detected face
    first_face = detections[0]
    box = first_face['box']  # [x, y, w, h]
    bounding_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
    
    # Extract the pre-calculated embedding array from the detection object
    embedding = first_face['embedding']
    
    return [bounding_box], embedding

def calculate_similarity(embedding1, embedding2):
    """
    Computes the cosine similarity between two face embedding vectors.
    """
    if embedding1 is None or embedding2 is None:
        return 0.0
        
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    cosine_similarity = dot_product / (norm1 * norm2)
    return float(cosine_similarity)