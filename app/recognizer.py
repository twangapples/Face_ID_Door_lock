"""Module for face recognition: encoding and matching."""

import face_recognition
import cv2
import numpy as np
from app.config import RECOGNITION_THRESHOLD


def encode_face(image_path):
    """
    Generate face encoding from an image file.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Face encoding (128-dim numpy array) or None if no face found
    """
    # Load image using face_recognition (uses RGB)
    image = face_recognition.load_image_file(image_path)
    
    # Find face locations
    face_locations = face_recognition.face_locations(image)
    
    if len(face_locations) == 0:
        return None
    
    # Generate encoding for the first face found
    encodings = face_recognition.face_encodings(image, face_locations)
    
    if len(encodings) > 0:
        return encodings[0]
    
    return None


def detect_and_encode_frame(frame):
    """
    Detect face in a live camera frame and return encoding.
    
    Args:
        frame: BGR frame from OpenCV
    
    Returns:
        Face encoding (128-dim numpy array) or None if no face found
    """
    # Convert BGR to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find face locations
    face_locations = face_recognition.face_locations(rgb_frame)
    
    if len(face_locations) == 0:
        return None
    
    # Generate encoding for the first face found
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    if len(encodings) > 0:
        return encodings[0]
    
    return None


def find_match(encoding, known_encodings_dict, threshold=None):
    """
    Find the best matching user for a given face encoding.
    
    Args:
        encoding: Face encoding to match (128-dim numpy array)
        known_encodings_dict: Dictionary mapping username to list of encodings
        threshold: Distance threshold (defaults to config value)
    
    Returns:
        Username if match found, None otherwise
    """
    if threshold is None:
        threshold = RECOGNITION_THRESHOLD
    
    if encoding is None or len(known_encodings_dict) == 0:
        return None
    
    best_match = None
    best_distance = float('inf')
    
    # Compare with all users
    for username, user_encodings in known_encodings_dict.items():
        if len(user_encodings) == 0:
            continue
        
        # Calculate distances to all encodings for this user
        distances = face_recognition.face_distance(user_encodings, encoding)
        
        # Use the minimum distance (best match) for this user
        min_distance = np.min(distances)
        
        if min_distance < best_distance:
            best_distance = min_distance
            best_match = username
    
    # Check if best match is below threshold
    if best_match is not None and best_distance <= threshold:
        return best_match
    
    return None

