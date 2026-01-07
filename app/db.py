"""Database module for storing and loading face encodings."""

import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from app.config import ENCODINGS_FILE, EMBEDDING_DIM
from app.utils import l2_distance


def load_encodings() -> Dict:
    """
    Load face encodings from the pickle file.
    
    Returns:
        Dictionary mapping username to encodings data:
        {"username": {"embeddings": [...], "created_at": "...", "face_images": [...]}, ...}
        Returns empty dict if file doesn't exist
    """
    if not ENCODINGS_FILE.exists():
        return {}
    
    try:
        with open(ENCODINGS_FILE, 'rb') as f:
            encodings_dict = pickle.load(f)
        return encodings_dict
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return {}


def save_encodings(encodings_dict: Dict):
    """
    Save face encodings to the pickle file.
    
    Args:
        encodings_dict: Dictionary mapping username to encodings data
    """
    # Ensure directory exists
    ENCODINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(encodings_dict, f)
    except Exception as e:
        raise RuntimeError(f"Error saving encodings: {e}")


def add_user_encodings(username: str, new_encodings: List[np.ndarray], face_images: Optional[List[str]] = None):
    """
    Add or update encodings for a user.
    
    Args:
        username: Name of the user
        new_encodings: List of new face encodings (512-D numpy arrays)
        face_images: Optional list of image paths used for enrollment
    """
    # Load existing encodings
    encodings_dict = load_encodings()
    
    # Get current timestamp
    now = datetime.now().isoformat()
    
    # Add or append encodings for this user
    if username in encodings_dict:
        user_data = encodings_dict[username]
        
        # Append to existing embeddings
        if 'embeddings' in user_data:
            user_data['embeddings'].extend(new_encodings)
        else:
            user_data['embeddings'] = new_encodings
        
        # Update metadata
        if face_images:
            if 'face_images' in user_data:
                user_data['face_images'].extend(face_images)
            else:
                user_data['face_images'] = face_images
    else:
        # Create new entry
        encodings_dict[username] = {
            'embeddings': new_encodings,
            'created_at': now,
            'face_images': face_images or []
        }
    
    # Save updated encodings
    save_encodings(encodings_dict)


def find_best_match(encoding: np.ndarray, known_encodings_dict: Dict, threshold: Optional[float] = None) -> Tuple[Optional[str], float]:
    """
    Find the best matching user for a given face encoding.
    
    Args:
        encoding: Face encoding to match (512-dim numpy array)
        known_encodings_dict: Dictionary mapping username to encodings data
        threshold: Distance threshold (optional, uses config default)
    
    Returns:
        Tuple of (username, distance) if match found, (None, best_distance) otherwise
    """
    from app.config import RECOGNITION_THRESHOLD
    
    if threshold is None:
        threshold = RECOGNITION_THRESHOLD
    
    if encoding is None or len(known_encodings_dict) == 0:
        return (None, float('inf'))
    
    best_match = None
    best_distance = float('inf')
    
    # Compare with all users
    for username, user_data in known_encodings_dict.items():
        # Extract embeddings from user data
        if not isinstance(user_data, dict) or 'embeddings' not in user_data:
            continue
        
        user_encodings = user_data['embeddings']
        
        if len(user_encodings) == 0:
            continue
        
        # Calculate distances to all encodings for this user
        distances = [l2_distance(enc, encoding) for enc in user_encodings]
        
        # Use the minimum distance (best match) for this user
        min_distance = np.min(distances)
        
        if min_distance < best_distance:
            best_distance = min_distance
            best_match = username
    
    # Check if best match is below threshold
    if best_match is not None and best_distance <= threshold:
        return (best_match, best_distance)
    
    return (None, best_distance)
