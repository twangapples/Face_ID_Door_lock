"""Database module for storing and loading face encodings."""

import pickle
from pathlib import Path
from app.config import ENCODINGS_FILE


def load_encodings():
    """
    Load face encodings from the pickle file.
    
    Returns:
        Dictionary mapping username to list of encodings:
        {"username": [embedding1, embedding2, ...], ...}
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


def save_encodings(encodings_dict):
    """
    Save face encodings to the pickle file.
    
    Args:
        encodings_dict: Dictionary mapping username to list of encodings
    """
    # Ensure directory exists
    ENCODINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(encodings_dict, f)
    except Exception as e:
        raise RuntimeError(f"Error saving encodings: {e}")


def add_user_encodings(username, new_encodings):
    """
    Add or update encodings for a user.
    
    Args:
        username: Name of the user
        new_encodings: List of new face encodings to add
    """
    # Load existing encodings
    encodings_dict = load_encodings()
    
    # Add or append encodings for this user
    if username in encodings_dict:
        # Append to existing encodings
        encodings_dict[username].extend(new_encodings)
    else:
        # Create new entry
        encodings_dict[username] = new_encodings
    
    # Save updated encodings
    save_encodings(encodings_dict)

