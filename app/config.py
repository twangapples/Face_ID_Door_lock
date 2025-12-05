"""Configuration settings for the face recognition system."""

import os
from pathlib import Path

# Base directory (parent of app/)
BASE_DIR = Path(__file__).parent.parent

# Recognition settings
RECOGNITION_THRESHOLD = 0.4  # Face distance threshold for matching
PHOTOS_PER_USER = 5  # Number of photos to capture during enrollment

# Camera settings
CAMERA_INDEX = 0  # Default webcam index

# File paths
DATA_DIR = BASE_DIR / "data"
USERS_DIR = DATA_DIR / "users"
ENCODINGS_FILE = DATA_DIR / "encodings.pkl"

# Ensure directories exist
USERS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

