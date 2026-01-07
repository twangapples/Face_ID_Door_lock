"""Configuration settings for the face recognition system."""

import os
from pathlib import Path

# Base directory (parent of app/)
BASE_DIR = Path(__file__).parent.parent

# Embedding model settings
EMBEDDING_MODEL = "insightface"  # Model type: "insightface"
EMBEDDING_DIM = 512  # Dimension of face embeddings

# Recognition settings
RECOGNITION_THRESHOLD = 0.7  # Face distance threshold for matching (tuned for 512-D vectors)
PHOTOS_PER_USER = 10  # Number of photos to capture during enrollment

# Camera settings
CAMERA_INDEX = 0  # Default webcam index
MAX_FRAME_RATE = 10  # Maximum frames per second to process

# Liveness detection settings (ONNX-based)
LIVENESS_MODEL_PATH = BASE_DIR / "models" / "liveness" / "AntiSpoofing_bin_1.5_128.onnx"  # Path to ONNX model from hairymax/Face-AntiSpoofing
LIVENESS_PROVIDERS = None  # None = default (CPU), or ["CUDAExecutionProvider", "CPUExecutionProvider"] for GPU
LIVENESS_SCORE_THRESHOLD = 0.99  # float [0,1], higher = stricter (tune with scripts/tune_liveness.py)
LIVENESS_BATCH = 1  # Batch size for inference (usually 1 for real-time)
LIVENESS_IGNORE_EXCEPTIONS = False  # If False, on exception, is_live() returns False (fail-closed)

# Legacy liveness settings (kept for backward compatibility)
LIVENESS_ONNX_PATH = LIVENESS_MODEL_PATH  # Alias for backward compatibility
LIVENESS_INPUT_SIZE = (128, 128)  # Deprecated: auto-detected from model

# File paths
DATA_DIR = BASE_DIR / "data"
USERS_DIR = DATA_DIR / "users"
ENCODINGS_FILE = DATA_DIR / "encodings.pkl"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
INSIGHTFACE_MODEL_DIR = MODELS_DIR / "insightface"
LIVENESS_MODEL_DIR = MODELS_DIR / "liveness"

# Ensure directories exist
USERS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
INSIGHTFACE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LIVENESS_MODEL_DIR.mkdir(parents=True, exist_ok=True)

