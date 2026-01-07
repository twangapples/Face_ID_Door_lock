"""Utility functions for face recognition system."""

import cv2
import numpy as np
from typing import Tuple, Optional
import time
import json
from pathlib import Path
from datetime import datetime


def crop_face_from_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int], pad: float = 0.2) -> np.ndarray:
    """
    Crop face from image using bounding box with optional padding.
    
    Args:
        image: Input image (BGR format from OpenCV)
        bbox: Bounding box as (top, left, bottom, right) or (x1, y1, x2, y2)
        pad: Padding factor (0.2 = 20% padding on each side)
    
    Returns:
        Cropped face image
    """
    h, w = image.shape[:2]
    
    # Handle different bbox formats
    if len(bbox) == 4:
        if bbox[0] < bbox[2]:  # (top, left, bottom, right)
            top, left, bottom, right = bbox
        else:  # (x1, y1, x2, y2)
            left, top, right, bottom = bbox
    else:
        raise ValueError(f"Invalid bbox format: {bbox}")
    
    # Calculate padding
    width = right - left
    height = bottom - top
    pad_w = int(width * pad)
    pad_h = int(height * pad)
    
    # Apply padding with boundary checks
    left = max(0, left - pad_w)
    top = max(0, top - pad_h)
    right = min(w, right + pad_w)
    bottom = min(h, bottom + pad_h)
    
    # Crop and return
    face_crop = image[top:bottom, left:right]
    return face_crop


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate L2 (Euclidean) distance between two vectors.
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(a - b))


class FrameRateLimiter:
    """Limits processing to a maximum frame rate."""
    
    def __init__(self, max_fps: float):
        """
        Initialize frame rate limiter.
        
        Args:
            max_fps: Maximum frames per second to process
        """
        self.max_fps = max_fps
        self.min_interval = 1.0 / max_fps if max_fps > 0 else 0
        self.last_time = 0
    
    def should_process(self) -> bool:
        """
        Check if enough time has passed to process the next frame.
        
        Returns:
            True if frame should be processed, False otherwise
        """
        if self.min_interval == 0:
            return True
        
        current_time = time.time()
        if current_time - self.last_time >= self.min_interval:
            self.last_time = current_time
            return True
        return False
    
    def reset(self):
        """Reset the limiter."""
        self.last_time = 0


def normalize_image(img: np.ndarray, mode: str = "default") -> np.ndarray:
    """
    Apply normalization to image.
    
    Args:
        img: Input image (RGB or BGR format, uint8)
        mode: Normalization mode:
            - "default": Scale to [0, 1] by dividing by 255.0
            - "imagenet": Apply ImageNet mean/std normalization
            - Other modes can be added as needed
    
    Returns:
        Normalized image (float32)
    """
    if img.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {img.dtype}")
    
    img_float = img.astype(np.float32)
    
    if mode == "default":
        # Simple [0, 1] normalization
        normalized = img_float / 255.0
    elif mode == "imagenet":
        # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # Assuming RGB input
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Normalize to [0, 1] first
        normalized = img_float / 255.0
        
        # Apply mean/std
        if len(normalized.shape) == 3 and normalized.shape[2] == 3:
            # RGB image
            normalized = (normalized - mean) / std
        else:
            # Grayscale or unexpected format, just use default
            normalized = img_float / 255.0
    else:
        # Unknown mode, use default
        normalized = img_float / 255.0
    
    return normalized


def save_denied_attempt(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                        username: Optional[str], score: float, 
                        recognition_distance: Optional[float] = None,
                        model_name: Optional[str] = None,
                        model_input_shape: Optional[Tuple] = None,
                        base_path: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Save denied attempt image and metadata for audit.
    
    Args:
        frame: Full frame image (BGR format)
        bbox: Bounding box (top, left, bottom, right)
        username: Username if recognized, None otherwise
        score: Liveness score
        recognition_distance: Recognition distance if available
        model_name: Name of liveness model used
        model_input_shape: Input shape of the model
        base_path: Base directory for saving (defaults to data/logs/denied)
    
    Returns:
        Tuple of (image_path, json_path) where files were saved
    """
    from app.config import LOGS_DIR
    
    if base_path is None:
        base_path = LOGS_DIR / "denied"
    
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove last 3 digits of microseconds
    username_str = username or "unknown"
    
    # Crop face from frame
    try:
        face_crop = crop_face_from_bbox(frame, bbox, pad=0.2)
    except Exception as e:
        # If cropping fails, use full frame
        face_crop = frame
    
    # Save image
    image_filename = f"{timestamp}_{username_str}.jpg"
    image_path = base_path / image_filename
    cv2.imwrite(str(image_path), face_crop)
    
    # Save metadata JSON
    json_filename = f"{timestamp}_{username_str}.json"
    json_path = base_path / json_filename
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "username": username,
        "liveness_score": float(score),
        "recognition_distance": float(recognition_distance) if recognition_distance is not None else None,
        "bbox": list(bbox),
        "model_name": model_name,
        "model_input_shape": list(model_input_shape) if model_input_shape else None,
        "image_path": str(image_path.relative_to(base_path.parent.parent))
    }
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return image_path, json_path

