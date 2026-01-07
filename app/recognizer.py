"""Module for face recognition: InsightFace-based encoding and matching."""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import insightface
from app.config import RECOGNITION_THRESHOLD, EMBEDDING_DIM
from app.utils import l2_distance


class InsightFaceRecognizer:
    """InsightFace-based face recognition wrapper."""
    
    def __init__(self, ctx_id: int = -1, det_size: Tuple[int, int] = (640, 640)):
        """
        Initialize InsightFace FaceAnalysis model.
        
        Args:
            ctx_id: -1 for CPU, 0+ for GPU index
            det_size: Detector input size (width, height)
        """
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.model = None
        self._initialized = False
    
    def prepare(self):
        """Prepare/warmup the model."""
        if not self._initialized:
            try:
                # Initialize InsightFace FaceAnalysis
                # This will automatically download models on first run
                self.model = insightface.app.FaceAnalysis(
                    name='buffalo_l',  # Use buffalo_l model (good balance of speed/accuracy)
                    providers=['CPUExecutionProvider'] if self.ctx_id == -1 else ['CUDAExecutionProvider']
                )
                self.model.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
                self._initialized = True
                print("InsightFace model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize InsightFace model: {e}")
    
    def detect_and_encode(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image and return list of face dictionaries.
        
        Args:
            image: Input image (BGR format from OpenCV)
        
        Returns:
            List of face dictionaries:
            [
                {
                    'bbox': (top, left, bottom, right),
                    'embedding': np.ndarray(shape=(512,), dtype=float),
                    'landmarks': np.ndarray(shape=(5,2))  # 5 facial landmarks
                },
                ...
            ]
        """
        if not self._initialized:
            self.prepare()
        
        if self.model is None:
            return []
        
        # InsightFace expects RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.model.get(rgb_image)
        
        result = []
        for face in faces:
            # Extract bounding box (x1, y1, x2, y2)
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Convert to (top, left, bottom, right) format
            top, left, bottom, right = y1, x1, y2, x2
            
            # Get embedding (512-D)
            embedding = face.normed_embedding
            
            # Get landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
            landmarks = face.kps if hasattr(face, 'kps') else None
            
            result.append({
                'bbox': (top, left, bottom, right),
                'embedding': embedding,
                'landmarks': landmarks
            })
        
        return result
    
    def encode_face_crop(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Return 512-D embedding for single face crop.
        
        Args:
            face_crop: Cropped face image (BGR format)
        
        Returns:
            512-D embedding or None if face not found
        """
        if not self._initialized:
            self.prepare()
        
        if self.model is None:
            return None
        
        # Convert to RGB
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Detect and encode
        faces = self.model.get(rgb_crop)
        
        if len(faces) > 0:
            return faces[0].normed_embedding
        
        return None


def encode_face(image_path: str, recognizer: Optional[InsightFaceRecognizer] = None) -> Optional[np.ndarray]:
    """
    Generate face encoding from an image file.
    
    Args:
        image_path: Path to the image file
        recognizer: Optional InsightFaceRecognizer instance (creates new if None)
    
    Returns:
        Face encoding (512-dim numpy array) or None if no face found
    """
    if recognizer is None:
        recognizer = InsightFaceRecognizer(ctx_id=-1)
        recognizer.prepare()
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Encode
    return recognizer.encode_face_crop(image)
