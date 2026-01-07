"""ONNX-based liveness detection with auto-detection of model metadata."""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import onnxruntime as ort
from pathlib import Path
import logging
import time
from app.config import (
    LIVENESS_MODEL_PATH, LIVENESS_PROVIDERS, LIVENESS_SCORE_THRESHOLD,
    LIVENESS_BATCH, LIVENESS_IGNORE_EXCEPTIONS, LOGS_DIR
)

logger = logging.getLogger(__name__)


class LivenessDetector:
    """ONNX-based liveness detection wrapper with auto-detection of model requirements."""
    
    def __init__(self, model_path: Optional[Path] = None, 
                 providers: Optional[List[str]] = None,
                 score_threshold: Optional[float] = None):
        """
        Initialize liveness detector with ONNX model.
        
        Args:
            model_path: Path to ONNX model file (defaults to config path)
            providers: ONNX execution providers (defaults to config or CPU)
            score_threshold: Liveness score threshold (defaults to config)
        """
        if model_path is None:
            model_path = LIVENESS_MODEL_PATH
        
        if providers is None:
            providers = LIVENESS_PROVIDERS
        
        if score_threshold is None:
            score_threshold = LIVENESS_SCORE_THRESHOLD
        
        self.model_path = Path(model_path)
        self.score_threshold = score_threshold
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.is_nchw = True  # Default to NCHW layout
        self.model_width = None
        self.model_height = None
        self._initialized = False
        self._load_time = None
        
        # Set up providers
        if providers is None:
            providers = ['CPUExecutionProvider']
            try:
                # Try to add CUDA if available
                providers.append('CUDAExecutionProvider')
            except:
                pass
        
        self.providers = providers
        
        # Check if model exists
        if not self.model_path.exists():
            logger.warning(f"Liveness model not found at {self.model_path}. Liveness detection will be disabled.")
            logger.warning("Please download ONNX model from hairymax/Face-AntiSpoofing and place it in models/liveness/")
            return
        
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model and auto-detect input requirements."""
        load_start = time.time()
        
        try:
            # Create session
            if self.providers:
                self.session = ort.InferenceSession(
                    str(self.model_path),
                    providers=self.providers
                )
            else:
                self.session = ort.InferenceSession(str(self.model_path))
            
            # Get input metadata
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            self.input_shape = list(input_meta.shape)
            
            # Auto-detect layout (NCHW vs NHWC)
            # NCHW: shape is [N, C, H, W] where C is typically 1 or 3 at position 1
            # NHWC: shape is [N, H, W, C] where C is at position 3
            if len(self.input_shape) == 4:
                # Check channel position
                if self.input_shape[1] in (1, 3) and self.input_shape[1] < self.input_shape[2]:
                    # Likely NCHW: [N, C, H, W]
                    self.is_nchw = True
                    self.model_height = self.input_shape[2] if self.input_shape[2] is not None and self.input_shape[2] > 0 else None
                    self.model_width = self.input_shape[3] if self.input_shape[3] is not None and self.input_shape[3] > 0 else None
                elif self.input_shape[3] in (1, 3):
                    # Likely NHWC: [N, H, W, C]
                    self.is_nchw = False
                    self.model_height = self.input_shape[1] if self.input_shape[1] is not None and self.input_shape[1] > 0 else None
                    self.model_width = self.input_shape[2] if self.input_shape[2] is not None and self.input_shape[2] > 0 else None
                else:
                    # Default assumption: NCHW
                    self.is_nchw = True
                    # Try to infer from shape
                    if self.input_shape[2] and self.input_shape[2] > 0:
                        self.model_height = self.input_shape[2]
                        self.model_width = self.input_shape[3] if len(self.input_shape) > 3 else self.input_shape[2]
                    else:
                        self.model_height = 128  # Default
                        self.model_width = 128
            else:
                # Fallback: assume NCHW with default size
                self.is_nchw = True
                self.model_height = 128
                self.model_width = 128
            
            # Handle dynamic dimensions
            if self.model_height is None or self.model_width is None:
                # Use defaults if dynamic
                self.model_height = self.model_height or 128
                self.model_width = self.model_width or 128
                logger.warning(f"Model has dynamic input dimensions, using default {self.model_width}x{self.model_height}")
            
            # Get output metadata
            output_meta = self.session.get_outputs()[0]
            self.output_name = output_meta.name
            
            self._load_time = time.time() - load_start
            logger.info(f"Liveness model loaded: input shape {self.input_shape}, "
                       f"size {self.model_width}x{self.model_height}, "
                       f"layout {'NCHW' if self.is_nchw else 'NHWC'}, "
                       f"providers {self.providers}, "
                       f"load time {self._load_time:.3f}s")
            
            # Warmup
            self._warmup()
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to load liveness model: {e}", exc_info=True)
            self.session = None
            self._initialized = False
    
    def _warmup(self):
        """Warmup model with dummy input to reduce first-call latency."""
        if self.session is None:
            return
        
        try:
            # Create dummy input based on detected shape
            if self.is_nchw:
                dummy_input = np.zeros((1, 3, self.model_height, self.model_width), dtype=np.float32)
            else:
                dummy_input = np.zeros((1, self.model_height, self.model_width, 3), dtype=np.float32)
            
            _ = self.session.run([self.output_name], {self.input_name: dummy_input})
            logger.debug("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess face crop for model input.
        
        Args:
            face_crop: Face image crop (BGR format from OpenCV)
        
        Returns:
            Preprocessed array ready for model input
        """
        if face_crop is None or face_crop.size == 0:
            raise ValueError("Empty face crop provided")
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        # Use INTER_AREA for downscaling, INTER_LINEAR for upscaling
        if rgb.shape[0] > self.model_height or rgb.shape[1] > self.model_width:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        
        resized = cv2.resize(rgb, (self.model_width, self.model_height), interpolation=interpolation)
        
        # Convert to float32 and normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Arrange axes based on detected layout
        if self.is_nchw:
            # Transpose from HWC to CHW: (H, W, C) -> (C, H, W)
            transposed = np.transpose(normalized, (2, 0, 1))
        else:
            # Keep NHWC layout
            transposed = normalized
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def predict_score(self, face_crop: np.ndarray) -> float:
        """
        Predict liveness score for face crop.
        
        Args:
            face_crop: Face image crop (BGR format)
        
        Returns:
            Liveness score in [0, 1] where higher = more likely live
            Returns 0.0 if model not loaded or error occurs (fail-closed)
        """
        if not self._initialized or self.session is None:
            if not LIVENESS_IGNORE_EXCEPTIONS:
                logger.warning("Liveness model not initialized, denying access (fail-closed)")
                return 0.0
            return 0.0
        
        inference_start = time.time()
        
        try:
            # Preprocess
            preprocessed = self.preprocess(face_crop)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: preprocessed})
            output = outputs[0]
            
            inference_time = time.time() - inference_start
            logger.debug(f"Liveness inference time: {inference_time:.3f}s")
            
            # Log raw output for debugging
            logger.debug(f"Raw model output: shape={output.shape}, dtype={output.dtype}, min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
            
            # Normalize output to [0, 1] where higher = more likely live
            prob_live = self._normalize_output(output)
            
            logger.info(f"Normalized liveness score: {prob_live:.4f}")
            
            return prob_live
            
        except Exception as e:
            logger.error(f"Liveness prediction error: {e}", exc_info=True)
            if not LIVENESS_IGNORE_EXCEPTIONS:
                # Fail-closed: return 0.0 (deny access)
                return 0.0
            return 0.0
    
    def _normalize_output(self, output: np.ndarray) -> float:
        """
        Normalize model output to probability of live face.
        
        Handles different output formats:
        - Binary output (0 or 1): 0 = spoof, 1 = live
        - (1, 2) shape: softmax probabilities [spoof, live] or [live, spoof]
        - (1,) or (1,1) shape: single score, apply sigmoid if needed
        - Logits: apply sigmoid/softmax as needed
        
        Args:
            output: Raw model output
        
        Returns:
            Probability of live face [0, 1] where higher = more likely live
        """
        output = np.array(output)
        original_shape = output.shape
        
        # Log raw output for debugging
        logger.debug(f"Normalizing output: shape={original_shape}, dtype={output.dtype}, min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        
        # Squeeze to remove single dimensions
        output = output.squeeze()
        
        # Handle different output formats
        if output.ndim == 0:
            # Scalar output
            score = float(output)
            logger.debug(f"Scalar output: {score}")
            
            # Check if binary (0 or 1)
            if score == 0.0 or score == 1.0:
                # Binary output: 1 = live, 0 = spoof
                prob_live = score
                logger.debug(f"Binary output detected: {prob_live}")
            elif score < 0 or score > 1:
                # Likely logits, apply sigmoid
                prob_live = 1.0 / (1.0 + np.exp(-score))  # sigmoid
                logger.debug(f"Applied sigmoid to logit {score} -> {prob_live}")
            else:
                # Already in [0, 1] range
                prob_live = score
                
        elif output.ndim == 1:
            if len(output) == 2:
                # Two-class output
                logger.debug(f"Two-class output: [{output[0]:.4f}, {output[1]:.4f}]")
                
                # Check if binary (0/1) format
                if np.all(np.isin(output, [0, 1])):
                    # Binary format: [spoof, live] where 1 = live
                    prob_live = float(output[1])
                    logger.debug(f"Binary two-class format: prob_live = {prob_live}")
                else:
                    # Apply softmax for probabilities or logits
                    exp_output = np.exp(output - np.max(output))  # Numerical stability
                    probs = exp_output / np.sum(exp_output)
                    logger.debug(f"Softmax probabilities: [{probs[0]:.4f}, {probs[1]:.4f}]")
                    
                    # Try to determine order by checking which is higher
                    # Common convention: [spoof, live] or [live, spoof]
                    # For hairymax models, typically [spoof, live] where second is live
                    if probs[1] >= probs[0]:
                        # Likely [spoof, live] - take second
                        prob_live = float(probs[1])
                    else:
                        # Could be [live, spoof] - take first
                        prob_live = float(probs[0])
                    logger.debug(f"Selected prob_live = {prob_live:.4f} from softmax")
            else:
                # Single value in array
                score = float(output[0])
                logger.debug(f"Single value in array: {score}")
                
                if score == 0.0 or score == 1.0:
                    # Binary
                    prob_live = score
                elif score < 0 or score > 1:
                    prob_live = 1.0 / (1.0 + np.exp(-score))  # sigmoid
                else:
                    prob_live = score
                    
        else:
            # Multi-dimensional: flatten and take first element
            score = float(output.flat[0])
            logger.debug(f"Multi-dimensional output, using first element: {score}")
            
            if score == 0.0 or score == 1.0:
                prob_live = score
            elif score < 0 or score > 1:
                prob_live = 1.0 / (1.0 + np.exp(-score))  # sigmoid
            else:
                prob_live = score
        
        # Ensure in [0, 1] range
        prob_live = max(0.0, min(1.0, prob_live))
        
        logger.debug(f"Final normalized score: {prob_live:.4f}")
        
        return prob_live
    
    def is_live(self, face_crop: np.ndarray, threshold: Optional[float] = None) -> bool:
        """
        Return True if face is live.
        
        Args:
            face_crop: Face image crop (BGR format)
            threshold: Liveness threshold (defaults to config value)
        
        Returns:
            True if liveness score >= threshold, False otherwise
            Returns False on error (fail-closed)
        """
        if threshold is None:
            threshold = self.score_threshold
        
        try:
            score = self.predict_score(face_crop)
            return score >= threshold
        except Exception as e:
            logger.error(f"Error in is_live(): {e}", exc_info=True)
            if not LIVENESS_IGNORE_EXCEPTIONS:
                return False  # Fail-closed
            return False

