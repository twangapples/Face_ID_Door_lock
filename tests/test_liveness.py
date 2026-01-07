"""Unit and integration tests for ONNX liveness detection."""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.liveness_onnx import LivenessDetector
from app.config import LIVENESS_MODEL_PATH, LIVENESS_SCORE_THRESHOLD


class TestLivenessDetector:
    """Test suite for LivenessDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a LivenessDetector instance."""
        # Only create if model exists, otherwise return None
        if not LIVENESS_MODEL_PATH.exists():
            pytest.skip("Liveness model not found - skipping tests that require model")
        return LivenessDetector()
    
    @pytest.fixture
    def dummy_face_crop(self):
        """Create a dummy face crop image for testing."""
        # Create a 128x128 RGB image (simulating a face)
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        # Convert to BGR for OpenCV
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    @pytest.mark.skipif(
        not LIVENESS_MODEL_PATH.exists(),
        reason="Liveness model not found"
    )
    def test_model_load(self, detector):
        """Test that the model loads successfully."""
        assert detector.session is not None, "Model session should be initialized"
        assert detector._initialized, "Detector should be marked as initialized"
        assert detector.input_name is not None, "Input name should be set"
        assert detector.output_name is not None, "Output name should be set"
        assert detector.model_width is not None, "Model width should be detected"
        assert detector.model_height is not None, "Model height should be detected"
    
    @pytest.mark.skipif(
        not LIVENESS_MODEL_PATH.exists(),
        reason="Liveness model not found"
    )
    def test_predict_score(self, detector, dummy_face_crop):
        """Test that predict_score returns a valid score."""
        score = detector.predict_score(dummy_face_crop)
        
        assert isinstance(score, float), "Score should be a float"
        assert 0.0 <= score <= 1.0, f"Score should be in [0, 1], got {score}"
    
    @pytest.mark.skipif(
        not LIVENESS_MODEL_PATH.exists(),
        reason="Liveness model not found"
    )
    def test_is_live_threshold_behavior(self, detector, dummy_face_crop, monkeypatch):
        """Test that is_live respects threshold parameter."""
        # Mock predict_score to return specific values
        test_cases = [
            (0.8, 0.6, True),   # score > threshold -> True
            (0.5, 0.6, False),  # score < threshold -> False
            (0.6, 0.6, True),   # score == threshold -> True
            (0.0, 0.6, False),  # score == 0 -> False
            (1.0, 0.6, True),   # score == 1 -> True
        ]
        
        for score_value, threshold, expected in test_cases:
            monkeypatch.setattr(detector, 'predict_score', lambda x: score_value)
            result = detector.is_live(dummy_face_crop, threshold=threshold)
            assert result == expected, \
                f"is_live(score={score_value}, threshold={threshold}) should be {expected}, got {result}"
    
    @pytest.mark.skipif(
        not LIVENESS_MODEL_PATH.exists(),
        reason="Liveness model not found"
    )
    def test_is_live_default_threshold(self, detector, dummy_face_crop, monkeypatch):
        """Test that is_live uses config threshold when None provided."""
        # Mock predict_score
        monkeypatch.setattr(detector, 'predict_score', lambda x: 0.7)
        
        # Should use detector's score_threshold (from config)
        result = detector.is_live(dummy_face_crop, threshold=None)
        expected = 0.7 >= detector.score_threshold
        assert result == expected
    
    def test_predict_score_fail_closed(self):
        """Test that predict_score returns 0.0 when model not initialized."""
        # Create detector without model (use non-existent path)
        detector = LivenessDetector(model_path=Path("nonexistent_model.onnx"))
        
        dummy_crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        score = detector.predict_score(dummy_crop)
        
        assert score == 0.0, "Should return 0.0 (fail-closed) when model not loaded"
    
    def test_is_live_fail_closed(self):
        """Test that is_live returns False when model not initialized."""
        # Create detector without model
        detector = LivenessDetector(model_path=Path("nonexistent_model.onnx"))
        
        dummy_crop = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = detector.is_live(dummy_crop)
        
        assert result is False, "Should return False (fail-closed) when model not loaded"
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        not LIVENESS_MODEL_PATH.exists(),
        reason="Liveness model not found"
    )
    def test_predict_real_vs_spoof(self, detector):
        """Integration test: compare scores for real vs spoof images."""
        fixtures_dir = Path(__file__).parent / "fixtures" / "liveness"
        real_dir = fixtures_dir / "real"
        spoof_dir = fixtures_dir / "spoof"
        
        # Find first available image in each directory
        real_images = list(real_dir.glob("*.jpg")) if real_dir.exists() else []
        spoof_images = list(spoof_dir.glob("*.jpg")) if spoof_dir.exists() else []
        
        if not real_images or not spoof_images:
            pytest.skip(f"Test fixtures not found - need images in {real_dir} and {spoof_dir}")
        
        # Load first image from each directory
        real_img = cv2.imread(str(real_images[0]))
        spoof_img = cv2.imread(str(spoof_images[0]))
        
        if real_img is None or spoof_img is None:
            pytest.skip("Failed to load test fixtures")
        
        # Get scores
        real_score = detector.predict_score(real_img)
        spoof_score = detector.predict_score(spoof_img)
        
        print(f"\nReal face score: {real_score:.4f}")
        print(f"Spoof face score: {spoof_score:.4f}")
        
        # Real face should have higher score than spoof
        # Note: This assumes the model is correctly calibrated
        # If this fails, the model may need recalibration or threshold tuning
        assert real_score >= spoof_score, \
            f"Real face score ({real_score:.4f}) should be >= spoof score ({spoof_score:.4f})"
    
    @pytest.mark.skipif(
        not LIVENESS_MODEL_PATH.exists(),
        reason="Liveness model not found"
    )
    def test_preprocess_shape(self, detector, dummy_face_crop):
        """Test that preprocessing produces correct shape."""
        preprocessed = detector.preprocess(dummy_face_crop)
        
        assert preprocessed.ndim == 4, "Preprocessed should be 4D (batch, channels, height, width)"
        assert preprocessed.shape[0] == 1, "Batch size should be 1"
        
        if detector.is_nchw:
            assert preprocessed.shape[1] == 3, "Should have 3 channels in NCHW format"
            assert preprocessed.shape[2] == detector.model_height
            assert preprocessed.shape[3] == detector.model_width
        else:
            assert preprocessed.shape[1] == detector.model_height
            assert preprocessed.shape[2] == detector.model_width
            assert preprocessed.shape[3] == 3, "Should have 3 channels in NHWC format"
    
    @pytest.mark.skipif(
        not LIVENESS_MODEL_PATH.exists(),
        reason="Liveness model not found"
    )
    def test_preprocess_normalization(self, detector, dummy_face_crop):
        """Test that preprocessing normalizes values to [0, 1]."""
        preprocessed = detector.preprocess(dummy_face_crop)
        
        assert preprocessed.dtype == np.float32, "Should be float32"
        assert preprocessed.min() >= 0.0, "Values should be >= 0"
        assert preprocessed.max() <= 1.0, "Values should be <= 1"
    
    @pytest.mark.skipif(
        not LIVENESS_MODEL_PATH.exists(),
        reason="Liveness model not found"
    )
    def test_normalize_output_softmax(self, detector):
        """Test output normalization for softmax format."""
        # Simulate (1, 2) output
        output = np.array([[0.2, 0.8]])  # [spoof, live] probabilities
        prob_live = detector._normalize_output(output)
        
        assert 0.0 <= prob_live <= 1.0
        # After softmax normalization, should extract live probability
    
    @pytest.mark.skipif(
        not LIVENESS_MODEL_PATH.exists(),
        reason="Liveness model not found"
    )
    def test_normalize_output_sigmoid(self, detector):
        """Test output normalization for sigmoid format."""
        # Simulate single value output (logits)
        output = np.array([2.0])  # Logit value
        prob_live = detector._normalize_output(output)
        
        assert 0.0 <= prob_live <= 1.0
        # Should apply sigmoid to convert logit to probability

