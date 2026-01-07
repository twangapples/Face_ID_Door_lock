"""Benchmark script to evaluate liveness detection performance."""

import sys
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.liveness_onnx import LivenessDetector
from app.config import LIVENESS_SCORE_THRESHOLD


def evaluate_liveness_model(detector: LivenessDetector, real_images: List[Path], 
                           spoof_images: List[Path]) -> dict:
    """
    Evaluate liveness model on real and spoof images.
    
    Args:
        detector: LivenessDetector instance
        real_images: List of paths to real face images
        spoof_images: List of paths to spoof (photo/screen) images
    
    Returns:
        Dictionary with evaluation metrics
    """
    if not detector._initialized:
        return {"error": "Liveness detector not initialized"}
    
    real_scores = []
    spoof_scores = []
    
    # Evaluate real images
    print("Evaluating real images...")
    for img_path in real_images:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            score = detector.predict_score(img)
            real_scores.append(score)
            print(f"  {img_path.name}: {score:.3f}")
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
    
    # Evaluate spoof images
    print("\nEvaluating spoof images...")
    for img_path in spoof_images:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            score = detector.predict_score(img)
            spoof_scores.append(score)
            print(f"  {img_path.name}: {score:.3f}")
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
    
    if len(real_scores) == 0 or len(spoof_scores) == 0:
        return {"error": "Insufficient data for evaluation"}
    
    # Compute metrics
    real_mean = np.mean(real_scores)
    real_std = np.std(real_scores)
    spoof_mean = np.mean(spoof_scores)
    spoof_std = np.std(spoof_scores)
    
    # ROC analysis: try different thresholds
    all_scores = real_scores + spoof_scores
    all_labels = [1] * len(real_scores) + [0] * len(spoof_scores)
    
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_threshold = None
    best_accuracy = 0
    
    for threshold in thresholds:
        predictions = [1 if score >= threshold else 0 for score in all_scores]
        accuracy = sum(p == l for p, l in zip(predictions, all_labels)) / len(all_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    # Compute metrics at current threshold
    current_threshold = LIVENESS_SCORE_THRESHOLD
    tp = sum(1 for s in real_scores if s >= current_threshold)
    fn = sum(1 for s in real_scores if s < current_threshold)
    tn = sum(1 for s in spoof_scores if s < current_threshold)
    fp = sum(1 for s in spoof_scores if s >= current_threshold)
    
    tpr = tp / len(real_scores) if len(real_scores) > 0 else 0  # True Positive Rate (sensitivity)
    fpr = fp / len(spoof_scores) if len(spoof_scores) > 0 else 0  # False Positive Rate
    tnr = tn / len(spoof_scores) if len(spoof_scores) > 0 else 0  # True Negative Rate (specificity)
    fnr = fn / len(real_scores) if len(real_scores) > 0 else 0  # False Negative Rate
    
    return {
        "real_scores": {
            "mean": real_mean,
            "std": real_std,
            "min": np.min(real_scores),
            "max": np.max(real_scores),
            "count": len(real_scores)
        },
        "spoof_scores": {
            "mean": spoof_mean,
            "std": spoof_std,
            "min": np.min(spoof_scores),
            "max": np.max(spoof_scores),
            "count": len(spoof_scores)
        },
        "best_threshold": best_threshold,
        "best_accuracy": best_accuracy,
        "current_threshold": current_threshold,
        "metrics_at_current": {
            "TPR (sensitivity)": tpr,
            "FPR": fpr,
            "TNR (specificity)": tnr,
            "FNR": fnr,
            "Accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        }
    }


def main():
    """Run liveness benchmark."""
    print("=== Liveness Detection Benchmark ===")
    print("This script evaluates liveness detection on real and spoof images\n")
    
    # Initialize detector
    print("Initializing liveness detector...")
    detector = LivenessDetector()
    
    if not detector._initialized:
        print("Error: Liveness detector not initialized.")
        print("Please ensure the liveness ONNX model is placed in models/liveness/")
        return
    
    print("✓ Liveness detector ready\n")
    
    # Get image directories
    base_dir = Path(__file__).parent.parent
    real_dir = base_dir / "tests" / "fixtures" / "liveness" / "real"
    spoof_dir = base_dir / "tests" / "fixtures" / "liveness" / "spoof"
    
    if not real_dir.exists() or not spoof_dir.exists():
        print("Test fixtures not found.")
        print(f"Expected directories:")
        print(f"  {real_dir}")
        print(f"  {spoof_dir}")
        print("\nPlease create these directories and add test images:")
        print("  - real/: images of real faces")
        print("  - spoof/: images of printed photos/screens")
        return
    
    # Find images
    real_images = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.jpeg")) + list(real_dir.glob("*.png"))
    spoof_images = list(spoof_dir.glob("*.jpg")) + list(spoof_dir.glob("*.jpeg")) + list(spoof_dir.glob("*.png"))
    
    if len(real_images) == 0 or len(spoof_images) == 0:
        print("Insufficient test images found.")
        print(f"  Real images: {len(real_images)}")
        print(f"  Spoof images: {len(spoof_images)}")
        return
    
    print(f"Found {len(real_images)} real images and {len(spoof_images)} spoof images\n")
    
    # Evaluate
    results = evaluate_liveness_model(detector, real_images, spoof_images)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Print results
    print("\n=== Results ===")
    print(f"\nReal images (should be high scores):")
    print(f"  Mean: {results['real_scores']['mean']:.3f} ± {results['real_scores']['std']:.3f}")
    print(f"  Range: [{results['real_scores']['min']:.3f}, {results['real_scores']['max']:.3f}]")
    
    print(f"\nSpoof images (should be low scores):")
    print(f"  Mean: {results['spoof_scores']['mean']:.3f} ± {results['spoof_scores']['std']:.3f}")
    print(f"  Range: [{results['spoof_scores']['min']:.3f}, {results['spoof_scores']['max']:.3f}]")
    
    print(f"\n=== Threshold Analysis ===")
    print(f"Current threshold: {results['current_threshold']:.3f}")
    metrics = results['metrics_at_current']
    print(f"  TPR (True Positive Rate / Sensitivity): {metrics['TPR (sensitivity)']*100:.2f}%")
    print(f"  TNR (True Negative Rate / Specificity): {metrics['TNR (specificity)']*100:.2f}%")
    print(f"  FPR (False Positive Rate): {metrics['FPR']*100:.2f}%")
    print(f"  FNR (False Negative Rate): {metrics['FNR']*100:.2f}%")
    print(f"  Accuracy: {metrics['Accuracy']*100:.2f}%")
    
    print(f"\nRecommended threshold: {results['best_threshold']:.3f}")
    print(f"  (achieves {results['best_accuracy']*100:.2f}% accuracy)")


if __name__ == "__main__":
    main()

