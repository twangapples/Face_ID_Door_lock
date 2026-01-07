"""Threshold tuning script for liveness detection.

Loads labeled dataset and computes ROC curve, AUC, and suggests optimal threshold.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from typing import List, Tuple, Optional
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.liveness_onnx import LivenessDetector
from app.config import LIVENESS_MODEL_PATH


def load_images_from_dir(directory: Path) -> List[np.ndarray]:
    """Load all images from a directory."""
    images = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    if not directory.exists():
        print(f"Warning: Directory {directory} does not exist")
        return images
    
    for img_path in directory.iterdir():
        if img_path.suffix.lower() in image_extensions:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Failed to load {img_path}")
    
    return images


def compute_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.
    
    Args:
        y_true: True labels (1 for live, 0 for spoof)
        y_scores: Predicted scores (higher = more likely live)
    
    Returns:
        fpr, tpr, thresholds
    """
    # Sort by score (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # Get unique thresholds
    thresholds = np.unique(y_scores_sorted)
    thresholds = np.append(thresholds, [0.0, 1.0])  # Add endpoints
    thresholds = np.sort(thresholds)[::-1]  # Descending
    
    fpr = []
    tpr = []
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        print("Error: Need both positive and negative samples")
        return np.array([]), np.array([]), np.array([])
    
    for threshold in thresholds:
        # Predictions: score >= threshold -> live (1)
        y_pred = (y_scores >= threshold).astype(int)
        
        # True positives, false positives
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        tpr.append(tp / n_pos if n_pos > 0 else 0.0)
        fpr.append(fp / n_neg if n_neg > 0 else 0.0)
    
    return np.array(fpr), np.array(tpr), thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute Area Under Curve (AUC) using trapezoidal rule."""
    if len(fpr) == 0 or len(tpr) == 0:
        return 0.0
    
    # Sort by FPR
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr_sorted, fpr_sorted)
    return float(auc)


def find_optimal_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray,
                          target_far: float = 0.005) -> Tuple[float, float, float]:
    """
    Find optimal threshold targeting a specific False Accept Rate (FAR).
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Corresponding thresholds
        target_far: Target False Accept Rate (default 0.5%)
    
    Returns:
        optimal_threshold, far_at_threshold, fnr_at_threshold
    """
    # Find threshold closest to target FAR
    best_idx = np.argmin(np.abs(fpr - target_far))
    optimal_threshold = thresholds[best_idx]
    far_at_threshold = fpr[best_idx]
    fnr_at_threshold = 1.0 - tpr[best_idx]  # False Negative Rate = 1 - TPR
    
    return optimal_threshold, far_at_threshold, fnr_at_threshold


def main():
    """Main tuning function."""
    parser = argparse.ArgumentParser(description="Tune liveness detection threshold")
    parser.add_argument("--real-dir", type=str, default="tune/real",
                       help="Directory containing real face images")
    parser.add_argument("--spoof-dir", type=str, default="tune/spoof",
                       help="Directory containing spoof face images")
    parser.add_argument("--target-far", type=float, default=0.005,
                       help="Target False Accept Rate (default: 0.005 = 0.5%%)")
    parser.add_argument("--save-plot", action="store_true",
                       help="Save ROC curve plot (requires matplotlib)")
    parser.add_argument("--plot-path", type=str, default="roc_curve.png",
                       help="Path to save ROC curve plot")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not LIVENESS_MODEL_PATH.exists():
        print(f"Error: Liveness model not found at {LIVENESS_MODEL_PATH}")
        print("Please download ONNX model from hairymax/Face-AntiSpoofing and place it in models/liveness/")
        return 1
    
    # Initialize detector
    print("Loading liveness detector...")
    detector = LivenessDetector()
    
    if not detector._initialized:
        print("Error: Failed to initialize liveness detector")
        return 1
    
    print(f"Model loaded: {detector.model_path.name}")
    print(f"Input size: {detector.model_width}x{detector.model_height}")
    print()
    
    # Load images
    base_dir = Path(__file__).parent.parent
    real_dir = base_dir / args.real_dir
    spoof_dir = base_dir / args.spoof_dir
    
    print(f"Loading real face images from {real_dir}...")
    real_images = load_images_from_dir(real_dir)
    print(f"Loaded {len(real_images)} real face images")
    
    print(f"Loading spoof face images from {spoof_dir}...")
    spoof_images = load_images_from_dir(spoof_dir)
    print(f"Loaded {len(spoof_images)} spoof face images")
    print()
    
    if len(real_images) == 0 or len(spoof_images) == 0:
        print("Error: Need both real and spoof images for tuning")
        print(f"  Real images: {len(real_images)}")
        print(f"  Spoof images: {len(spoof_images)}")
        return 1
    
    # Get scores
    print("Computing liveness scores...")
    real_scores = []
    for img in real_images:
        score = detector.predict_score(img)
        real_scores.append(score)
    
    spoof_scores = []
    for img in spoof_images:
        score = detector.predict_score(img)
        spoof_scores.append(score)
    
    real_scores = np.array(real_scores)
    spoof_scores = np.array(spoof_scores)
    
    print(f"Real face scores: mean={real_scores.mean():.3f}, std={real_scores.std():.3f}, "
          f"min={real_scores.min():.3f}, max={real_scores.max():.3f}")
    print(f"Spoof face scores: mean={spoof_scores.mean():.3f}, std={spoof_scores.std():.3f}, "
          f"min={spoof_scores.min():.3f}, max={spoof_scores.max():.3f}")
    print()
    
    # Prepare labels and scores
    y_true = np.concatenate([
        np.ones(len(real_scores)),  # 1 for live
        np.zeros(len(spoof_scores))  # 0 for spoof
    ])
    y_scores = np.concatenate([real_scores, spoof_scores])
    
    # Compute ROC curve
    print("Computing ROC curve...")
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_scores)
    
    if len(fpr) == 0:
        print("Error: Failed to compute ROC curve")
        return 1
    
    # Compute AUC
    auc = compute_auc(fpr, tpr)
    print(f"AUC: {auc:.4f}")
    print()
    
    # Find optimal threshold
    optimal_threshold, far_at_threshold, fnr_at_threshold = find_optimal_threshold(
        fpr, tpr, thresholds, target_far=args.target_far
    )
    
    print("=" * 60)
    print("THRESHOLD TUNING RESULTS")
    print("=" * 60)
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Target FAR: {args.target_far:.1%}")
    print(f"  Achieved FAR: {far_at_threshold:.4%}")
    print(f"  False Negative Rate (FNR): {fnr_at_threshold:.4%}")
    print(f"  True Positive Rate (TPR): {1.0 - fnr_at_threshold:.4%}")
    print()
    print(f"AUC: {auc:.4f}")
    print()
    print("RECOMMENDATION:")
    print(f"  Set LIVENESS_SCORE_THRESHOLD = {optimal_threshold:.4f} in app/config.py")
    print("=" * 60)
    
    # Save plot if requested
    if args.save_plot:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'r--', label='Random classifier')
            plt.xlabel('False Positive Rate (FAR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('ROC Curve for Liveness Detection')
            plt.legend()
            plt.grid(True)
            
            # Mark optimal threshold point
            optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
            plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', 
                    label=f'Optimal threshold = {optimal_threshold:.4f}')
            plt.legend()
            
            plot_path = Path(__file__).parent.parent / args.plot_path
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\nROC curve saved to {plot_path}")
        except ImportError:
            print("\nWarning: matplotlib not available, skipping plot generation")
            print("  Install with: pip install matplotlib")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

