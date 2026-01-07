"""Benchmark script to compute recognition distances for threshold tuning."""

import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import load_encodings
from app.utils import l2_distance


def main():
    """Compute intra-class and inter-class distances."""
    print("=== Recognition Distance Benchmark ===")
    print("Computing intra-class (same person) and inter-class (different person) distances\n")
    
    # Load encodings
    encodings_dict = load_encodings()
    
    if len(encodings_dict) < 2:
        print("Need at least 2 users with multiple encodings for meaningful statistics.")
        return
    
    # Collect all encodings by user
    user_encodings = {}
    for username, user_data in encodings_dict.items():
        if isinstance(user_data, dict) and 'embeddings' in user_data:
            embeddings = user_data['embeddings']
            if len(embeddings) > 0:
                user_encodings[username] = embeddings
    
    if len(user_encodings) < 2:
        print("Need at least 2 users with encodings.")
        return
    
    # Compute intra-class distances (same person)
    intra_distances = []
    for username, embeddings in user_encodings.items():
        if len(embeddings) < 2:
            continue
        
        # Compare all pairs within this user
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = l2_distance(embeddings[i], embeddings[j])
                intra_distances.append(dist)
    
    # Compute inter-class distances (different people)
    inter_distances = []
    usernames = list(user_encodings.keys())
    for i in range(len(usernames)):
        for j in range(i + 1, len(usernames)):
            user1 = usernames[i]
            user2 = usernames[j]
            
            # Compare all pairs between users
            for emb1 in user_encodings[user1]:
                for emb2 in user_encodings[user2]:
                    dist = l2_distance(emb1, emb2)
                    inter_distances.append(dist)
    
    # Statistics
    if len(intra_distances) > 0:
        intra_mean = np.mean(intra_distances)
        intra_std = np.std(intra_distances)
        intra_min = np.min(intra_distances)
        intra_max = np.max(intra_distances)
        intra_median = np.median(intra_distances)
        
        print("Intra-class distances (same person):")
        print(f"  Mean:   {intra_mean:.4f}")
        print(f"  Std:    {intra_std:.4f}")
        print(f"  Median: {intra_median:.4f}")
        print(f"  Min:    {intra_min:.4f}")
        print(f"  Max:    {intra_max:.4f}")
        print(f"  Count:  {len(intra_distances)}")
    
    if len(inter_distances) > 0:
        inter_mean = np.mean(inter_distances)
        inter_std = np.std(inter_distances)
        inter_min = np.min(inter_distances)
        inter_max = np.max(inter_distances)
        inter_median = np.median(inter_distances)
        
        print("\nInter-class distances (different people):")
        print(f"  Mean:   {inter_mean:.4f}")
        print(f"  Std:    {inter_std:.4f}")
        print(f"  Median: {inter_median:.4f}")
        print(f"  Min:    {inter_min:.4f}")
        print(f"  Max:    {inter_max:.4f}")
        print(f"  Count:  {len(inter_distances)}")
    
    # Threshold recommendation
    if len(intra_distances) > 0 and len(inter_distances) > 0:
        # Find threshold that maximizes separation
        all_distances = sorted(intra_distances + inter_distances)
        
        best_threshold = None
        best_separation = 0
        
        for threshold in np.arange(0.2, 1.0, 0.05):
            # False reject rate (intra-class above threshold)
            frr = sum(1 for d in intra_distances if d > threshold) / len(intra_distances)
            # False accept rate (inter-class below threshold)
            far = sum(1 for d in inter_distances if d <= threshold) / len(inter_distances)
            
            # Separation score (lower is better for both)
            separation = frr + far
            
            if separation < best_separation or best_threshold is None:
                best_separation = separation
                best_threshold = threshold
        
        print(f"\n=== Threshold Recommendation ===")
        print(f"Suggested threshold: {best_threshold:.3f}")
        print(f"  (minimizes FRR + FAR)")
        print(f"\nAt threshold {best_threshold:.3f}:")
        frr = sum(1 for d in intra_distances if d > best_threshold) / len(intra_distances)
        far = sum(1 for d in inter_distances if d <= best_threshold) / len(inter_distances)
        print(f"  False Reject Rate (FRR): {frr*100:.2f}%")
        print(f"  False Accept Rate (FAR):  {far*100:.2f}%")
        print(f"\nCurrent config threshold: {sys.path[0] if sys.path else 'check config.py'}")


if __name__ == "__main__":
    main()

