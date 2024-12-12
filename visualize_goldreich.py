import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from goldreich import goldreich_reduction
import math
import os
import json

def analyze_distribution_transformation(D: Dict[str, float], p_samples: List[int], epsilon: float) -> Dict[str, Dict]:
    n = len(D)
    required_samples = int(16 * math.sqrt(n) / epsilon**2)
    p_samples = p_samples[:required_samples]
    
    # Create directory for saving visualizations
    os.makedirs("goldreich_visual", exist_ok=True)
    
    # Original distribution
    original_hist = Counter(p_samples)
    original_dist = {i: original_hist.get(i, 0) / len(p_samples) for i in range(1, n+1)}
    
    # After Goldreich reduction
    mapped_samples, m = goldreich_reduction(D, p_samples, epsilon)
    
    # Final distribution
    final_hist = Counter(mapped_samples)
    final_dist = {i: final_hist.get(i, 0) / len(mapped_samples) for i in range(1, m+1)}
    
    # Statistics
    filtered = [s for s in mapped_samples if 1 <= s <= m]
    collisions = sum(c * (c-1) // 2 for c in Counter(filtered).values())
    total = len(filtered) * (len(filtered) - 1) // 2
    collision_rate = collisions / total if total > 0 else 0
    
    # Plot original distribution
    plt.figure(figsize=(10, 6))
    plt.bar(original_dist.keys(), original_dist.values(), alpha=0.6, label="Original Distribution")
    plt.xlabel("Element")
    plt.ylabel("Probability")
    plt.title("Original Distribution")
    plt.legend()
    #plt.savefig("goldreich_visual/original_distribution.png")
    plt.close()
    
    # Plot final distribution
    plt.figure(figsize=(10, 6))
    plt.bar(final_dist.keys(), final_dist.values(), alpha=0.6, label="Final Distribution")
    plt.xlabel("Element")
    plt.ylabel("Probability")
    plt.title("Final Distribution")
    plt.legend()
    #plt.savefig("goldreich_visual/final_distribution.png")
    plt.close()
    
    # Plot comparison of original and final distributions
    plt.figure(figsize=(10, 6))
    plt.bar(original_dist.keys(), original_dist.values(), alpha=0.6, label="Original Distribution")
    plt.bar(final_dist.keys(), final_dist.values(), alpha=0.6, label="Final Distribution")
    plt.xlabel("Element")
    plt.ylabel("Probability")
    plt.title("Comparison of Original and Final Distributions")
    plt.legend()
    #plt.savefig("goldreich_visual/comparison_distribution.png")
    plt.close()
    
    # Save statistics to a text file
    """
    with open("goldreich_visual/distribution_stats.txt", "w") as f:
        f.write("Distribution stats:\n")
        f.write(f"Original range: {min(original_dist.values()):.6f} to {max(original_dist.values()):.6f}\n")
        f.write(f"Final range: {min(final_dist.values()):.6f} to {max(final_dist.values()):.6f}\n")
        f.write(f"Collision rate: {collision_rate:.6f}\n")
        f.write(f"Sample sizes - Original: {len(p_samples)}, Mapped: {len(mapped_samples)}\n")
    """
    
    return {
        'original': original_dist,
        'final': final_dist,
        'stats': {
            'collision_rate': collision_rate
        }
    }

def visualize_batch_results(test_dir: str, sample_dir: str, epsilon: float):
    distribution_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".json")])
    
    for dist_file in distribution_files:
        base = os.path.splitext(dist_file)[0]
        
        with open(os.path.join(test_dir, dist_file)) as f:
            D = json.load(f)
        
        # Analyze close samples
        with open(os.path.join(sample_dir, f"{base}_X_close.json")) as f:
            close = json.load(f)["samples"]
        close_results = analyze_distribution_transformation(D, close, epsilon)
        
        # Create a new figure for each distribution
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"{base} Distribution Analysis")
        
        # Plot close samples
        plt.subplot(221)
        plt.bar(list(close_results['original'].keys()), list(close_results['original'].values()), alpha=0.6)
        plt.title('Close Samples - Original')
        plt.ylabel('Probability')
        
        plt.subplot(223)
        plt.bar(list(close_results['final'].keys()), list(close_results['final'].values()), alpha=0.6)
        plt.title(f"Close Samples - After Reduction (CR: {close_results['stats']['collision_rate']:.6f})")
        plt.ylabel('Probability')
        
        # Analyze and plot far samples
        with open(os.path.join(sample_dir, f"{base}_X_far.json")) as f:
            far = json.load(f)["samples"]
        far_results = analyze_distribution_transformation(D, far, epsilon)
        
        plt.subplot(222)
        plt.bar(list(far_results['original'].keys()), list(far_results['original'].values()), alpha=0.6)
        plt.title('Far Samples - Original')
        
        plt.subplot(224)
        plt.bar(list(far_results['final'].keys()), list(far_results['final'].values()), alpha=0.6)
        plt.title(f"Far Samples - After Reduction (CR: {far_results['stats']['collision_rate']:.6f})")
        
        plt.tight_layout()
        plt.savefig(f"goldreich_visual/{base}_distribution_analysis.png")
        plt.close()

if __name__ == "__main__":
    test_dir = './D'
    sample_dir = './X'
    epsilon = 0.1
    visualize_batch_results(test_dir, sample_dir, epsilon)