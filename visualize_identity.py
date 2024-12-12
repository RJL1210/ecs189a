import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import os

def visualize_identity_test(
    q: Dict[str, float],
    samples: List[int],
    epsilon: float,
    title: str,
    output_dir: str = "identity_visual"
):
    """
    Creates visualization for identity testing showing distribution comparison
    and Z statistic components.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup
    n = len(q)
    q_array = np.array([q[str(i+1)] for i in range(n)])
    
    # Calculate empirical distribution
    counts = Counter(samples)
    empirical_dist = np.zeros(n)
    for i in range(n):
        empirical_dist[i] = counts.get(i+1, 0) / len(samples)
    
    # Calculate Z statistic components
    threshold = epsilon / (50 * n)
    set_A = q_array >= threshold
    k = len(samples)
    
    Z_contributions = np.zeros(n)
    Z = 0
    for i in range(n):
        if set_A[i]:
            kqi = k * q_array[i]
            if kqi > 0:
                contribution = ((empirical_dist[i] * k - kqi) ** 2 - empirical_dist[i] * k) / kqi
                Z_contributions[i] = contribution
                Z += contribution
    
    # Normalize Z contributions for visualization
    if np.max(np.abs(Z_contributions)) > 0:
        Z_contributions = Z_contributions / np.max(np.abs(Z_contributions))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 2])
    
    # Plot 1: Distribution comparison
    x = np.arange(1, n+1)
    ax1.plot(x, q_array, color='blue', alpha=0.6, label='Target (q)')
    ax1.plot(x, empirical_dist, color='red', alpha=0.6, label='Observed (p)')
    ax1.set_xlabel('Domain')
    ax1.set_ylabel('Probability')
    ax1.set_title(title)
    ax1.legend()
    
    # Plot 2: Set A and Z contributions
    acceptance_threshold = (k * epsilon ** 2) / 10
    ax2.bar(x[set_A], np.ones(np.sum(set_A)), alpha=0.3, color='green', label='Set A (q[i] ≥ 0.000003)')
    ax2.bar(x, Z_contributions, alpha=0.6, color='red', label='Z contributions (scaled)')
    ax2.set_xlabel('Domain')
    ax2.set_ylabel('Probability / Scaled Contribution')
    ax2.set_title(f'Set A and Z Statistic Components\nZ={Z:.2f}, threshold={acceptance_threshold:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png"))
    plt.close()

def batch_visualize_identity_tests(test_dir: str, sample_dir: str, epsilon: float = 0.1):
    """
    Process all test cases and create visualizations.
    """
    import json
    
    for dist_file in sorted(os.listdir(test_dir)):
        if not dist_file.endswith('.json'):
            continue
            
        base_name = os.path.splitext(dist_file)[0]
        
        with open(os.path.join(test_dir, dist_file)) as f:
            q = json.load(f)
        
        for sample_type in ['close', 'far']:
            with open(os.path.join(sample_dir, f"{base_name}_X_{sample_type}.json")) as f:
                samples = json.load(f)["samples"]
            
            title = f"{base_name} Distribution Comparison ({sample_type} samples)"
            visualize_identity_test(q, samples, epsilon, title)

if __name__ == "__main__":
    batch_visualize_identity_tests("D", "X", epsilon=0.1)