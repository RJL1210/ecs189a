import numpy as np
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
import os

def identity_tester(q: Dict[str, float], samples: List[int], epsilon: float, plot: bool = False) -> Tuple[bool, dict]:
    """
    Algorithm 12.1: Identity Tester with detailed statistics.
    
    Returns:
        Tuple[bool, dict]: (decision, statistics)
    """
    n = len(q)
    m = int(np.ceil(16 * np.sqrt(n) / epsilon ** 2))
    
    k = np.random.poisson(m)
    if k > 2*m:
        return False, {'Z_statistic': None, 'early_exit': True}
        
    k = min(k, len(samples))
    samples = samples[:k]
    
    # Count occurrences
    Ni = np.zeros(n)
    for sample in samples:
        Ni[sample - 1] += 1
    
    # Compute A
    q_array = np.array([q[str(i + 1)] for i in range(n)])
    threshold = epsilon / (50 * n)
    A = np.where(q_array >= threshold)[0]
    
    # Compute Z
    Z = 0
    for i in A:
        kqi = k * q_array[i]
        if kqi > 0:
            Z += ((Ni[i] - kqi) ** 2 - Ni[i]) / kqi
    
    # Acceptance threshold
    acceptance_threshold = (k * epsilon ** 2) / 10
    decision = Z <= acceptance_threshold
    
    # Calculate statistics similar to Goldreich's
    stats = {
        'Z_statistic': Z,
        'threshold': acceptance_threshold,
        'sample_size': k,
        'expected_samples': m,
        'set_A_size': len(A),
        'support_size': np.sum(Ni > 0),  # Analogous to Goldreich's support size
        'tvd_estimate': 0.5 * np.sum(np.abs(Ni/k - q_array))  # Total variation distance
    }
    
    #if plot:
        #plot_distribution_comparison(q, Ni, k, n)
    
    return decision, stats

def batch_compare_testers(test_dir: str, sample_dir: str, epsilon: float = 0.1):
    """Run both testers with comparable output format"""
    from goldreich_tester import majority_rules_test
    
    for dist_file in sorted(os.listdir(test_dir)):
        if not dist_file.endswith('.json'):
            continue
            
        print(f"\nTesting {os.path.splitext(dist_file)[0]}:")
        with open(os.path.join(test_dir, dist_file)) as f:
            q = json.load(f)
        
        for sample_type in ['close', 'far']:
            with open(os.path.join(sample_dir, f"{os.path.splitext(dist_file)[0]}_X_{sample_type}.json")) as f:
                samples = json.load(f)["samples"]
            
            print(f"\n{sample_type.upper()} samples:")
            
            # Identity tester
            id_decision, id_stats = identity_tester(q, samples, epsilon)
            print("Identity Tester:")
            print(f"  Decision: {'Accept' if id_decision else 'Reject'}")
            print(f"  Z statistic: {id_stats['Z_statistic']:.6f}")
            print(f"  Threshold: {id_stats['threshold']:.6f}")
            print(f"  Sample size: {id_stats['sample_size']} (expected: {id_stats['expected_samples']})")
            print(f"  Support size: {id_stats['support_size']}")
            print(f"  TVD estimate: {id_stats['tvd_estimate']:.6f}")
            
            # Goldreich tester
            gold_decision, gold_stats = majority_rules_test(q, samples, epsilon)
            print("\nGoldreich Tester:")
            print(f"  Decision: {'Accept' if gold_decision else 'Reject'}")
            print(f"  Votes: {gold_stats['positive_votes']}/{gold_stats['total_trials']}")
            print(f"  Collision rate: {gold_stats['avg_collision_rate']:.6f}")
            print(f"  Support size: {gold_stats['avg_support_size']:.1f}")
        

if __name__ == "__main__":
    # Run all tests with 5 repetitions per test and plot for the first run
    batch_compare_testers("D", "X", epsilon=0.1)
