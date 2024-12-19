import math
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from goldreich import goldreich_reduction
from visualize_goldreich import visualize_batch_results

DEBUG = False

from typing import List
from collections import Counter

def lecture11_collision_tester(samples: List[int], n: int, epsilon: float) -> bool:
    # Apply sample complexity limit here
    required = int(math.sqrt(n) / (epsilon * epsilon))
    samples = samples[:required]  # Now we limit samples just for collision testing
    
    # Rest of collision test remains the same
    filtered = [s for s in samples if 1 <= s <= n]
    m = len(filtered)
    
    if m < 2:
        return True
    
    freq = Counter(filtered)
    collision_count = sum(c * (c-1) // 2 for c in freq.values())
    total_pairs = (m * (m-1)) // 2
    
    if total_pairs == 0:
        return True
    
    C = collision_count / total_pairs
    expected_rate = 1.0/n
    relative_rate = C/expected_rate
    
    tolerance = 1 + epsilon
    return relative_rate <= tolerance
    
    
def majority_rules_test(D: Dict[str, float], p_samples: List[int], 
                       epsilon: float, num_trials: int = 1) -> Tuple[bool, dict]:
    """
    Single trial test with basic statistics.
    """
    n = len(D)
    sample_complexity = int(2 * np.ceil(np.sqrt(n) / (epsilon * epsilon)))
    if len(p_samples) <= sample_complexity:
        raise ValueError("Sample size too small")
    
    p_samples = p_samples[:sample_complexity] #estimate amount of required samples before goldreichs since itll cut about 50% of the samples
    
    mapped_samples, m = goldreich_reduction(D, p_samples, epsilon)
    result = lecture11_collision_tester(mapped_samples, m, epsilon/3)
    
    # Calculate basic statistics
    valid = [s for s in mapped_samples if 1 <= s <= m]
    freq = Counter(valid)
    collisions = sum(c * (c-1) // 2 for c in freq.values())
    total = len(valid) * (len(valid) - 1) // 2
    collision_rate = collisions / total if total > 0 else 0
    
    stats = {
        'positive_votes': 1 if result else 0,
        'total_trials': 1,
        'vote_ratio': 1.0 if result else 0.0,
        'avg_collision_rate': collision_rate,
        'avg_support_size': len(freq),
        'min_support': len(freq),
        'max_support': len(freq),
        'expected_collision_rate': 1.0/m,
        'relative_collision_rate': collision_rate * m
    }
    
    return result, stats

def batch_test(test_dir: str, sample_dir: str, epsilon: float = 0.1):
    """
    Test samples against their target distributions.
    
    Args:
        test_dir: Directory containing target distributions
        sample_dir: Directory containing sample sets
        epsilon: Proximity parameter
    """
    import os, json
    
    results = {}
    for dist_file in sorted(os.listdir(test_dir)):
        if not dist_file.endswith('.json'):
            continue
            
        base = os.path.splitext(dist_file)[0]
        
        # Load target distribution
        with open(os.path.join(test_dir, dist_file)) as f:
            D = json.load(f)
        
        # Test both close and far samples against target distribution
        for sample_type in ['close', 'far']:
            with open(os.path.join(sample_dir, f"{base}_X_{sample_type}.json")) as f:
                samples = json.load(f)["samples"]
            
            decision, stats = majority_rules_test(D, samples, epsilon)
            
            if base not in results:
                results[base] = {}
            results[base][sample_type] = {
                'decision': decision,
                'stats': stats
            }
            
            print(f"\nTesting {base} ({sample_type}):")
            print(f"Decision: {'Uniform' if decision else 'Not Uniform'}")
            print(f"Votes: {stats['positive_votes']}/{stats['total_trials']} ({stats['vote_ratio']:.2%})")
            print(f"Avg collision rate: {stats['avg_collision_rate']:.6f}")
            print(f"Support size: {stats['avg_support_size']:.1f} ({stats['min_support']}-{stats['max_support']})")
    
    return results


if __name__ == "__main__":
    test_dir = './D'
    sample_dir = './X'
    epsilon = 0.1
    
    # Run batch tests
    results = batch_test(test_dir, sample_dir, epsilon)
    
    # Visualize results
    #WILL GENERATE LOT OF IMAGES
    #visualize_batch_results(test_dir, sample_dir, epsilon)