import math
from collections import Counter
from typing import Dict, List, Tuple
from goldreich import goldreich_reduction

def lecture11_collision_tester(samples: List[int], m: int, epsilon: float, delta: float = 0.05) -> bool:
    """
    Modified collision tester using Bernstein's inequality for the threshold.
    
    Args:
        samples: List of sample values
        m: Domain size
        epsilon: Proximity parameter
        delta: Confidence parameter (default 0.05 for 95% confidence)
    """
    # Filter valid samples
    filtered = [s for s in samples if 1 <= s <= m]
    n = len(filtered)
    
    if n < 2:
        return True
    
    # Count collisions
    freq = Counter(filtered)
    collision_count = sum(c * (c-1) // 2 for c in freq.values())
    M = (n * (n-1)) // 2  # number of pairs
    
    if M == 0:
        return True
    
    # Calculate collision rate
    C = collision_count / M
    
    # Base threshold for uniformity
    base_threshold = (1 + epsilon**2) / m
    
    # Bernstein bound
    # Use base_threshold as estimate of variance (since collisions are rare)
    variance = base_threshold
    L = 1.0  # maximum value of |X - μ|
    
    # Solve for t in Bernstein's inequality:
    # 2exp(-t²/(2σ² + 2Lt/3)) = delta
    # This is a cubic equation, we'll use numerical solution
    def bernstein_bound(t):
        return 2 * math.exp(-M * t**2 / (2 * variance + 2 * L * t/3)) - delta
    
    # Binary search for t
    left, right = 0, 1
    for _ in range(16):  # 20 iterations should give enough precision
        t = (left + right) / 2
        if bernstein_bound(t) > 0:
            left = t
        else:
            right = t
    
    bernstein_term = right
    threshold = base_threshold + bernstein_term
    
    return C <= threshold


def majority_rules_test(D: Dict[str, float], p_samples: List[int], 
                       epsilon: float, num_trials: int = 11) -> Tuple[bool, dict]:
    results = []
    collision_rates = []
    support_sizes = []
    
    n = len(D)
    required_samples = int(16 * math.sqrt(n) / epsilon**2)
    p_samples = p_samples[:required_samples]
    
    for _ in range(num_trials):
        mapped_samples, m = goldreich_reduction(D, p_samples, epsilon)
        result = lecture11_collision_tester(mapped_samples, m, epsilon/3)
        results.append(result)
        
        # Calculate statistics
        valid = [s for s in mapped_samples if 1 <= s <= m]
        freq = Counter(valid)
        collisions = sum(c * (c-1) // 2 for c in freq.values())
        total = len(valid) * (len(valid) - 1) // 2
        rate = collisions / total if total > 0 else 0
        collision_rates.append(rate)
        
        support = len(set(valid))
        support_sizes.append(support)
    
    # Calculate majority decision
    positive_votes = sum(results)
    majority_decision = positive_votes > num_trials // 2
    
    stats = {
        'positive_votes': positive_votes,
        'total_trials': num_trials,
        'vote_ratio': positive_votes / num_trials,
        'avg_collision_rate': sum(collision_rates) / len(collision_rates),
        'avg_support_size': sum(support_sizes) / len(support_sizes),
        'min_support': min(support_sizes),
        'max_support': max(support_sizes)
    }
    
    return majority_decision, stats

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
