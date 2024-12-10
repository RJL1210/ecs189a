import random
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import statistics
import glob, os, json
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False # Set to False to reduce verbosity
VERBOSE = False  # Set to True for deeper insights

# Helper Functions for Debugging
def debug(msg: str, level: str = "INFO"):
    """Print debug messages based on the DEBUG flag."""
    if DEBUG:
        print(f"[{level}] {msg}")

def verbose_debug(msg: str):
    """Print verbose debug messages based on the VERBOSE flag."""
    if VERBOSE:
        print(f"[VERBOSE] {msg}")

def warn(msg: str):
    """Print warning messages."""
    print(f"[WARNING] {msg}")

def log_to_file(filename: str, content: str):
    """Log messages to a file for further analysis."""
    with open(filename, 'a') as log_file:
        log_file.write(content + '\n')    

def apply_uniform_filter(p_samples: List[int], n: int) -> List[int]:
    """Simple 50-50 mixing with uniform as per paper"""
    return [s if random.random() < 0.5 else random.randint(1, n) 
            for s in p_samples]

def mix_distribution_with_uniform(D: Dict[str, float]) -> Dict[str, float]:
    """50-50 mixing with uniform distribution"""
    n = len(D)
    return {str(i): 0.5 * D.get(str(i), 0.0) + 0.5/n 
            for i in range(1, n+1)}

def quantize_distribution(q_prime: Dict[str, float], gamma: float = 1/6) -> Dict[str, float]:
    """
    Create m-grained distribution using floor and redistribution.
    Simplified as per paper's description.
    """
    n = len(q_prime)
    
    # First pass: floor values
    q_double_prime = {}
    total_mass = 0.0
    
    for i in range(1, n+1):
        m_i = math.floor(q_prime[str(i)] * n / gamma)
        q_double_prime[str(i)] = (m_i * gamma) / n
        total_mass += q_double_prime[str(i)]
    
    # Distribute remaining mass to largest remainders
    remaining = 1.0 - total_mass
    if remaining > 0:
        remainders = [(i, q_prime[str(i)] * n / gamma - math.floor(q_prime[str(i)] * n / gamma)) 
                     for i in range(1, n+1)]
        remainders.sort(key=lambda x: x[1], reverse=True)
        
        units = int(remaining * n / gamma)
        for i in range(units):
            if i < len(remainders):
                idx = str(remainders[i][0])
                q_double_prime[idx] += gamma / n
    
    return q_double_prime

def validate_graining(q_double_prime, n, gamma):
    for key, value in q_double_prime.items():
        multiple = value * n / gamma
        if abs(multiple - round(multiple)) >= 1e-10:
            warn(f"Graining error at {key}: {value}")

def transform_samples(p_prime: List[int], q_prime: Dict[str, float], 
                     q_double_prime: Dict[str, float]) -> List[int]:
    """Transform samples using ratio q''/q'"""
    return [s for s in p_prime if random.random() < 
            q_double_prime[str(s)] / (q_prime[str(s)] + 1e-10)]

def analyze_probability_flow(samples: List[int], n: int, stage: str):
    """Track probability changes through the reduction chain"""
    hist = np.histogram(samples, bins=n, range=(1, n+1))[0]
    hist = hist / len(samples)
    
    print(f"\nProbability analysis at {stage}:")
    print(f"  Min non-zero prob: {np.min(hist[hist>0]):.8f}")
    print(f"  Max prob: {np.max(hist):.8f}")
    print(f"  Std dev: {np.std(hist):.8f}")
    print(f"  Unique probs: {len(np.unique(hist[hist>0]))}")

def transform_p_prime_to_p_double_prime(p_prime_samples: List[int], 
                                      q_prime: Dict[str, float],
                                      q_double_prime: Dict[str, float], 
                                      gamma: float = 1/6) -> List[int]:
    """Transform p' into p'' using q''(i)/q'(i)"""
    p_double_prime = []
    for s in p_prime_samples:
        q_p = q_prime[str(s)]
        q_pp = q_double_prime[str(s)]
        if q_p > 0:
            keep_prob = q_pp / q_p
            if random.random() < keep_prob:
                p_double_prime.append(s)
    return p_double_prime

def algorithm_8(D: Dict[str, float], p_samples: List[int], 
                     gamma: float=1/6, is_close: bool=False) -> Tuple[Dict[str, float], List[int]]:
    n = len(D)
    #analyze_probability_flow(p_samples, n, "Input")
    
    p_prime = apply_uniform_filter(p_samples, n)
    #analyze_probability_flow(p_prime, n, "After Filter")
    
    q_prime = mix_distribution_with_uniform(D)
    #print("\nq_prime analysis:")
    q_values = [q_prime[str(i)] for i in range(1, n+1)]
    #print(f"  Range: {min(q_values):.8f} to {max(q_values):.8f}")
    
    q_double_prime = quantize_distribution(q_prime, gamma)
    
    # Use transform_p_prime_to_p_double_prime instead of transform_samples
    p_double_prime = transform_p_prime_to_p_double_prime(p_prime, q_prime, q_double_prime, gamma)
    #Aanalyze_probability_flow(p_double_prime, n, "After Transform")
    
    return q_double_prime, p_double_prime

def algorithm_5(q_double_prime: Dict[str, float], 
                  p_double_prime_samples: List[int], 
                  gamma: float = 1/6) -> Tuple[List[int], int]:
    n = len(q_double_prime)
    m = int(n / gamma)  # Should be 4500
    
    # Ensure we use all m positions
    m_i = [0] * n
    positions_left = m
    
    # First allocate minimum positions to each used index
    used_indices = set(p_double_prime_samples)
    min_positions = positions_left // len(used_indices)
    for i in used_indices:
        m_i[i-1] = min_positions
        positions_left -= min_positions
    
    # Distribute remaining based on q''
    if positions_left > 0:
        weights = [q_double_prime[str(i)] for i in range(1, n+1)]
        total_weight = sum(weights)
        for i in range(n):
            if i+1 in used_indices:
                additional = int((weights[i]/total_weight) * positions_left)
                m_i[i] += additional
                positions_left -= additional
    
    # Map samples using full range
    ranges = []
    current = 1
    for size in m_i:
        if size > 0:
            ranges.append((current, current + size - 1))
            current += size
        else:
            ranges.append((0, 0))
    
    mapped_samples = []
    for s in p_double_prime_samples:
        start, end = ranges[s-1]
        if start > 0:
            mapped = random.randint(start, end)
            mapped_samples.append(mapped)
            
    return mapped_samples, m



def collision_tester(samples: List[int], m: int, epsilon: float, delta: float = 0.05) -> bool:
    """
    Basic collision test as per lecture notes.
    Tests if collision rate is close to 1/m.
    """
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
    
    # Hoeffding bound
    hoeffding_term = math.sqrt(math.log(2/delta)/(2*M))
    threshold = base_threshold + hoeffding_term
    
    if DEBUG:
        print(f"Collision rate: {C:.6f}")
        print(f"Base threshold: {base_threshold:.6f}")
        print(f"Hoeffding term: {hoeffding_term:.6f}")
        print(f"Final threshold: {threshold:.6f}")
    
    return C <= threshold

def total_variation_distance(p_samples: List[int], n: int) -> float:
    """
    Calculate the total variation distance between the empirical distribution of p_samples
    and the uniform distribution over [1, n].
    
    Args:
        p_samples: List of samples from the distribution P.
        n: The size of the domain.
        
    Returns:
        The total variation distance.
    """
    # Calculate the empirical distribution of p_samples
    empirical_dist = np.zeros(n)
    for sample in p_samples:
        empirical_dist[sample - 1] += 1
    empirical_dist /= len(p_samples)
    
    # Calculate the uniform distribution over [1, n]
    uniform_dist = np.ones(n) / n
    
    # Calculate the total variation distance
    tvd = 0.5 * np.sum(np.abs(empirical_dist - uniform_dist))
    
    return tvd

def analyze_distribution_structure(samples: List[int], n: int, stage: str):
    """Analyze distribution structure at each stage"""
    freq = np.zeros(n)
    for s in samples:
        if isinstance(s, int) and 1 <= s <= n:
            freq[s-1] += 1
        else:
            print(f"Warning: Sample {s} out of bounds for stage {stage}")
    freq /= len(samples)
    
    # Calculate key statistics
    mean = np.mean(freq)
    std = np.std(freq)
    max_prob = np.max(freq)
    min_prob = np.min(freq[freq > 0])  # Minimum non-zero probability
    
    print(f"{stage} Distribution Analysis:")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std Dev: {std:.6f}")
    print(f"  Max probability: {max_prob:.6f}")
    print(f"  Min non-zero prob: {min_prob:.6f}")
    print(f"  Support size: {np.sum(freq > 0)}")

def goldreich_reduction(D: Dict[str, float], p_samples: List[int], 
                       epsilon: float, gamma: float = 1/6) -> bool:
    n = len(D)
    required = int(16 * math.sqrt(n) / epsilon**2)
    p_samples = p_samples[:required]
    
    #analyze_distribution_structure(p_samples, n, "Initial")
    
    # Use algorithm_8_debug to get detailed probability flow
    q_double_prime, p_double_prime = algorithm_8(D, p_samples, gamma)
    
    # Run Algorithm 5 - maps to [m]
    mapped_samples, m = algorithm_5(q_double_prime, p_double_prime, gamma)
    #analyze_distribution_structure(mapped_samples, m, "After Mapping")
    
    # Test uniformity
    return collision_tester(mapped_samples, m, epsilon/3)

def majority_rules_test(D: Dict[str, float], p_samples: List[int], 
                       epsilon: float, num_trials: int = 11, 
                       delta: float = 0.05) -> Tuple[bool, dict]:
    """
    Run multiple trials with Hoeffding bound.
    """
    results = []
    collision_rates = []
    support_sizes = []
    
    n = len(D)
    required_samples = int(16 * math.sqrt(n) / epsilon**2)
    p_samples = p_samples[:required_samples]
    
    for trial in range(num_trials):
        # Randomly subsample 90% of data for each trial
        trial_size = int(0.9 * len(p_samples))
        trial_samples = random.sample(p_samples, trial_size)
        
        # Run the standard Goldreich reduction
        q_double_prime, p_double_prime = algorithm_8(D, trial_samples)
        mapped_samples, m = algorithm_5(q_double_prime, p_double_prime)
        
        # Test with Hoeffding bound
        result = collision_tester(mapped_samples, m, epsilon/3, delta)
        results.append(result)
        
        if DEBUG:
            # Calculate statistics for this trial
            valid = [s for s in mapped_samples if 1 <= s <= m]
            if len(valid) >= 2:
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
    
    # Compile statistics
    stats = {
        'positive_votes': positive_votes,
        'total_trials': num_trials,
        'vote_ratio': positive_votes / num_trials
    }
    
    if collision_rates:
        stats.update({
            'avg_collision_rate': np.mean(collision_rates),
            'std_collision_rate': np.std(collision_rates),
            'avg_support_size': np.mean(support_sizes),
            'min_support': min(support_sizes),
            'max_support': max(support_sizes)
        })
        
        print("\nMajority Rules Test Results:")
        print(f"Positive votes: {positive_votes}/{num_trials} ({stats['vote_ratio']:.2%})")
        print(f"Average collision rate: {stats['avg_collision_rate']:.6f}")
        print(f"Average support size: {stats['avg_support_size']:.1f}")
        print(f"Support size range: {stats['min_support']} - {stats['max_support']}")
    
    return majority_decision, stats

def batch_test_with_correct_targets(test_dir: str, mixed_dir: str, sample_dir: str, 
                                  epsilon: float = 0.1, num_trials: int = 11):
    """Modified batch testing that uses correct target distributions"""
    results = {}
    
    for dist_file in sorted(os.listdir(test_dir)):
        if not dist_file.endswith('.json'):
            continue
            
        base = os.path.splitext(dist_file)[0]
        print(f"\nTesting {base}:")
        
        # Load original distribution for far testing
        with open(os.path.join(test_dir, dist_file)) as f:
            D_original = json.load(f)
            
        # Load mixed distribution for close testing
        with open(os.path.join(mixed_dir, f"{base}_mixed.json")) as f:
            D_mixed = json.load(f)
        
        # Test close samples against mixed distribution
        with open(os.path.join(sample_dir, f"{base}_X_close.json")) as f:
            close = json.load(f)["samples"]
        print("Testing close samples:")
        close_decision, close_stats = majority_rules_test(D_mixed, close, epsilon, num_trials)
        
        # Test far samples against original distribution
        with open(os.path.join(sample_dir, f"{base}_X_far.json")) as f:
            far = json.load(f)["samples"]
        print("\nTesting far samples:")
        far_decision, far_stats = majority_rules_test(D_original, far, epsilon, num_trials)
        
        results[base] = {
            'close': {
                'decision': close_decision,
                'stats': close_stats
            },
            'far': {
                'decision': far_decision,
                'stats': far_stats
            }
        }
    
    return results


# Example Debug Usage
if __name__ == "__main__":
    # Example Usage with Debugging
    DEBUG = True
    VERBOSE = False

    # Run batch testing
    results = batch_test_with_correct_targets(
        test_dir="./D",
        mixed_dir="./D_mixed",
        sample_dir="./X",
        epsilon=0.1,
        num_trials=11
    )