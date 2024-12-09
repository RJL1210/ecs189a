import random
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import statistics
import glob, os, json
import numpy as np
import matplotlib.pyplot as plt

DEBUG = True  # Set to False to reduce verbosity
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

# Functions with Enhanced Debugging
def get_uniform_distribution(n: int) -> Dict[str, float]:
    """Get uniform distribution over [n]"""
    debug(f"Creating uniform distribution for n={n}.", level="STEP")
    return {str(i): 1.0/n for i in range(1, n+1)}

def mix_distribution_with_uniform(D: Dict[str, float]) -> Dict[str, float]:
    n = len(D)
    uniform_prob = 1.0 / n
    q_prime = {}
    for i in range(1, n+1):
        q_i = D.get(str(i), 0.0)
        q_prime[str(i)] = 0.5*q_i + 0.5*uniform_prob  # Changed to 50-50 mix
    return q_prime

def quantize_distribution(q_prime: Dict[str, float], gamma: float = 1/6) -> Dict[str, float]:
    """
    Enhanced quantization with better rounding properties
    """
    n = len(q_prime)
    q_double_prime = {}
    total_mass = 0.0
    
    # First pass: compute floor values
    for i in range(1, n+1):
        q_pi = q_prime[str(i)]
        m_i = math.floor(q_pi * n / gamma)
        q_double_prime[str(i)] = (m_i * gamma) / n
        total_mass += q_double_prime[str(i)]
    
    # Second pass: distribute remaining mass
    remaining_mass = 1.0 - total_mass
    if remaining_mass > 0:
        # Distribute to largest remainder values
        remainders = [(i, q_prime[str(i)] * n / gamma - math.floor(q_prime[str(i)] * n / gamma)) 
                     for i in range(1, n+1)]
        remainders.sort(key=lambda x: x[1], reverse=True)
        
        mass_per_unit = gamma / n
        units_to_distribute = int(remaining_mass / mass_per_unit + 0.5)
        
        for i in range(units_to_distribute):
            if i < len(remainders):
                idx = str(remainders[i][0])
                q_double_prime[idx] += mass_per_unit
    
    validate_graining(q_double_prime, n, gamma)
    debug(f"Quantized distribution q'' created with improved balance")
    return q_double_prime

def validate_graining(q_double_prime, n, gamma):
    for key, value in q_double_prime.items():
        multiple = value * n / gamma
        #if abs(multiple - round(multiple)) >= 1e-10:
            #warn(f"Graining error at {key}: {value}")


def transform_p_prime_to_p_double_prime(p_prime_samples: List[int], 
                                        q_prime: Dict[str, float],
                                        q_double_prime: Dict[str, float], 
                                        gamma: float = 1/6) -> List[int]:
    """
    Transform p' into p'' using q''(i)/q'(i).
    """
    p_double_prime = []
    for s in p_prime_samples:
        q_p = q_prime[str(s)]
        q_pp = q_double_prime[str(s)]
        if q_p > 0:
            keep_prob = q_pp / q_p
            if random.random() < keep_prob:
                p_double_prime.append(s)
    debug(f"Transformed p' into p'' with {len(p_double_prime)} samples retained.", level="STEP")
    return p_double_prime


def apply_uniform_filter(p_samples: List[int], n: int) -> List[int]:
    """
    Apply uniform filter to create p' from p.
    """
    p_prime = []
    for s in p_samples:
        if random.random() < 0.5:
            p_prime.append(s)
        else:
            p_prime.append(random.randint(1, n))
    debug(f"Uniform filter applied. p' size = {len(p_prime)} (Original size: {len(p_samples)}).", level="STEP")
    return p_prime

# Core Algorithms with Enhanced Debugging
def algorithm_8(D: Dict[str, float], p_samples: List[int], gamma: float=1/6) -> Tuple[Dict[str, float], List[int]]:
    """
    Implements Algorithm 8 with enhanced debugging.
    """
    debug("Starting Algorithm 8.", level="STEP")

    # Step 1: p'
    p_prime = apply_uniform_filter(p_samples, len(D))
    
    # Step 2: q'
    q_prime = mix_distribution_with_uniform(D)
    
    # Step 3: q''
    q_double_prime = quantize_distribution(q_prime, gamma)
    
    # Step 4: p''
    p_double_prime = transform_p_prime_to_p_double_prime(p_prime, q_prime, q_double_prime, gamma)

    # Summary
    debug(f"Algorithm 8 completed: q'' mass = {sum(q_double_prime.values()):.6f}, p'' size = {len(p_double_prime)}.", level="SUMMARY")
    return q_double_prime, p_double_prime

def algorithm_5(q_double_prime: Dict[str, float], 
                p_double_prime_samples: List[int], 
                gamma: float = 1/6) -> Tuple[List[int], int]:
    """
    Implements Algorithm 5:
    - Input: q'' (m-grained) over [n], p'' samples from p'' over [n].
    - Output: Samples mapped to [m+1] for uniformity testing.
    """
    n = len(q_double_prime)
    m = int(n / gamma)  # m = n/gamma, should be integral if gamma divides 1/6 and so forth
    
    # Compute m_i = q''(i)*m
    m_i = []
    total_qpp = 0.0
    for i in range(1, n+1):
        val = q_double_prime[str(i)]
        total_qpp += val
        # Since q'' is m-grained, q''(i)*m should be an integer
        m_val = int(round(val * m))
        m_i.append(m_val)
    
    # Add the leftover element (m+1)
    m_plus_1 = int(round((1 - total_qpp) * m))
    
    # Compute prefix sums for m_i
    prefix_sum = [0]*(n+1)
    for i in range(1, n+1):
        prefix_sum[i] = prefix_sum[i-1] + m_i[i-1]
    
    # The final domain size is m_total + 1, where m_total = prefix_sum[n] + m_plus_1
    m_total = prefix_sum[n] + m_plus_1
    domain_size = m_total + 1
    
    mapped_samples = []
    for s in p_double_prime_samples:
        i = s
        if m_i[i-1] > 0:
            j_x = random.randint(1, m_i[i-1])
            rank = prefix_sum[i-1] + j_x
            mapped_samples.append(rank)
        else:
            # Map to the leftover element (m+1)
            mapped_samples.append(domain_size)

    return mapped_samples, domain_size

def lecture11_collision_tester(samples: List[int], m: int, epsilon: float) -> bool:
    filtered = [s for s in samples if 1 <= s <= m]
    M = len(filtered)
    
    if M < 2:
        return True
    
    # Count collisions
    freq = Counter(filtered)
    collision_count = sum(c * (c-1) // 2 for c in freq.values())
    total_pairs = M * (M-1) // 2
    
    if total_pairs == 0:
        return True
        
    C = collision_count / total_pairs
    uniform_rate = 1.0/m
    
    # Key insight from lecture: 
    # - Uniform distribution has collision rate very close to 1/m
    # - Far distributions have rate > (1 + ε²)/m
    # Use this gap directly
    threshold = uniform_rate * (1 + epsilon**2)
    
    if DEBUG:
        print(f"[lecture11_collision_tester] Statistics:")
        print(f"  - Valid samples: {M}")
        print(f"  - Collision rate C: {C:.15f}")
        print(f"  - Uniform rate (1/m): {uniform_rate:.15f}")
        print(f"  - Threshold: {threshold:.15f}")
        print(f"  - Gap from uniform: {(C - uniform_rate)/uniform_rate:.15f}")
    
    return C <= threshold

def calculate_sample_complexity(n: int, epsilon: float) -> int:
    # We need O(sqrt(n)/epsilon^2) samples
    return int(8 * math.sqrt(n) / (epsilon * epsilon))

def goldreich_reduction(D: Dict[str, float], p_samples: List[int], epsilon: float, gamma: float = 1/6) -> bool:
    
    """
    Modified Goldreich reduction using the Lecture 11 collision tester.
    """
    n = len(D)
    # Calculate required samples - O(sqrt(n)/epsilon^2)
    required_samples = int(8 * math.sqrt(n) / (epsilon * epsilon))
    
    if DEBUG:
        print(f"\n[goldreich_reduction] Starting test with:")
        print(f"  - Required samples: {required_samples}")
        print(f"  - Input samples: {len(p_samples)}")
        print(f"  - Epsilon: {epsilon}")
        print(f"  - Gamma: {gamma}")
    
    # Use only required number of samples
    p_samples = p_samples[:required_samples]
    
    # Run Algorithm 8 - creates q'' and p''
    q_double_prime, p_double_prime = algorithm_8(D, p_samples, gamma)
    
    # Run Algorithm 5 - maps to [m]
    mapped_samples, m = algorithm_5(q_double_prime, p_double_prime)
    
    # Test uniformity using lecture 11 tester
    # Note: The lecture suggests using epsilon/3 due to the reduction
    return lecture11_collision_tester(mapped_samples, m, epsilon/3)


def test_goldreich_reduction(test_dir="./D", sample_dir="./X", epsilon=0.1, gamma=1/6):
    """
    Test the goldreich_reduction implementation with optimal sample complexity.
    """
    distribution_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".json")])
    results = []

    for dist_file in distribution_files:
        dist_path = os.path.join(test_dir, dist_file)
        
        with open(dist_path, "r") as f:
            D = json.load(f)
        
        # Calculate optimal sample size for this distribution
        n = len(D)
        optimal_samples = calculate_sample_complexity(n, epsilon)
        
        if DEBUG:
            print(f"\nTesting {dist_file} with optimal {optimal_samples} samples")
        
        # Load and process samples according to optimal size
        base_name = os.path.splitext(dist_file)[0]
        close_sample_file = f"{base_name}_X_close.json"
        far_sample_file = f"{base_name}_X_far.json"
        
        close_sample_path = os.path.join(sample_dir, close_sample_file)
        far_sample_path = os.path.join(sample_dir, far_sample_file)
        
        if not os.path.exists(close_sample_path) or not os.path.exists(far_sample_path):
            print(f"[WARNING] Missing sample files for {dist_file}. Skipping...")
            continue
            
        with open(close_sample_path, "r") as f:
            close_samples = json.load(f)["samples"][:optimal_samples]
            
        with open(far_sample_path, "r") as f:
            far_samples = json.load(f)["samples"][:optimal_samples]
            
        # Test both close and far samples
        is_uniform_close = goldreich_reduction(D, close_samples, epsilon, gamma)
        results.append((dist_file, "close", is_uniform_close))
        
        is_uniform_far = goldreich_reduction(D, far_samples, epsilon, gamma)
        results.append((dist_file, "far", is_uniform_far))

    # Print results
    print("\nTest Results:")
    for dist_file, sample_type, is_uniform in results:
        print(f"  Distribution: {dist_file}, Sample Type: {sample_type}, Result: {'Uniform' if is_uniform else 'Not Uniform'}")


def test_lecture11_implementation(test_dir="./D", sample_dir="./X", epsilon=0.05, gamma=1/6):
    """
    Test the implementation with both original and Lecture 11 collision testers.
    """
    distribution_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".json")])
    results = []

    for dist_file in distribution_files:
        dist_path = os.path.join(test_dir, dist_file)
        
        with open(dist_path, "r") as f:
            D = json.load(f)
        
        # Calculate optimal sample size
        n = len(D)
        optimal_samples = int(8 * math.sqrt(n) / (epsilon * epsilon))
        
        if DEBUG:
            print(f"\nTesting {dist_file} with optimal {optimal_samples} samples")
        
        base_name = os.path.splitext(dist_file)[0]
        close_sample_file = f"{base_name}_X_close.json"
        far_sample_file = f"{base_name}_X_far.json"
        
        close_sample_path = os.path.join(sample_dir, close_sample_file)
        far_sample_path = os.path.join(sample_dir, far_sample_file)
        
        if not os.path.exists(close_sample_path) or not os.path.exists(far_sample_path):
            print(f"[WARNING] Missing sample files for {dist_file}. Skipping...")
            continue
            
        with open(close_sample_path, "r") as f:
            close_samples = json.load(f)["samples"][:optimal_samples]
            
        with open(far_sample_path, "r") as f:
            far_samples = json.load(f)["samples"][:optimal_samples]
            
        # Test both close and far samples
        is_uniform_close = goldreich_reduction(D, close_samples, epsilon, gamma)
        results.append((dist_file, "close", is_uniform_close))
        
        is_uniform_far = goldreich_reduction(D, far_samples, epsilon, gamma)
        results.append((dist_file, "far", is_uniform_far))

    print("\nTest Results:")
    for dist_file, sample_type, is_uniform in results:
        print(f"  Distribution: {dist_file}, Sample Type: {sample_type}, Result: {'Uniform' if is_uniform else 'Not Uniform'}")

def collision_test_with_repetition(samples, m, epsilon, num_trials=5):
    results = [lecture11_collision_tester(samples, m, epsilon) 
               for _ in range(num_trials)]
    return sum(results) > num_trials/2

# Example Debug Usage
if __name__ == "__main__":
    # Example Usage with Debugging
    DEBUG = True
    VERBOSE = False

    # Test the goldreich_reduction function
    test_lecture11_implementation()