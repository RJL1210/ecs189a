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
    """
    Creates q' from D:
    q'(i) = 0.5*D(i) + 0.5*(1/n)
    """
    n = len(D)
    uniform_prob = 1.0 / n
    q_prime = {}
    for i in range(1, n+1):
        q_i = D.get(str(i), 0.0)
        q_prime[str(i)] = 0.5*q_i + 0.5*uniform_prob
    debug(f"Mixed distribution q' created for n={n}.", level="STEP")
    verbose_debug(f"q': {q_prime}")
    return q_prime


def quantize_distribution(q_prime: Dict[str, float], gamma: float = 1/6) -> Dict[str, float]:
    """
    Compute q'' from q':
    q''(i) = (floor(q'(i)*n/gamma)*gamma)/n
    """
    n = len(q_prime)
    q_double_prime = {}
    total_mass = 0.0
    for i in range(1, n+1):
        q_pi = q_prime[str(i)]
        m_i = math.floor(q_pi * n / gamma)
        q_double_prime[str(i)] = (m_i * gamma) / n
        total_mass += q_double_prime[str(i)]

    # Normalize q''
    for i in range(1, n+1):
        q_double_prime[str(i)] /= total_mass

    # Validation and Summary
    validate_graining(q_double_prime, n, gamma)
    debug(f"Quantized distribution q'' created. Total mass = {total_mass:.6f}, Deviation = {abs(1 - total_mass):.6f}")
    verbose_debug(f"q'': {q_double_prime}")
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


def collision_tester_uniformity(samples: List[int], m: int, epsilon: float) -> bool:
    """
    Balanced collision tester using Hoeffding's inequality.
    For close distributions, should accept with probability ~2/3.
    For far distributions, should reject with high probability.
    """
    # Filter valid samples
    filtered = [s for s in samples if 1 <= s <= m]
    M = len(filtered)
    
    if M < 2:
        if DEBUG:
            print(f"[collision_tester] Not enough valid samples: {M} (needed at least 2)")
        return True
    
    # Count frequencies
    freq = Counter(filtered)
    
    # Compute collisions
    collision_count = sum(c * (c-1) // 2 for c in freq.values())
    total_pairs = M * (M-1) // 2
    
    if total_pairs == 0:
        return True
        
    # Compute collision statistic
    C = collision_count / total_pairs
    
    # Base threshold from lecture
    base_threshold = (1 + epsilon**2) / m
    
    # Compute Hoeffding's inequality term
    hoeffding_term = math.sqrt((2 * base_threshold * (1 - base_threshold)) / M)
    
    # Final threshold
    threshold = base_threshold + hoeffding_term
    
    if DEBUG:
        print(f"[collision_tester] Statistics:")
        print(f"  - Valid samples: {M}")
        print(f"  - Unique values: {len(freq)}")
        print(f"  - Collision rate C: {C:.15f}")
        print(f"  - Base threshold: {base_threshold:.15f}")
        print(f"  - Hoeffding term: {hoeffding_term:.15f}")
        print(f"  - Final threshold: {threshold:.15f}")
        print(f"  - Decision: {'Accept' if C <= threshold else 'Reject'}")
        if C <= threshold:
            print(f"  - Below threshold by: {threshold - C:.15f}")
        else:
            print(f"  - Above threshold by: {C - threshold:.15f}")
    
    return C <= threshold


def goldreich_reduction(D: Dict[str, float], p_samples: List[int], epsilon: float, gamma: float = 1/6) -> bool:
    """
    Enhanced testing function with better diagnostics
    """
    if DEBUG:
        print("\n[goldreich_reduction] Starting test with:")
        print(f"  - Sample size: {len(p_samples)}")
        print(f"  - Epsilon: {epsilon}")
        print(f"  - Gamma: {gamma}")
    
    # Run Algorithm 8
    q_double_prime, p_double_prime = algorithm_8(D, p_samples, gamma)
    print(f"[DEBUG] After Algorithm 8: Size of p'': {len(p_double_prime)}, Domain size: {len(q_double_prime)}")

    
    # Run Algorithm 5
    mapped_samples, m = algorithm_5(q_double_prime, p_double_prime)
    
    if DEBUG:
        print(f"[DEBUG] Input sample size: {len(p_samples)}")
        print(f"[DEBUG] p'' size after Algorithm 8: {len(p_double_prime)}")
        print(f"[DEBUG] Mapped samples size after Algorithm 5: {len(mapped_samples)}")
        print(f"[DEBUG] Final domain size for collision testing: {m}")
    


    # Run collision tester
    return collision_tester_uniformity(mapped_samples, m, epsilon)

def test_goldreich_reduction(test_dir="./D", sample_dir="./X", epsilon=0.001, gamma=1/6):
    """
    Test the goldreich_reduction implementation using generated distributions and samples.
    
    Args:
        test_dir (str): Path to the directory containing distribution JSON files.
        sample_dir (str): Path to the directory containing sample JSON files.
        epsilon (float): Distance parameter for the reduction.
        gamma (float): Graining parameter for the reduction.
        
    Returns:
        None
    """

    # Fetch all distribution files
    distribution_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".json")])
    results = []

    for dist_file in distribution_files:
        dist_path = os.path.join(test_dir, dist_file)

        # Load distribution D
        with open(dist_path, "r") as f:
            D = json.load(f)

        # Dynamically determine close and far sample file names
        base_name = os.path.splitext(dist_file)[0]
        close_sample_file = f"{base_name}_X_close.json"
        far_sample_file = f"{base_name}_X_far.json"

        close_sample_path = os.path.join(sample_dir, close_sample_file)
        far_sample_path = os.path.join(sample_dir, far_sample_file)

        if not os.path.exists(close_sample_path):
            print(f"[WARNING] Missing close sample file: {close_sample_path}. Skipping...")
            continue

        if not os.path.exists(far_sample_path):
            print(f"[WARNING] Missing far sample file: {far_sample_path}. Skipping...")
            continue

        with open(close_sample_path, "r") as f:
            close_samples = json.load(f)["samples"]

        with open(far_sample_path, "r") as f:
            far_samples = json.load(f)["samples"]

        # Test close samples
        print(f"Testing close samples for {dist_file}...")
        is_uniform_close = goldreich_reduction(D, close_samples, epsilon, gamma)
        results.append((dist_file, "close", is_uniform_close))

        # Test far samples
        print(f"Testing far samples for {dist_file}...")
        is_uniform_far = goldreich_reduction(D, far_samples, epsilon, gamma)
        results.append((dist_file, "far", is_uniform_far))

    # Summarize results
    print("\nTest Results:")
    for dist_file, sample_type, is_uniform in results:
        print(f"  Distribution: {dist_file}, Sample Type: {sample_type}, Result: {'Uniform' if is_uniform else 'Not Uniform'}")

    # Summarize results
    print("\nTest Results:")
    for dist_file, sample_type, is_uniform in results:
        print(f"  Distribution: {dist_file}, Sample Type: {sample_type}, Result: {'Uniform' if is_uniform else 'Not Uniform'}")

    # Add the following code here

    # Additional checks for D_far_exact distribution
    far_exact_dist_file = "D_far_exact.json"
    far_exact_dist_path = os.path.join(test_dir, far_exact_dist_file)

    with open(far_exact_dist_path, "r") as f:
        D_far_exact = json.load(f)

     # Print q' and q'' distributions for D_far_exact
    q_prime = mix_distribution_with_uniform(D_far_exact)
    print("q' distribution for D_far_exact:")
    print(q_prime)

    q_double_prime = quantize_distribution(q_prime, gamma)
    print("q'' distribution for D_far_exact:")
    print(q_double_prime)


    # Calculate total variation distance
    def total_variation_distance(p, q):
        return sum(abs(p_i - q_i) for p_i, q_i in zip(p, q)) / 2

    uniform_dist = [1/len(D_far_exact)] * len(D_far_exact)
    tvd = total_variation_distance(list(D_far_exact.values()), uniform_dist)
    print(f"\nTotal Variation Distance (D_far_exact vs. Uniform): {tvd}")

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(D_far_exact)+1), D_far_exact.values())
    plt.xlabel('Element')
    plt.ylabel('Probability')
    plt.title('D_far_exact Distribution')
    plt.show()

    # Generate multiple sets of far samples and test separately
    num_sample_sets = 5
    for i in range(num_sample_sets):
        print(f"\nTesting far samples for D_far_exact (Set {i+1})...")
        far_samples = np.random.choice(range(1, len(D_far_exact)+1), size=100000, p=list(D_far_exact.values()))
        is_uniform_far = goldreich_reduction(D_far_exact, far_samples, epsilon, gamma)
        print(f"  Result: {'Uniform' if is_uniform_far else 'Not Uniform'}")

# Example Debug Usage
if __name__ == "__main__":
    # Example Usage with Debugging
    DEBUG = True
    VERBOSE = False

    # Test the goldreich_reduction function
    test_goldreich_reduction()
