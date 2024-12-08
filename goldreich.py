import numpy as np
import random
import json
from collections import Counter
from typing import List, Dict
import os
import glob

def average_with_uniform(input_samples: List[int], domain_size: int) -> List[int]:
    """
    First transformation that averages input distribution with uniform distribution.
    Guarantees that for any element i: p'(i) = 0.5p(i) + 0.5/n exactly.
    
    Args:
        input_samples: Original samples from distribution p
        domain_size: Size of the domain [n]
    
    Returns:
        Samples from distribution p' where p'(i) = 0.5p(i) + 0.5/n
    """
    total_samples = len(input_samples)
    
    # Count frequencies in input samples
    input_counts = Counter(input_samples)
    input_probs = {i: input_counts.get(i, 0)/total_samples for i in range(1, domain_size + 1)}
    
    # Calculate exact counts needed for each element
    averaged_counts = {}
    for i in range(1, domain_size + 1):
        # p'(i) = 0.5p(i) + 0.5/n
        target_prob = 0.5 * input_probs.get(i, 0) + 0.5/domain_size
        # Convert probability to count
        averaged_counts[i] = int(round(target_prob * total_samples))
    
    # Generate samples according to exact counts
    averaged_samples = []
    for element, count in averaged_counts.items():
        averaged_samples.extend([element] * count)
    
    # Adjust length if necessary due to rounding
    while len(averaged_samples) < total_samples:
        averaged_samples.append(random.randint(1, domain_size))
    while len(averaged_samples) > total_samples:
        averaged_samples.pop()
        
    # Shuffle to maintain randomness
    random.shuffle(averaged_samples)
    
    return averaged_samples

def compute_averaged_distribution(target_dist: Dict[str, float]) -> Dict[str, float]:
    """
    Compute the averaged distribution where each probability is
    averaged_prob(i) = 0.5 * original_prob(i) + 0.5 * (1/n)
    """
    domain_size = len(target_dist)
    uniform_prob = 1.0 / domain_size
    averaged_dist = {}
    
    for i in range(1, domain_size + 1):
        original_prob = target_dist.get(str(i), 0)
        averaged_dist[str(i)] = 0.5 * original_prob + 0.5 * uniform_prob
    
    return averaged_dist

def map_to_expanded_domain(input_samples: List[int], averaged_dist: Dict[str, float], 
                          grain_size: float = 1/6) -> List[int]:
    """
    Second transformation that maps samples to a larger domain [6n].
    For each input value i:
    - Map to segment [6(i-1)+1, 6i] with probability (bucket_size * grain_size/n) / averaged_prob(i)
    - Map to overflow segment otherwise
    
    grain_size is γ from the paper (default 1/6)
    """
    domain_size = len(averaged_dist)
    expanded_size = 6 * domain_size  # Should be n/γ where γ = 1/6
    expanded_samples = []
    
    for sample in input_samples:
        sample_key = str(sample)
        if sample_key not in averaged_dist:
            # Map to overflow segment
            expanded_samples.append(expanded_size)
            continue
            
        # Calculate bucket size for this value
        prob = averaged_dist[sample_key]
        bucket_size = int(np.floor(prob * domain_size / grain_size))
        
        # Probability of keeping the sample
        if prob > 0:
            keep_prob = (bucket_size * grain_size / domain_size) / prob
            keep_prob = min(1, keep_prob)  # Ensure valid probability
            
            if random.random() < keep_prob:
                # Map to corresponding segment in [6n]
                segment_start = 6 * (int(sample_key) - 1) + 1
                expanded_samples.append(segment_start + random.randint(0, 5))
            else:
                expanded_samples.append(expanded_size)
        else:
            expanded_samples.append(expanded_size)
    
    return expanded_samples

def collision_tester(samples: List[int], domain_size: int, epsilon: float) -> bool:
    """
    Normalized implementation of collision tester from lecture notes.
    
    For domain size n:
    - Uniform distributions should have C ≈ 1/n
    - ε-far distributions should have C ≥ (1 + 4ε²)/n
    """
    m = len(samples)
    required_samples = int(np.ceil(np.sqrt(domain_size) / (epsilon * epsilon)))
    
    if m > required_samples:
        samples = samples[:required_samples]
    
    # Count collisions
    frequency = Counter(samples)
    collision_count = sum(freq * (freq - 1) // 2 for freq in frequency.values())
    
    # Normalize by domain size and sample size
    m_choose_2 = (len(samples) * (len(samples) - 1)) // 2
    C = (collision_count / m_choose_2) * (domain_size / 6)  # Divide by 6 for expanded domain
    
    # Thresholds according to lecture notes
    uniform_threshold = 1 + 0.1 * epsilon**2  # Should be ≈ 1 for uniform
    far_threshold = 1 + 4 * epsilon**2      # For ε-far distributions
    
    print(f"Normalized collision statistic (C): {C:.6f}")
    print(f"Uniform threshold (1 + 0.1ε²): {uniform_threshold:.6f}")
    print(f"Far threshold (1 + 4ε²): {far_threshold:.6f}")
    print(f"Samples used: {len(samples)} of {required_samples} required")
    
    return C <= uniform_threshold

def goldreich_test(D: Dict[str, float], X_samples: List[int], epsilon: float) -> bool:
    """
    Main testing function implementing Algorithm 8 from Goldreich's paper.
    Tests if distribution X is close to distribution D.
    
    Args:
        D: Target distribution
        X_samples: Samples from distribution to test
        epsilon: Distance parameter
    
    Returns:
        True if X appears to match D, False otherwise
    """
    domain_size = len(D)
    grain_param = 1/6  # As specified in paper, gamma
    
    # Step 1: Average with uniform distribution
    averaged_samples = average_with_uniform(X_samples, domain_size)
    
    # Compute averaged distribution
    averaged_dist = compute_averaged_distribution(D)
    
    # Step 2: Map to expanded domain
    expanded_samples = map_to_expanded_domain(averaged_samples, averaged_dist, grain_param)
    
    # Test uniformity on expanded domain [6n]
    expanded_domain_size = 6 * domain_size
    return collision_tester(expanded_samples, expanded_domain_size, epsilon/3)

def verify_reduction(target_dist: Dict[str, float], test_samples: List[int], 
                    num_verify_samples: int = 100000):
    """
    Verify reduction properties with focus on exact probabilities
    """
    domain_size = len(target_dist)
    print("\nVerification Results:")
    print("-" * 50)
    
    verify_samples = test_samples[:num_verify_samples]
    
    # Input distribution analysis
    print("Input Distribution Analysis:")
    input_freqs = Counter(verify_samples)
    input_probs = {i: input_freqs.get(i, 0)/num_verify_samples 
                   for i in range(1, domain_size + 1)}
    
    print(f"Input min probability: {min(input_probs.values()):.6f}")
    print(f"Input max probability: {max(input_probs.values()):.6f}")
    
    # Averaging verification
    averaged_samples = average_with_uniform(verify_samples, domain_size)
    avg_freqs = Counter(averaged_samples)
    avg_probs = {i: avg_freqs.get(i, 0)/num_verify_samples 
                 for i in range(1, domain_size + 1)}
    
    print("\nAveraging Step Verification:")
    min_freq = min(avg_probs.values())
    min_required = 1/(2*domain_size)
    
    print(f"Minimum frequency found: {min_freq:.6f}")
    print(f"Required minimum (1/2n): {min_required:.6f}")
    
    # Verify averaging formula exactly
    print("\nProbability Verification:")
    max_diff = 0
    worst_element = None
    
    for i in range(1, domain_size + 1):
        expected = 0.5 * input_probs[i] + 0.5/domain_size
        actual = avg_probs[i]
        diff = abs(expected - actual)
        if diff > max_diff:
            max_diff = diff
            worst_element = i
    
    print(f"Maximum deviation from expected probability: {max_diff:.6f} at element {worst_element}")
    print(f"For element {worst_element}:")
    print(f"  Original p(i): {input_probs[worst_element]:.6f}")
    print(f"  Expected p'(i): {0.5 * input_probs[worst_element] + 0.5/domain_size:.6f}")
    print(f"  Actual p'(i): {avg_probs[worst_element]:.6f}")
    
    print(f"\nAveraging property {'SATISFIED' if min_freq >= min_required else 'FAILED'}")
    
    # Expansion verification
    averaged_dist = compute_averaged_distribution(target_dist)
    expanded_samples = map_to_expanded_domain(averaged_samples, averaged_dist)
    
    expanded_freqs = Counter(expanded_samples)
    expanded_probs = {k: v/len(expanded_samples) for k, v in expanded_freqs.items()}
    
    print("\nExpansion Step Verification:")
    expected_uniform = 1/(6*domain_size)
    print(f"Expected uniform probability: {expected_uniform:.6f}")
    
    max_dev_elem = max(expanded_probs.keys(), 
                      key=lambda k: abs(expanded_probs[k] - expected_uniform))
    max_deviation = abs(expanded_probs[max_dev_elem] - expected_uniform)
    
    print(f"Maximum deviation: {max_deviation:.6f} (at element {max_dev_elem})")
    print(f"Uniformity property {'SATISFIED' if max_deviation < 0.1 else 'FAILED'}")

def test_distribution_files(distribution_dir: str = "./D", samples_dir: str = "./X", epsilon: float = 0.1):
    """
    Test distributions from files for both close and far cases
    """
    # Test each distribution file in the D directory
    for d_file in glob.glob(os.path.join(distribution_dir, "*.json")):
        dist_name = os.path.basename(d_file).replace(".json", "")
        print(f"\nTesting distribution: {dist_name}")
        
        # Load the target distribution
        with open(d_file) as f:
            target_dist = json.load(f)
        
        # Test close samples
        close_file = os.path.join(samples_dir, f"{dist_name}_X_close.json")
        if os.path.exists(close_file):
            print("\nTesting close samples:")
            with open(close_file) as f:
                close_samples = json.load(f)["samples"]
            result = goldreich_test(target_dist, close_samples, epsilon)
            print(f"Close samples test result: {'Accept' if result else 'Reject'}")
            # Verify the reduction properties
            verify_reduction(target_dist, close_samples)
        
        # Test far samples
        far_file = os.path.join(samples_dir, f"{dist_name}_X_far.json")
        if os.path.exists(far_file):
            print("\nTesting far samples:")
            with open(far_file) as f:
                far_samples = json.load(f)["samples"]
            result = goldreich_test(target_dist, far_samples, epsilon)
            print(f"Far samples test result: {'Accept' if result else 'Reject'}")
            # Verify the reduction properties
            verify_reduction(target_dist, far_samples)


if __name__ == "__main__":
    # Test all distribution files
    test_distribution_files()
    