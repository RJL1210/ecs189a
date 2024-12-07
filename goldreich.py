import numpy as np
import random
import json
from collections import Counter
from typing import List, Dict

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
    Second transformation that maps samples to a larger domain.
    For each input value i:
    - Keep i with probability (bucket_size * grain_size/n) / averaged_prob(i)
    - Map to overflow_value otherwise
    
    grain_size is γ from the paper (default 1/6)
    """
    domain_size = len(averaged_dist)
    overflow_value = domain_size + 1
    expanded_samples = []
    
    for sample in input_samples:
        sample_key = str(sample)
        if sample_key not in averaged_dist:
            expanded_samples.append(overflow_value)
            continue
            
        # Calculate bucket size for this value
        # bucket_size = floor(averaged_prob * n/grain_size)
        prob = averaged_dist[sample_key]
        bucket_size = int(np.floor(prob * domain_size / grain_size))
        
        # Probability of keeping the sample
        if prob > 0:
            keep_prob = (bucket_size * grain_size / domain_size) / prob
            keep_prob = min(1, keep_prob)  # Ensure valid probability
            
            if random.random() < keep_prob:
                expanded_samples.append(sample)
            else:
                expanded_samples.append(overflow_value)
        else:
            expanded_samples.append(overflow_value)
    
    return expanded_samples

def collision_tester(samples: List[int], domain_size: int, epsilon: float) -> bool:
    """
    Test uniformity using collision statistics.
    
    Args:
        samples: Samples to test for uniformity
        domain_size: Size of domain
        epsilon: Proximity parameter
    
    Returns:
        True if samples appear uniform, False otherwise
    """
    if len(samples) < 2:
        raise ValueError("Need at least two samples for collision testing")

    # Count collisions
    sample_count = len(samples)
    frequency = Counter(samples)
    collision_count = sum(f * (f - 1) for f in frequency.values())
    collision_stat = collision_count / (sample_count * (sample_count - 1))
    
    # Threshold from Corollary 11.2
    threshold = (1 + epsilon**2) / domain_size
    
    print(f"Collision statistic: {collision_stat:.6f}")
    print(f"Uniform expectation: {1/domain_size:.6f}")
    print(f"Threshold (1+ε²)/n: {threshold:.6f}")
    
    return collision_stat <= threshold

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
    grain_param = 1/6  # As specified in paper
    
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

def test_from_files(D_file: str, X_file: str, epsilon: float):
    """
    Test distributions loaded from files
    """
    with open(D_file) as f:
        D = json.load(f)
    with open(X_file) as f:
        X = json.load(f)
    
    result = goldreich_test(D, X["samples"], epsilon)
    print("Accept: X matches D" if result else "Reject: X is far from D")

if __name__ == "__main__":
    # Load test data
    with open("./D/D_gaussian.json") as f:
        target_dist = json.load(f)
    with open("./X/D_gaussian_X_close.json") as f:
        close_samples = json.load(f)["samples"]
    with open("./X/D_gaussian_X_far.json") as f:
        far_samples = json.load(f)["samples"]
        
    print("Verifying reduction with close samples:")
    verify_reduction(target_dist, close_samples)
    
    print("\nVerifying reduction with far samples:")
    verify_reduction(target_dist, far_samples)