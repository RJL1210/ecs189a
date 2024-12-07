import numpy as np
import random
import json
from collections import Counter
from typing import List, Dict

def average_with_uniform(samples: List[int], domain_size: int) -> List[int]:
    """
    First filter (F') that averages input distribution with uniform distribution.
    With 0.5 probability keeps the sample, otherwise draws from uniform.
    
    Args:
        samples: Original samples to transform
        domain_size: Size of the distribution domain [n]
    
    Returns:
        List of transformed samples
    """
    transformed_samples = []
    for sample in samples:
        if random.random() < 0.5:
            transformed_samples.append(sample)
        else:
            transformed_samples.append(random.randint(1, domain_size))
    return transformed_samples

def compute_averaged_distribution(original_dist: Dict[str, float]) -> Dict[str, float]:
    """
    Compute the averaged distribution q' = F'(q) where q'(i) = 0.5q(i) + 0.5/n.
    This ensures each element has probability at least 1/2n.
    
    Args:
        original_dist: Original distribution D
    
    Returns:
        Averaged distribution q'
    """
    domain_size = len(original_dist)
    averaged_dist = {}
    uniform_prob = 1.0 / domain_size
    
    for i in range(1, domain_size + 1):
        # Average with uniform: q'(i) = 0.5 * q(i) + 0.5/n
        original_prob = original_dist.get(str(i), 0)
        averaged_dist[str(i)] = 0.5 * original_prob + 0.5 * uniform_prob
    
    return averaged_dist

def map_to_expanded_domain(samples: List[int], averaged_dist: Dict[str, float], 
                          grain_param: float = 1/6) -> List[int]:
    """
    Second filter (F''_q') that maps samples to an expanded domain.
    Maps i to either itself or domain_size + 1 based on calculated probabilities.
    
    Args:
        samples: Input samples
        averaged_dist: The averaged distribution q'
        grain_param: Graining parameter γ (default 1/6 per paper)
    
    Returns:
        Samples mapped to expanded domain
    """
    domain_size = len(averaged_dist)
    mapped_samples = []
    
    for sample in samples:
        sample_str = str(sample)
        if sample_str not in averaged_dist:
            mapped_samples.append(domain_size + 1)
            continue
            
        # Calculate floor(q'(i)*n/γ)
        sample_count = int(np.floor(averaged_dist[sample_str] * domain_size / grain_param))
        
        # Probability of keeping sample i is (mi*γ/n)/q'(i)
        if averaged_dist[sample_str] > 0:
            keep_prob = (sample_count * grain_param / domain_size) / averaged_dist[sample_str]
            keep_prob = min(1, keep_prob)  # Ensure probability doesn't exceed 1
            
            if random.random() < keep_prob:
                mapped_samples.append(sample)
            else:
                mapped_samples.append(domain_size + 1)
        else:
            mapped_samples.append(domain_size + 1)
            
    return mapped_samples

def collision_tester(samples: List[int], domain_size: int, epsilon: float) -> bool:
    """
    Test uniformity using collision statistics.
    
    Args:
        samples: Samples to test for uniformity
        domain_size: Size of expanded domain (6n)
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
    
    # Expected probability for uniform distribution
    uniform_prob = 1 / domain_size
    
    print("Collision Statistic:", collision_stat)
    print("Expected Uniform Probability:", uniform_prob)
    print("Difference:", abs(collision_stat - uniform_prob))
    print("Epsilon:", epsilon)
    
    return abs(collision_stat - uniform_prob) <= epsilon

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
    test_from_files("./D/D_gaussian.json", "./X/D_gaussian_X_far.json", 0.05)
    test_from_files("./D/D_gaussian.json", "./X/D_gaussian_X_close.json", 0.05)