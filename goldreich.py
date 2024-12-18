import random
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np


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
    
    # Apply a uniform filter to p_samples to create p_prime
    p_prime = apply_uniform_filter(p_samples, n)

    # Mix the distribution D with a uniform distribution to create q_prime
    q_prime = mix_distribution_with_uniform(D)

    # Quantize the mixed distribution q' to create q'' with m-grained values
    q_double_prime = quantize_distribution(q_prime, gamma)

    # Transform p' into p'' using q''(i)/q'(i)
    p_double_prime = transform_p_prime_to_p_double_prime(p_prime, q_prime, q_double_prime, gamma)
    
    return q_double_prime, p_double_prime

def algorithm_5(q_double_prime: Dict[str, float], 
                p_double_prime_samples: List[int], 
                gamma: float = 1/6) -> Tuple[List[int], int]:
    n = len(q_double_prime)
    m = int(n / gamma)
    
    m_i = {}
    total_allocated = 0
    
    for i in range(1, n+1):
        str_i = str(i)
        m_i[str_i] = int(q_double_prime[str_i] * m)
        total_allocated += m_i[str_i]
    
    # Distribute remaining positions
    remaining = m - total_allocated
    if remaining > 0:
        fractional_parts = [(str(i), (q_double_prime[str(i)] * m) % 1) 
                           for i in range(1, n+1)]
        fractional_parts.sort(key=lambda x: x[1], reverse=True)
        
        # Only distribute up to number of available positions
        for i in range(min(remaining, len(fractional_parts))):
            m_i[fractional_parts[i][0]] += 1
    
    # Compute prefix sums for mapping
    prefix_sums = [0]
    curr_sum = 0
    for i in range(1, n+1):
        curr_sum += m_i[str(i)]
        prefix_sums.append(curr_sum)
    
    # Map samples
    mapped_samples = []
    for s in p_double_prime_samples:
        str_s = str(s)
        if str_s in m_i and m_i[str_s] > 0:
            # Map to a random position in the assigned range
            start = prefix_sums[s-1] + 1
            end = prefix_sums[s]
            if start <= end:
                mapped_value = random.randint(start, end)
                if mapped_value <= m:  # Ensure we stay within [m]
                    mapped_samples.append(mapped_value)
    
    if len(mapped_samples) == 0:
        # If no samples were mapped, this is likely not uniform
        return mapped_samples, m
        
    return mapped_samples, m

def goldreich_reduction(D: Dict[str, float], p_samples: List[int], 
                       epsilon: float, gamma: float = 1/6) -> bool:
    n = len(D)
    required = int(np.ceil(np.sqrt(n) / (epsilon * epsilon)))
    p_samples = p_samples[:required]
    
    q_double_prime, p_double_prime = algorithm_8(D, p_samples, gamma)
    
    # Run Algorithm 5 - maps to [m]
    mapped_samples, m = algorithm_5(q_double_prime, p_double_prime, gamma)
    
    # Test uniformity
    return mapped_samples, m

