import random
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


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

def goldreich_reduction(D: Dict[str, float], p_samples: List[int], 
                       epsilon: float, gamma: float = 1/6) -> bool:
    n = len(D)
    required = int(16 * math.sqrt(n) / epsilon**2)
    p_samples = p_samples[:required]
    
    q_double_prime, p_double_prime = algorithm_8(D, p_samples, gamma)
    
    # Run Algorithm 5 - maps to [m]
    mapped_samples, m = algorithm_5(q_double_prime, p_double_prime, gamma)
    
    # Test uniformity
    return mapped_samples, m

