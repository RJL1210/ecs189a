import numpy as np
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
import os

def identity_tester(q: Dict[str, float], samples: List[int], epsilon: float) -> Tuple[bool, dict]:
    """
    Algorithm 12.1: Identity Tester with detailed statistics.
    
    Returns:
        Tuple[bool, dict]: (decision, statistics)
    """  
    n = len(q)
    m = int(np.ceil(16 * np.sqrt(n) / epsilon ** 2))
    
    k = np.random.poisson(m)
    if k > 2*m:
        return False, {'Z_statistic': None, 'early_exit': True}
    
    k = min(k, len(samples))
    samples = samples[:k]
    
    # Count occurrences
    Ni = np.zeros(n)
    for sample in samples:
        Ni[sample - 1] += 1
    
    # Compute A
    q_array = np.array([q[str(i + 1)] for i in range(n)])
    threshold = epsilon / (50 * n)
    A = np.where(q_array >= threshold)[0]
    
    # Compute Z
    Z = 0
    Z_contributions = np.zeros(n)
    for i in A:
        kqi = k * q_array[i]
        if kqi > 0:
            contribution = ((Ni[i] - kqi) ** 2 - Ni[i]) / kqi
            Z_contributions[i] = contribution
            Z += contribution
    
    # Acceptance threshold
    acceptance_threshold = k * epsilon ** 2 / 5
    
    # Enhanced support size calculations
    empirical_support = np.sum(Ni > 0)  # Elements we actually saw
    theoretical_support = len(A)  # Elements in set A
    expected_support = sum(1 - (1 - q_array[i])**k for i in A)  # Expected number of unique elements
    
    # Calculate TVD
    emp_dist = Ni / k
    tvd = 0.5 * np.sum(np.abs(emp_dist - q_array))
    
    stats = {
        'Z_statistic': Z,
        'threshold': acceptance_threshold,
        'sample_size': k,
        'expected_samples': m,
        'set_A_size': len(A),
        'empirical_support': empirical_support,  # Elements we actually saw
        'theoretical_support': theoretical_support,  # Size of set A
        'expected_support': expected_support,  # Expected number of unique elements
        'tvd_estimate': tvd
    }
    
    return Z <= acceptance_threshold, stats