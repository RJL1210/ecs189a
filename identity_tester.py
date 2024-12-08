import numpy as np
from typing import Dict, List
import json

def identity_tester(q: Dict[str, float], samples: List[int], epsilon: float) -> bool:
    """
    Algorithm 12.1: Identity Tester
    Tests if samples come from distribution q or are ε-far from it.
    
    Args:
        q: Known distribution (maps string element IDs to probabilities)
        samples: List of samples from distribution to test
        epsilon: Distance parameter
        
    Returns:
        True if samples appear to come from q, False if they appear ε-far
    """
    # Get domain size n
    n = len(q)
    
    # Calculate sample complexity - O(√n/ε²) from lecture notes
    # Adding constant factor based on proof
    m = int(np.ceil(1000 * np.sqrt(n) / (epsilon * epsilon)))
    
    # Step 1: Draw k ~ Poi(m) samples
    k = np.random.poisson(m)
    k = min(k, len(samples))  # Don't use more samples than available
    samples = samples[:k]
    
    # Step 2: Count occurrences
    Ni = np.zeros(n)
    for sample in samples:
        Ni[sample-1] += 1
    
    # Step 3: Compute A = {i ∈ [n] | qi ≥ ε/(50n)}
    q_array = np.array([q[str(i+1)] for i in range(n)])
    threshold = epsilon / (50 * n)
    A = np.where(q_array >= threshold)[0]
    
    # Step 4: Compute Z = Σ((Ni - mqi)² - Ni) / mqi for i ∈ A
    Z = 0
    total_qi_A = 0
    for i in A:
        mqi = k * q_array[i]  # Using k instead of m since we use k samples
        if mqi > 0:  # Avoid division by zero
            Z += ((Ni[i] - mqi)**2 - Ni[i]) / mqi
            total_qi_A += q_array[i]
    
    # Print debug info
    print(f"\nDebug Information:")
    print(f"Sample complexity (m): {m}")
    print(f"Actual samples used (k): {k}")
    print(f"Domain size (n): {n}")
    print(f"Size of set A: {len(A)}")
    print(f"Total probability mass in A: {total_qi_A:.6f}")
    print(f"Z statistic: {Z:.6f}")
    print(f"Threshold (kε²/10): {(k * epsilon**2) / 10:.6f}")
    print(f"Ratio Z/threshold: {Z/((k * epsilon**2) / 10):.6f}")
    
    # Step 5: Accept if Z ≤ mε²/10
    return Z <= (k * epsilon**2) / 10

def test_on_files(D_file: str, X_file: str, epsilon: float = 0.1) -> bool:
    """Helper function to test on distribution files"""
    with open(D_file) as f:
        D = json.load(f)
    with open(X_file) as f:
        X = json.load(f)
    
    return identity_tester(D, X["samples"], epsilon)

def run_all_tests(distributions: List[str] = ["gaussian", "l2_far", "l2_close"], 
                  epsilon: float = 0.1):
    """Run tests on all distributions"""
    for dist in distributions:
        print(f"\nTesting {dist} distribution:")
        
        # Test far distribution
        print("\nTesting far distribution:")
        far_result = test_on_files(
            f"./D/D_{dist}.json", 
            f"./X/D_{dist}_X_far.json", 
            epsilon
        )
        print(f"Result: {'Accept' if far_result else 'Reject'}")
        
        # Test close distribution
        print("\nTesting close distribution:")
        close_result = test_on_files(
            f"./D/D_{dist}.json",
            f"./X/D_{dist}_X_close.json",
            epsilon
        )
        print(f"Result: {'Accept' if close_result else 'Reject'}")

if __name__ == "__main__":
    run_all_tests()