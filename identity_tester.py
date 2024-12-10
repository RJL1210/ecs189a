import numpy as np
from typing import Dict, List
import json
from collections import Counter
import matplotlib.pyplot as plt

def identity_tester(q: Dict[str, float], samples: List[int], epsilon: float, plot: bool = False) -> bool:
    """
    Algorithm 12.1: Identity Tester with optional plotting.
    
    Args:
        q: Known distribution (maps string element IDs to probabilities)
        samples: List of samples from distribution to test
        epsilon: Distance parameter
        plot: If True, generates a plot comparing theoretical and observed distributions.
        
    Returns:
        True if samples appear to come from q, False if they appear ε-far.
    """
    # Get domain size n
    n = len(q)
    
    # Calculate sample complexity - Θ(√n/ε²)
    m = int(np.ceil(25 * np.sqrt(n) / (epsilon * epsilon)))
    
    # Step 1: Draw k ~ Poi(m) samples
    k = np.random.poisson(m)
    k = min(k, len(samples))  # Don't use more samples than available
    samples = samples[:k]
    
    # Step 2: Count occurrences
    Ni = np.zeros(n)
    for sample in samples:
        Ni[sample - 1] += 1  # Assuming samples are 1-indexed
    
    # Step 3: Compute A = {i ∈ [n] | qi ≥ ε/(50n)}
    q_array = np.array([q[str(i + 1)] for i in range(n)])
    threshold = epsilon / (50 * n)
    A = np.where(q_array >= threshold)[0]
    
    # Step 4: Compute Z = Σ((Ni - kqi)² - Ni) / kqi for i ∈ A
    Z = 0
    for i in A:
        kqi = k * q_array[i]
        if kqi > 0:  # Avoid division by zero
            Z += ((Ni[i] - kqi) ** 2 - Ni[i]) / kqi
    
    # Optional plotting
    if plot:
        plot_distribution_comparison(q, Ni, k, n)
    
    # Step 5: Accept if Z ≤ kε²/10
    return Z <= (k * epsilon ** 2) / 10

def plot_distribution_comparison(q: Dict[str, float], Ni: np.ndarray, k: int, n: int):
    """
    Plot the theoretical distribution versus observed frequencies.
    
    Args:
        q: Known distribution as a dictionary.
        Ni: Observed sample counts.
        k: Number of samples used.
        n: Domain size.
    """
    domain = list(range(1, n + 1))  # Assuming 1-indexed domain
    q_array = np.array([q[str(i)] for i in domain])
    observed_freq = Ni / k  # Normalize observed counts
    
    plt.figure(figsize=(12, 6))
    plt.bar(domain, q_array, alpha=0.6, label="Theoretical Distribution (q)", color="blue")
    plt.bar(domain, observed_freq, alpha=0.6, label="Observed Frequencies (Normalized)", color="orange")
    plt.xlabel("Domain Elements")
    plt.ylabel("Probability")
    plt.title("Theoretical vs. Observed Distribution")
    plt.legend()
    plt.show()

def test_on_files(D_file: str, X_file: str, epsilon: float = 0.1, repeats: int = 1, plot: bool = False) -> str:
    """
    Helper function to test on distribution files multiple times and take majority result.
    
    Args:
        D_file: Path to the known distribution file.
        X_file: Path to the samples file.
        epsilon: Distance parameter.
        repeats: Number of times to repeat the test.
        plot: If True, generates a plot for the first run.
        
    Returns:
        Majority result as 'Accept' or 'Reject'.
    """
    with open(D_file) as f:
        D = json.load(f)
    with open(X_file) as f:
        X = json.load(f)
    
    results = []
    for i in range(repeats):
        result = identity_tester(D, X["samples"], epsilon, plot=(plot and i == 0))
        results.append('Accept' if result else 'Reject')
    
    # Determine majority result
    counter = Counter(results)
    majority_result = counter.most_common(1)[0][0]
    return majority_result

def run_all_tests(distributions: List[str] = ["bimodal", "exponential", "gaussian", "heavy_tailed", "uniform_exact", "uniform_skew"], 
                  epsilon: float = 0.1, repeats: int = 1, plot: bool = False):
    """
    Run tests on all distributions with repeated trials and take majority decision.
    
    Args:
        distributions: List of distributions to test.
        epsilon: Distance parameter.
        repeats: Number of repetitions per test.
        plot: If True, generates plots for the first run of each test.
    """
    for dist in distributions:
        print(f"\nTesting {dist} distribution:")
        
        # Test far distribution
        print("\nTesting far distribution:")
        far_result = test_on_files(
            f"./D/D_{dist}.json", 
            f"./X/D_{dist}_X_far.json", 
            epsilon, 
            repeats,
            plot
        )
        print(f"Result: {far_result}")
        
        # Test close distribution
        print("\nTesting close distribution:")
        close_result = test_on_files(
            f"./D/D_{dist}.json",
            f"./X/D_{dist}_X_close.json",
            epsilon, 
            repeats,
            plot
        )
        print(f"Result: {close_result}")

if __name__ == "__main__":
    # Run all tests with 5 repetitions per test and plot for the first run
    run_all_tests(repeats=5, plot=False)
