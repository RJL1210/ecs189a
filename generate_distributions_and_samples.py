import numpy as np
import os
import json

def uniform_exact(n):
    return np.ones(n) / n

def bimodal_distribution(n, peak1=0.3, peak2=0.7, std=0.1):
    x = np.linspace(0, 1, n)
    dist1 = np.exp(-0.5 * ((x - peak1) / std)**2)
    dist2 = np.exp(-0.5 * ((x - peak2) / std)**2)
    dist = dist1 + dist2
    return dist / np.sum(dist)

def gaussian_distribution(n, mean=None, std=None):
    if mean is None:
        mean = n / 2
    if std is None:
        std = n / 10  # Default spread
    x = np.arange(n)
    dist = np.exp(-0.5 * ((x - mean) / std)**2)
    return dist / np.sum(dist)

def exponential_distribution(n, scale=None):
    if scale is None:
        scale = n / 10
    x = np.arange(n)
    dist = np.exp(-x / scale)
    return dist / np.sum(dist)

def uniform_skew_distribution(n):
    dist = np.linspace(1, n, n)
    return dist / np.sum(dist)

def generate_nice_distributions(domain_size=800, sample_size=1000000):
    os.makedirs("./D", exist_ok=True)
    os.makedirs("./X", exist_ok=True)

    n = domain_size
    uniform = np.ones(n) / n
    
    distributions = {
        'uniform_exact': uniform,
        'bimodal': bimodal_distribution(n),
        'gaussian': gaussian_distribution(n),
        'exponential': exponential_distribution(n),
        'uniform_skew': uniform_skew_distribution(n)
    }
    
    for name, dist in distributions.items():
        # Store target distribution
        D = {str(i+1): float(p) for i, p in enumerate(dist)}
        with open(f"./D/D_{name}.json", "w") as f:
            json.dump(D, f)
        
        # Generate close samples (90-10 mix with uniform)
        close_probs = 0.95 * uniform + 0.05 * dist
        close_probs /= np.sum(close_probs)
        close_samples = np.random.choice(range(1, n+1), size=sample_size, p=close_probs)
        with open(f"./X/D_{name}_X_close.json", "w") as f:
            json.dump({"samples": close_samples.tolist()}, f)
        
        # Generate far samples
        dist /= np.sum(dist)
        far_samples = np.random.choice(range(1, n+1), size=sample_size, p=dist)
        with open(f"./X/D_{name}_X_far.json", "w") as f:
            json.dump({"samples": far_samples.tolist()}, f)

if __name__ == "__main__":
    generate_nice_distributions(domain_size = 750, sample_size = 100000)