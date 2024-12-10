import numpy as np
import os
import json

def uniform_exact(n):
    return np.ones(n) / n

def far_exact_distribution(n):
    dist = np.zeros(n)
    dist[0] = 0.7
    dist[1:] = 0.3 / (n - 1)
    return dist

def bimodal_distribution(n):
    dist = np.zeros(n)
    third = n // 3
    dist[:third] = 0.2 / third
    dist[third:2*third] = 1.0 / third
    dist[2*third:] = 2.8 / third
    return dist

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
        scale = n / 10  # Default scale
    x = np.arange(n)
    dist = np.exp(-x / scale)
    return dist / np.sum(dist)


def heavy_tailed_distribution(n, alpha=2):
    x = np.arange(1, n + 1)
    dist = 1 / (x**alpha)
    return dist / np.sum(dist)


def uniform_skew_distribution(n):
    dist = np.linspace(1, n, n)  # Gradually increasing probabilities
    return dist / np.sum(dist)

def generate_nice_distributions(domain_size=500, sample_size=3000000):
    os.makedirs("./D", exist_ok=True)
    os.makedirs("./D_mixed", exist_ok=True)  # New directory for mixed distributions
    os.makedirs("./X", exist_ok=True)

    n = domain_size
    uniform = np.ones(n) / n
    
    distributions = {
        'uniform_exact': uniform,
        'far_exact': far_exact_distribution(n),
        'bimodal': bimodal_distribution(n),
        'gaussian': gaussian_distribution(n),
        'exponential': exponential_distribution(n),
        'heavy_tailed': heavy_tailed_distribution(n),
        'uniform_skew': uniform_skew_distribution(n)
    }
    
    for name, dist in distributions.items():
        # Store original distribution
        D = {str(i+1): float(p) for i, p in enumerate(dist)}
        with open(f"./D/D_{name}.json", "w") as f:
            json.dump(D, f)
        
        # Create and store mixed distribution (90-10)
        close_probs = 0.9 * uniform + 0.1 * dist
        close_probs /= np.sum(close_probs)  # Normalize
        D_mixed = {str(i+1): float(p) for i, p in enumerate(close_probs)}
        with open(f"./D_mixed/D_{name}_mixed.json", "w") as f:
            json.dump(D_mixed, f)
        
        # Generate samples
        close_samples = np.random.choice(range(1, n+1), size=sample_size, p=close_probs)
        with open(f"./X/D_{name}_X_close.json", "w") as f:
            json.dump({"samples": close_samples.tolist()}, f)
        
        dist /= np.sum(dist)  # Normalize the distribution
        far_samples = np.random.choice(range(1, n+1), size=sample_size, p=dist)
        with open(f"./X/D_{name}_X_far.json", "w") as f:
            json.dump({"samples": far_samples.tolist()}, f)

if __name__ == "__main__":
    generate_nice_distributions(domain_size=750, sample_size=3000000)