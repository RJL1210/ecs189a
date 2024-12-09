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

def generate_nice_distributions(domain_size=1000, sample_size=3000000):
    os.makedirs("./D", exist_ok=True)
    os.makedirs("./X", exist_ok=True)

    n = domain_size
    uniform = np.ones(n) / n
    
    distributions = {
        'uniform_exact': uniform,
        'far_exact': far_exact_distribution(n),
        'bimodal': bimodal_distribution(n)
    }
    
    for name, dist in distributions.items():
        D = {str(i+1): float(p) for i, p in enumerate(dist)}
        with open(f"./D/D_{name}.json", "w") as f:
            json.dump(D, f)
        
        close_probs = 0.9 * uniform + 0.1 * dist
        close_probs /= np.sum(close_probs)  # Normalize the probabilities
        close_samples = np.random.choice(range(1, n+1), size=sample_size, p=close_probs)
        with open(f"./X/D_{name}_X_close.json", "w") as f:
            json.dump({"samples": close_samples.tolist()}, f)
        
        dist /= np.sum(dist)  # Normalize the distribution
        far_samples = np.random.choice(range(1, n+1), size=sample_size, p=dist)
        with open(f"./X/D_{name}_X_far.json", "w") as f:
            json.dump({"samples": far_samples.tolist()}, f)

if __name__ == "__main__":
    generate_nice_distributions(domain_size=1000, sample_size=3000000)