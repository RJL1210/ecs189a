import numpy as np
import json
import os
import glob
from typing import Dict, List

def generate_D(domain_size: int, skew_type: str = "gaussian") -> Dict[str, float]:
    """
    Generate a distribution D with proper normalization and bounds checking.
    """
    if skew_type == "random":
        # Generate random probabilities with some structure
        probabilities = np.random.dirichlet(np.ones(domain_size))
    elif skew_type == "skewed":
        # Generate Zipf distribution with controlled skew
        s = 1.07  # Zipf parameter (closer to 1 means less extreme)
        probabilities = 1.0 / np.power(np.arange(1, domain_size + 1), s)
    elif skew_type == "gaussian":
        # Generate truncated gaussian
        mean = domain_size / 2
        stddev = domain_size / 6
        probabilities = np.exp(-0.5 * ((np.arange(domain_size) - mean) / stddev) ** 2)
    else:
        raise ValueError("Invalid skew_type")
    
    # Ensure proper normalization
    probabilities = probabilities / np.sum(probabilities)
    
    # Convert to dictionary
    return {str(i + 1): float(p) for i, p in enumerate(probabilities)}

def generate_X_close_to_D(D: Dict[str, float], num_samples: int) -> List[int]:
    """
    Generate samples that are close to distribution D.
    """
    domain = list(map(int, D.keys()))
    probabilities = list(D.values())
    
    # Add small random noise while maintaining closeness
    noise = np.random.normal(0, 0.01, len(probabilities))
    noisy_probs = np.array(probabilities) + noise
    noisy_probs = np.maximum(noisy_probs, 0)  # Ensure non-negative
    noisy_probs = noisy_probs / np.sum(noisy_probs)  # Renormalize
    
    return np.random.choice(domain, size=num_samples, p=noisy_probs).tolist()

def generate_X_far_from_D(D: Dict[str, float], num_samples: int) -> List[int]:
    """
    Generate samples that are ε-far from D in total variation distance.
    Uses a more extreme perturbation to ensure distance after reduction.
    """
    domain = list(map(int, D.keys()))
    n = len(domain)
    
    # Use a larger epsilon for generation to ensure distance after reduction
    epsilon = 0.8  # Much larger perturbation
    
    # Create a very skewed distribution
    skewed_probs = np.zeros(n)
    subset_size = max(1, n // 10)  # Concentrate on smaller subset
    skewed_probs[:subset_size] = 1.0 / subset_size
    
    # Mix original distribution with skewed distribution
    far_probs = (1 - epsilon) * np.array(list(D.values())) + epsilon * skewed_probs
    
    # Ensure normalization
    far_probs = far_probs / np.sum(far_probs)
    
    return np.random.choice(domain, size=num_samples, p=far_probs).tolist()

def generate_distributions(domain_size: int = 100, num_samples: int = 100000):
    """
    Generate and save all distributions and samples.
    """
    os.makedirs("./D", exist_ok=True)
    os.makedirs("./X", exist_ok=True)
    
    # Generate different types of distributions
    for skew_type in ["random", "skewed", "gaussian"]:
        D = generate_D(domain_size, skew_type)
        
        # Save D
        with open(f"./D/D_{skew_type}.json", "w") as f:
            json.dump(D, f)
            
        # Generate and save X_close
        X_close = generate_X_close_to_D(D, num_samples)
        with open(f"./X/D_{skew_type}_X_close.json", "w") as f:
            json.dump({"samples": X_close}, f)
            
        # Generate and save X_far
        X_far = generate_X_far_from_D(D, num_samples)
        with open(f"./X/D_{skew_type}_X_far.json", "w") as f:
            json.dump({"samples": X_far}, f)
        
        print(f"Generated files for {skew_type} distribution")

if __name__ == "__main__":
    # Generate with reasonable defaults
    generate_distributions(domain_size=100, num_samples=100000)