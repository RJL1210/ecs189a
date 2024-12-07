import numpy as np
import json
import os
from typing import Dict, List
from goldreich import test_from_files

def generate_test_distributions(domain_size: int = 100, sample_size: int = 100000):
    """
    Generate test distributions with safer far distribution generation
    """
    os.makedirs("./D", exist_ok=True)
    os.makedirs("./X", exist_ok=True)
    
    def create_gaussian():
        mean = domain_size / 2
        std = domain_size / 6
        probs = np.exp(-0.5 * ((np.arange(1, domain_size + 1) - mean) / std) ** 2)
        return probs / np.sum(probs)
    
    def create_powerlaw():
        alpha = 1.2
        probs = 1.0 / (np.arange(1, domain_size + 1) ** alpha)
        return probs / np.sum(probs)
    
    def create_bimodal():
        mean1, mean2 = domain_size / 4, 3 * domain_size / 4
        std = domain_size / 15
        probs = (np.exp(-0.5 * ((np.arange(1, domain_size + 1) - mean1) / std) ** 2) * 2 + 
                np.exp(-0.5 * ((np.arange(1, domain_size + 1) - mean2) / std) ** 2))
        return probs / np.sum(probs)
    
    def create_noisy_uniform():
        probs = np.ones(domain_size)
        noise = np.random.normal(0, 0.3, domain_size)
        probs += noise
        probs = np.maximum(probs, 0.0001)  # Ensure no zeros
        return probs / np.sum(probs)
    
    def create_step():
        probs = np.ones(domain_size)
        probs[:domain_size//2] *= 3
        probs[domain_size//2:] *= 0.2
        return probs / np.sum(probs)
        
    def generate_far_distribution(probs):
        """
        Generate a distribution that's guaranteed to be far from input distribution
        """
        # Method 1: Reverse the probabilities (with smoothing to avoid division by zero)
        smoothed_probs = probs + 0.0001
        reverse_probs = 1.0 / smoothed_probs
        reverse_probs = reverse_probs / np.sum(reverse_probs)
        
        # Method 2: Shift the distribution
        shift_probs = np.roll(probs, domain_size//2)
        
        # Use whichever method creates a distribution more different from original
        dist1 = np.sum(np.abs(probs - reverse_probs))
        dist2 = np.sum(np.abs(probs - shift_probs))
        
        return reverse_probs if dist1 > dist2 else shift_probs
    
    # Generate distributions
    distributions = {
        "gaussian": create_gaussian(),
        "powerlaw": create_powerlaw(),
        "bimodal": create_bimodal(),
        "noisy_uniform": create_noisy_uniform(),
        "step": create_step()
    }
    
    for dist_name, probs in distributions.items():
        # Save distribution D
        D = {str(i+1): float(p) for i, p in enumerate(probs)}
        with open(f"./D/D_{dist_name}.json", "w") as f:
            json.dump(D, f)
        
        # Generate close samples (small perturbation)
        close_probs = probs * (1 + np.random.normal(0, 0.05, domain_size))  # Reduced noise
        close_probs = np.maximum(close_probs, 0)
        close_probs = close_probs / np.sum(close_probs)
        
        # Generate far samples
        far_probs = generate_far_distribution(probs)
        
        # Generate samples
        close_samples = np.random.choice(range(1, domain_size + 1), 
                                       size=sample_size, 
                                       p=close_probs)
        
        far_samples = np.random.choice(range(1, domain_size + 1), 
                                     size=sample_size, 
                                     p=far_probs)
        
        # Save samples
        with open(f"./X/D_{dist_name}_X_close.json", "w") as f:
            json.dump({"samples": close_samples.tolist()}, f)
            
        with open(f"./X/D_{dist_name}_X_far.json", "w") as f:
            json.dump({"samples": far_samples.tolist()}, f)

def run_tests(epsilon: float = 0.05):
    """
    Run tests with more detailed output
    """
    successes = 0
    total = 0
    
    for dist_name in ["gaussian", "powerlaw", "bimodal", "noisy_uniform", "step"]:
        print(f"\nTesting {dist_name} distribution:")
        
        print("Testing far distribution:")
        far_result = test_from_files(f"./D/D_{dist_name}.json", 
                                   f"./X/D_{dist_name}_X_far.json", epsilon)
        if not far_result:  # Should reject
            successes += 1
        total += 1
            
        print("\nTesting close distribution:")
        close_result = test_from_files(f"./D/D_{dist_name}.json", 
                                     f"./X/D_{dist_name}_X_close.json", epsilon)
        if close_result:  # Should accept
            successes += 1
        total += 1
            
        print("-" * 50)
    
    print(f"\nOverall Success Rate: {successes}/{total} tests passed")

if __name__ == "__main__":
    print("Generating test distributions...")
    generate_test_distributions(domain_size=100, sample_size=100000)
    
    print("\nRunning tests...")
    run_tests(epsilon=0.05)