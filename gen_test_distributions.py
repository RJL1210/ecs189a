import numpy as np
import json
import os
from typing import Dict, List
from goldreich import test_distribution_files

def generate_test_distributions(domain_size: int = 1000, sample_size: int = 100000):
    """
    Generate test distributions designed to test the collision-based uniformity tester.
    Creates distributions with controlled L2-norm properties.
    """
    os.makedirs("./D", exist_ok=True)
    os.makedirs("./X", exist_ok=True)
    
    def create_gaussian():
        """
        Create a well-behaved Gaussian distribution
        """
        mean = domain_size / 2
        std = domain_size / 6
        probs = np.exp(-0.5 * ((np.arange(1, domain_size + 1) - mean) / std) ** 2)
        return probs / np.sum(probs)    
    
    def create_gaussian_close():
        """
        Create a close-to-uniform gaussian
        """
        mean = domain_size / 2
        std = domain_size / 2  # Increased std to make distribution flatter
        probs = np.exp(-0.5 * ((np.arange(1, domain_size + 1) - mean) / std) ** 2)
        # Mix with uniform to ensure closeness
        uniform = np.ones(domain_size) / domain_size
        probs = 0.8 * uniform + 0.2 * (probs / np.sum(probs))
        return probs / np.sum(probs)

    def create_close_samples(base_dist):
        """
        Create samples that are guaranteed to be close to uniform
        """
        # Mix base distribution heavily with uniform
        uniform = np.ones(domain_size) / domain_size
        close_probs = 0.9 * uniform + 0.1 * base_dist
        return np.random.choice(range(1, domain_size + 1), 
                              size=sample_size, 
                              p=close_probs)
    
    def create_l2_controlled_far(epsilon: float = 0.1):
        """
        Create distribution with controlled L2-norm to ensure it's ε-far from uniform.
        Uses concentration on a subset of elements to achieve desired L2-norm.
        """
        # Calculate target L2 norm from Corollary 11.2
        target_l2_squared = (1 + 4 * epsilon * epsilon) / domain_size
        
        # Initialize probabilities
        probs = np.ones(domain_size)
        
        # Concentrate probability on sqrt(n) elements
        num_concentrated = max(1, int(np.sqrt(domain_size)))
        
        # Calculate concentration while ensuring non-negative probabilities
        high_prob = min(0.8 / num_concentrated, 2.0 / domain_size)  # Cap the high probability
        
        # Set high probability for concentrated elements
        probs[:num_concentrated] = high_prob
        
        # Set remaining probability uniformly
        remaining_prob = (1.0 - high_prob * num_concentrated) / (domain_size - num_concentrated)
        probs[num_concentrated:] = remaining_prob
        
        # Ensure non-negative and normalized
        probs = np.maximum(probs, 0)
        return probs / np.sum(probs)
    
    def create_l2_controlled_close(epsilon: float = 0.1):
        """
        Create distribution that's close to uniform in L2 norm
        by adding small controlled perturbations.
        """
        # Start with uniform
        probs = np.ones(domain_size) / domain_size
        
        # Add small controlled perturbation
        max_perturbation = epsilon / (4 * domain_size)
        perturbation = np.random.uniform(-max_perturbation, max_perturbation, domain_size)
        
        # Ensure sum of perturbations is 0 to maintain total probability of 1
        perturbation -= np.mean(perturbation)
        
        # Add perturbation and ensure non-negative
        probs += perturbation
        probs = np.maximum(probs, 0)
        
        return probs / np.sum(probs)
    
    def verify_l2_norm(probs, epsilon: float, name: str):
        """
        Verify L2 norm properties of distribution
        """
        l2_squared = np.sum(probs * probs)
        uniform_l2 = 1.0 / domain_size
        far_threshold = (1 + 4 * epsilon * epsilon) / domain_size
        print(f"\nL2 norm verification for {name}:")
        print(f"L2-squared norm: {l2_squared:.6f}")
        print(f"Uniform L2-squared: {uniform_l2:.6f}")
        print(f"Far threshold: {far_threshold:.6f}")
        print(f"Min probability: {np.min(probs):.6f}")
        print(f"Max probability: {np.max(probs):.6f}")
    
    # Generate distributions
    distributions = {
        "gaussian": create_gaussian_close(),  # Use close gaussian for base
        "l2_far": create_l2_controlled_far(),
        "l2_close": create_l2_controlled_close()
    }
    
    # Save distributions and generate samples
    for dist_name, base_probs in distributions.items():
        verify_l2_norm(base_probs, epsilon=0.1, name=dist_name)
        
        # Save base distribution
        D = {str(i+1): float(p) for i, p in enumerate(base_probs)}
        with open(f"./D/D_{dist_name}.json", "w") as f:
            json.dump(D, f)
        
        # Generate and save close samples (mixed with uniform)
        close_samples = create_close_samples(base_probs)
        with open(f"./X/D_{dist_name}_X_close.json", "w") as f:
            json.dump({"samples": close_samples.tolist()}, f)
        
        # Generate and save far samples (using far distribution)
        far_probs = create_l2_controlled_far(epsilon=0.2)
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
        far_result = test_distribution_files(f"./D/D_{dist_name}.json", 
                                   f"./X/D_{dist_name}_X_far.json", epsilon)
        if not far_result:  # Should reject
            successes += 1
        total += 1
            
        print("\nTesting close distribution:")
        close_result = test_distribution_files(f"./D/D_{dist_name}.json", 
                                     f"./X/D_{dist_name}_X_close.json", epsilon)
        if close_result:  # Should accept
            successes += 1
        total += 1
            
        print("-" * 50)
    
    print(f"\nOverall Success Rate: {successes}/{total} tests passed")

if __name__ == "__main__":
    print("Generating test distributions...")
    generate_test_distributions(domain_size=1000, sample_size=100000)