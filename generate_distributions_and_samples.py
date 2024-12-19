import numpy as np
import os
import json
import scipy.stats

def tvd(p, q):
    """Calculate Total Variation Distance with error checking"""
    if not (0.99 <= np.sum(p) <= 1.01) or not (0.99 <= np.sum(q) <= 1.01):
        raise ValueError("Distributions must sum to approximately 1")
    return 0.5 * np.sum(np.abs(p - q))

def normalize(p):
    """Safely normalize a distribution"""
    p = np.maximum(p, 0)  # Ensure non-negative
    sum_p = np.sum(p)
    if sum_p == 0:
        raise ValueError("Cannot normalize zero vector")
    return p / sum_p

def generate_discrete_distributions(n: int):
    """Generate proper discrete distributions over [n]"""
    distributions = {}
    
    # 1. Uniform discrete
    distributions['uniform'] = np.ones(n) / n
    
    # 2. Binomial converted to distribution over [n]
    p = 0.5
    binomial = np.array([scipy.stats.binom.pmf(i, n-1, p) for i in range(n)])
    distributions['binomial'] = normalize(binomial)
    
    # 3. Geometric distribution truncated to [n]
    p = 0.3
    geometric = np.array([(1-p)**(i) * p for i in range(n)])
    distributions['geometric'] = normalize(geometric)
    
    # 4. Two-point distribution (concentrated on two elements)
    two_point = np.zeros(n)
    two_point[n//4] = 0.7
    two_point[3*n//4] = 0.3
    distributions['two_point'] = two_point
    
    # 5. Step distribution (half uniform at one level, half at another)
    step = np.ones(n)
    step[:n//2] *= 2
    step[n//2:] *= 0.5
    distributions['step'] = normalize(step)
    
    return distributions

def make_close_distribution(q, epsilon):
    """
    Create a distribution that's epsilon-close to q in TVD.
    Uses mixture with uniform distribution.
    """
    n = len(q)
    u = np.ones(n) / n
    p_close = (1 - epsilon)*q + epsilon*u
    p_close = normalize(p_close)
    
    # Verify TVD
    actual_tvd = tvd(p_close, q)
    if actual_tvd > epsilon + 1e-9:
        raise ValueError(f"Failed to create close distribution. TVD: {actual_tvd}")
    
    return p_close

def make_far_distribution_discrete(q, epsilon):
    """Create epsilon-far distribution by moving probability mass"""
    n = len(q)
    p_far = q.copy()
    
    # Check if distribution is highly concentrated
    sorted_probs = np.sort(q)[::-1]  
    if sorted_probs[0] + sorted_probs[1] > 0.8:  # If top 2 points have >80% of mass
        # Find the two highest probability points
        top_indices = np.argsort(q)[-2:]
        p_far = q.copy()  # Start with copy instead of zeros
        
        # Move just enough mass to make it epsilon-far
        for idx in top_indices:
            original_mass = p_far[idx]
            mass_to_move = original_mass * epsilon
            new_idx = (idx + n//3) % n
            
            # Move only epsilon fraction of the mass
            p_far[idx] -= mass_to_move
            p_far[new_idx] += mass_to_move
            
        return normalize(p_far)
    
    # Original code for other distributions
    def try_move_mass(from_low_to_high: bool):
        p_attempt = q.copy()
        sorted_indices = np.argsort(q)
        if not from_low_to_high:
            sorted_indices = sorted_indices[::-1]
            
        mass_to_move = epsilon
        idx1 = 0
        idx2 = n-1
        
        while mass_to_move > 0 and idx1 < idx2:
            moveable = min(p_attempt[sorted_indices[idx1]], mass_to_move)
            p_attempt[sorted_indices[idx1]] -= moveable
            p_attempt[sorted_indices[idx2]] += moveable
            mass_to_move -= moveable
            
            if p_attempt[sorted_indices[idx1]] <= 1e-10:
                idx1 += 1
            if mass_to_move <= 1e-10:
                break
            idx2 -= 1
            
        return normalize(p_attempt), tvd(normalize(p_attempt), q)
    
    # Try both directions
    p_low_to_high, tvd_low_to_high = try_move_mass(True)
    p_high_to_low, tvd_high_to_low = try_move_mass(False)
    
    if tvd_low_to_high >= epsilon:
        return p_low_to_high
    elif tvd_high_to_low >= epsilon:
        return p_high_to_low
    
    # Fallback to aggressive approach
    sorted_indices = np.argsort(q)
    p_far = q.copy()
    mid = n // 2
    mass_to_move = epsilon / 2
    p_far[sorted_indices[0]] += mass_to_move
    p_far[sorted_indices[-1]] += mass_to_move
    p_far[sorted_indices[mid-1:mid+1]] -= mass_to_move
    
    return normalize(p_far)

def generate_nice_distributions(domain_size=800, sample_size=1000000, epsilon=0.1):
    os.makedirs("./D", exist_ok=True)
    os.makedirs("./X", exist_ok=True)

    n = domain_size
    distributions = generate_discrete_distributions(n)
    
    results = {}
    for name, q in distributions.items():
        #print(f"\nProcessing {name} distribution...")
        
        # Store target distribution
        D = {str(i+1): float(p) for i, p in enumerate(q)}
        
        try:
            # Generate close and far distributions
            p_close = make_close_distribution(q, epsilon)
            p_far = make_far_distribution_discrete(q, epsilon)
            
            # Verify TVD
            close_tvd = tvd(p_close, q)
            far_tvd = tvd(p_far, q)
            """
            print(f"Close TVD: {close_tvd:.6f}")
            print(f"Far TVD: {far_tvd:.6f}")
            """
            
            # Generate samples
            close_samples = np.random.choice(range(1, n+1), size=sample_size, p=p_close)
            far_samples = np.random.choice(range(1, n+1), size=sample_size, p=p_far)
            
            # Save everything
            with open(f"./D/D_{name}.json", "w") as f:
                json.dump(D, f)
            
            with open(f"./X/D_{name}_X_close.json", "w") as f:
                json.dump({"samples": close_samples.tolist()}, f)
            
            with open(f"./X/D_{name}_X_far.json", "w") as f:
                json.dump({"samples": far_samples.tolist()}, f)
            
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
            continue
            
    return results

if __name__ == "__main__":
    domain_size = 100
    epsilon = 0.1
    sample_size = 3 * int(np.ceil(np.sqrt(domain_size) / (epsilon * epsilon)))
    generate_nice_distributions(domain_size=domain_size, sample_size=sample_size, epsilon=epsilon)
