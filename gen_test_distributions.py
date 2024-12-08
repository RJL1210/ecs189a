import numpy as np
import json
import os

def generate_nice_distributions(domain_size=1000, sample_size=100000):
    """
    Generate distributions that are more complex than uniform but still have nice,
    rational probabilities that sum to exactly 1. This helps avoid approximation issues.
    """

    os.makedirs("./D", exist_ok=True)
    os.makedirs("./X", exist_ok=True)

    n = domain_size
    uniform = np.ones(n) / n

    def save_distribution_and_samples(dist_name, probs):
        """
        Given a distribution 'probs', save the distribution D_<dist_name>.json and
        generate close and far samples.
        
        Close samples: Mix with uniform heavily (90% uniform + 10% this distribution).
        Far samples: Use a separate far distribution (already defined as far_exact) or just
        create a different pattern and use it for far samples.
        """
        D = {str(i+1): float(p) for i, p in enumerate(probs)}
        with open(f"./D/D_{dist_name}.json", "w") as f:
            json.dump(D, f)

        # Close samples: 90% uniform, 10% probs
        close_probs = 0.9*uniform + 0.1*probs
        close_l1_distance = sum(abs(p - 1/n) for p in close_probs)
        print(f"[DEBUG] Close distribution l1-distance: {close_l1_distance}")
        close_samples = np.random.choice(range(1, n+1), size=sample_size, p=close_probs)
        with open(f"./X/D_{dist_name}_X_close.json", "w") as f:
            json.dump({"samples": close_samples.tolist()}, f)

        # Far samples: For demonstration, use the far_exact distribution from before:
        far_probs = far_exact_distribution(n)
        far_samples = np.random.choice(range(1, n+1), size=sample_size, p=far_probs)
        with open(f"./X/D_{dist_name}_X_far.json", "w") as f:
            json.dump({"samples": far_samples.tolist()}, f)

    def uniform_exact(n):
        # Exactly uniform, each = 1/n
        return np.ones(n)/n

    def far_exact_distribution(n):
        """
        A far-from-uniform distribution:
        - First element: 0.5 probability
        - Remaining n-1 elements share 0.5 equally, each = 0.5/(n-1)
        All are rational multiples of 1/n (since 0.5 = (n/2)* (1/n) if n divides nicely).
        """
        dist = np.zeros(n)
        dist[0] = 0.5
        dist[1:] = 0.5/(n-1)
        return dist

    def bimodal_distribution(n):
        """
        Bimodal distribution:
        - First half of the domain: each element gets probability = 1/(2n)
        - Second half of the domain: each element gets probability = 3/(2n)
        
        Check sum:
        First half sum: (n/2)* (1/(2n)) = 1/4
        Second half sum: (n/2)* (3/(2n)) = 3/4
        Total = 1/4 + 3/4 = 1

        We now have a distribution that puts less mass on the first half and more on the second half,
        clearly not uniform but still nicely rational.
        """
        dist = np.zeros(n)
        half = n//2
        dist[:half] = 1/(2*n)   # Low probability block
        dist[half:] = 3/(2*n)   # High probability block
        return dist

    def step_distribution(n):
        """
        Step distribution with 4 equal blocks of n/4 each:
        
        Block 1: each = 1/(2n)
        Block 2: each = 1/(2n)
        Block 3: each = 3/(2n)
        Block 4: each = 3/(2n)

        Check sum:
        Each block has n/4 elements.
        Block 1 sum: (n/4)* (1/(2n)) = 1/8
        Block 2 sum = 1/8 total so far=1/4
        Block 3 sum: (n/4)*(3/(2n))=3/8 total so far=1/4+3/8=2/8+3/8=5/8
        Block 4 sum: 3/8 total=5/8+3/8=8/8=1

        This creates a "step" pattern: first half is lower probability, second half is higher.
        """
        dist = np.zeros(n)
        quarter = n//4
        dist[0:quarter] = 1/(2*n)
        dist[quarter:2*quarter] = 1/(2*n)
        dist[2*quarter:3*quarter] = 3/(2*n)
        dist[3*quarter:4*quarter] = 3/(2*n)
        return dist

    # Create distributions
    uniform_dist = uniform_exact(n)
    bimodal_dist = bimodal_distribution(n)
    step_dist = step_distribution(n)
    far_dist = far_exact_distribution(n)  # already defined
    # Also can re-use the close pattern by making tiny rational shifts:
    # For variety, let's just use these three main ones.

    # Save distributions and generate samples
    print("Creating uniform_exact distribution")
    save_distribution_and_samples("uniform_exact", uniform_dist)

    print("Creating bimodal distribution")
    save_distribution_and_samples("bimodal", bimodal_dist)

    print("Creating step distribution")
    save_distribution_and_samples("step", step_dist)

    print("Creating far_exact distribution")
    save_distribution_and_samples("far_exact", far_dist)

if __name__ == "__main__":
    generate_nice_distributions(domain_size=1000, sample_size=100000)
