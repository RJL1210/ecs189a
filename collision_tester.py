import numpy as np
import json
from collections import Counter
from typing import List
from goldreich import collision_tester

"""
JUST A HELPER FUNCTION TO TEST THE COLLISION TESTER
"""

def test_collision_tester():
    """
    Test collision tester directly on our generated distributions
    """
    domain_size = 100  # Matches what we used in distribution generation
    epsilon = 0.1
    
    print("Testing Collision Tester")
    print("=" * 50)
    
    # Test each distribution type
    for dist_type in ["gaussian", "l2_far", "l2_close"]:
        print(f"\nTesting {dist_type} distribution:")
        print("-" * 30)
        
        # Load and test close samples
        with open(f"./X/D_{dist_type}_X_close.json") as f:
            close_samples = json.load(f)["samples"]
        print("Close samples:")
        result = collision_tester(close_samples, domain_size, epsilon)
        print(f"Result: {'Accept' if result else 'Reject'}")
        print(f"Expected: Accept")
        
        # Load and test far samples
        print("\nFar samples:")
        with open(f"./X/D_{dist_type}_X_far.json") as f:
            far_samples = json.load(f)["samples"]
        result = collision_tester(far_samples, domain_size, epsilon)
        print(f"Result: {'Accept' if result else 'Reject'}")
        print(f"Expected: Reject")
        
        # Calculate actual L2 norm of samples for verification
        close_freqs = Counter(close_samples)
        far_freqs = Counter(far_samples)
        
        close_l2 = sum((freq/len(close_samples))**2 for freq in close_freqs.values())
        far_l2 = sum((freq/len(far_samples))**2 for freq in far_freqs.values())
        
        print(f"\nEmpirical L2 norms:")
        print(f"Close samples L2: {close_l2:.6f}")
        print(f"Far samples L2: {far_l2:.6f}")
        print(f"Uniform L2: {1/domain_size:.6f}")

if __name__ == "__main__":
    test_collision_tester()
