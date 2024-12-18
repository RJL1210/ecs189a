import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import json
import os
from identity_tester import identity_tester

def visualize_modified_identity_test(q, samples, epsilon, title, output_dir="identity_visual"):
    """
    Visualization for the modified identity test.
    """
    os.makedirs(output_dir, exist_ok=True)

    decision, stats = identity_tester(q, samples, epsilon)

    # Extract visualization data
    q_array = np.array([q[str(i+1)] for i in range(len(q))])
    counts = Counter(samples)
    empirical_dist = np.array([counts.get(i+1, 0) / len(samples) for i in range(len(q))])

    Z_contributions = [c['contribution'] for c in stats['components']]

    # Create visualization
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    # Plot empirical vs target distributions
    ax[0].plot(range(1, len(q)+1), q_array, label='Target Distribution', color='blue')
    ax[0].plot(range(1, len(q)+1), empirical_dist, label='Empirical Distribution', color='red')
    ax[0].set_title(f"{title} - Distribution Comparison")
    ax[0].legend()

    # Plot Z contributions
    ax[1].bar(range(1, len(Z_contributions)+1), Z_contributions, label='Z Contributions')
    ax[1].set_title(f"{title} - Z Contributions")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}.png"))
    plt.close()


if __name__ == "__main__":
    # Load test cases
    with open("./D/D_gaussian.json") as f:
        q = json.load(f)

    with open("./X/D_gaussian_X_close.json") as f:
        samples = json.load(f)["samples"]

    epsilon = 0.1
    visualize_modified_identity_test(q, samples, epsilon, "Gaussian Distribution Close Test")