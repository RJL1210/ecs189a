import matplotlib.pyplot as plt
import json
import numpy as np
from typing import Dict, List, Tuple

def identity_tester(q: Dict[str, float], samples: List[int], 
                           epsilon: float) -> Tuple[bool, dict]:
    """
    Modified identity tester with improved handling of low probability events
    """
    n = len(q)
    m = int(np.ceil(np.sqrt(n) / (epsilon * epsilon)))
    
    k = np.random.poisson(m)
    if k > 2*m:
        return False, {'Z_statistic': None, 'early_exit': True}
    
    k = min(k, len(samples))
    samples = samples[:k]
    
    # Count occurrences
    Ni = np.zeros(n)
    for sample in samples:
        Ni[sample - 1] += 1
    
    # More aggressive threshold for set A
    q_array = np.array([q[str(i + 1)] for i in range(n)])
    #min_expected = 1.0  # Require at least 1 expected observation
    #threshold = max(epsilon / (50 * n), min_expected / m)
    threshold = epsilon / (50 * n)
    A = np.where(q_array >= threshold)[0]
    
    # Modified Z calculation
    Z = 0.0
    Z_contributions = []
    
    for i in A:
        mqi = m * q_array[i]
        ni = Ni[i]
        
        if mqi > 0:
            # Calculate contribution with dampening for very small mqi
            raw_contribution = ((ni - mqi)**2 - ni) / mqi
            
            # Dampen contribution for very small expected counts
            if mqi < 1:
                dampening = mqi  # Linear dampening for small expected counts
                contribution = raw_contribution * dampening
            else:
                contribution = raw_contribution
                
            Z_contributions.append({
                'index': i,
                'qi': q_array[i],
                'ni': ni,
                'mqi': mqi,
                'contribution': contribution,
                'raw_contribution': raw_contribution
            })
            Z += contribution
    
    # Acceptance threshold as per lecture notes
    acceptance_threshold = m * epsilon**2 / 10
    
    # Sort contributions
    Z_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    stats = {
        'Z_statistic': Z,
        'threshold': acceptance_threshold,
        'ratio': Z/acceptance_threshold if acceptance_threshold > 0 else float('inf'),
        'sample_size': k,
        'expected_samples': m,
        'set_A_size': len(A),
        'A_threshold': threshold,
        'components': Z_contributions
    }
    
    return Z <= acceptance_threshold, stats


def analyze_and_visualize_results(results, epsilon):
    """
    Generate plots and analysis for modified identity test results.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    m_factors = list(results.keys())
    acceptance_rates = [results[m]['acceptance_rate'] for m in m_factors]
    ax.plot(m_factors, acceptance_rates, marker='o', label='Acceptance Rate')
    
    ax.axhline(y=2 / 3, color='red', linestyle='--', label='Theoretical Threshold')
    ax.set_title("Acceptance Rate vs. m Factor")
    ax.set_xlabel("m Factor")
    ax.set_ylabel("Acceptance Rate")
    ax.legend()
    plt.show()

    return fig


if __name__ == "__main__":
    # Example distributions and samples
    with open("./D/D_bimodal.json") as f:
        q = json.load(f)

    with open("./X/D_bimodal_X_close.json") as f:
        close_samples = json.load(f)["samples"]

    epsilon = 0.1
    results = run_multiple_trials(q, close_samples, epsilon)
    fig = analyze_and_visualize_results(results, epsilon)
    fig.savefig("modified_tester_analysis.png")