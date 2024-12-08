import numpy as np
from typing import Dict, List
import json
from goldreich import goldreich_test
from identity_tester import identity_tester

def compare_testers(D_file: str, X_file: str, epsilon: float = 0.1, trials: int = 10) -> Dict:
    """
    Compare results between identity_tester and goldreich_test over multiple trials
    
    Args:
        D_file: Path to distribution file
        X_file: Path to samples file
        epsilon: Distance parameter
        trials: Number of trials to run
        
    Returns:
        Dictionary with comparison statistics
    """
    # Load files
    with open(D_file) as f:
        D = json.load(f)
    with open(X_file) as f:
        X = json.load(f)
    
    results = {
        "identity_accepts": 0,
        "goldreich_accepts": 0,
        "agreements": 0,
        "trials": trials
    }
    
    # Run multiple trials
    for i in range(trials):
        print(f"\nTrial {i+1}:")
        
        # Run both tests
        identity_result = identity_tester(D, X["samples"], epsilon)
        goldreich_result = goldreich_test(D, X["samples"], epsilon)
        
        # Update statistics
        if identity_result:
            results["identity_accepts"] += 1
        if goldreich_result:
            results["goldreich_accepts"] += 1
        if identity_result == goldreich_result:
            results["agreements"] += 1
    
    # Calculate percentages
    results["identity_accept_rate"] = results["identity_accepts"] / trials * 100
    results["goldreich_accept_rate"] = results["goldreich_accepts"] / trials * 100
    results["agreement_rate"] = results["agreements"] / trials * 100
    
    return results

def run_comparison_tests(distributions: List[str] = ["gaussian", "l2_far", "l2_close"], 
                        epsilon: float = 0.1,
                        trials: int = 10):
    """
    Run comparison tests on all distributions
    
    Args:
        distributions: List of distribution types to test
        epsilon: Distance parameter
        trials: Number of trials per test
    """
    overall_results = {
        "total_trials": 0,
        "total_agreements": 0,
        "distribution_results": {}
    }
    
    for dist in distributions:
        print(f"\nTesting {dist} distribution:")
        
        # Test far distribution
        print("\nFar distribution results:")
        far_results = compare_testers(
            f"./D/D_{dist}.json",
            f"./X/D_{dist}_X_far.json",
            epsilon,
            trials
        )
        
        print(f"Identity tester accept rate: {far_results['identity_accept_rate']:.1f}%")
        print(f"Goldreich test accept rate: {far_results['goldreich_accept_rate']:.1f}%")
        print(f"Agreement rate: {far_results['agreement_rate']:.1f}%")
        
        overall_results["distribution_results"][f"{dist}_far"] = far_results
        
        # Test close distribution
        print("\nClose distribution results:")
        close_results = compare_testers(
            f"./D/D_{dist}.json",
            f"./X/D_{dist}_X_close.json",
            epsilon,
            trials
        )
        
        print(f"Identity tester accept rate: {close_results['identity_accept_rate']:.1f}%")
        print(f"Goldreich test accept rate: {close_results['goldreich_accept_rate']:.1f}%")
        print(f"Agreement rate: {close_results['agreement_rate']:.1f}%")
        
        overall_results["distribution_results"][f"{dist}_close"] = close_results
        
        # Update overall statistics
        overall_results["total_trials"] += 2 * trials
        overall_results["total_agreements"] += (far_results["agreements"] + 
                                              close_results["agreements"])
    
    # Print overall statistics
    print("\nOverall Results:")
    print("-" * 50)
    print(f"Total trials: {overall_results['total_trials']}")
    agreement_rate = (overall_results["total_agreements"] / 
                     overall_results["total_trials"] * 100)
    print(f"Overall agreement rate: {agreement_rate:.1f}%")
    
    return overall_results

if __name__ == "__main__":
    # Run tests with 10 trials per distribution
    overall_results = run_comparison_tests(trials=10)