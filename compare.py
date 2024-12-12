import os
import json
from identity_tester import identity_tester
from goldreich_tester import majority_rules_test
from visualize_goldreich import visualize_batch_results  
from visualize_identity import batch_visualize_identity_tests

def batch_compare_testers(test_dir: str, sample_dir: str, epsilon: float = 0.1):
    """Run both testers with comparable output format"""
    from goldreich_tester import majority_rules_test
    
    for dist_file in sorted(os.listdir(test_dir)):
        if not dist_file.endswith('.json'):
            continue
            
        print(f"\nTesting {os.path.splitext(dist_file)[0]}:")
        with open(os.path.join(test_dir, dist_file)) as f:
            q = json.load(f)
        
        for sample_type in ['close', 'far']:
            with open(os.path.join(sample_dir, f"{os.path.splitext(dist_file)[0]}_X_{sample_type}.json")) as f:
                samples = json.load(f)["samples"]
            
            print(f"\n{sample_type.upper()} samples:")
            
            # Identity tester
            
            id_decision, id_stats = identity_tester(q, samples, epsilon)
            print("Identity Tester:")
            print(f"  Decision: {'Accept' if id_decision else 'Reject'}")
            print(f"  Z statistic: {id_stats['Z_statistic']:.6f}")
            print(f"  Threshold: {id_stats['threshold']:.6f}")
            print(f"  Sample size: {id_stats['sample_size']} (expected: {id_stats['expected_samples']})")
            print(f"  Support size: {id_stats['empirical_support']:.1f}/{id_stats['set_A_size']} (expected: {id_stats['expected_support']:.1f})")
            print(f"  TVD estimate: {id_stats['tvd_estimate']:.6f}")
            
            
            # Goldreich tester
            
            gold_decision, gold_stats = majority_rules_test(q, samples, epsilon)
            
            print("\nGoldreich Tester:")
            print(f"  Decision: {'Accept' if gold_decision else 'Reject'}")
            print(f"  Votes: {gold_stats['positive_votes']}/{gold_stats['total_trials']}")
            print(f"  Collision rate: {gold_stats['avg_collision_rate']:.6f}")
            print(f"  Support size: {gold_stats['avg_support_size']:.1f}")
            

if __name__ == "__main__":
    # Run all tests with 5 repetitions per test and plot for the first run
    batch_compare_testers("D", "X", epsilon=0.1)

    # Run both visualizations
    #visualize_batch_results("D", "X", epsilon=0.1)
    #batch_visualize_identity_tests("D", "X", epsilon=0.1)