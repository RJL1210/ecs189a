import os
import json
import numpy as np
from identity_tester import identity_tester
from goldreich_tester import majority_rules_test
from visualize_goldreich import visualize_batch_results  
from visualize_identity import visualize_modified_identity_test
from generate_distributions_and_samples import generate_nice_distributions

def batch_compare_testers(test_dir: str, sample_dir: str, epsilon: float = 0.1, num_repetitions: int = 5):
    """Run both testers with comparable output format and majority rule decision"""
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

            # Identity Tester with majority rule
            id_accept_count = 0
            for _ in range(num_repetitions):
                id_decision, id_stats = identity_tester(q, samples, epsilon)
                id_accept_count += int(id_decision)

            final_id_decision = id_accept_count > (num_repetitions / 2)
            print("Identity Tester:")
            print(f"  Final Decision: {'Accept' if final_id_decision else 'Reject'}")
            print(f"  Acceptance Count: {id_accept_count}/{num_repetitions}")
            print(f" Z_statistic: {id_stats['Z_statistic']}")
            print(f" threshold: {id_stats['threshold']}")
            print(f" sample_size: {id_stats['sample_size']}")

            # Goldreich Tester with majority rule
            gold_accept_count = 0
            for _ in range(num_repetitions):
                gold_decision, gold_stats = majority_rules_test(q, samples, epsilon)
                gold_accept_count += int(gold_decision)

            final_gold_decision = gold_accept_count > (num_repetitions / 2)
            print("Goldreich Tester:")
            print(f"  Final Decision: {'Accept' if final_gold_decision else 'Reject'}")
            print(f"  Acceptance Count: {gold_accept_count}/{num_repetitions}")
            print(f" gold_stats: {gold_stats}")


generate = True
visual = False

if __name__ == "__main__":

    if generate == True:
        domain_size = 1000
        epsilon = 0.1
        sample_size = 10 * int(np.ceil(np.sqrt(domain_size) / (epsilon * epsilon)))
        generate_nice_distributions(domain_size=domain_size, sample_size=sample_size, epsilon=epsilon)
        print("Distributions and samples generated.")


    #Run batch tests
    batch_compare_testers("D", "X", epsilon=0.1, num_repetitions=501)

    if visual == True:
        for dist_file in os.listdir("D"):
            if dist_file.endswith('.json'):
                with open(os.path.join("D", dist_file)) as f:
                    q = json.load(f)

                for sample_type in ['close', 'far']:
                    with open(os.path.join("X", f"{os.path.splitext(dist_file)[0]}_X_{sample_type}.json")) as f:
                        samples = json.load(f)["samples"]

                    visualize_modified_identity_test(q, samples, 0.1, f"{os.path.splitext(dist_file)[0]}_{sample_type}")
        
        visualize_batch_results("D", "X", epsilon=0.1)