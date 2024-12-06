import numpy as np
import json

def json_parse(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

def grain_distribution(D, m):
    # TODO: Implement graining logic (optional if D is m-grained)
    pass

def expand_domain(D, m, domain_size):
    
    # check if sum is 1
    assert np.isclose(D.sum(), 1), "Probabilities do not sum to 1"

    # convert probabilities to counts based on m
    counts = (D * m).astype(int)
    expanded_domain = {}
    index = 1

    for i, count in enumerate(counts):  # Fixed range(enumerate()) issue here
        # map each element to range in [6n]
        expanded_domain[i + 1] = list(range(index, index + count))
        index += count

    if index - 1 > 6 * domain_size:
        raise ValueError("Mapping exceeds domain size [6n], check your Distribution")
    
    return expanded_domain

def map_samples(X, expanded_domain):
    # Transform samples from X to expanded domain
    pass

def collision_tester(samples, expanded_size):
    # TODO: Implement collision tester (uniformity testing)
    pass

def goldreich_file(D_file, X_File, m, epsilon):
    D_json = json_parse(D_file)
    D = np.array(list(D_json.values()))  # Correct conversion of dictionary values to NumPy array
    
    domain_size = len(D_json)

    print(f"D: {D}")
    print(f"m: {m}")
    print(f"Domain size: {domain_size}")

    print(expand_domain(D, m, domain_size))

if __name__ == "__main__":
    goldreich_file("./test.json", 10, 0.1)