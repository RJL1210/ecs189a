import numpy as np
import random
import json

def grain_distribution(D, m):

    total = sum(D.values())
    assert np.isclose(total, 1), "Probabilities in D do not sum to 1"


    # grain each probability to nearest multiple of 1/m
    grained_D = {}
    for key, value in D.items():
        grained_value = round(value * m) / m
        grained_D[key] = grained_value

    # normalize grained distribution to ensure sum is 1
    normalization_factor = sum(grained_D.values())
    for key in grained_D:
        grained_D[key] /= normalization_factor

    return grained_D

def expand_domain(D, m, domain_size):

    if isinstance(D, dict):
        D = np.array(list(D.values()))
    elif not isinstance(D, np.ndarray):
        raise TypeError("D must be a NumPy array or dictionary")
    

    # check if sum is 1
    assert np.isclose(D.sum(), 1), "Probabilities do not sum to 1"

    # convert probabilities to counts based on m
    counts = (D * m).astype(int)
    expanded_domain = {}
    index = 1

    for i, count in enumerate(counts):  
        # map each element to range in [6n]
        expanded_domain[str(i + 1)] = list(range(index, index + count))
        index += count

    if index - 1 > 6 * domain_size:
        raise ValueError("Mapping exceeds domain size [6n], check your Distribution")
    
    return expanded_domain

def map_samples(X, expanded_domain):
    # transform 

    transformed_samples = []

    for x in X:
        if str(x) in expanded_domain:
            # choose mapped index from expanded at random
            transformed_samples.append(random.choice(expanded_domain[str(x)]))
        else:
            raise ValueError(f"Sample {x} not found in original domain of D")

    return transformed_samples

def collision_tester(samples, domain_size, epsilon):
    """
    collision tester for uniformity using epsilon as the threshold.

    args:
        samples (list or numpy.ndarray): the samples drawn from the distribution.
        domain_size (int): the size of the domain (e.g., 6n for goldreich).
        epsilon (float): distance threshold for testing.

    returns:
        bool: true if the samples are uniform, false otherwise.
    """
    # number of samples
    s = len(samples)
    if s < 2:
        raise ValueError("need at least two samples for collision testing.")

    # count the frequency of each sample
    from collections import Counter
    frequency = Counter(samples)

    # compute the collision statistic
    collision_stat = sum(f * (f - 1) for f in frequency.values()) / (s * (s - 1))

    # compute the expected uniform collision probability
    uniform_collision_prob = 1 / domain_size

    print("Collision Statistic:", collision_stat)
    print("Expected Uniform Probability:", uniform_collision_prob)
    print("Difference:", abs(collision_stat - uniform_collision_prob))


    # check if the collision statistic is within epsilon of the uniform probability
    return abs(collision_stat - uniform_collision_prob) <= epsilon

def calculate_samples(domain_size, epsilon, k=1):
    
    # calculate the number of samples needed based on epsilon and domain size.

    if epsilon <= 0:
        raise ValueError("epsilon must be greater than 0.")
    
    return int(np.ceil(k * (np.sqrt(domain_size) / epsilon**2)))

def goldreich_file(D_file, X_file, m, epsilon):
    """
    args:
        D_file: path to json file for the fixed distribution D.
        X_file: path to json file for the unknown distribution X.
        m (int): graining parameter.
        epsilon (float): distance threshold.
    """
    # load D and X
    def json_parse(json_file):
        with open(json_file) as f:
            return json.load(f)

    D_json = json_parse(D_file)
    X_json = json_parse(X_file)

    D = np.array(list(D_json.values()))  # convert D to numpy array
    X_samples = X_json["samples"]  # extract samples from X json

    # calculate domain size and expanded domain size
    domain_size = len(D_json)
    expanded_domain_size = 6 * domain_size

    # grain the distribution D
    grained_D = grain_distribution(D_json, m)
    print(f"Grained D: {grained_D}")

    # expand domain for D
    expanded_D = expand_domain(D, m, domain_size)

    # map X samples to expanded domain
    mapped_X = map_samples(X_samples, expanded_D)
    print(f"Mapped X Samples: {mapped_X[:10]}")  # debug: print the first 10 mapped samples

    # calculate the number of samples based on epsilon
    samples_needed = calculate_samples(expanded_domain_size, epsilon)
    # ensure enough samples are available
    if len(X_samples) < samples_needed:
        raise ValueError(f"not enough samples in X. required: {samples_needed}, provided: {len(X_samples)}")

    # use the first `samples_needed` samples from mapped X
    test_samples = mapped_X[:samples_needed]

    # perform collision testing
    is_uniform = collision_tester(test_samples, expanded_domain_size, epsilon)
    if is_uniform:
        print("accept: X matches D")
    else:
        print("reject: X is far from D")



if __name__ == "__main__":
    goldreich_file("./non_grained_d.json", "./X_far.json", 10, 0.06)
    goldreich_file("./non_grained_d.json", "./X_close.json", 10, 0.06)

# 0.05 doesnt work with the non grained D, 0.05 works well with grained D