import numpy as np
import json
import random


num_samples = 42427 # CHANGE THIS ACCORDING TO EPSILON/ HOW MANY SAMPLES IT REQUIRES
with open ("./non_grained_d.json") as f:
    D =  json.load(f)

def generate_X_close_to_D(D, num_samples):
    domain = list(D.keys())  # Elements of the domain
    probabilities = list(D.values())  # Probabilities of D
    # Generate samples using D's probabilities
    return np.random.choice(domain, size=num_samples, p=probabilities).tolist()

X_close = generate_X_close_to_D(D, num_samples)

with open("X_close.json", "w") as f:
    json.dump({"samples": X_close}, f)




def generate_X_far_from_D(D, num_samples):

    domain = list(D.keys())
    # Create a biased probability distribution for X
    far_probabilities = [0.8 if i == "1" else 0.1 for i in domain]  # Skew heavily toward "1"
    far_probabilities = np.array(far_probabilities) / np.sum(far_probabilities)  # Normalize
    # Generate samples using the biased probabilities
    return np.random.choice(domain, size=num_samples, p=far_probabilities).tolist()

X_far = generate_X_far_from_D(D, num_samples)

with open("X_far.json", "w") as f:
    json.dump({"samples": X_far}, f)