import matplotlib.pyplot as plt
import numpy as np
from goldreich import quantize_distribution

def far_exact_distribution(n):
    dist = np.zeros(n)
    dist[0] = 0.7
    dist[1:] = 0.3 / (n - 1)
    return dist

def test_quantization_effect(n):
    dist = far_exact_distribution(n)
    dist_dict = {str(i): dist[i] for i in range(n)}  # Convert to dictionary with string keys
    quantized_dist_dict = quantize_distribution(dist_dict)  # Call the actual function
    quantized_dist = np.array([quantized_dist_dict.get(str(i), 0) for i in range(n)])  # Convert back to array
    plt.bar(range(n), dist, alpha=0.5, label="Original")
    plt.bar(range(n), quantized_dist, alpha=0.5, label="Quantized")
    plt.legend()
    plt.title("Quantization Effect on Far Exact")
    plt.show()

n = 100
dist = far_exact_distribution(n)
plt.bar(range(n), dist)
plt.title("Far Exact Distribution")
plt.xlabel("Element")
plt.ylabel("Probability")
plt.show()
test_quantization_effect(n)