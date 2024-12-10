# Goldreich Distribution Tester

This project implements a uniformity testing algorithm based on Goldreich's reduction for distribution testing. The goal is to determine whether a given distribution is close to uniform or far from it, using a sublinear number of samples.

The implementation closely follows Goldreich's reduction framework, applying quantization, mapping, and collision-based testing to evaluate the uniformity of a distribution. By leveraging this approach, the project provides an efficient solution for analyzing large distributions without requiring the entire dataset.

## Introduction

Goldreich's uniformity testing framework is a sublinear algorithmic approach to determining whether a given distribution is uniform or far from it in Total Variation Distance (TVD). This project simulates multiple synthetic distributions, applies reduction techniques, and visualizes results to validate the tester's effectiveness.

## Features

- Generates a variety of synthetic distributions.
- Reduces distributions using Goldreich's quantization and mapping algorithms.
- Tests proximity to uniformity using collision statistics.
- Visualizes results to analyze distribution transformations.

## Project Structure

```plaintext
├── D/                                      # Generated target distributions (JSON)
├── X/                                      # Generated sample sets (JSON)
├── goldreich.py                            # Core implementation of reduction algorithms
├── goldreich_tester.py                     # Uniformity testing logic (From lecture 11, algorithm 11.3)
├── generate_distributions_and_samples.py   # Script to generate distributions
├── run_goldreich.py                        # Main script for executing tests
├── visualization.py                        # Visualization of distributions
├── README.md                               # Project documentation
```

## Usage

This project includes scripts to generate distributions, test them for uniformity, and visualize the results. Below are the steps to use these scripts effectively.

### Generating Distributions

To create target distributions and sample sets for testing, run:

```bash
python generate_distributions_and_samples.py
```

This script generates:

- Target distributions: Stored in the D/ directory.
- Sample sets: Stored in the X/ directory.
- Each distribution type has two sample sets:

Close samples: A 90-10 blend of the target distribution and uniform distribution.
Far samples: Samples drawn entirely from the target distribution.

### Running Tests

Once the distributions and samples are ready, you can run the Goldreich uniformity tester:

```bash
python run_goldreich.py
```

This script will:

- Load distributions and samples.
- Apply the Goldreich reduction algorithm.
- Test for uniformity using collision rates.
- Output decisions for each distribution (Uniform or Not Uniform) with statistics like collision rates and support size.

#### Example Test Output

```text
Testing D_bimodal (close):
Decision: Uniform
Votes: 11/11 (100.00%)
Avg collision rate: 0.000230
Support size: 4498.9 (4496-4500)

Testing D_bimodal (far):
Decision: Not Uniform
Votes: 0/11 (0.00%)
Avg collision rate: 0.000268
Support size: 4483.3 (4479-4489)
```

### Visualizing Results

To visualize how distributions change after applying the Goldreich reduction:

```bash
python visualization.py
```

This script generates visualizations for:

Original Distribution: The target distribution before sampling or reduction.
Final Distribution: The reduced distribution after applying Goldreich's algorithm.
Comparison: Overlays the original and final distributions for direct comparison.
Output Location
Generated visualizations are saved in the goldreich_visual/ directory:

original_distribution.png: Shows the original distribution.
final_distribution.png: Shows the reduced distribution.
comparison_distribution.png: Overlays the original and reduced distributions
