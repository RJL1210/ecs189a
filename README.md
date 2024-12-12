# Distribution Testing Toolkit

## Overview

This toolkit provides a set of Python scripts for testing the uniformity of discrete probability distributions using two different statistical testing methods:

1. Identity Tester (based on Algorithm 12.1)
2. Goldreich Collision Tester

## Prerequisites

- Python 3.7+
- NumPy
- Matplotlib (for visualization)

## Installation

1. Clone the repository
2. Install required dependencies:
   ```
   pip install numpy matplotlib
   ```

## Usage Workflow

### 1. Generate Distributions and Samples

First, run `generate_distributions_and_samples.py` to create test distributions and corresponding sample sets:

```bash
python generate_distributions_and_samples.py
```

This script does the following:

- Creates a `./D` directory with JSON files representing different probability distributions
- Creates an `./X` directory with sample sets for each distribution
- Generates two types of samples for each distribution:
  - `close` samples: Slightly mixed with uniform distribution
  - `far` samples: Drawn directly from the target distribution

Distribution types generated:

- Uniform
- Bimodal
- Gaussian
- Exponential
- Uniform Skew

### 2. Run Tests and Generate Visualizations

Use `compare.py` to run both the Identity and Goldreich testers and generate visualizations:

```bash
python compare.py
```

This script will:

- Load distributions from `./D`
- Load sample sets from `./X`
- Run both Identity and Goldreich testers
- Generate visualizations for:
  - Goldreich tester batch results
  - Identity tester batch results
- Print detailed test results for each distribution and sample type

## Understanding the Test Methods

### Identity Tester (`identity_tester.py`)

- Uses a statistical approach based on sample frequency
- Computes a Z-statistic to determine distribution uniformity
- Outputs decision (Accept/Reject) and detailed statistics

### Goldreich Tester (`goldreich_tester.py`)

- Uses a collision-based method
- Applies several reduction techniques to test uniformity
- Uses majority voting across multiple trials
- Provides collision rate and support size statistics

## Parameters

Key parameters you can adjust:

- `epsilon`: Proximity parameter (default: 0.1)
  - Controls the sensitivity of the uniformity test
- `domain_size`: Size of the distribution (default: 800)
- `sample_size`: Number of samples to generate (default: 3,000,000)

## Customization

You can modify `generate_distributions_and_samples.py` to:

- Add new distribution types
- Change distribution parameters
- Adjust sample generation methods

## Example Workflow

```bash
# Step 1: Generate distributions and samples
python generate_distributions_and_samples.py

# Step 2: Run tests and generate visualizations
python compare.py
```

## Output

The script will generate:

- Console output with test results for each distribution
- Visualization plots in the current directory
  - Goldreich tester results
  - Identity tester results

## Notes

- Not all distributions will look perfectly uniform
- The testers provide probabilistic guarantees
- Adjust `epsilon` to control test sensitivity

## Troubleshooting

- Ensure all dependencies are installed
- Check Python version compatibility
- Verify input JSON files are correctly formatted

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
