## MNIST Neural Network from scratch in Rust

A hand-written neural network implementation in Rust for classifying handwritten digits from the MNIST dataset. Built from scratch using only `ndarray` for matrix operations - no ML frameworks.

### Architecture

- **Input Layer:** 784 neurons (28x28 pixel images normalized to [0, 1])
- **Hidden Layer:** 128 neurons with ReLU activation
- **Output Layer:** 10 neurons with softmax activation (digits 0-9)
- **Loss Function:** Cross-entropy
- **Optimization:** Mini-batch gradient descent (batch size: 60)
- **Weight Initialization:** He (Kaiming) initialization

### Features

- CLI interface with `train` and `test` commands
- Shuffled training data each epoch for better generalization
- Custom binary model format for saving/loading weights
- Colorized terminal output with training progress and test results

### Usage

```bash
# Train the model
cargo run -- train

# Test the trained model
cargo run -- test
```

### Current Accuracy

─────────────────────────────────────────
[-] Test Results
─────────────────────────────────────────
  Samples tested : 1000
  Correct        : 941
  Incorrect      : 59
  Accuracy       : 94.10%  [PASS]
─────────────────────────────────────────

### Requirements

- MNIST dataset files in:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`
  - `t10k-images.idx3-ubyte`
  - `t10k-labels.idx1-ubyte`

