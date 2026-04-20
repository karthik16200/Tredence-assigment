# Tredence-assigment
This notebook implements a custom prunable neural network that learns which weights are important through learnable gate parameters. The network is trained on CIFAR-10 with a sparsity-inducing regularization loss.

# Prunable Neural Network with Learnable Gates

This project implements a custom neural network architecture designed for **structured pruning** through learnable gate mechanisms. By integrating a sigmoid-based gating system directly into the linear layers, the model can learn which weight connections are redundant during the training process, effectively optimizing for both accuracy and sparsity.

## 📁 Project Structure

- `prunable_neural_network.ipynb`: The primary Jupyter Notebook containing the implementation, training loops, and sparsity analysis.
- `README.md`: Project documentation.

## 🚀 Key Features

- **Custom PrunableLinear Layer**: A drop-in replacement for `nn.Linear` that includes learnable gate scores.
- **Sparsity Regularization**: Integration of a sparsity-inducing loss term ($\lambda$) that penalizes active gates, encouraging the network to "turn off" unnecessary parameters.
- **Dynamic Gating**: Utilizes a Sigmoid activation on gate scores to maintain values between $[0, 1]$, allowing for gradient-based optimization of the pruning mask.
- **CIFAR-10 Implementation**: A complete pipeline for training and evaluating the prunable architecture on the CIFAR-10 dataset.
- **Performance Visualization**: Includes tools to plot gate distributions and compare accuracy vs. sparsity across different lambda values.

## 🛠️ Technical Overview

### 1. PrunableLinear Layer
The core of the project is the `PrunableLinear` module. Unlike standard layers, it maintains two sets of parameters:
- **Weights (W):** The standard trainable weight matrix.
- **Gate Scores (G):** Parameters that determine the importance of each connection.

During the forward pass:

Weight_pruned = W * sigma(G)

where sigma is the sigmoid function and $\odot$ is the element-wise product.

### 2. Sparsity Loss
To encourage pruning, the total loss function is defined as:

Loss = Loss_{classification} + lambda_sum (sigma(G))

By tuning the hyperparameter lambda, you can control the trade-off between model performance and the degree of pruning.

## 💻 Requirements

- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Matplotlib
