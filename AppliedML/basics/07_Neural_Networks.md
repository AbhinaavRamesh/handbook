# Neural Networks

> **Deep learning fundamentals** — backpropagation, activations, architectures

---

## One-Sentence Description

Neural networks learn hierarchical representations by stacking layers of linear transformations and non-linear activations, training via gradient descent with backpropagation.

---

## Core Architecture

### Single Neuron

```
Inputs: x₁, x₂, ..., xₙ
Weights: w₁, w₂, ..., wₙ
Bias: b

z = Σ wᵢxᵢ + b = w·x + b    (linear combination)
a = σ(z)                     (activation function)
```

### Feedforward Network

```
Input Layer → Hidden Layer(s) → Output Layer

x → [W₁, b₁] → σ → [W₂, b₂] → σ → ... → ŷ
```

**Forward pass**:
```
h₁ = σ(W₁x + b₁)      # First hidden layer
h₂ = σ(W₂h₁ + b₂)     # Second hidden layer
ŷ = σ(W₃h₂ + b₃)      # Output layer
```

---

## Activation Functions

### Why Non-linearity?

Without activation functions, stacking layers is useless:
```
W₂(W₁x) = (W₂W₁)x = Wx  (still linear)
```

Non-linear activations enable learning complex patterns.

### Common Activations

| Activation | Formula | Range | Use Case |
|------------|---------|-------|----------|
| **Sigmoid** | 1/(1 + e⁻ˣ) | (0, 1) | Output for binary classification |
| **Tanh** | (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ) | (-1, 1) | Hidden layers (centered) |
| **ReLU** | max(0, x) | [0, ∞) | Hidden layers (default) |
| **Leaky ReLU** | max(αx, x) | (-∞, ∞) | Avoid dead neurons |
| **Softmax** | eˣⁱ/Σeˣʲ | (0, 1), sum=1 | Output for multi-class |

### ReLU Advantages

- **No vanishing gradient** (for positive inputs)
- **Sparse activation** (only some neurons fire)
- **Computationally efficient** (just max operation)
- **Biological plausibility**

### ReLU Disadvantages

- **Dead neurons** — Neurons can "die" if inputs always negative
- **Not zero-centered** — Can slow convergence
- Fix: Use Leaky ReLU, ELU, or GELU

---

## Backpropagation

### Core Idea

Use chain rule to compute gradients layer by layer, from output back to input.

### Forward Pass

```python
def forward(X, weights, biases):
    activations = [X]
    z_values = []

    for W, b in zip(weights, biases):
        z = activations[-1] @ W + b
        z_values.append(z)
        a = relu(z)  # or other activation
        activations.append(a)

    return activations, z_values
```

### Backward Pass

```python
def backward(y, activations, z_values, weights):
    gradients_W = []
    gradients_b = []

    # Output layer gradient
    delta = activations[-1] - y  # For cross-entropy + softmax

    # Propagate backwards
    for i in reversed(range(len(weights))):
        grad_W = activations[i].T @ delta / len(y)
        grad_b = np.mean(delta, axis=0)
        gradients_W.insert(0, grad_W)
        gradients_b.insert(0, grad_b)

        if i > 0:  # Not input layer
            delta = (delta @ weights[i].T) * relu_derivative(z_values[i-1])

    return gradients_W, gradients_b
```

### Key Equations

For layer l:
```
δˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ ⊙ σ'(zˡ)    (error at layer l)
∂L/∂Wˡ = aˡ⁻¹ᵀδˡ            (weight gradient)
∂L/∂bˡ = δˡ                 (bias gradient)
```

---

## Loss Functions

### Binary Cross-Entropy

```
L = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

Use with sigmoid output.

### Categorical Cross-Entropy

```
L = -Σᵢ yᵢ log(ŷᵢ)
```

Use with softmax output. y is one-hot encoded.

### Mean Squared Error

```
L = (1/n) Σ(y - ŷ)²
```

Use for regression.

---

## Training Challenges

### Vanishing Gradient

**Problem**: Gradients shrink exponentially through layers (sigmoid, tanh)

**Solutions**:
- ReLU activation
- Residual connections (skip connections)
- Batch normalization
- Proper weight initialization

### Exploding Gradient

**Problem**: Gradients grow exponentially

**Solutions**:
- Gradient clipping
- Proper weight initialization
- Batch normalization

### Weight Initialization

| Method | Formula | Use With |
|--------|---------|----------|
| **Xavier/Glorot** | W ~ N(0, 2/(nᵢₙ + nₒᵤₜ)) | Sigmoid, Tanh |
| **He** | W ~ N(0, 2/nᵢₙ) | ReLU |
| **LeCun** | W ~ N(0, 1/nᵢₙ) | SELU |

---

## Regularization

### L2 Regularization (Weight Decay)

```
L_total = L_data + λ Σ||W||²
```

Penalizes large weights.

### Dropout

Randomly set neurons to zero during training:
```python
mask = np.random.binomial(1, keep_prob, size=layer.shape)
layer = layer * mask / keep_prob  # Scale to maintain expectation
```

- Forces redundant representations
- Approximates ensemble of networks
- Typical rates: 0.2-0.5

### Batch Normalization

Normalize activations within mini-batch:
```
μ = mean(x)
σ² = var(x)
x̂ = (x - μ) / √(σ² + ε)
y = γx̂ + β  (learnable scale and shift)
```

Benefits:
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization

---

## Architectures

### Fully Connected (Dense)

- Every neuron connected to all neurons in adjacent layers
- Good for: tabular data, small inputs
- Bad for: images, sequences (too many parameters)

### Convolutional Neural Networks (CNN)

- Local connectivity: neurons see small patches
- Weight sharing: same filter across positions
- Good for: images, spatial data
- Key layers: Conv2D, MaxPool, BatchNorm

### Recurrent Neural Networks (RNN)

- Connections form cycles (memory)
- Process sequential data
- Variants: LSTM, GRU (solve vanishing gradient)
- Good for: text, time series

### Transformers

- Attention mechanism: learn which inputs to focus on
- Parallelizable (unlike RNN)
- State-of-the-art for NLP, increasingly for vision
- Key: Self-attention, positional encoding

---

## Hyperparameters

| Hyperparameter | Effect | Typical Values |
|----------------|--------|----------------|
| **Learning rate** | Step size in gradient descent | 1e-4 to 1e-2 |
| **Batch size** | Samples per gradient update | 32, 64, 128, 256 |
| **Hidden units** | Capacity per layer | 64, 128, 256, 512 |
| **Layers** | Network depth | 2-10 for most tasks |
| **Dropout rate** | Regularization strength | 0.1-0.5 |
| **Optimizer** | Update algorithm | Adam (default) |

### Learning Rate Schedules

- **Constant**: Fixed LR
- **Step decay**: Reduce by factor at epochs
- **Exponential**: LR = LR₀ × γᵉᵖᵒᶜʰ
- **Cosine annealing**: Smooth decay
- **Warmup**: Start small, increase, then decay

---

## Interview Questions

### Q1: "Explain backpropagation."

**Strong answer**:
> "Backpropagation computes gradients of the loss with respect to all weights using the chain rule.
>
> The process:
> 1. **Forward pass**: Compute activations layer by layer, storing intermediate values
> 2. **Compute loss**: Compare predictions to targets
> 3. **Backward pass**: Starting from output, compute gradients using chain rule:
>    - δˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ ⊙ σ'(zˡ)
>    - ∂L/∂W = aᵀδ
> 4. **Update weights**: W = W - α × gradient
>
> The key insight is reusing computations — we compute the gradient at each layer once and propagate it backward, rather than computing each weight's gradient independently."

### Q2: "Why might ReLU cause problems and how do you fix them?"

**Strong answer**:
> "ReLU's main problem is **dying neurons** — if inputs are always negative, the gradient is always zero, and the neuron never updates.
>
> This can happen due to:
> - Poor initialization (weights too large negative)
> - High learning rate (weights jump to bad region)
> - Data distribution (inputs naturally negative)
>
> Solutions:
> 1. **Leaky ReLU**: f(x) = max(0.01x, x) — small gradient for negative inputs
> 2. **ELU/GELU**: Smooth alternatives with better properties
> 3. **Proper initialization**: He initialization for ReLU
> 4. **Lower learning rate**: Prevent large weight jumps
> 5. **Batch normalization**: Keeps activations in good range"

### Q3: "What's the difference between batch size effects?"

**Strong answer**:
> "Batch size affects both optimization and generalization:
>
> **Small batch (e.g., 32)**:
> - Noisy gradients → acts as regularization
> - More updates per epoch → can be faster early
> - Better generalization (research shows)
> - Higher variance in training
>
> **Large batch (e.g., 1024+)**:
> - Stable gradients → precise updates
> - Fewer updates per epoch
> - Can get stuck in sharp minima (worse generalization)
> - More efficient GPU utilization
>
> In practice, I'd start with batch size 64-128 and adjust based on GPU memory and validation performance. If using very large batches, warmup the learning rate."

### Q4: "How do you prevent overfitting in neural networks?"

**Strong answer**:
> "Multiple complementary approaches:
>
> 1. **Data**:
>    - More training data
>    - Data augmentation (for images: flips, crops, color jitter)
>
> 2. **Architecture**:
>    - Fewer layers/units (reduce capacity)
>    - Dropout (randomly zero activations)
>    - Batch normalization (also regularizes)
>
> 3. **Training**:
>    - L2 regularization (weight decay)
>    - Early stopping (monitor validation loss)
>    - Learning rate scheduling
>
> 4. **Ensemble**:
>    - Train multiple models, average predictions
>
> I'd typically use dropout (0.2-0.5) + weight decay (1e-4) + early stopping as a baseline. For images, data augmentation is crucial."

---

## Code Reference

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu'):
        self.weights = []
        self.biases = []

        # He initialization
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = self.activations[-1] @ W + b
            self.z_values.append(z)

            if i == len(self.weights) - 1:  # Output layer
                a = self.softmax(z)
            else:
                a = self.relu(z)
            self.activations.append(a)

        return self.activations[-1]

    def backward(self, y, lr=0.01):
        m = len(y)
        delta = self.activations[-1] - y  # Cross-entropy + softmax gradient

        for i in reversed(range(len(self.weights))):
            grad_W = self.activations[i].T @ delta / m
            grad_b = np.mean(delta, axis=0)

            self.weights[i] -= lr * grad_W
            self.biases[i] -= lr * grad_b

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu_derivative(self.z_values[i-1])

    def fit(self, X, y, epochs=100, lr=0.01, batch_size=32):
        n_samples = len(X)
        y_onehot = np.eye(len(np.unique(y)))[y]

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                self.forward(X_batch)
                self.backward(y_batch, lr)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
```

---

## Quick Reference Card

```
NEURAL NETWORK
─────────────────────────────────────────────────
Architecture: Input → Hidden(s) → Output
Forward:      z = Wx + b, a = σ(z)
Backward:     Chain rule to compute gradients

ACTIVATIONS
─────────────────────────────────────────────────
ReLU:     max(0, x)      — hidden layers (default)
Sigmoid:  1/(1+e⁻ˣ)      — binary output
Softmax:  eˣⁱ/Σeˣʲ       — multi-class output

TRAINING
─────────────────────────────────────────────────
Loss:     Cross-entropy (classification), MSE (regression)
Optimizer: Adam (default), SGD with momentum
Init:     He (ReLU), Xavier (sigmoid/tanh)

REGULARIZATION
─────────────────────────────────────────────────
Dropout:  Zero random activations (0.2-0.5)
L2:       Weight decay (1e-4 to 1e-2)
BatchNorm: Normalize activations

HYPERPARAMETERS
─────────────────────────────────────────────────
Learning rate: 1e-4 to 1e-2
Batch size:    32, 64, 128
Hidden units:  64, 128, 256, 512
Layers:        2-5 for most tasks
```

---

**Previous**: [← 06_KNN](./06_KNN.md) | **Next**: [08_Clustering →](./08_Clustering.md)
