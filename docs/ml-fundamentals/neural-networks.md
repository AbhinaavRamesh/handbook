# Neural Networks

> **Deep learning fundamentals** — backpropagation, activations, architectures

---

## One-Sentence Summary

Neural networks learn hierarchical representations by stacking layers of linear transformations and non-linear activations, training via gradient descent with backpropagation.

---

## Core Architecture

### Single Neuron

A single neuron computes:

$$z = \sum_{i} w_i x_i + b = \mathbf{w} \cdot \mathbf{x} + b$$

$$a = \sigma(z)$$

Where $\sigma$ is the activation function.

### Feedforward Network

**Architecture:** Input Layer $\rightarrow$ Hidden Layer(s) $\rightarrow$ Output Layer

**Forward pass** through $L$ layers:

$$\mathbf{h}^{(l)} = \sigma(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})$$

---

## Activation Functions

### Why Non-linearity?

Without activations, stacking layers collapses to a single linear transformation:

$$\mathbf{W}_2(\mathbf{W}_1\mathbf{x}) = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x} = \mathbf{W}\mathbf{x}$$

Non-linear activations enable learning complex, non-linear decision boundaries.

![Activation Functions Comparison](/images/nn-activations.svg)

### Common Activations

| Activation | Formula | Range | Use Case |
|------------|---------|-------|----------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | $(0, 1)$ | Binary classification output |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | Hidden layers (zero-centered) |
| **ReLU** | $f(x) = \max(0, x)$ | $[0, \infty)$ | Hidden layers (default choice) |
| **Leaky ReLU** | $f(x) = \max(\alpha x, x)$ | $(-\infty, \infty)$ | Prevents dead neurons |
| **Softmax** | $\sigma_i(\mathbf{x}) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | $(0, 1)$, sum=1 | Multi-class output |

### ReLU: Pros and Cons

**Advantages:**
- No vanishing gradient (for positive inputs)
- Sparse activation (only some neurons fire)
- Computationally efficient (just max operation)

**Disadvantages:**
- **Dead neurons** — neurons with always-negative inputs never update
- **Not zero-centered** — can slow convergence

**Fixes:** Leaky ReLU, ELU, GELU

---

## Backpropagation

### Core Idea

Use the **chain rule** to compute gradients layer by layer, from output back to input.

### Key Equations

For layer $l$:

| Quantity | Formula |
|----------|---------|
| Error at layer $l$ | $\delta^{(l)} = (\mathbf{W}^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(\mathbf{z}^{(l)})$ |
| Weight gradient | $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = (\mathbf{a}^{(l-1)})^T \delta^{(l)}$ |
| Bias gradient | $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \delta^{(l)}$ |

### Implementation

```python
def forward(X, weights, biases):
    activations = [X]
    z_values = []
    for W, b in zip(weights, biases):
        z = activations[-1] @ W + b
        z_values.append(z)
        activations.append(relu(z))
    return activations, z_values

def backward(y, activations, z_values, weights):
    delta = activations[-1] - y  # Cross-entropy + softmax
    gradients_W, gradients_b = [], []

    for i in reversed(range(len(weights))):
        grad_W = activations[i].T @ delta / len(y)
        grad_b = np.mean(delta, axis=0)
        gradients_W.insert(0, grad_W)
        gradients_b.insert(0, grad_b)
        if i > 0:
            delta = (delta @ weights[i].T) * relu_derivative(z_values[i-1])

    return gradients_W, gradients_b
```

---

## Loss Functions

| Loss | Formula | Use Case |
|------|---------|----------|
| **Binary Cross-Entropy** | $\mathcal{L} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$ | Binary classification (sigmoid) |
| **Categorical Cross-Entropy** | $\mathcal{L} = -\sum_i y_i \log(\hat{y}_i)$ | Multi-class (softmax, one-hot $y$) |
| **Mean Squared Error** | $\mathcal{L} = \frac{1}{n} \sum(y - \hat{y})^2$ | Regression |

---

## Training Challenges

### Vanishing Gradient

**Problem:** Gradients shrink exponentially through layers (sigmoid, tanh)

**Solutions:**
- ReLU activation
- Residual/skip connections
- Batch normalization
- Proper weight initialization

### Exploding Gradient

**Problem:** Gradients grow exponentially

**Solutions:**
- Gradient clipping
- Proper weight initialization
- Batch normalization

### Weight Initialization

| Method | Formula | Use With |
|--------|---------|----------|
| **Xavier/Glorot** | $W \sim \mathcal{N}(0, \frac{2}{n_{in} + n_{out}})$ | Sigmoid, Tanh |
| **He** | $W \sim \mathcal{N}(0, \frac{2}{n_{in}})$ | ReLU |
| **LeCun** | $W \sim \mathcal{N}(0, \frac{1}{n_{in}})$ | SELU |

---

## Regularization

### L2 Regularization (Weight Decay)

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \sum ||\mathbf{W}||^2$$

Penalizes large weights, encouraging simpler models.

### Dropout

Randomly zero neurons during training:

```python
mask = np.random.binomial(1, keep_prob, size=layer.shape)
layer = layer * mask / keep_prob  # Scale to maintain expectation
```

- **Effect:** Forces redundant representations, approximates ensemble
- **Typical rates:** 0.2-0.5

### Batch Normalization

Normalize activations within each mini-batch:

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

**Benefits:**
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization

---

## Training Curves

Understanding training dynamics is crucial for diagnosing model issues:

![Training Loss Curves](/images/nn-training-curves.svg)

**Good fit:** Training and validation loss both decrease and converge together.

**Overfitting:** Training loss continues decreasing while validation loss increases. Use early stopping at the checkpoint where validation loss is lowest.

---

## Decision Boundaries

Neural networks can learn complex, non-linear decision boundaries that simple models cannot represent:

![XOR Decision Boundary](/images/nn-xor-boundary.svg)

The XOR problem is a classic example: no single line can separate the classes, but a neural network with one hidden layer can learn the non-linear boundary.

---

## Architectures

| Architecture | Key Feature | Best For |
|--------------|-------------|----------|
| **Fully Connected** | Every neuron connected to all in adjacent layers | Tabular data, small inputs |
| **CNN** | Local connectivity + weight sharing | Images, spatial data |
| **RNN/LSTM/GRU** | Recurrent connections (memory) | Text, time series |
| **Transformer** | Self-attention mechanism | NLP, vision (ViT) |

---

## Hyperparameters

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| **Learning rate** | Step size in gradient descent | $10^{-4}$ to $10^{-2}$ |
| **Batch size** | Samples per gradient update | 32, 64, 128, 256 |
| **Hidden units** | Capacity per layer | 64, 128, 256, 512 |
| **Layers** | Network depth | 2-10 for most tasks |
| **Dropout rate** | Regularization strength | 0.1-0.5 |
| **Optimizer** | Update algorithm | Adam (default) |

### Learning Rate Schedules

- **Step decay:** Reduce by factor at specific epochs
- **Exponential:** $\text{LR} = \text{LR}_0 \times \gamma^{\text{epoch}}$
- **Cosine annealing:** Smooth decay following cosine curve
- **Warmup:** Start small, increase, then decay

---

## Interview Questions

### Q1: "Explain backpropagation."

> "Backpropagation computes gradients of the loss with respect to all weights using the chain rule.
>
> **Process:**
> 1. **Forward pass** — compute activations layer by layer, storing intermediate values
> 2. **Compute loss** — compare predictions to targets
> 3. **Backward pass** — compute gradients using: $\delta^{(l)} = (\mathbf{W}^{(l+1)})^T\delta^{(l+1)} \odot \sigma'(\mathbf{z}^{(l)})$
> 4. **Update weights** — $\mathbf{W} = \mathbf{W} - \alpha \nabla_W \mathcal{L}$
>
> The key insight is reusing computations — we compute each layer's gradient once and propagate backward."

### Q2: "Why might ReLU cause problems?"

> "**Problem:** Dying neurons — if inputs are always negative, gradient is always zero, neuron never updates.
>
> **Causes:**
> - Poor initialization (large negative weights)
> - High learning rate (weights jump to bad region)
>
> **Fixes:**
> - Leaky ReLU: $f(x) = \max(0.01x, x)$
> - He initialization
> - Batch normalization"

### Q3: "How does batch size affect training?"

> | Small Batch (32) | Large Batch (1024+) |
> |------------------|---------------------|
> | Noisy gradients (regularization effect) | Stable gradients |
> | More updates per epoch | Fewer updates |
> | Better generalization | Risk of sharp minima |
> | Higher variance | Better GPU utilization |
>
> Start with 64-128, adjust based on GPU memory and validation performance.

### Q4: "How do you prevent overfitting?"

> **Data:** More training data, data augmentation
>
> **Architecture:** Fewer layers/units, dropout, batch normalization
>
> **Training:** L2 regularization, early stopping, LR scheduling
>
> **Ensemble:** Train multiple models, average predictions
>
> Typical baseline: dropout (0.2-0.5) + weight decay ($10^{-4}$) + early stopping.

---

## Code Reference

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        # He initialization
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return (x > 0).astype(float)
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = self.activations[-1] @ W + b
            self.z_values.append(z)
            a = self.softmax(z) if i == len(self.weights) - 1 else self.relu(z)
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, y, lr=0.01):
        m = len(y)
        delta = self.activations[-1] - y
        for i in reversed(range(len(self.weights))):
            self.weights[i] -= lr * (self.activations[i].T @ delta / m)
            self.biases[i] -= lr * np.mean(delta, axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu_derivative(self.z_values[i-1])

    def fit(self, X, y, epochs=100, lr=0.01, batch_size=32):
        y_onehot = np.eye(len(np.unique(y)))[y]
        for _ in range(epochs):
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i+batch_size]
                self.forward(X[batch_idx])
                self.backward(y_onehot[batch_idx], lr)

    def predict(self, X): return np.argmax(self.forward(X), axis=1)
```

---

## Quick Reference Card

```
NEURAL NETWORK
---------------------------------------------------
Architecture: Input -> Hidden(s) -> Output
Forward:      z = Wx + b, a = sigma(z)
Backward:     Chain rule to compute gradients

ACTIVATIONS
---------------------------------------------------
ReLU:     max(0, x)         -- hidden layers (default)
Sigmoid:  1/(1+e^-x)        -- binary output
Softmax:  e^xi / sum(e^xj)  -- multi-class output

TRAINING
---------------------------------------------------
Loss:      Cross-entropy (classification), MSE (regression)
Optimizer: Adam (default), SGD with momentum
Init:      He (ReLU), Xavier (sigmoid/tanh)

REGULARIZATION
---------------------------------------------------
Dropout:   Zero random activations (0.2-0.5)
L2:        Weight decay (1e-4 to 1e-2)
BatchNorm: Normalize activations

HYPERPARAMETERS
---------------------------------------------------
Learning rate: 1e-4 to 1e-2
Batch size:    32, 64, 128
Hidden units:  64, 128, 256, 512
Layers:        2-5 for most tasks
```
