---
title: Neural Network (MLP) Implementation
description: Build a multi-layer perceptron from scratch with forward pass, backpropagation, and training loop.
---

# Neural Network (MLP) Implementation

> **Understand neural networks by building one from scratch**

Multi-Layer Perceptrons (MLPs) are the foundation of deep learning. Understanding how to implement one from scratch - including forward propagation, backpropagation, and gradient descent - is essential for ML interviews at top tech companies.

## Network Architecture Overview

```mermaid
graph LR
    subgraph Input["Input Layer"]
        x1((x1))
        x2((x2))
        x3((x3))
    end

    subgraph Hidden1["Hidden Layer 1"]
        h1((h1))
        h2((h2))
        h3((h3))
        h4((h4))
    end

    subgraph Hidden2["Hidden Layer 2"]
        h5((h5))
        h6((h6))
        h7((h7))
    end

    subgraph Output["Output Layer"]
        y1((y1))
        y2((y2))
    end

    x1 --> h1 & h2 & h3 & h4
    x2 --> h1 & h2 & h3 & h4
    x3 --> h1 & h2 & h3 & h4

    h1 --> h5 & h6 & h7
    h2 --> h5 & h6 & h7
    h3 --> h5 & h6 & h7
    h4 --> h5 & h6 & h7

    h5 --> y1 & y2
    h6 --> y1 & y2
    h7 --> y1 & y2
```

## Mathematical Foundation

### Forward Propagation

For each layer $l$, the forward pass computes:

$$z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g^{[l]}(z^{[l]})$$

Where:
- $W^{[l]}$ is the weight matrix for layer $l$
- $b^{[l]}$ is the bias vector for layer $l$
- $g^{[l]}$ is the activation function for layer $l$
- $a^{[l-1]}$ is the activation from the previous layer (input for first layer)

### Backpropagation

The backward pass computes gradients using the chain rule:

$$\frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial z^{[l]}} \cdot (a^{[l-1]})^T$$
$$\frac{\partial L}{\partial b^{[l]}} = \frac{\partial L}{\partial z^{[l]}}$$
$$\frac{\partial L}{\partial a^{[l-1]}} = (W^{[l]})^T \cdot \frac{\partial L}{\partial z^{[l]}}$$

```mermaid
graph TB
    subgraph Forward["Forward Pass"]
        direction LR
        A[Input X] --> B[Z1 = W1*X + b1]
        B --> C[A1 = ReLU Z1]
        C --> D[Z2 = W2*A1 + b2]
        D --> E[A2 = Softmax Z2]
        E --> F[Loss L]
    end

    subgraph Backward["Backward Pass"]
        direction RL
        G[dL/dA2] --> H[dL/dZ2]
        H --> I[dL/dW2, dL/db2]
        H --> J[dL/dA1]
        J --> K[dL/dZ1]
        K --> L[dL/dW1, dL/db1]
    end

    F -.-> G
```

---

## Complete Implementation from Scratch

```python
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
import pickle

class ActivationFunction:
    """
    Collection of activation functions and their derivatives.

    Each activation function is critical for introducing non-linearity
    into the network, allowing it to learn complex patterns.
    """

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation: σ(z) = 1 / (1 + e^(-z))

        Properties:
        - Output range: (0, 1)
        - Smooth gradient
        - Suffers from vanishing gradients for large |z|
        - Historically used, now less common except for output layers

        Args:
            z: Input array of any shape

        Returns:
            Activated values in range (0, 1)
        """
        # Clip to prevent overflow
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    @staticmethod
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        """
        Derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))

        Note: Takes activation 'a' as input (already computed sigmoid)
        This is more efficient during backpropagation.

        Args:
            a: Sigmoid activation values

        Returns:
            Derivative values
        """
        return a * (1 - a)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """
        ReLU activation: f(z) = max(0, z)

        Properties:
        - Computationally efficient
        - Helps with vanishing gradient problem
        - Can suffer from "dying ReLU" problem
        - Most commonly used activation for hidden layers

        Args:
            z: Input array of any shape

        Returns:
            Activated values (non-negative)
        """
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU: f'(z) = 1 if z > 0, else 0

        Args:
            z: Pre-activation values (z, not a)

        Returns:
            Derivative values (0 or 1)
        """
        return (z > 0).astype(float)

    @staticmethod
    def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Leaky ReLU: f(z) = z if z > 0, else alpha * z

        Properties:
        - Addresses dying ReLU problem
        - Small slope for negative values
        - alpha is typically 0.01

        Args:
            z: Input array
            alpha: Slope for negative values

        Returns:
            Activated values
        """
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Derivative of Leaky ReLU.

        Args:
            z: Pre-activation values
            alpha: Slope for negative values

        Returns:
            Derivative values
        """
        return np.where(z > 0, 1, alpha)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """
        Hyperbolic tangent: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

        Properties:
        - Output range: (-1, 1)
        - Zero-centered (unlike sigmoid)
        - Still suffers from vanishing gradients
        - Often used in RNNs and LSTMs

        Args:
            z: Input array

        Returns:
            Activated values in range (-1, 1)
        """
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(a: np.ndarray) -> np.ndarray:
        """
        Derivative of tanh: tanh'(z) = 1 - tanh²(z)

        Args:
            a: Tanh activation values

        Returns:
            Derivative values
        """
        return 1 - a ** 2

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """
        Softmax activation: softmax(z)_i = e^(z_i) / Σ e^(z_j)

        Properties:
        - Output is a probability distribution (sums to 1)
        - Used for multi-class classification output layer
        - Numerically stable implementation using max subtraction

        Args:
            z: Input array, shape (n_classes, batch_size)

        Returns:
            Probability distribution over classes
        """
        # Subtract max for numerical stability
        z_shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    @staticmethod
    def linear(z: np.ndarray) -> np.ndarray:
        """
        Linear (identity) activation: f(z) = z

        Used for regression output layers.

        Args:
            z: Input array

        Returns:
            Same as input
        """
        return z

    @staticmethod
    def linear_derivative(z: np.ndarray) -> np.ndarray:
        """
        Derivative of linear activation: f'(z) = 1

        Args:
            z: Input array

        Returns:
            Array of ones
        """
        return np.ones_like(z)

    @staticmethod
    def elu(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        Exponential Linear Unit: f(z) = z if z > 0, else alpha * (e^z - 1)

        Properties:
        - Smooth for negative values
        - Can produce negative outputs
        - Helps with vanishing gradients

        Args:
            z: Input array
            alpha: Scale for negative values

        Returns:
            Activated values
        """
        return np.where(z > 0, z, alpha * (np.exp(z) - 1))

    @staticmethod
    def elu_derivative(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        Derivative of ELU.

        Args:
            z: Pre-activation values
            alpha: Scale for negative values

        Returns:
            Derivative values
        """
        return np.where(z > 0, 1, alpha * np.exp(z))

    @staticmethod
    def swish(z: np.ndarray) -> np.ndarray:
        """
        Swish activation: f(z) = z * sigmoid(z)

        Properties:
        - Self-gated activation
        - Smooth and non-monotonic
        - Often outperforms ReLU

        Args:
            z: Input array

        Returns:
            Activated values
        """
        return z * ActivationFunction.sigmoid(z)

    @staticmethod
    def swish_derivative(z: np.ndarray) -> np.ndarray:
        """
        Derivative of Swish.

        Args:
            z: Pre-activation values

        Returns:
            Derivative values
        """
        sig = ActivationFunction.sigmoid(z)
        return sig + z * sig * (1 - sig)


class WeightInitializer:
    """
    Weight initialization strategies.

    Proper initialization is crucial for:
    - Preventing vanishing/exploding gradients
    - Enabling faster convergence
    - Achieving better final performance
    """

    @staticmethod
    def zeros(shape: Tuple[int, int]) -> np.ndarray:
        """
        Initialize weights to zeros.

        WARNING: Not recommended for weights (breaks symmetry).
        Use only for biases.

        Args:
            shape: (n_out, n_in) weight matrix shape

        Returns:
            Zero-initialized weights
        """
        return np.zeros(shape)

    @staticmethod
    def random_normal(shape: Tuple[int, int],
                      mean: float = 0.0,
                      std: float = 0.01) -> np.ndarray:
        """
        Initialize with random normal distribution.

        Simple but can cause vanishing/exploding gradients
        in deep networks.

        Args:
            shape: Weight matrix shape
            mean: Mean of distribution
            std: Standard deviation

        Returns:
            Randomly initialized weights
        """
        return np.random.randn(*shape) * std + mean

    @staticmethod
    def xavier_uniform(shape: Tuple[int, int]) -> np.ndarray:
        """
        Xavier/Glorot uniform initialization.

        Designed for tanh and sigmoid activations.
        Maintains variance across layers.

        Formula: W ~ U(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))

        Args:
            shape: (n_out, n_in) weight matrix shape

        Returns:
            Xavier-initialized weights
        """
        n_out, n_in = shape
        limit = np.sqrt(6.0 / (n_in + n_out))
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    def xavier_normal(shape: Tuple[int, int]) -> np.ndarray:
        """
        Xavier/Glorot normal initialization.

        Normal distribution variant of Xavier initialization.

        Formula: W ~ N(0, sqrt(2/(n_in + n_out)))

        Args:
            shape: (n_out, n_in) weight matrix shape

        Returns:
            Xavier-initialized weights
        """
        n_out, n_in = shape
        std = np.sqrt(2.0 / (n_in + n_out))
        return np.random.randn(*shape) * std

    @staticmethod
    def he_uniform(shape: Tuple[int, int]) -> np.ndarray:
        """
        He/Kaiming uniform initialization.

        Designed for ReLU activations.
        Accounts for ReLU zeroing half the values.

        Formula: W ~ U(-sqrt(6/n_in), sqrt(6/n_in))

        Args:
            shape: (n_out, n_in) weight matrix shape

        Returns:
            He-initialized weights
        """
        n_out, n_in = shape
        limit = np.sqrt(6.0 / n_in)
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    def he_normal(shape: Tuple[int, int]) -> np.ndarray:
        """
        He/Kaiming normal initialization.

        Most commonly used for ReLU networks.

        Formula: W ~ N(0, sqrt(2/n_in))

        Args:
            shape: (n_out, n_in) weight matrix shape

        Returns:
            He-initialized weights
        """
        n_out, n_in = shape
        std = np.sqrt(2.0 / n_in)
        return np.random.randn(*shape) * std

    @staticmethod
    def lecun_normal(shape: Tuple[int, int]) -> np.ndarray:
        """
        LeCun normal initialization.

        Designed for SELU activations.

        Formula: W ~ N(0, sqrt(1/n_in))

        Args:
            shape: (n_out, n_in) weight matrix shape

        Returns:
            LeCun-initialized weights
        """
        n_out, n_in = shape
        std = np.sqrt(1.0 / n_in)
        return np.random.randn(*shape) * std

    @staticmethod
    def orthogonal(shape: Tuple[int, int], gain: float = 1.0) -> np.ndarray:
        """
        Orthogonal initialization.

        Creates orthogonal matrices which help with gradient flow.
        Particularly useful for RNNs.

        Args:
            shape: Weight matrix shape
            gain: Scaling factor

        Returns:
            Orthogonally initialized weights
        """
        n_out, n_in = shape
        flat_shape = (n_out, n_in) if n_out > n_in else (n_in, n_out)

        # Generate random matrix
        a = np.random.randn(*flat_shape)

        # QR decomposition
        q, r = np.linalg.qr(a)

        # Make Q uniform
        d = np.diag(r)
        ph = np.sign(d)
        q *= ph

        if n_out < n_in:
            q = q.T

        return gain * q[:n_out, :n_in]


class LossFunction:
    """
    Collection of loss functions and their derivatives.

    Loss functions measure the discrepancy between predictions
    and true labels, guiding the optimization process.
    """

    @staticmethod
    def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Mean Squared Error loss.

        L = (1/n) * Σ(y_pred - y_true)²

        Used for regression tasks.

        Args:
            y_pred: Predictions, shape (n_features, n_samples)
            y_true: True values, same shape

        Returns:
            Scalar loss value
        """
        n_samples = y_true.shape[1]
        return np.sum((y_pred - y_true) ** 2) / (2 * n_samples)

    @staticmethod
    def mse_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Derivative of MSE loss with respect to predictions.

        dL/dy_pred = (y_pred - y_true) / n

        Args:
            y_pred: Predictions
            y_true: True values

        Returns:
            Gradient with respect to predictions
        """
        n_samples = y_true.shape[1]
        return (y_pred - y_true) / n_samples

    @staticmethod
    def binary_cross_entropy(y_pred: np.ndarray,
                             y_true: np.ndarray,
                             epsilon: float = 1e-15) -> float:
        """
        Binary Cross-Entropy loss.

        L = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]

        Used for binary classification.

        Args:
            y_pred: Predicted probabilities (0, 1)
            y_true: True labels {0, 1}
            epsilon: Small value for numerical stability

        Returns:
            Scalar loss value
        """
        n_samples = y_true.shape[1]

        # Clip predictions for numerical stability
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.sum(
            y_true * np.log(y_pred_clipped) +
            (1 - y_true) * np.log(1 - y_pred_clipped)
        ) / n_samples

        return loss

    @staticmethod
    def binary_cross_entropy_derivative(y_pred: np.ndarray,
                                        y_true: np.ndarray,
                                        epsilon: float = 1e-15) -> np.ndarray:
        """
        Derivative of Binary Cross-Entropy.

        dL/dy_pred = -(y/p - (1-y)/(1-p)) / n

        Args:
            y_pred: Predicted probabilities
            y_true: True labels
            epsilon: Numerical stability constant

        Returns:
            Gradient with respect to predictions
        """
        n_samples = y_true.shape[1]
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

        return (-(y_true / y_pred_clipped) +
                ((1 - y_true) / (1 - y_pred_clipped))) / n_samples

    @staticmethod
    def categorical_cross_entropy(y_pred: np.ndarray,
                                  y_true: np.ndarray,
                                  epsilon: float = 1e-15) -> float:
        """
        Categorical Cross-Entropy loss.

        L = -(1/n) * ΣΣ y_true * log(y_pred)

        Used for multi-class classification with softmax output.

        Args:
            y_pred: Predicted probabilities, shape (n_classes, n_samples)
            y_true: One-hot encoded labels, same shape
            epsilon: Numerical stability constant

        Returns:
            Scalar loss value
        """
        n_samples = y_true.shape[1]
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.sum(y_true * np.log(y_pred_clipped)) / n_samples

    @staticmethod
    def categorical_cross_entropy_derivative(y_pred: np.ndarray,
                                             y_true: np.ndarray) -> np.ndarray:
        """
        Derivative of Categorical Cross-Entropy.

        When combined with softmax, simplifies to: y_pred - y_true

        Args:
            y_pred: Predicted probabilities (after softmax)
            y_true: One-hot encoded labels

        Returns:
            Gradient with respect to pre-softmax values
        """
        n_samples = y_true.shape[1]
        return (y_pred - y_true) / n_samples

    @staticmethod
    def huber_loss(y_pred: np.ndarray,
                   y_true: np.ndarray,
                   delta: float = 1.0) -> float:
        """
        Huber loss (smooth L1 loss).

        Combines MSE and MAE - quadratic for small errors, linear for large.
        Less sensitive to outliers than MSE.

        Args:
            y_pred: Predictions
            y_true: True values
            delta: Threshold for switching between quadratic and linear

        Returns:
            Scalar loss value
        """
        n_samples = y_true.shape[1]
        error = y_pred - y_true

        is_small = np.abs(error) <= delta

        squared_loss = 0.5 * error ** 2
        linear_loss = delta * np.abs(error) - 0.5 * delta ** 2

        loss = np.where(is_small, squared_loss, linear_loss)
        return np.sum(loss) / n_samples

    @staticmethod
    def huber_loss_derivative(y_pred: np.ndarray,
                              y_true: np.ndarray,
                              delta: float = 1.0) -> np.ndarray:
        """
        Derivative of Huber loss.

        Args:
            y_pred: Predictions
            y_true: True values
            delta: Threshold

        Returns:
            Gradient with respect to predictions
        """
        n_samples = y_true.shape[1]
        error = y_pred - y_true

        is_small = np.abs(error) <= delta

        gradient = np.where(is_small, error, delta * np.sign(error))
        return gradient / n_samples


class Layer:
    """
    A single fully-connected (dense) layer in the neural network.

    Stores weights, biases, and intermediate values for backpropagation.
    """

    def __init__(self,
                 n_input: int,
                 n_output: int,
                 activation: str = 'relu',
                 weight_init: str = 'he_normal',
                 use_bias: bool = True):
        """
        Initialize a dense layer.

        Args:
            n_input: Number of input features
            n_output: Number of output features (neurons)
            activation: Activation function name
            weight_init: Weight initialization strategy
            use_bias: Whether to use bias terms
        """
        self.n_input = n_input
        self.n_output = n_output
        self.activation_name = activation
        self.use_bias = use_bias

        # Initialize weights
        self.W = self._init_weights((n_output, n_input), weight_init)
        self.b = np.zeros((n_output, 1)) if use_bias else None

        # Storage for forward/backward pass
        self.z = None  # Pre-activation
        self.a = None  # Post-activation
        self.a_prev = None  # Input from previous layer

        # Gradients
        self.dW = None
        self.db = None

        # For momentum/Adam
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b) if use_bias else None
        self.sW = np.zeros_like(self.W)
        self.sb = np.zeros_like(self.b) if use_bias else None

        # Set activation function
        self._set_activation(activation)

    def _init_weights(self, shape: Tuple[int, int], method: str) -> np.ndarray:
        """Initialize weights using specified method."""
        initializers = {
            'zeros': WeightInitializer.zeros,
            'random': WeightInitializer.random_normal,
            'xavier_uniform': WeightInitializer.xavier_uniform,
            'xavier_normal': WeightInitializer.xavier_normal,
            'glorot_uniform': WeightInitializer.xavier_uniform,
            'glorot_normal': WeightInitializer.xavier_normal,
            'he_uniform': WeightInitializer.he_uniform,
            'he_normal': WeightInitializer.he_normal,
            'lecun_normal': WeightInitializer.lecun_normal,
            'orthogonal': WeightInitializer.orthogonal,
        }

        if method not in initializers:
            raise ValueError(f"Unknown initialization: {method}")

        return initializers[method](shape)

    def _set_activation(self, name: str):
        """Set activation function and its derivative."""
        activations = {
            'sigmoid': (ActivationFunction.sigmoid,
                       ActivationFunction.sigmoid_derivative),
            'relu': (ActivationFunction.relu,
                    ActivationFunction.relu_derivative),
            'leaky_relu': (ActivationFunction.leaky_relu,
                          ActivationFunction.leaky_relu_derivative),
            'tanh': (ActivationFunction.tanh,
                    ActivationFunction.tanh_derivative),
            'softmax': (ActivationFunction.softmax, None),
            'linear': (ActivationFunction.linear,
                      ActivationFunction.linear_derivative),
            'elu': (ActivationFunction.elu,
                   ActivationFunction.elu_derivative),
            'swish': (ActivationFunction.swish,
                     ActivationFunction.swish_derivative),
        }

        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")

        self.activation, self.activation_derivative = activations[name]

    def forward(self, a_prev: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the layer.

        Computes: z = W * a_prev + b, then a = activation(z)

        Args:
            a_prev: Input from previous layer, shape (n_input, batch_size)
            training: Whether in training mode (store intermediate values)

        Returns:
            Activated output, shape (n_output, batch_size)
        """
        # Linear transformation
        self.z = np.dot(self.W, a_prev)
        if self.use_bias:
            self.z += self.b

        # Activation
        self.a = self.activation(self.z)

        # Store for backprop
        if training:
            self.a_prev = a_prev

        return self.a

    def backward(self, da: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer.

        Computes gradients dW, db and propagates gradient to previous layer.

        Args:
            da: Gradient of loss with respect to this layer's activation
                Shape: (n_output, batch_size)

        Returns:
            Gradient with respect to previous layer's activation
            Shape: (n_input, batch_size)
        """
        batch_size = da.shape[1]

        # Compute gradient through activation
        if self.activation_name == 'softmax':
            # For softmax + cross-entropy, da is already dz
            dz = da
        elif self.activation_name in ['sigmoid', 'tanh']:
            # These use activation values for derivative
            dz = da * self.activation_derivative(self.a)
        else:
            # ReLU, Leaky ReLU, etc. use pre-activation
            dz = da * self.activation_derivative(self.z)

        # Compute gradients for weights and biases
        self.dW = np.dot(dz, self.a_prev.T) / batch_size
        if self.use_bias:
            self.db = np.sum(dz, axis=1, keepdims=True) / batch_size

        # Compute gradient for previous layer
        da_prev = np.dot(self.W.T, dz)

        return da_prev

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get layer parameters."""
        params = {'W': self.W}
        if self.use_bias:
            params['b'] = self.b
        return params

    def set_params(self, params: Dict[str, np.ndarray]):
        """Set layer parameters."""
        self.W = params['W']
        if self.use_bias and 'b' in params:
            self.b = params['b']

    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get computed gradients."""
        grads = {'dW': self.dW}
        if self.use_bias:
            grads['db'] = self.db
        return grads


class Optimizer:
    """
    Base class for optimization algorithms.
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.t = 0  # Time step for Adam

    def update(self, layer: Layer):
        """Update layer parameters. Override in subclasses."""
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.

    Update rule:
    - Without momentum: W = W - lr * dW
    - With momentum: v = momentum * v - lr * dW, W = W + v
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 nesterov: bool = False):
        """
        Initialize SGD optimizer.

        Args:
            learning_rate: Step size
            momentum: Momentum factor (0 = no momentum)
            nesterov: Use Nesterov accelerated gradient
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov

    def update(self, layer: Layer):
        """Update layer parameters using SGD."""
        if self.momentum > 0:
            # Update velocity
            layer.vW = self.momentum * layer.vW - self.learning_rate * layer.dW

            if self.nesterov:
                # Nesterov: look ahead
                layer.W += self.momentum * layer.vW - self.learning_rate * layer.dW
            else:
                layer.W += layer.vW

            if layer.use_bias:
                layer.vb = self.momentum * layer.vb - self.learning_rate * layer.db
                if self.nesterov:
                    layer.b += self.momentum * layer.vb - self.learning_rate * layer.db
                else:
                    layer.b += layer.vb
        else:
            # Vanilla SGD
            layer.W -= self.learning_rate * layer.dW
            if layer.use_bias:
                layer.b -= self.learning_rate * layer.db


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).

    Combines momentum (first moment) and RMSprop (second moment).
    Very effective and widely used.

    Update rules:
    m = beta1 * m + (1 - beta1) * dW
    v = beta2 * v + (1 - beta2) * dW^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    W = W - lr * m_hat / (sqrt(v_hat) + epsilon)
    """

    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8):
        """
        Initialize Adam optimizer.

        Args:
            learning_rate: Step size (typically 0.001)
            beta1: Exponential decay for first moment (typically 0.9)
            beta2: Exponential decay for second moment (typically 0.999)
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, layer: Layer):
        """Update layer parameters using Adam."""
        self.t += 1

        # Update biased first moment estimate
        layer.vW = self.beta1 * layer.vW + (1 - self.beta1) * layer.dW

        # Update biased second moment estimate
        layer.sW = self.beta2 * layer.sW + (1 - self.beta2) * (layer.dW ** 2)

        # Bias correction
        vW_corrected = layer.vW / (1 - self.beta1 ** self.t)
        sW_corrected = layer.sW / (1 - self.beta2 ** self.t)

        # Update weights
        layer.W -= self.learning_rate * vW_corrected / (np.sqrt(sW_corrected) + self.epsilon)

        if layer.use_bias:
            layer.vb = self.beta1 * layer.vb + (1 - self.beta1) * layer.db
            layer.sb = self.beta2 * layer.sb + (1 - self.beta2) * (layer.db ** 2)

            vb_corrected = layer.vb / (1 - self.beta1 ** self.t)
            sb_corrected = layer.sb / (1 - self.beta2 ** self.t)

            layer.b -= self.learning_rate * vb_corrected / (np.sqrt(sb_corrected) + self.epsilon)


class RMSprop(Optimizer):
    """
    RMSprop optimizer.

    Adapts learning rate based on moving average of squared gradients.

    Update rules:
    v = rho * v + (1 - rho) * dW^2
    W = W - lr * dW / (sqrt(v) + epsilon)
    """

    def __init__(self,
                 learning_rate: float = 0.001,
                 rho: float = 0.9,
                 epsilon: float = 1e-8):
        """
        Initialize RMSprop optimizer.

        Args:
            learning_rate: Step size
            rho: Decay rate for moving average
            epsilon: Numerical stability constant
        """
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon

    def update(self, layer: Layer):
        """Update layer parameters using RMSprop."""
        # Update squared gradient average
        layer.sW = self.rho * layer.sW + (1 - self.rho) * (layer.dW ** 2)

        # Update weights
        layer.W -= self.learning_rate * layer.dW / (np.sqrt(layer.sW) + self.epsilon)

        if layer.use_bias:
            layer.sb = self.rho * layer.sb + (1 - self.rho) * (layer.db ** 2)
            layer.b -= self.learning_rate * layer.db / (np.sqrt(layer.sb) + self.epsilon)


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer.

    Adapts learning rate for each parameter based on historical gradients.
    Learning rate decreases over time (can be problematic for long training).
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon

    def update(self, layer: Layer):
        """Update layer parameters using AdaGrad."""
        # Accumulate squared gradients
        layer.sW += layer.dW ** 2

        # Update weights
        layer.W -= self.learning_rate * layer.dW / (np.sqrt(layer.sW) + self.epsilon)

        if layer.use_bias:
            layer.sb += layer.db ** 2
            layer.b -= self.learning_rate * layer.db / (np.sqrt(layer.sb) + self.epsilon)


class Regularizer:
    """
    Regularization techniques to prevent overfitting.
    """

    @staticmethod
    def l2_penalty(layers: List[Layer], lambda_: float) -> float:
        """
        Compute L2 regularization penalty.

        L2 = (lambda/2) * Σ||W||²

        Args:
            layers: List of network layers
            lambda_: Regularization strength

        Returns:
            Regularization penalty to add to loss
        """
        penalty = 0.0
        for layer in layers:
            penalty += np.sum(layer.W ** 2)
        return (lambda_ / 2) * penalty

    @staticmethod
    def l2_gradient(layer: Layer, lambda_: float) -> np.ndarray:
        """
        Compute L2 regularization gradient.

        dL2/dW = lambda * W

        Args:
            layer: Network layer
            lambda_: Regularization strength

        Returns:
            Gradient contribution from L2 regularization
        """
        return lambda_ * layer.W

    @staticmethod
    def l1_penalty(layers: List[Layer], lambda_: float) -> float:
        """
        Compute L1 regularization penalty.

        L1 = lambda * Σ|W|

        Args:
            layers: List of network layers
            lambda_: Regularization strength

        Returns:
            Regularization penalty
        """
        penalty = 0.0
        for layer in layers:
            penalty += np.sum(np.abs(layer.W))
        return lambda_ * penalty

    @staticmethod
    def l1_gradient(layer: Layer, lambda_: float) -> np.ndarray:
        """
        Compute L1 regularization gradient.

        dL1/dW = lambda * sign(W)

        Args:
            layer: Network layer
            lambda_: Regularization strength

        Returns:
            Gradient contribution from L1 regularization
        """
        return lambda_ * np.sign(layer.W)


class Dropout:
    """
    Dropout regularization.

    Randomly sets a fraction of activations to zero during training.
    Helps prevent overfitting by reducing co-adaptation of neurons.
    """

    def __init__(self, rate: float = 0.5):
        """
        Initialize dropout.

        Args:
            rate: Fraction of neurons to drop (0 to 1)
        """
        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be in [0, 1)")
        self.rate = rate
        self.mask = None

    def forward(self, a: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply dropout during forward pass.

        Args:
            a: Activation values
            training: Whether in training mode

        Returns:
            Masked activations (scaled during training)
        """
        if not training or self.rate == 0:
            return a

        # Create mask (inverted dropout - scale during training)
        self.mask = (np.random.rand(*a.shape) > self.rate) / (1 - self.rate)
        return a * self.mask

    def backward(self, da: np.ndarray) -> np.ndarray:
        """
        Backward pass through dropout.

        Args:
            da: Upstream gradient

        Returns:
            Gradient (masked with same pattern as forward)
        """
        if self.mask is None:
            return da
        return da * self.mask


class BatchNormalization:
    """
    Batch Normalization layer.

    Normalizes activations to have zero mean and unit variance,
    then applies learnable scale (gamma) and shift (beta).

    Helps with:
    - Faster training
    - Higher learning rates
    - Reducing internal covariate shift
    """

    def __init__(self, n_features: int, momentum: float = 0.99, epsilon: float = 1e-8):
        """
        Initialize batch normalization.

        Args:
            n_features: Number of features to normalize
            momentum: Momentum for running mean/variance
            epsilon: Numerical stability constant
        """
        self.n_features = n_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Learnable parameters
        self.gamma = np.ones((n_features, 1))  # Scale
        self.beta = np.zeros((n_features, 1))   # Shift

        # Running statistics for inference
        self.running_mean = np.zeros((n_features, 1))
        self.running_var = np.ones((n_features, 1))

        # Cache for backprop
        self.x_norm = None
        self.std = None
        self.mean = None
        self.x = None

        # Gradients
        self.dgamma = None
        self.dbeta = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through batch normalization.

        Args:
            x: Input, shape (n_features, batch_size)
            training: Whether in training mode

        Returns:
            Normalized and scaled output
        """
        self.x = x

        if training:
            # Compute batch statistics
            self.mean = np.mean(x, axis=1, keepdims=True)
            self.var = np.var(x, axis=1, keepdims=True)
            self.std = np.sqrt(self.var + self.epsilon)

            # Normalize
            self.x_norm = (x - self.mean) / self.std

            # Update running statistics
            self.running_mean = (self.momentum * self.running_mean +
                                (1 - self.momentum) * self.mean)
            self.running_var = (self.momentum * self.running_var +
                               (1 - self.momentum) * self.var)
        else:
            # Use running statistics for inference
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # Scale and shift
        return self.gamma * self.x_norm + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass through batch normalization.

        Args:
            dout: Upstream gradient

        Returns:
            Gradient with respect to input
        """
        batch_size = dout.shape[1]

        # Gradients for learnable parameters
        self.dgamma = np.sum(dout * self.x_norm, axis=1, keepdims=True)
        self.dbeta = np.sum(dout, axis=1, keepdims=True)

        # Gradient for normalized input
        dx_norm = dout * self.gamma

        # Gradient for variance
        dvar = np.sum(dx_norm * (self.x - self.mean) * -0.5 *
                      (self.var + self.epsilon) ** (-1.5), axis=1, keepdims=True)

        # Gradient for mean
        dmean = np.sum(dx_norm * -1 / self.std, axis=1, keepdims=True)
        dmean += dvar * np.mean(-2 * (self.x - self.mean), axis=1, keepdims=True)

        # Gradient for input
        dx = dx_norm / self.std
        dx += dvar * 2 * (self.x - self.mean) / batch_size
        dx += dmean / batch_size

        return dx


class MLP:
    """
    Multi-Layer Perceptron neural network.

    A complete implementation supporting:
    - Arbitrary architecture (number of layers and neurons)
    - Multiple activation functions
    - Various optimizers (SGD, Adam, RMSprop)
    - Regularization (L1, L2, Dropout)
    - Batch normalization
    - Mini-batch training
    """

    def __init__(self,
                 layer_sizes: List[int],
                 activations: List[str] = None,
                 weight_init: str = 'he_normal',
                 use_batch_norm: bool = False,
                 dropout_rates: List[float] = None):
        """
        Initialize MLP network.

        Args:
            layer_sizes: List of layer sizes, including input and output
                         e.g., [784, 256, 128, 10] for MNIST
            activations: Activation function for each layer (except input)
                         Defaults to ReLU for hidden, softmax for output
            weight_init: Weight initialization strategy
            use_batch_norm: Whether to use batch normalization
            dropout_rates: Dropout rate for each hidden layer
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.use_batch_norm = use_batch_norm

        # Set default activations
        if activations is None:
            activations = ['relu'] * (self.n_layers - 1) + ['softmax']

        # Validate input
        if len(activations) != self.n_layers:
            raise ValueError("Number of activations must match number of layers")

        # Create layers
        self.layers = []
        for i in range(self.n_layers):
            layer = Layer(
                n_input=layer_sizes[i],
                n_output=layer_sizes[i + 1],
                activation=activations[i],
                weight_init=weight_init
            )
            self.layers.append(layer)

        # Batch normalization layers
        self.batch_norms = []
        if use_batch_norm:
            for i in range(self.n_layers - 1):  # Not for output layer
                bn = BatchNormalization(layer_sizes[i + 1])
                self.batch_norms.append(bn)

        # Dropout layers
        self.dropouts = []
        if dropout_rates is not None:
            for rate in dropout_rates:
                self.dropouts.append(Dropout(rate))

        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward propagation through the entire network.

        Args:
            X: Input data, shape (n_features, batch_size)
            training: Whether in training mode

        Returns:
            Network output, shape (n_output, batch_size)
        """
        a = X

        for i, layer in enumerate(self.layers):
            a = layer.forward(a, training)

            # Apply batch normalization (except for output layer)
            if self.use_batch_norm and i < len(self.batch_norms):
                a = self.batch_norms[i].forward(a, training)

            # Apply dropout (except for output layer)
            if self.dropouts and i < len(self.dropouts):
                a = self.dropouts[i].forward(a, training)

        return a

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray,
                 lambda_l2: float = 0.0, lambda_l1: float = 0.0) -> float:
        """
        Backward propagation through the entire network.

        Computes gradients for all parameters using chain rule.

        Args:
            y_pred: Network predictions
            y_true: True labels
            lambda_l2: L2 regularization strength
            lambda_l1: L1 regularization strength

        Returns:
            Loss value
        """
        # Compute loss
        loss = LossFunction.categorical_cross_entropy(y_pred, y_true)

        # Add regularization penalty
        if lambda_l2 > 0:
            loss += Regularizer.l2_penalty(self.layers, lambda_l2)
        if lambda_l1 > 0:
            loss += Regularizer.l1_penalty(self.layers, lambda_l1)

        # Initial gradient (softmax + cross-entropy)
        da = LossFunction.categorical_cross_entropy_derivative(y_pred, y_true)

        # Backpropagate through layers
        for i in reversed(range(self.n_layers)):
            # Dropout gradient
            if self.dropouts and i < len(self.dropouts):
                da = self.dropouts[i].backward(da)

            # Batch norm gradient
            if self.use_batch_norm and i < len(self.batch_norms):
                da = self.batch_norms[i].backward(da)

            # Layer gradient
            da = self.layers[i].backward(da)

            # Add regularization gradient
            if lambda_l2 > 0:
                self.layers[i].dW += Regularizer.l2_gradient(self.layers[i], lambda_l2)
            if lambda_l1 > 0:
                self.layers[i].dW += Regularizer.l1_gradient(self.layers[i], lambda_l1)

        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Input data, shape (n_features, n_samples)

        Returns:
            Predicted class probabilities
        """
        return self.forward(X, training=False)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input data

        Returns:
            Predicted class indices
        """
        probs = self.predict(X)
        return np.argmax(probs, axis=0)

    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Args:
            X: Input data
            y: One-hot encoded labels

        Returns:
            Accuracy as fraction
        """
        predictions = self.predict_classes(X)
        true_classes = np.argmax(y, axis=0)
        return np.mean(predictions == true_classes)

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            optimizer: Optimizer = None,
            lambda_l2: float = 0.0,
            lambda_l1: float = 0.0,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
            early_stopping_patience: int = None,
            verbose: bool = True,
            shuffle: bool = True):
        """
        Train the neural network.

        Args:
            X_train: Training data, shape (n_features, n_samples)
            y_train: Training labels (one-hot), shape (n_classes, n_samples)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            optimizer: Optimizer instance (default: Adam)
            lambda_l2: L2 regularization strength
            lambda_l1: L1 regularization strength
            X_val: Validation data
            y_val: Validation labels
            early_stopping_patience: Stop if val loss doesn't improve
            verbose: Print progress
            shuffle: Shuffle data each epoch

        Returns:
            Training history dictionary
        """
        if optimizer is None:
            optimizer = Adam(learning_rate=0.001)

        n_samples = X_train.shape[1]
        n_batches = max(1, n_samples // batch_size)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0.0

            # Shuffle data
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X_train[:, indices]
                y_shuffled = y_train[:, indices]
            else:
                X_shuffled = X_train
                y_shuffled = y_train

            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = X_shuffled[:, start_idx:end_idx]
                y_batch = y_shuffled[:, start_idx:end_idx]

                # Forward pass
                y_pred = self.forward(X_batch, training=True)

                # Backward pass
                batch_loss = self.backward(y_pred, y_batch, lambda_l2, lambda_l1)
                epoch_loss += batch_loss

                # Update parameters
                for layer in self.layers:
                    optimizer.update(layer)

                # Update batch norm parameters
                if self.use_batch_norm:
                    for bn in self.batch_norms:
                        bn.gamma -= optimizer.learning_rate * bn.dgamma
                        bn.beta -= optimizer.learning_rate * bn.dbeta

            # Compute epoch metrics
            avg_loss = epoch_loss / n_batches
            train_acc = self.compute_accuracy(X_train, y_train)

            self.history['train_loss'].append(avg_loss)
            self.history['train_accuracy'].append(train_acc)

            # Validation
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = LossFunction.categorical_cross_entropy(val_pred, y_val)
                val_acc = self.compute_accuracy(X_val, y_val)

                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)

                # Early stopping
                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch + 1}")
                            break

            # Print progress
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.4f}"
                if X_val is not None:
                    msg += f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
                print(msg)

        return self.history

    def save(self, filepath: str):
        """Save model to file."""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'weights': [layer.get_params() for layer in self.layers],
            'history': self.history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load(self, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        for layer, params in zip(self.layers, model_data['weights']):
            layer.set_params(params)
        self.history = model_data['history']

    def summary(self):
        """Print model architecture summary."""
        print("=" * 60)
        print("Model Summary")
        print("=" * 60)

        total_params = 0
        for i, layer in enumerate(self.layers):
            n_params = layer.W.size + (layer.b.size if layer.use_bias else 0)
            total_params += n_params
            print(f"Layer {i + 1}: {layer.n_input} -> {layer.n_output} "
                  f"({layer.activation_name}) - {n_params:,} params")

        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print("=" * 60)


class LearningRateScheduler:
    """
    Learning rate scheduling strategies.
    """

    @staticmethod
    def step_decay(initial_lr: float, epoch: int,
                   drop_rate: float = 0.5, drop_every: int = 10) -> float:
        """
        Step decay: reduce LR by factor every N epochs.

        Args:
            initial_lr: Starting learning rate
            epoch: Current epoch number
            drop_rate: Factor to reduce LR by
            drop_every: Reduce every N epochs

        Returns:
            New learning rate
        """
        return initial_lr * (drop_rate ** (epoch // drop_every))

    @staticmethod
    def exponential_decay(initial_lr: float, epoch: int,
                          decay_rate: float = 0.95) -> float:
        """
        Exponential decay: LR = initial_lr * decay_rate^epoch

        Args:
            initial_lr: Starting learning rate
            epoch: Current epoch number
            decay_rate: Decay factor

        Returns:
            New learning rate
        """
        return initial_lr * (decay_rate ** epoch)

    @staticmethod
    def cosine_annealing(initial_lr: float, epoch: int,
                         total_epochs: int, min_lr: float = 0.0) -> float:
        """
        Cosine annealing: smooth decrease following cosine curve.

        Args:
            initial_lr: Starting learning rate
            epoch: Current epoch number
            total_epochs: Total number of epochs
            min_lr: Minimum learning rate

        Returns:
            New learning rate
        """
        return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs)) / 2

    @staticmethod
    def warmup(initial_lr: float, epoch: int, warmup_epochs: int) -> float:
        """
        Linear warmup: gradually increase LR during initial epochs.

        Args:
            initial_lr: Target learning rate
            epoch: Current epoch number
            warmup_epochs: Number of warmup epochs

        Returns:
            Current learning rate
        """
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        return initial_lr


def create_mini_batches(X: np.ndarray, y: np.ndarray,
                        batch_size: int, shuffle: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create mini-batches for training.

    Args:
        X: Input data, shape (n_features, n_samples)
        y: Labels, shape (n_classes, n_samples)
        batch_size: Size of each mini-batch
        shuffle: Whether to shuffle data

    Returns:
        List of (X_batch, y_batch) tuples
    """
    n_samples = X.shape[1]

    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[:, indices]
        y = y[:, indices]

    batches = []
    n_complete_batches = n_samples // batch_size

    for i in range(n_complete_batches):
        start = i * batch_size
        end = start + batch_size
        batches.append((X[:, start:end], y[:, start:end]))

    # Handle remaining samples
    if n_samples % batch_size != 0:
        start = n_complete_batches * batch_size
        batches.append((X[:, start:], y[:, start:]))

    return batches


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Convert class indices to one-hot encoded vectors.

    Args:
        y: Class indices, shape (n_samples,)
        n_classes: Total number of classes

    Returns:
        One-hot encoded array, shape (n_classes, n_samples)
    """
    n_samples = len(y)
    one_hot = np.zeros((n_classes, n_samples))
    one_hot[y, np.arange(n_samples)] = 1
    return one_hot


def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.2,
                     shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.

    Args:
        X: Input data
        y: Labels
        test_size: Fraction of data for testing
        shuffle: Whether to shuffle before splitting

    Returns:
        X_train, X_test, y_train, y_test
    """
    n_samples = X.shape[1]
    n_test = int(n_samples * test_size)

    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[:, indices]
        y = y[:, indices]

    X_test = X[:, :n_test]
    X_train = X[:, n_test:]
    y_test = y[:, :n_test]
    y_train = y[:, n_test:]

    return X_train, X_test, y_train, y_test


def normalize_data(X: np.ndarray,
                   mean: np.ndarray = None,
                   std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data to zero mean and unit variance.

    Args:
        X: Input data, shape (n_features, n_samples)
        mean: Pre-computed mean (optional)
        std: Pre-computed std (optional)

    Returns:
        Normalized X, mean, std
    """
    if mean is None:
        mean = np.mean(X, axis=1, keepdims=True)
    if std is None:
        std = np.std(X, axis=1, keepdims=True)
        std[std == 0] = 1  # Prevent division by zero

    X_normalized = (X - mean) / std
    return X_normalized, mean, std
```

---

## Gradient Flow Visualization

```mermaid
graph TB
    subgraph ForwardPass["Forward Pass (Computing Activations)"]
        direction LR
        X[Input X] --> |"W1, b1"| Z1["Z1 = W1*X + b1"]
        Z1 --> |"ReLU"| A1["A1 = ReLU(Z1)"]
        A1 --> |"W2, b2"| Z2["Z2 = W2*A1 + b2"]
        Z2 --> |"ReLU"| A2["A2 = ReLU(Z2)"]
        A2 --> |"W3, b3"| Z3["Z3 = W3*A2 + b3"]
        Z3 --> |"Softmax"| Y["Y = Softmax(Z3)"]
        Y --> L["Loss = -sum(y*log(Y))"]
    end

    subgraph BackwardPass["Backward Pass (Computing Gradients)"]
        direction RL
        dL["dL = 1"] --> dY["dL/dY"]
        dY --> dZ3["dL/dZ3 = Y - y"]
        dZ3 --> dW3["dL/dW3 = dZ3 * A2.T"]
        dZ3 --> dA2["dL/dA2 = W3.T * dZ3"]
        dA2 --> dZ2_["dL/dZ2 = dA2 * ReLU'(Z2)"]
        dZ2_ --> dW2["dL/dW2 = dZ2 * A1.T"]
        dZ2_ --> dA1["dL/dA1 = W2.T * dZ2"]
        dA1 --> dZ1_["dL/dZ1 = dA1 * ReLU'(Z1)"]
        dZ1_ --> dW1["dL/dW1 = dZ1 * X.T"]
    end
```

---

## Complete Training Example

```python
"""
Complete example: Training an MLP on synthetic data.
This demonstrates all components working together.
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_spiral_data(n_samples: int = 1000, n_classes: int = 3, noise: float = 0.1):
    """
    Generate spiral dataset for classification.

    Args:
        n_samples: Total number of samples
        n_classes: Number of spiral arms (classes)
        noise: Noise level

    Returns:
        X: Data points, shape (2, n_samples)
        y: One-hot labels, shape (n_classes, n_samples)
    """
    samples_per_class = n_samples // n_classes
    X = np.zeros((2, n_samples))
    y = np.zeros(n_samples, dtype=int)

    for class_idx in range(n_classes):
        start_idx = class_idx * samples_per_class
        end_idx = start_idx + samples_per_class

        r = np.linspace(0.0, 1, samples_per_class)
        t = (np.linspace(0, 4, samples_per_class) +
             class_idx * 4 / n_classes +
             np.random.randn(samples_per_class) * noise)

        X[0, start_idx:end_idx] = r * np.sin(t * 2.5)
        X[1, start_idx:end_idx] = r * np.cos(t * 2.5)
        y[start_idx:end_idx] = class_idx

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[:, indices]
    y = y[indices]

    return X, one_hot_encode(y, n_classes)


def visualize_decision_boundary(model, X, y, resolution=100):
    """
    Visualize the decision boundary of a trained model.

    Args:
        model: Trained MLP model
        X: Training data
        y: One-hot labels
        resolution: Grid resolution
    """
    # Create mesh grid
    x_min, x_max = X[0].min() - 0.5, X[0].max() + 0.5
    y_min, y_max = X[1].min() - 0.5, X[1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Predict for each point in grid
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    predictions = model.predict_classes(grid_points)
    predictions = predictions.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, predictions, alpha=0.3, cmap='viridis')
    plt.scatter(X[0], X[1], c=np.argmax(y, axis=0), cmap='viridis', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.colorbar(label='Class')
    plt.show()


def plot_training_history(history):
    """
    Plot training and validation metrics over epochs.

    Args:
        history: Dictionary with training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    if history['val_loss']:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(history['train_accuracy'], label='Train Accuracy')
    if history['val_accuracy']:
        axes[1].plot(history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """
    Main training pipeline demonstrating MLP usage.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate data
    print("Generating spiral dataset...")
    X, y = generate_spiral_data(n_samples=1500, n_classes=3, noise=0.15)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)

    print(f"Training samples: {X_train.shape[1]}")
    print(f"Validation samples: {X_val.shape[1]}")
    print(f"Test samples: {X_test.shape[1]}")

    # Normalize data
    X_train, mean, std = normalize_data(X_train)
    X_val, _, _ = normalize_data(X_val, mean, std)
    X_test, _, _ = normalize_data(X_test, mean, std)

    # Create model
    print("\nBuilding model...")
    model = MLP(
        layer_sizes=[2, 64, 32, 16, 3],  # 2 inputs, 3 hidden layers, 3 outputs
        activations=['relu', 'relu', 'relu', 'softmax'],
        weight_init='he_normal',
        use_batch_norm=True,
        dropout_rates=[0.2, 0.2, 0.1]
    )

    model.summary()

    # Train model
    print("\nTraining model...")
    optimizer = Adam(learning_rate=0.01)

    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        optimizer=optimizer,
        lambda_l2=0.0001,
        X_val=X_val,
        y_val=y_val,
        early_stopping_patience=20,
        verbose=True
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy = model.compute_accuracy(X_test, y_test)
    test_pred = model.predict(X_test)
    test_loss = LossFunction.categorical_cross_entropy(test_pred, y_test)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Visualize results
    print("\nVisualizing results...")
    plot_training_history(history)
    visualize_decision_boundary(model, X_test, y_test)

    # Save model
    model.save('spiral_classifier.pkl')
    print("Model saved!")


if __name__ == "__main__":
    main()
```

---

## Step-by-Step Backpropagation Derivation

### Single Neuron Example

```python
"""
Detailed backpropagation derivation for a single neuron.
This helps build intuition before moving to full networks.
"""

def single_neuron_backprop():
    """
    Demonstrates backpropagation for a single neuron.

    Neuron computes: y = sigmoid(w*x + b)
    Loss: L = (y - target)^2 / 2
    """
    # Forward pass
    x = 2.0       # Input
    w = 0.5       # Weight
    b = 0.1       # Bias
    target = 1.0  # True value

    # Step 1: Compute z = w*x + b
    z = w * x + b
    print(f"z = w*x + b = {w}*{x} + {b} = {z}")

    # Step 2: Compute activation a = sigmoid(z)
    a = 1 / (1 + np.exp(-z))
    print(f"a = sigmoid(z) = {a:.4f}")

    # Step 3: Compute loss
    loss = 0.5 * (a - target) ** 2
    print(f"Loss = 0.5 * (a - target)^2 = {loss:.4f}")

    # Backward pass (computing gradients)
    print("\n--- Backward Pass ---")

    # dL/da = a - target
    dL_da = a - target
    print(f"dL/da = a - target = {dL_da:.4f}")

    # da/dz = sigmoid'(z) = a * (1 - a)
    da_dz = a * (1 - a)
    print(f"da/dz = a*(1-a) = {da_dz:.4f}")

    # dL/dz = dL/da * da/dz (chain rule)
    dL_dz = dL_da * da_dz
    print(f"dL/dz = dL/da * da/dz = {dL_dz:.4f}")

    # dz/dw = x
    dz_dw = x
    print(f"dz/dw = x = {dz_dw}")

    # dL/dw = dL/dz * dz/dw (chain rule)
    dL_dw = dL_dz * dz_dw
    print(f"dL/dw = dL/dz * dz/dw = {dL_dw:.4f}")

    # dz/db = 1
    dz_db = 1
    print(f"dz/db = 1")

    # dL/db = dL/dz * dz/db
    dL_db = dL_dz * dz_db
    print(f"dL/db = dL/dz * dz/db = {dL_db:.4f}")

    # Update weights
    learning_rate = 0.1
    w_new = w - learning_rate * dL_dw
    b_new = b - learning_rate * dL_db

    print(f"\n--- Update ---")
    print(f"w_new = w - lr * dL/dw = {w} - {learning_rate}*{dL_dw:.4f} = {w_new:.4f}")
    print(f"b_new = b - lr * dL/db = {b} - {learning_rate}*{dL_db:.4f} = {b_new:.4f}")

    # Verify: new loss should be lower
    z_new = w_new * x + b_new
    a_new = 1 / (1 + np.exp(-z_new))
    loss_new = 0.5 * (a_new - target) ** 2
    print(f"\nNew prediction: {a_new:.4f}")
    print(f"New loss: {loss_new:.4f} (was {loss:.4f})")


# Run demonstration
single_neuron_backprop()
```

### Two-Layer Network Derivation

```python
"""
Complete backpropagation for a 2-layer network.
Shows matrix operations and gradient flow.
"""

def two_layer_backprop_example():
    """
    Network architecture:
    - Input: 3 features
    - Hidden: 4 neurons (ReLU)
    - Output: 2 neurons (Softmax)

    Forward:
        Z1 = W1 @ X + b1
        A1 = ReLU(Z1)
        Z2 = W2 @ A1 + b2
        A2 = Softmax(Z2)
        L = CrossEntropy(A2, Y)

    Backward:
        dZ2 = A2 - Y
        dW2 = (1/m) * dZ2 @ A1.T
        db2 = (1/m) * sum(dZ2, axis=1)
        dA1 = W2.T @ dZ2
        dZ1 = dA1 * ReLU'(Z1)
        dW1 = (1/m) * dZ1 @ X.T
        db1 = (1/m) * sum(dZ1, axis=1)
    """
    np.random.seed(42)

    # Dimensions
    n_input = 3    # Number of input features
    n_hidden = 4   # Number of hidden neurons
    n_output = 2   # Number of output classes
    m = 5          # Batch size

    # Initialize weights
    W1 = np.random.randn(n_hidden, n_input) * 0.1
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(n_output, n_hidden) * 0.1
    b2 = np.zeros((n_output, 1))

    # Sample input and labels
    X = np.random.randn(n_input, m)
    Y = np.array([[1, 0, 1, 0, 1],
                  [0, 1, 0, 1, 0]])  # One-hot

    print("=== Forward Pass ===\n")

    # Layer 1
    Z1 = np.dot(W1, X) + b1
    print(f"Z1 = W1 @ X + b1")
    print(f"Z1 shape: {Z1.shape}")

    A1 = np.maximum(0, Z1)  # ReLU
    print(f"A1 = ReLU(Z1)")
    print(f"A1 shape: {A1.shape}")

    # Layer 2
    Z2 = np.dot(W2, A1) + b2
    print(f"\nZ2 = W2 @ A1 + b2")
    print(f"Z2 shape: {Z2.shape}")

    # Softmax
    exp_Z2 = np.exp(Z2 - np.max(Z2, axis=0, keepdims=True))
    A2 = exp_Z2 / np.sum(exp_Z2, axis=0, keepdims=True)
    print(f"A2 = Softmax(Z2)")
    print(f"A2 shape: {A2.shape}")

    # Loss
    epsilon = 1e-15
    L = -np.sum(Y * np.log(A2 + epsilon)) / m
    print(f"\nCross-entropy loss: {L:.4f}")

    print("\n=== Backward Pass ===\n")

    # Output layer gradient
    # For softmax + cross-entropy, gradient simplifies to A2 - Y
    dZ2 = A2 - Y
    print(f"dZ2 = A2 - Y")
    print(f"dZ2 shape: {dZ2.shape}")

    # Gradients for W2 and b2
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    print(f"\ndW2 = (1/m) * dZ2 @ A1.T")
    print(f"dW2 shape: {dW2.shape}")
    print(f"db2 shape: {db2.shape}")

    # Propagate to hidden layer
    dA1 = np.dot(W2.T, dZ2)
    print(f"\ndA1 = W2.T @ dZ2")
    print(f"dA1 shape: {dA1.shape}")

    # Through ReLU
    dZ1 = dA1 * (Z1 > 0).astype(float)
    print(f"dZ1 = dA1 * ReLU'(Z1)")
    print(f"dZ1 shape: {dZ1.shape}")

    # Gradients for W1 and b1
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    print(f"\ndW1 = (1/m) * dZ1 @ X.T")
    print(f"dW1 shape: {dW1.shape}")
    print(f"db1 shape: {db1.shape}")

    print("\n=== Gradient Update ===\n")

    learning_rate = 0.01
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Verify loss decreased
    Z1_new = np.dot(W1, X) + b1
    A1_new = np.maximum(0, Z1_new)
    Z2_new = np.dot(W2, A1_new) + b2
    exp_Z2_new = np.exp(Z2_new - np.max(Z2_new, axis=0, keepdims=True))
    A2_new = exp_Z2_new / np.sum(exp_Z2_new, axis=0, keepdims=True)
    L_new = -np.sum(Y * np.log(A2_new + epsilon)) / m

    print(f"Initial loss: {L:.4f}")
    print(f"New loss: {L_new:.4f}")
    print(f"Loss decreased by: {L - L_new:.4f}")


two_layer_backprop_example()
```

---

## Common Interview Questions and Solutions

### Question 1: Implement Forward Pass Only

```python
def forward_pass_simple(X, weights_list, biases_list, activations_list):
    """
    Implement forward pass for an MLP.

    Interview-style implementation focusing on clarity.

    Args:
        X: Input data, shape (n_features, n_samples)
        weights_list: List of weight matrices
        biases_list: List of bias vectors
        activations_list: List of activation function names

    Returns:
        Final output and list of all activations
    """
    activation_funcs = {
        'relu': lambda z: np.maximum(0, z),
        'sigmoid': lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500))),
        'tanh': lambda z: np.tanh(z),
        'softmax': lambda z: np.exp(z - np.max(z, axis=0, keepdims=True)) /
                            np.sum(np.exp(z - np.max(z, axis=0, keepdims=True)), axis=0, keepdims=True),
        'linear': lambda z: z
    }

    A = X
    activations = [A]  # Store for potential backprop

    for W, b, act_name in zip(weights_list, biases_list, activations_list):
        # Linear transformation
        Z = np.dot(W, A) + b

        # Activation
        A = activation_funcs[act_name](Z)
        activations.append(A)

    return A, activations
```

### Question 2: Implement Gradient Descent Update

```python
def gradient_descent_step(params, grads, learning_rate):
    """
    Perform one step of gradient descent.

    Args:
        params: Dictionary of parameters {'W1': ..., 'b1': ..., ...}
        grads: Dictionary of gradients {'dW1': ..., 'db1': ..., ...}
        learning_rate: Step size

    Returns:
        Updated parameters
    """
    updated_params = {}

    for key in params:
        grad_key = 'd' + key  # e.g., 'W1' -> 'dW1'
        updated_params[key] = params[key] - learning_rate * grads[grad_key]

    return updated_params


def gradient_descent_with_momentum(params, grads, velocities,
                                   learning_rate, momentum=0.9):
    """
    Gradient descent with momentum.

    v = momentum * v - learning_rate * gradient
    param = param + v

    Args:
        params: Current parameters
        grads: Gradients
        velocities: Previous velocities
        learning_rate: Step size
        momentum: Momentum coefficient

    Returns:
        Updated parameters and velocities
    """
    updated_params = {}
    updated_velocities = {}

    for key in params:
        grad_key = 'd' + key
        vel_key = 'v' + key

        # Update velocity
        updated_velocities[vel_key] = (momentum * velocities.get(vel_key, 0) -
                                        learning_rate * grads[grad_key])

        # Update parameter
        updated_params[key] = params[key] + updated_velocities[vel_key]

    return updated_params, updated_velocities
```

### Question 3: Implement Batch Normalization Forward Pass

```python
def batch_norm_forward(x, gamma, beta, epsilon=1e-8, training=True,
                       running_mean=None, running_var=None, momentum=0.9):
    """
    Batch normalization forward pass.

    Normalizes activations, then applies learnable scale and shift.

    Args:
        x: Input, shape (n_features, batch_size)
        gamma: Scale parameter
        beta: Shift parameter
        epsilon: Numerical stability
        training: Whether in training mode
        running_mean: Running mean for inference
        running_var: Running variance for inference
        momentum: Momentum for running statistics

    Returns:
        Normalized output, cache for backprop, updated statistics
    """
    if training:
        # Compute batch statistics
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + epsilon)

        # Scale and shift
        out = gamma * x_norm + beta

        # Update running statistics
        if running_mean is not None:
            running_mean = momentum * running_mean + (1 - momentum) * mean
        else:
            running_mean = mean

        if running_var is not None:
            running_var = momentum * running_var + (1 - momentum) * var
        else:
            running_var = var

        # Cache for backprop
        cache = (x, x_norm, mean, var, gamma, beta, epsilon)

    else:
        # Use running statistics for inference
        x_norm = (x - running_mean) / np.sqrt(running_var + epsilon)
        out = gamma * x_norm + beta
        cache = None

    return out, cache, running_mean, running_var
```

### Question 4: Implement Xavier Initialization from Scratch

```python
def xavier_init(n_in, n_out, uniform=True):
    """
    Xavier/Glorot initialization.

    Maintains variance across layers for tanh/sigmoid activations.

    Theory: For activations with unit derivative around zero,
    we want Var(output) = Var(input). This requires:
    Var(W) = 2 / (n_in + n_out)

    Args:
        n_in: Number of input features
        n_out: Number of output features
        uniform: Use uniform distribution (vs normal)

    Returns:
        Initialized weight matrix
    """
    if uniform:
        # Uniform distribution: U(-limit, limit)
        # For uniform, Var = (limit^2) / 3
        # So limit = sqrt(6 / (n_in + n_out))
        limit = np.sqrt(6.0 / (n_in + n_out))
        return np.random.uniform(-limit, limit, (n_out, n_in))
    else:
        # Normal distribution: N(0, std^2)
        # std = sqrt(2 / (n_in + n_out))
        std = np.sqrt(2.0 / (n_in + n_out))
        return np.random.randn(n_out, n_in) * std


def he_init(n_in, n_out, uniform=True):
    """
    He/Kaiming initialization.

    Designed for ReLU activations which zero out half the values.

    Theory: ReLU sets negative values to zero, so we need
    Var(W) = 2 / n_in to maintain variance.

    Args:
        n_in: Number of input features
        n_out: Number of output features
        uniform: Use uniform distribution

    Returns:
        Initialized weight matrix
    """
    if uniform:
        limit = np.sqrt(6.0 / n_in)
        return np.random.uniform(-limit, limit, (n_out, n_in))
    else:
        std = np.sqrt(2.0 / n_in)
        return np.random.randn(n_out, n_in) * std
```

---

## Numerical Gradient Checking

```python
def numerical_gradient(f, x, epsilon=1e-7):
    """
    Compute numerical gradient using finite differences.

    Used to verify analytical gradient computations.

    Formula: df/dx ≈ [f(x + ε) - f(x - ε)] / (2ε)

    Args:
        f: Function that takes x and returns a scalar
        x: Point at which to compute gradient
        epsilon: Small perturbation

    Returns:
        Numerical gradient array (same shape as x)
    """
    grad = np.zeros_like(x)

    # Iterate over all elements
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        original_value = x[idx]

        # Compute f(x + epsilon)
        x[idx] = original_value + epsilon
        f_plus = f(x)

        # Compute f(x - epsilon)
        x[idx] = original_value - epsilon
        f_minus = f(x)

        # Compute gradient
        grad[idx] = (f_plus - f_minus) / (2 * epsilon)

        # Restore original value
        x[idx] = original_value

        it.iternext()

    return grad


def gradient_check(model, X, y, epsilon=1e-7):
    """
    Check gradients by comparing analytical and numerical gradients.

    Args:
        model: MLP model
        X: Input batch
        y: True labels
        epsilon: Perturbation size

    Returns:
        Maximum relative difference between gradients
    """
    # Compute analytical gradients
    y_pred = model.forward(X)
    model.backward(y_pred, y)

    max_diff = 0

    for layer_idx, layer in enumerate(model.layers):
        print(f"\nChecking Layer {layer_idx + 1}")

        # Check weight gradients
        analytical_grad = layer.dW.flatten()
        numerical_grad = np.zeros_like(analytical_grad)

        for i in range(len(analytical_grad)):
            # Get original weight
            orig_shape = layer.W.shape
            W_flat = layer.W.flatten()
            original = W_flat[i]

            # f(W + epsilon)
            W_flat[i] = original + epsilon
            layer.W = W_flat.reshape(orig_shape)
            y_pred_plus = model.forward(X)
            loss_plus = LossFunction.categorical_cross_entropy(y_pred_plus, y)

            # f(W - epsilon)
            W_flat[i] = original - epsilon
            layer.W = W_flat.reshape(orig_shape)
            y_pred_minus = model.forward(X)
            loss_minus = LossFunction.categorical_cross_entropy(y_pred_minus, y)

            # Numerical gradient
            numerical_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

            # Restore
            W_flat[i] = original
            layer.W = W_flat.reshape(orig_shape)

        # Compute relative difference
        diff = np.abs(analytical_grad - numerical_grad)
        denominator = np.maximum(np.abs(analytical_grad) + np.abs(numerical_grad), 1e-8)
        relative_diff = diff / denominator

        max_relative_diff = np.max(relative_diff)
        max_diff = max(max_diff, max_relative_diff)

        if max_relative_diff < 1e-5:
            print(f"  Weight gradients: PASS (max diff: {max_relative_diff:.2e})")
        else:
            print(f"  Weight gradients: FAIL (max diff: {max_relative_diff:.2e})")

    return max_diff
```

---

## Performance Optimizations

### Vectorized Operations

```python
def vectorized_forward_backward(X, y, W1, b1, W2, b2, learning_rate):
    """
    Fully vectorized forward and backward pass.

    Handles entire batch in parallel using NumPy broadcasting.
    This is 10-100x faster than loop-based implementations.
    """
    m = X.shape[1]  # Batch size

    # === Forward Pass ===
    # Layer 1
    Z1 = W1 @ X + b1            # (n_hidden, m)
    A1 = np.maximum(0, Z1)       # ReLU

    # Layer 2
    Z2 = W2 @ A1 + b2            # (n_output, m)
    exp_Z2 = np.exp(Z2 - np.max(Z2, axis=0, keepdims=True))
    A2 = exp_Z2 / np.sum(exp_Z2, axis=0, keepdims=True)  # Softmax

    # Loss
    loss = -np.sum(y * np.log(A2 + 1e-15)) / m

    # === Backward Pass ===
    # Output layer
    dZ2 = A2 - y                 # (n_output, m)
    dW2 = dZ2 @ A1.T / m         # (n_output, n_hidden)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    # Hidden layer
    dA1 = W2.T @ dZ2             # (n_hidden, m)
    dZ1 = dA1 * (Z1 > 0)         # ReLU derivative
    dW1 = dZ1 @ X.T / m          # (n_hidden, n_input)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    # === Update ===
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2, loss
```

### Memory-Efficient Implementation

```python
class MemoryEfficientMLP:
    """
    Memory-efficient MLP that doesn't store intermediate activations.

    Uses gradient checkpointing - recomputes forward pass during backward.
    Trades compute for memory.
    """

    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            b = np.zeros((n_out, 1))
            self.weights.append(W)
            self.biases.append(b)

    def forward_single_layer(self, a, layer_idx, is_output=False):
        """Forward through single layer without storing."""
        z = self.weights[layer_idx] @ a + self.biases[layer_idx]
        if is_output:
            # Softmax for output
            exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        else:
            # ReLU for hidden
            return np.maximum(0, z)

    def forward(self, X):
        """Full forward pass."""
        a = X
        for i in range(len(self.weights)):
            a = self.forward_single_layer(a, i, i == len(self.weights) - 1)
        return a

    def backward_with_recompute(self, X, y, learning_rate):
        """
        Backward pass with gradient checkpointing.

        Recomputes activations as needed instead of storing all.
        """
        m = X.shape[1]
        n_layers = len(self.weights)

        # Recompute all activations (we need them for gradients)
        activations = [X]
        pre_activations = []
        a = X

        for i in range(n_layers):
            z = self.weights[i] @ a + self.biases[i]
            pre_activations.append(z)

            if i == n_layers - 1:
                exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
                a = exp_z / np.sum(exp_z, axis=0, keepdims=True)
            else:
                a = np.maximum(0, z)
            activations.append(a)

        # Backward
        da = activations[-1] - y  # Softmax + CE gradient

        for i in reversed(range(n_layers)):
            # Compute gradients
            dW = da @ activations[i].T / m
            db = np.sum(da, axis=1, keepdims=True) / m

            if i > 0:
                da = self.weights[i].T @ da
                da = da * (pre_activations[i-1] > 0)  # ReLU derivative

            # Update
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

        # Clear intermediate results
        del activations, pre_activations
```

---

## Regularization Techniques Visualization

```mermaid
graph TB
    subgraph Regularization["Regularization Techniques"]
        L2["L2 Regularization<br/>Loss += λ/2 * ||W||²"]
        L1["L1 Regularization<br/>Loss += λ * ||W||"]
        DO["Dropout<br/>Randomly zero activations"]
        BN["Batch Normalization<br/>Normalize layer inputs"]
        ES["Early Stopping<br/>Stop when val loss increases"]
    end

    subgraph Effects["Effects on Training"]
        L2 --> S1["Smaller weights"]
        L1 --> S2["Sparse weights"]
        DO --> S3["Prevents co-adaptation"]
        BN --> S4["Stable training"]
        ES --> S5["Prevents overfitting"]
    end
```

---

## Summary and Key Takeaways

### Core Concepts

1. **Forward Propagation**: Sequential matrix multiplications with activations
2. **Backpropagation**: Chain rule applied backwards through the network
3. **Weight Initialization**: Critical for gradient flow (Xavier for tanh/sigmoid, He for ReLU)
4. **Activation Functions**: ReLU most common, softmax for classification output
5. **Loss Functions**: Cross-entropy for classification, MSE for regression

### Implementation Checklist

```python
"""
MLP Implementation Checklist for Interviews:

1. Forward Pass:
   - Linear: Z = W @ A_prev + b
   - Activation: A = g(Z)
   - Store intermediate values for backprop

2. Backward Pass:
   - Start with loss gradient
   - Apply chain rule through each layer
   - Compute dW, db for each layer
   - Propagate dA to previous layer

3. Weight Updates:
   - Simple: W = W - lr * dW
   - Momentum: v = β*v - lr*dW, W = W + v
   - Adam: Adaptive learning rates per parameter

4. Regularization:
   - L2: Add λ*W to gradient
   - Dropout: Random zeroing during training
   - Batch Norm: Normalize layer inputs

5. Training Loop:
   - Shuffle data each epoch
   - Process in mini-batches
   - Monitor training/validation loss
   - Early stopping if needed
"""
```

### Time and Space Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Forward Pass | O(sum of n_i * n_{i+1}) | O(sum of activations) |
| Backward Pass | O(sum of n_i * n_{i+1}) | O(sum of gradients) |
| SGD Update | O(total parameters) | O(1) |
| Adam Update | O(total parameters) | O(2 * total params) |
| Batch Norm | O(batch_size * features) | O(features) |

### Common Interview Follow-ups

1. **Why use mini-batches instead of full batch?**
   - Faster convergence (more frequent updates)
   - Regularization effect (noise in gradients)
   - Memory efficiency for large datasets

2. **Why initialize weights randomly?**
   - Break symmetry (all neurons learn different features)
   - Zero initialization causes all neurons to compute same gradients

3. **What causes vanishing/exploding gradients?**
   - Vanishing: Sigmoid/tanh in deep networks
   - Exploding: Large initial weights, no normalization
   - Solutions: ReLU, batch norm, residual connections, proper initialization

4. **When to use which optimizer?**
   - SGD + Momentum: Good generalization, needs tuning
   - Adam: Fast convergence, good default
   - RMSprop: Good for RNNs, non-stationary problems

5. **How does dropout regularize?**
   - Prevents co-adaptation of neurons
   - Approximates ensemble of networks
   - Forces redundant representations
