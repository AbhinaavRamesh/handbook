---
title: Batch Normalization and Dropout Implementation
description: Implement BatchNorm and Dropout layers from scratch with training vs inference modes.
---

# Batch Normalization and Dropout Implementation

> **Essential regularization techniques for deep learning**

## Overview

Batch Normalization and Dropout are two fundamental techniques that have revolutionized deep learning training. BatchNorm addresses internal covariate shift by normalizing activations, while Dropout provides regularization by randomly dropping units during training.

## 1. Batch Normalization from Scratch

### Core Implementation

```python
import numpy as np
from typing import Tuple, Optional, Dict

class BatchNormalization:
    """
    Batch Normalization layer implementation.

    Normalizes activations using batch statistics during training
    and running statistics during inference.

    Formula:
        x_hat = (x - mean) / sqrt(var + eps)
        y = gamma * x_hat + beta

    Parameters:
        num_features: Number of features/channels
        eps: Small constant for numerical stability
        momentum: Momentum for running mean/var updates
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Learnable parameters (scale and shift)
        if self.affine:
            self.gamma = np.ones(num_features)  # Scale parameter
            self.beta = np.zeros(num_features)  # Shift parameter

            # Gradients for learnable parameters
            self.dgamma = np.zeros(num_features)
            self.dbeta = np.zeros(num_features)

        # Running statistics for inference
        if self.track_running_stats:
            self.running_mean = np.zeros(num_features)
            self.running_var = np.ones(num_features)
            self.num_batches_tracked = 0

        # Cache for backward pass
        self.cache = {}

        # Training mode flag
        self.training = True

    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for batch normalization.

        Args:
            x: Input tensor of shape (N, D) or (N, C, H, W)

        Returns:
            Normalized output of same shape as input
        """
        # Handle different input dimensions
        original_shape = x.shape

        if x.ndim == 4:
            # Conv layer: (N, C, H, W) -> normalize over N, H, W
            return self._forward_conv(x)
        else:
            # FC layer: (N, D) -> normalize over N
            return self._forward_fc(x)

    def _forward_fc(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for fully connected layers."""
        N, D = x.shape

        if self.training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)

            # Normalize
            x_centered = x - batch_mean
            std = np.sqrt(batch_var + self.eps)
            x_norm = x_centered / std

            # Scale and shift
            if self.affine:
                out = self.gamma * x_norm + self.beta
            else:
                out = x_norm

            # Update running statistics
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                   self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + \
                                  self.momentum * batch_var
                self.num_batches_tracked += 1

            # Cache values for backward pass
            self.cache = {
                'x': x,
                'x_norm': x_norm,
                'x_centered': x_centered,
                'std': std,
                'batch_mean': batch_mean,
                'batch_var': batch_var
            }

        else:
            # Use running statistics for inference
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

            if self.affine:
                out = self.gamma * x_norm + self.beta
            else:
                out = x_norm

        return out

    def _forward_conv(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layers (BatchNorm2d)."""
        N, C, H, W = x.shape

        if self.training:
            # Compute statistics over N, H, W (per channel)
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)

            # Normalize
            x_centered = x - batch_mean
            std = np.sqrt(batch_var + self.eps)
            x_norm = x_centered / std

            # Scale and shift (gamma, beta have shape (C,))
            if self.affine:
                gamma_reshaped = self.gamma.reshape(1, C, 1, 1)
                beta_reshaped = self.beta.reshape(1, C, 1, 1)
                out = gamma_reshaped * x_norm + beta_reshaped
            else:
                out = x_norm

            # Update running statistics
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                   self.momentum * batch_mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + \
                                  self.momentum * batch_var.squeeze()
                self.num_batches_tracked += 1

            # Cache for backward
            self.cache = {
                'x': x,
                'x_norm': x_norm,
                'x_centered': x_centered,
                'std': std,
                'batch_mean': batch_mean,
                'batch_var': batch_var
            }

        else:
            # Inference mode
            mean_reshaped = self.running_mean.reshape(1, C, 1, 1)
            var_reshaped = self.running_var.reshape(1, C, 1, 1)

            x_norm = (x - mean_reshaped) / np.sqrt(var_reshaped + self.eps)

            if self.affine:
                gamma_reshaped = self.gamma.reshape(1, C, 1, 1)
                beta_reshaped = self.beta.reshape(1, C, 1, 1)
                out = gamma_reshaped * x_norm + beta_reshaped
            else:
                out = x_norm

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass for batch normalization.

        Args:
            dout: Gradient of loss w.r.t. output

        Returns:
            Gradient of loss w.r.t. input
        """
        if dout.ndim == 4:
            return self._backward_conv(dout)
        else:
            return self._backward_fc(dout)

    def _backward_fc(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass for FC layers."""
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        x_centered = self.cache['x_centered']
        std = self.cache['std']

        N, D = x.shape

        # Gradients for gamma and beta
        if self.affine:
            self.dbeta = np.sum(dout, axis=0)
            self.dgamma = np.sum(dout * x_norm, axis=0)

            # Gradient through scale
            dx_norm = dout * self.gamma
        else:
            dx_norm = dout

        # Gradient through normalization
        # This is the tricky part - need to account for the fact that
        # mean and variance depend on all inputs

        # Method 1: Expanded form (more intuitive)
        dvar = np.sum(dx_norm * x_centered * (-0.5) * (std ** -3), axis=0)
        dmean = np.sum(dx_norm * (-1 / std), axis=0) + \
                dvar * np.mean(-2 * x_centered, axis=0)

        dx = dx_norm / std + dvar * 2 * x_centered / N + dmean / N

        return dx

    def _backward_fc_optimized(self, dout: np.ndarray) -> np.ndarray:
        """
        Optimized backward pass using single formula.

        Derived from chain rule, combining all terms.
        """
        x_norm = self.cache['x_norm']
        std = self.cache['std']

        N = dout.shape[0]

        if self.affine:
            self.dbeta = np.sum(dout, axis=0)
            self.dgamma = np.sum(dout * x_norm, axis=0)
            dx_norm = dout * self.gamma
        else:
            dx_norm = dout

        # Optimized formula
        dx = (1.0 / N) * (1.0 / std) * (
            N * dx_norm -
            np.sum(dx_norm, axis=0) -
            x_norm * np.sum(dx_norm * x_norm, axis=0)
        )

        return dx

    def _backward_conv(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass for conv layers."""
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        x_centered = self.cache['x_centered']
        std = self.cache['std']

        N, C, H, W = x.shape
        m = N * H * W  # Number of elements per channel

        if self.affine:
            # Gradients for gamma and beta
            self.dbeta = np.sum(dout, axis=(0, 2, 3))
            self.dgamma = np.sum(dout * x_norm, axis=(0, 2, 3))

            gamma_reshaped = self.gamma.reshape(1, C, 1, 1)
            dx_norm = dout * gamma_reshaped
        else:
            dx_norm = dout

        # Gradient through normalization (conv version)
        dvar = np.sum(dx_norm * x_centered * (-0.5) * (std ** -3),
                      axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dx_norm * (-1 / std), axis=(0, 2, 3), keepdims=True) + \
                dvar * np.mean(-2 * x_centered, axis=(0, 2, 3), keepdims=True)

        dx = dx_norm / std + dvar * 2 * x_centered / m + dmean / m

        return dx

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get learnable parameters."""
        params = {}
        if self.affine:
            params['gamma'] = self.gamma
            params['beta'] = self.beta
        return params

    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get parameter gradients."""
        grads = {}
        if self.affine:
            grads['dgamma'] = self.dgamma
            grads['dbeta'] = self.dbeta
        return grads


class BatchNorm1d(BatchNormalization):
    """Batch normalization for 2D inputs (N, Features)."""
    pass


class BatchNorm2d(BatchNormalization):
    """Batch normalization for 4D inputs (N, C, H, W)."""
    pass
```

### Mathematical Derivation of Backward Pass

```python
def batchnorm_backward_derivation():
    """
    Detailed derivation of BatchNorm backward pass.

    Forward pass:
        1. mu = (1/N) * sum(x_i)
        2. var = (1/N) * sum((x_i - mu)^2)
        3. x_hat = (x - mu) / sqrt(var + eps)
        4. y = gamma * x_hat + beta

    Backward pass (given dL/dy):

    Step 1: Gradients for gamma and beta
        dL/dbeta = sum(dL/dy)
        dL/dgamma = sum(dL/dy * x_hat)

    Step 2: Gradient through scale (gamma)
        dL/dx_hat = dL/dy * gamma

    Step 3: Gradient through normalization
        This requires careful application of chain rule since
        mu and var both depend on all x_i.

        Let s = sqrt(var + eps)

        dL/dvar = sum(dL/dx_hat * (x - mu) * (-1/2) * (var + eps)^(-3/2))

        dL/dmu = sum(dL/dx_hat * (-1/s)) + dL/dvar * (-2/N) * sum(x - mu)

        dL/dx = dL/dx_hat * (1/s) + dL/dvar * (2/N) * (x - mu) + dL/dmu * (1/N)

    Optimized single formula:
        dL/dx = (1/N) * (1/s) * (
            N * dL/dx_hat -
            sum(dL/dx_hat) -
            x_hat * sum(dL/dx_hat * x_hat)
        )
    """
    pass
```

## 2. Dropout Implementation

### Standard and Inverted Dropout

```python
class Dropout:
    """
    Dropout layer implementation.

    During training, randomly zeroes elements with probability p.
    During inference, returns input unchanged (with inverted dropout)
    or scales by (1-p) (with standard dropout).

    Inverted Dropout:
        - Scale activations by 1/(1-p) during training
        - No scaling needed during inference
        - This is the standard modern approach

    Standard Dropout:
        - No scaling during training
        - Scale by (1-p) during inference
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        Args:
            p: Probability of dropping a unit (0 to 1)
            inplace: If True, modify input in place
        """
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")

        self.p = p
        self.inplace = inplace
        self.training = True
        self.mask = None

    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with inverted dropout.

        Args:
            x: Input tensor of any shape

        Returns:
            Output tensor with dropout applied (training) or unchanged (inference)
        """
        if not self.training or self.p == 0:
            return x

        # Create binary mask
        self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float64)

        # Apply mask and scale (inverted dropout)
        scale = 1.0 / (1.0 - self.p)

        if self.inplace:
            x *= self.mask * scale
            return x
        else:
            return x * self.mask * scale

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass for dropout.

        Gradient flows only through non-dropped units.
        """
        if not self.training or self.p == 0:
            return dout

        scale = 1.0 / (1.0 - self.p)
        return dout * self.mask * scale

    def forward_standard(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with standard (non-inverted) dropout.

        Scaling happens at inference time instead.
        """
        if not self.training:
            # Scale at inference time
            return x * (1.0 - self.p)

        if self.p == 0:
            return x

        # Create mask without scaling during training
        self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float64)
        return x * self.mask


class Dropout2d:
    """
    Spatial Dropout for convolutional layers.

    Drops entire channels instead of individual elements.
    This is more effective for conv layers where adjacent
    pixels are highly correlated.
    """

    def __init__(self, p: float = 0.5):
        self.p = p
        self.training = True
        self.mask = None

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass - drops entire channels.

        Args:
            x: Input of shape (N, C, H, W)

        Returns:
            Output with entire channels dropped
        """
        if not self.training or self.p == 0:
            return x

        N, C, H, W = x.shape

        # Create channel-wise mask (N, C, 1, 1)
        self.mask = (np.random.rand(N, C, 1, 1) > self.p).astype(np.float64)

        # Broadcast mask across spatial dimensions
        scale = 1.0 / (1.0 - self.p)

        return x * self.mask * scale

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass for spatial dropout."""
        if not self.training or self.p == 0:
            return dout

        scale = 1.0 / (1.0 - self.p)
        return dout * self.mask * scale


class AlphaDropout:
    """
    Alpha Dropout for Self-Normalizing Neural Networks (SNNs).

    Maintains mean and variance of inputs, designed to work
    with SELU activation function.
    """

    def __init__(self, p: float = 0.5):
        self.p = p
        self.training = True

        # SELU parameters
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

        # Derived constants for alpha dropout
        self.alpha_p = -self.alpha * self.scale

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass maintaining self-normalizing property."""
        if not self.training or self.p == 0:
            return x

        mask = np.random.rand(*x.shape) > self.p

        # Calculate affine transformation parameters
        a = 1.0 / np.sqrt(self.p + (1 - self.p) * self.alpha_p ** 2)
        b = -a * (1 - self.p) * self.alpha_p

        # Apply dropout
        out = np.where(mask, x, self.alpha_p)

        # Apply affine transformation to maintain mean/variance
        return a * out + b
```

## 3. Layer Normalization

### Comparison with Batch Normalization

```python
class LayerNormalization:
    """
    Layer Normalization implementation.

    Unlike BatchNorm, normalizes across features for each sample.
    Does not depend on batch size, making it suitable for:
    - RNNs/LSTMs/Transformers
    - Small batch sizes
    - Online learning

    For input (N, D):
        BatchNorm: normalize across N for each feature
        LayerNorm: normalize across D for each sample
    """

    def __init__(
        self,
        normalized_shape: Tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        """
        Args:
            normalized_shape: Input shape from expected input, starting
                from the last dimension. For (N, C, H, W), use (C, H, W)
            eps: Small constant for numerical stability
            elementwise_affine: Whether to learn gamma/beta
        """
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = np.ones(normalized_shape)
            self.beta = np.zeros(normalized_shape)
            self.dgamma = np.zeros(normalized_shape)
            self.dbeta = np.zeros(normalized_shape)

        self.cache = {}
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for layer normalization.

        Normalizes over the last len(normalized_shape) dimensions.
        """
        # Determine axes to normalize over
        ndim = len(self.normalized_shape)
        axes = tuple(range(-ndim, 0))

        # Compute statistics per sample
        mean = np.mean(x, axis=axes, keepdims=True)
        var = np.var(x, axis=axes, keepdims=True)

        # Normalize
        x_centered = x - mean
        std = np.sqrt(var + self.eps)
        x_norm = x_centered / std

        # Scale and shift
        if self.elementwise_affine:
            out = self.gamma * x_norm + self.beta
        else:
            out = x_norm

        # Cache for backward
        self.cache = {
            'x_norm': x_norm,
            'x_centered': x_centered,
            'std': std,
            'axes': axes
        }

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass for layer normalization."""
        x_norm = self.cache['x_norm']
        x_centered = self.cache['x_centered']
        std = self.cache['std']
        axes = self.cache['axes']

        # Number of elements being normalized
        n = np.prod([x_norm.shape[i] for i in range(-len(self.normalized_shape), 0)])

        if self.elementwise_affine:
            # Gradients for parameters
            # Sum over batch dimension(s)
            batch_axes = tuple(range(len(dout.shape) - len(self.normalized_shape)))
            self.dbeta = np.sum(dout, axis=batch_axes)
            self.dgamma = np.sum(dout * x_norm, axis=batch_axes)

            dx_norm = dout * self.gamma
        else:
            dx_norm = dout

        # Gradient through normalization
        dvar = np.sum(dx_norm * x_centered * (-0.5) * (std ** -3),
                      axis=axes, keepdims=True)
        dmean = np.sum(dx_norm * (-1 / std), axis=axes, keepdims=True) + \
                dvar * np.mean(-2 * x_centered, axis=axes, keepdims=True)

        dx = dx_norm / std + dvar * 2 * x_centered / n + dmean / n

        return dx


class RMSNorm:
    """
    Root Mean Square Layer Normalization.

    Simplified version of LayerNorm that only normalizes by RMS,
    without centering (no mean subtraction).

    Used in modern architectures like LLaMA.

    Formula: y = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.dgamma = np.zeros(normalized_shape)
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for RMS normalization."""
        # Compute RMS
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)

        x_norm = x / rms
        out = self.gamma * x_norm

        self.cache = {'x': x, 'x_norm': x_norm, 'rms': rms}

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass for RMS normalization."""
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        rms = self.cache['rms']

        n = x.shape[-1]

        # Gradient for gamma
        batch_axes = tuple(range(len(dout.shape) - 1))
        self.dgamma = np.sum(dout * x_norm, axis=batch_axes)

        dx_norm = dout * self.gamma

        # Gradient through RMS normalization
        # d/dx (x/rms) = 1/rms - x * d(rms)/dx / rms^2
        # d(rms)/dx = x / (n * rms)

        dx = dx_norm / rms - x_norm * np.mean(dx_norm * x_norm, axis=-1, keepdims=True)

        return dx
```

## 4. Complete Neural Network with BatchNorm and Dropout

```python
class NeuralNetworkWithRegularization:
    """
    Multi-layer neural network with BatchNorm and Dropout.

    Architecture: FC -> BN -> ReLU -> Dropout -> ... -> Output
    """

    def __init__(
        self,
        layer_dims: list,
        dropout_rate: float = 0.5,
        use_batchnorm: bool = True,
        activation: str = 'relu'
    ):
        """
        Args:
            layer_dims: List of layer dimensions [input, hidden1, ..., output]
            dropout_rate: Dropout probability for hidden layers
            use_batchnorm: Whether to use batch normalization
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.activation = activation

        # Initialize weights and biases
        self.params = {}
        self.bn_layers = {}
        self.dropout_layers = {}

        for l in range(1, self.num_layers + 1):
            # Xavier/He initialization
            if activation == 'relu':
                scale = np.sqrt(2.0 / layer_dims[l-1])  # He initialization
            else:
                scale = np.sqrt(1.0 / layer_dims[l-1])  # Xavier initialization

            self.params[f'W{l}'] = np.random.randn(
                layer_dims[l-1], layer_dims[l]
            ) * scale
            self.params[f'b{l}'] = np.zeros(layer_dims[l])

            # BatchNorm for all except output layer
            if use_batchnorm and l < self.num_layers:
                self.bn_layers[l] = BatchNormalization(layer_dims[l])

            # Dropout for all except output layer
            if dropout_rate > 0 and l < self.num_layers:
                self.dropout_layers[l] = Dropout(dropout_rate)

        self.cache = {}
        self.training = True

    def train(self, mode: bool = True):
        """Set training mode for all layers."""
        self.training = mode
        for bn in self.bn_layers.values():
            bn.train(mode)
        for dropout in self.dropout_layers.values():
            dropout.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def _activation_forward(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _activation_backward(self, da: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Backward pass through activation."""
        if self.activation == 'relu':
            return da * (z > 0)
        elif self.activation == 'tanh':
            return da * (1 - np.tanh(z) ** 2)
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return da * s * (1 - s)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            X: Input of shape (N, D_in)

        Returns:
            Output of shape (N, D_out)
        """
        self.cache['A0'] = X
        A = X

        for l in range(1, self.num_layers + 1):
            # Linear transformation
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']
            Z = A @ W + b
            self.cache[f'Z{l}'] = Z

            if l < self.num_layers:
                # Hidden layer: BN -> Activation -> Dropout

                # Batch normalization
                if self.use_batchnorm:
                    Z = self.bn_layers[l].forward(Z)
                    self.cache[f'Z_bn{l}'] = Z

                # Activation
                A = self._activation_forward(Z)
                self.cache[f'A{l}'] = A

                # Dropout
                if self.dropout_rate > 0:
                    A = self.dropout_layers[l].forward(A)
                    self.cache[f'A_drop{l}'] = A
            else:
                # Output layer (no BN, activation, or dropout)
                A = Z

        return A

    def backward(self, dout: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Backward pass through the network.

        Args:
            dout: Gradient of loss w.r.t. output (N, D_out)

        Returns:
            Dictionary of gradients for all parameters
        """
        grads = {}
        N = dout.shape[0]

        dA = dout

        for l in range(self.num_layers, 0, -1):
            if l < self.num_layers:
                # Dropout backward
                if self.dropout_rate > 0:
                    dA = self.dropout_layers[l].backward(dA)

                # Activation backward
                Z = self.cache[f'Z_bn{l}'] if self.use_batchnorm else self.cache[f'Z{l}']
                dZ = self._activation_backward(dA, Z)

                # BatchNorm backward
                if self.use_batchnorm:
                    dZ = self.bn_layers[l].backward(dZ)
            else:
                dZ = dA

            # Linear backward
            A_prev = self.cache[f'A{l-1}']
            W = self.params[f'W{l}']

            grads[f'dW{l}'] = A_prev.T @ dZ / N
            grads[f'db{l}'] = np.mean(dZ, axis=0)

            if l > 1:
                dA = dZ @ W.T

        # Add BatchNorm gradients
        if self.use_batchnorm:
            for l, bn in self.bn_layers.items():
                bn_grads = bn.get_gradients()
                grads[f'dgamma{l}'] = bn_grads['dgamma']
                grads[f'dbeta{l}'] = bn_grads['dbeta']

        return grads

    def update_params(self, grads: Dict[str, np.ndarray], lr: float = 0.01):
        """Update parameters using gradients."""
        for l in range(1, self.num_layers + 1):
            self.params[f'W{l}'] -= lr * grads[f'dW{l}']
            self.params[f'b{l}'] -= lr * grads[f'db{l}']

            if self.use_batchnorm and l < self.num_layers:
                self.bn_layers[l].gamma -= lr * grads[f'dgamma{l}']
                self.bn_layers[l].beta -= lr * grads[f'dbeta{l}']
```

## 5. Testing and Verification

```python
def test_batch_normalization():
    """Test BatchNorm implementation against expected behavior."""
    print("Testing Batch Normalization...")

    np.random.seed(42)

    # Test 1: Forward pass (training mode)
    bn = BatchNormalization(num_features=4)
    x = np.random.randn(32, 4)

    out = bn.forward(x)

    # Check that output has approximately zero mean and unit variance
    mean = np.mean(out, axis=0)
    var = np.var(out, axis=0)

    print(f"Output mean (should be ~0): {mean}")
    print(f"Output variance (should be ~1): {var}")

    assert np.allclose(mean, 0, atol=1e-5), "Mean should be close to 0"
    assert np.allclose(var, 1, atol=1e-2), "Variance should be close to 1"

    # Test 2: Backward pass (gradient check)
    dout = np.random.randn(*out.shape)
    dx = bn.backward(dout)

    # Numerical gradient check
    eps = 1e-5
    numerical_grad = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_plus = x.copy()
            x_plus[i, j] += eps
            bn_check = BatchNormalization(num_features=4)
            bn_check.gamma = bn.gamma.copy()
            bn_check.beta = bn.beta.copy()
            out_plus = bn_check.forward(x_plus)
            loss_plus = np.sum(out_plus * dout)

            x_minus = x.copy()
            x_minus[i, j] -= eps
            bn_check2 = BatchNormalization(num_features=4)
            bn_check2.gamma = bn.gamma.copy()
            bn_check2.beta = bn.beta.copy()
            out_minus = bn_check2.forward(x_minus)
            loss_minus = np.sum(out_minus * dout)

            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)

    grad_diff = np.max(np.abs(dx - numerical_grad))
    print(f"Max gradient difference: {grad_diff}")
    assert grad_diff < 1e-4, "Gradient check failed"

    # Test 3: Inference mode uses running statistics
    bn.eval()
    x_test = np.random.randn(16, 4)
    out_eval = bn.forward(x_test)

    # Should use running mean/var, not batch statistics
    print(f"Running mean: {bn.running_mean}")
    print(f"Running var: {bn.running_var}")

    print("BatchNorm tests passed!")


def test_dropout():
    """Test Dropout implementation."""
    print("\nTesting Dropout...")

    np.random.seed(42)

    # Test 1: Training mode
    dropout = Dropout(p=0.5)
    x = np.ones((1000, 100))

    out = dropout.forward(x)

    # About 50% should be zero
    zero_fraction = np.mean(out == 0)
    print(f"Fraction of zeros (should be ~0.5): {zero_fraction}")

    # Non-zero values should be scaled by 2 (inverted dropout)
    non_zero_values = out[out != 0]
    print(f"Non-zero values (should be 2.0): {np.mean(non_zero_values)}")

    # Mean should be approximately 1 (same as input)
    print(f"Output mean (should be ~1): {np.mean(out)}")

    # Test 2: Inference mode
    dropout.eval()
    out_eval = dropout.forward(x)

    # Should be unchanged in eval mode (inverted dropout)
    assert np.allclose(out_eval, x), "Eval mode should return input unchanged"
    print("Eval mode correctly returns input unchanged")

    # Test 3: Backward pass
    dropout.train()
    out = dropout.forward(x)
    dout = np.ones_like(out)
    dx = dropout.backward(dout)

    # Gradient should flow only through non-dropped units
    assert np.allclose((dx != 0), (out != 0)), "Gradient mask should match forward mask"
    print("Backward pass correctly routes gradients")

    print("Dropout tests passed!")


def test_layer_normalization():
    """Test LayerNorm implementation."""
    print("\nTesting Layer Normalization...")

    np.random.seed(42)

    # Test 1: Forward pass
    ln = LayerNormalization(normalized_shape=(64,))
    x = np.random.randn(32, 64)

    out = ln.forward(x)

    # Check per-sample statistics
    mean = np.mean(out, axis=1)
    var = np.var(out, axis=1)

    print(f"Per-sample means (should be ~0): max={np.max(np.abs(mean))}")
    print(f"Per-sample variances (should be ~1): mean={np.mean(var)}")

    assert np.allclose(mean, 0, atol=1e-5), "Means should be close to 0"
    assert np.allclose(var, 1, atol=1e-2), "Variances should be close to 1"

    print("LayerNorm tests passed!")


def test_full_network():
    """Test full network with BatchNorm and Dropout."""
    print("\nTesting Full Network with Regularization...")

    np.random.seed(42)

    # Create network
    net = NeuralNetworkWithRegularization(
        layer_dims=[784, 256, 128, 10],
        dropout_rate=0.5,
        use_batchnorm=True,
        activation='relu'
    )

    # Generate dummy data
    X = np.random.randn(64, 784)
    y = np.eye(10)[np.random.randint(0, 10, 64)]  # One-hot labels

    # Forward pass (training)
    net.train()
    logits = net.forward(X)

    # Softmax and cross-entropy loss
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    loss = -np.mean(np.sum(y * np.log(probs + 1e-8), axis=1))

    print(f"Initial loss: {loss}")

    # Backward pass
    dout = probs - y
    grads = net.backward(dout)

    # Update parameters
    net.update_params(grads, lr=0.01)

    # Forward pass again
    logits2 = net.forward(X)
    exp_logits2 = np.exp(logits2 - np.max(logits2, axis=1, keepdims=True))
    probs2 = exp_logits2 / np.sum(exp_logits2, axis=1, keepdims=True)
    loss2 = -np.mean(np.sum(y * np.log(probs2 + 1e-8), axis=1))

    print(f"Loss after update: {loss2}")
    print(f"Loss decreased: {loss2 < loss}")

    # Test inference mode
    net.eval()
    logits_eval = net.forward(X)
    print(f"Inference output shape: {logits_eval.shape}")

    print("Full network tests passed!")


if __name__ == "__main__":
    test_batch_normalization()
    test_dropout()
    test_layer_normalization()
    test_full_network()
    print("\n All tests passed!")
```

## 6. When to Use Each Technique

### Batch Normalization

| Use Case | Recommendation |
|----------|----------------|
| Deep CNNs | Highly recommended |
| Large batch sizes (32+) | Excellent choice |
| Training speed | Significantly faster convergence |
| Higher learning rates | Enables larger LRs |
| Small batch sizes (<8) | Consider LayerNorm or GroupNorm |
| RNNs/Transformers | Use LayerNorm instead |

### Dropout

| Use Case | Recommendation |
|----------|----------------|
| Fully connected layers | Standard approach (p=0.5) |
| Convolutional layers | Lower rate (p=0.1-0.3) or Dropout2d |
| After BatchNorm | Apply after BN, before activation |
| Modern architectures | Often replaced by other regularization |
| Small datasets | Essential for preventing overfitting |
| Large models | Critical regularization technique |

### Layer Normalization

| Use Case | Recommendation |
|----------|----------------|
| Transformers | Standard choice |
| RNNs/LSTMs | Works better than BatchNorm |
| Small batch sizes | Independent of batch |
| Online learning | No running statistics needed |
| Variable sequence lengths | Natural handling |

## 7. Key Insights Summary

```
+-------------------+------------------+-------------------+
|     Property      |    BatchNorm     |     LayerNorm     |
+-------------------+------------------+-------------------+
| Normalizes over   | Batch dimension  | Feature dimension |
| Running stats     | Yes              | No                |
| Batch dependency  | Yes              | No                |
| Best for          | CNNs             | Transformers/RNNs |
| Training/Eval diff| Yes              | No                |
+-------------------+------------------+-------------------+

Dropout Key Points:
- Inverted dropout: scale during training, not inference
- Creates ensemble effect (exponentially many subnetworks)
- Don't use with BatchNorm in same position (controversial)
- Spatial dropout for conv layers
- Alpha dropout for self-normalizing networks
```

## 8. Common Pitfalls and Solutions

```python
"""
Common issues and how to avoid them:

1. BatchNorm with small batches:
   - Problem: Noisy batch statistics
   - Solution: Use GroupNorm or LayerNorm
   - Or: Increase batch size with gradient accumulation

2. Dropout at test time:
   - Problem: Forgetting to switch to eval mode
   - Solution: Always call model.eval() before inference

3. BatchNorm momentum confusion:
   - Problem: PyTorch uses 0.1 default, others use 0.9
   - Note: PyTorch momentum = 1 - TensorFlow momentum

4. Order of operations:
   - Recommended: Conv -> BN -> Activation -> Dropout
   - Don't apply dropout before BatchNorm

5. Dropout and determinism:
   - Problem: Different results each forward pass
   - Solution: Set random seed for reproducibility

6. BatchNorm with varying batch sizes:
   - Problem: First batch sets running stats scale
   - Solution: Warm up with representative batches
"""
```
