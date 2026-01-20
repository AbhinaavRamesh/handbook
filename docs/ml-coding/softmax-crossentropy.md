---
title: Softmax and Cross-Entropy Implementation
description: Implement softmax activation and cross-entropy loss from scratch with numerical stability tricks.
---

# Softmax and Cross-Entropy Implementation

> **Essential building blocks for neural network classification**

---

## Problem Statement

Implement the softmax activation function and cross-entropy loss from scratch. These are fundamental components in neural network classification tasks.

**Requirements:**
1. Implement numerically stable softmax function
2. Implement cross-entropy loss for multi-class classification
3. Implement the combined softmax + cross-entropy for efficiency
4. Compute gradients for backpropagation
5. Handle edge cases (numerical overflow, zero probabilities)

**Why This Matters:**
- Softmax converts logits to probabilities
- Cross-entropy measures the difference between predicted and true distributions
- Combined implementation is more numerically stable and computationally efficient
- Understanding these is crucial for building and debugging neural networks

---

## When This Is Asked

| Company | Context | Frequency |
|---------|---------|-----------|
| Google | ML fundamentals, numerical stability | Very High |
| Meta | Classification systems, recommendation | High |
| Amazon | ML infrastructure, SageMaker internals | High |
| OpenAI | Deep learning foundations | Very High |
| Apple | On-device ML optimization | Medium |
| Netflix | Recommendation systems | Medium |

**Interview Scenarios:**
- "Implement softmax without using any ML library functions"
- "Why is naive softmax implementation problematic?"
- "Derive the gradient of cross-entropy with softmax"
- "How would you make this numerically stable?"

---

## Mathematical Foundation

### Softmax Function

The softmax function converts a vector of real numbers (logits) into a probability distribution:

```
softmax(z)_i = exp(z_i) / sum_j(exp(z_j))
```

**Properties:**
- Output values are in range (0, 1)
- Output values sum to 1
- Preserves relative ordering of inputs
- Differentiable everywhere

### Cross-Entropy Loss

Cross-entropy measures the difference between two probability distributions:

```
H(p, q) = -sum_i(p_i * log(q_i))
```

For classification with one-hot encoded labels:
```
L = -log(q_y)  where y is the true class
```

### Gradient Computation

For softmax output `s` and cross-entropy loss `L`:

```
dL/dz_i = s_i - y_i
```

Where `y` is the one-hot encoded true label. This elegant result makes the combined gradient computation very simple.

### Log-Sum-Exp Trick

To prevent numerical overflow in softmax:

```
log(sum(exp(z_i))) = max(z) + log(sum(exp(z_i - max(z))))
```

This keeps the exponents small and prevents overflow.

---

## Implementation

### Softmax (From Scratch)

```python
import numpy as np
from typing import Union, Optional, Tuple
import warnings


class Softmax:
    """
    Softmax activation function with numerical stability.

    Converts logits to probabilities using the formula:
    softmax(z)_i = exp(z_i) / sum_j(exp(z_j))

    Uses the log-sum-exp trick for numerical stability.
    """

    def __init__(self, axis: int = -1):
        """
        Initialize Softmax.

        Args:
            axis: Axis along which to compute softmax (default: -1, last axis)
        """
        self.axis = axis
        self._input = None  # Cache for backward pass
        self._output = None

    def forward(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute softmax of input logits.

        Args:
            logits: Input array of any shape

        Returns:
            Probability distribution with same shape as input
        """
        # Cache input for backward pass
        self._input = logits

        # Numerical stability: subtract max to prevent overflow
        # This doesn't change the result due to softmax properties
        shifted = logits - np.max(logits, axis=self.axis, keepdims=True)

        # Compute exponentials
        exp_shifted = np.exp(shifted)

        # Normalize to get probabilities
        self._output = exp_shifted / np.sum(exp_shifted, axis=self.axis, keepdims=True)

        return self._output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute gradient of softmax.

        The Jacobian of softmax is:
        dS_i/dz_j = S_i * (delta_ij - S_j)

        Args:
            grad_output: Gradient from subsequent layer

        Returns:
            Gradient with respect to input logits
        """
        if self._output is None:
            raise RuntimeError("Must call forward before backward")

        s = self._output

        # For each sample, compute: grad_output @ Jacobian
        # Jacobian[i,j] = s[i] * (delta[i,j] - s[j])
        # This simplifies to: s * (grad_output - sum(grad_output * s))

        sum_grad_s = np.sum(grad_output * s, axis=self.axis, keepdims=True)
        grad_input = s * (grad_output - sum_grad_s)

        return grad_input

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        """Allow using instance as a function."""
        return self.forward(logits)


def softmax_naive(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Naive softmax implementation (NOT numerically stable).

    WARNING: This can overflow for large logits!
    Only use for educational purposes.

    Args:
        logits: Input logits
        axis: Axis for softmax computation

    Returns:
        Softmax probabilities
    """
    exp_logits = np.exp(logits)  # Can overflow!
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)


def softmax_stable(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax function.

    Uses the max-subtraction trick to prevent overflow.

    Args:
        logits: Input logits of any shape
        axis: Axis along which to compute softmax

    Returns:
        Softmax probabilities with same shape as input
    """
    # Subtract max for numerical stability
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute log of softmax in a numerically stable way.

    log_softmax(z)_i = z_i - log(sum(exp(z_j)))
                     = z_i - max(z) - log(sum(exp(z_j - max(z))))

    Args:
        logits: Input logits
        axis: Axis for computation

    Returns:
        Log probabilities
    """
    max_logits = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp
```

### Cross-Entropy Loss

```python
class CrossEntropyLoss:
    """
    Cross-entropy loss for classification.

    Supports both:
    - Hard labels (class indices)
    - Soft labels (probability distributions)
    """

    def __init__(
        self,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        epsilon: float = 1e-12
    ):
        """
        Initialize CrossEntropyLoss.

        Args:
            reduction: 'none', 'mean', or 'sum'
            label_smoothing: Smoothing factor (0 = no smoothing)
            ignore_index: Index to ignore in loss computation
            epsilon: Small constant for numerical stability
        """
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.epsilon = epsilon

        # Cache for backward pass
        self._probs = None
        self._targets = None
        self._mask = None

    def forward(
        self,
        probs: np.ndarray,
        targets: np.ndarray
    ) -> Union[float, np.ndarray]:
        """
        Compute cross-entropy loss.

        Args:
            probs: Predicted probabilities, shape (N, C) or (N,)
            targets: Target labels (indices or one-hot), shape (N,) or (N, C)

        Returns:
            Loss value(s) depending on reduction
        """
        # Ensure 2D for batch processing
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)

        batch_size, num_classes = probs.shape

        # Clip probabilities for numerical stability
        probs_clipped = np.clip(probs, self.epsilon, 1 - self.epsilon)

        # Convert targets to one-hot if needed
        if targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1):
            targets_flat = targets.flatten().astype(int)
            targets_onehot = np.zeros((batch_size, num_classes))

            # Create mask for valid indices
            valid_mask = targets_flat != self.ignore_index
            valid_indices = targets_flat[valid_mask]

            # Set one-hot values
            targets_onehot[valid_mask, valid_indices] = 1.0
            self._mask = valid_mask
        else:
            targets_onehot = targets
            self._mask = np.ones(batch_size, dtype=bool)

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_onehot = self._apply_label_smoothing(
                targets_onehot, num_classes
            )

        # Cache for backward pass
        self._probs = probs_clipped
        self._targets = targets_onehot

        # Compute cross-entropy: -sum(y * log(p))
        log_probs = np.log(probs_clipped)
        loss_per_sample = -np.sum(targets_onehot * log_probs, axis=1)

        # Apply mask
        loss_per_sample = np.where(self._mask, loss_per_sample, 0.0)

        # Apply reduction
        if self.reduction == 'none':
            return loss_per_sample
        elif self.reduction == 'sum':
            return np.sum(loss_per_sample)
        else:  # mean
            valid_count = np.sum(self._mask)
            if valid_count == 0:
                return 0.0
            return np.sum(loss_per_sample) / valid_count

    def backward(self) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss with respect to probabilities.

        dL/dp_i = -y_i / p_i

        Returns:
            Gradient with respect to input probabilities
        """
        if self._probs is None:
            raise RuntimeError("Must call forward before backward")

        # Gradient: -y / p
        grad = -self._targets / self._probs

        # Apply mask
        grad = grad * self._mask.reshape(-1, 1)

        # Apply reduction scaling
        if self.reduction == 'mean':
            valid_count = np.sum(self._mask)
            if valid_count > 0:
                grad = grad / valid_count

        return grad

    def _apply_label_smoothing(
        self,
        targets: np.ndarray,
        num_classes: int
    ) -> np.ndarray:
        """
        Apply label smoothing to targets.

        Smoothed label = (1 - alpha) * one_hot + alpha / num_classes

        Args:
            targets: One-hot encoded targets
            num_classes: Number of classes

        Returns:
            Smoothed targets
        """
        smooth_value = self.label_smoothing / num_classes
        return (1 - self.label_smoothing) * targets + smooth_value

    def __call__(
        self,
        probs: np.ndarray,
        targets: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Allow using instance as a function."""
        return self.forward(probs, targets)


def cross_entropy_loss(
    probs: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-12
) -> float:
    """
    Simple cross-entropy loss function.

    Args:
        probs: Predicted probabilities, shape (N, C)
        targets: Target class indices, shape (N,)
        epsilon: Small constant for numerical stability

    Returns:
        Mean cross-entropy loss
    """
    batch_size = probs.shape[0]

    # Clip for stability
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)

    # Get probability of true class for each sample
    true_class_probs = probs_clipped[np.arange(batch_size), targets]

    # Cross-entropy: -log(p_true)
    loss = -np.log(true_class_probs)

    return np.mean(loss)


def binary_cross_entropy(
    probs: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-12
) -> float:
    """
    Binary cross-entropy loss.

    BCE = -[y * log(p) + (1-y) * log(1-p)]

    Args:
        probs: Predicted probabilities, shape (N,)
        targets: Binary targets (0 or 1), shape (N,)
        epsilon: Small constant for numerical stability

    Returns:
        Mean binary cross-entropy loss
    """
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)

    loss = -(
        targets * np.log(probs_clipped) +
        (1 - targets) * np.log(1 - probs_clipped)
    )

    return np.mean(loss)
```

### Combined Softmax + Cross-Entropy (Numerically Stable)

```python
class SoftmaxCrossEntropyLoss:
    """
    Combined Softmax and Cross-Entropy loss.

    This combined implementation is:
    1. More numerically stable
    2. More computationally efficient
    3. Has simpler gradients

    The loss is computed as:
    L = -z_y + log(sum(exp(z_i)))
      = -z_y + max(z) + log(sum(exp(z_i - max(z))))
    """

    def __init__(
        self,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        ignore_index: int = -100
    ):
        """
        Initialize SoftmaxCrossEntropyLoss.

        Args:
            reduction: 'none', 'mean', or 'sum'
            label_smoothing: Smoothing factor
            ignore_index: Index to ignore
        """
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

        # Cache for backward
        self._logits = None
        self._targets = None
        self._softmax_output = None
        self._mask = None

    def forward(
        self,
        logits: np.ndarray,
        targets: np.ndarray
    ) -> Union[float, np.ndarray]:
        """
        Compute combined softmax + cross-entropy loss.

        Uses log-sum-exp trick for numerical stability.

        Args:
            logits: Raw scores, shape (N, C)
            targets: Target class indices, shape (N,)

        Returns:
            Loss value(s)
        """
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        batch_size, num_classes = logits.shape

        # Cache for backward
        self._logits = logits

        # Handle targets
        if targets.ndim == 1:
            targets_flat = targets.astype(int)
        else:
            targets_flat = targets.flatten().astype(int)

        self._targets = targets_flat

        # Create mask for valid indices
        self._mask = targets_flat != self.ignore_index

        # Compute log-sum-exp with numerical stability
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted_logits = logits - max_logits
        log_sum_exp = max_logits.squeeze() + np.log(
            np.sum(np.exp(shifted_logits), axis=1)
        )

        # Get logits of true class
        # Handle ignore_index by using 0 (will be masked anyway)
        safe_targets = np.where(self._mask, targets_flat, 0)
        logits_true_class = logits[np.arange(batch_size), safe_targets]

        # Apply label smoothing if needed
        if self.label_smoothing > 0:
            # With smoothing: L = (1-alpha)*L_hard + alpha*L_uniform
            # L_uniform = log(C) - mean(logits) + log_sum_exp
            loss_hard = log_sum_exp - logits_true_class
            loss_uniform = log_sum_exp - np.mean(logits, axis=1)
            loss_per_sample = (
                (1 - self.label_smoothing) * loss_hard +
                self.label_smoothing * loss_uniform
            )
        else:
            # Standard cross-entropy: -log(softmax(z)_y) = log_sum_exp - z_y
            loss_per_sample = log_sum_exp - logits_true_class

        # Apply mask
        loss_per_sample = np.where(self._mask, loss_per_sample, 0.0)

        # Cache softmax output for backward pass
        self._softmax_output = softmax_stable(logits, axis=1)

        # Apply reduction
        if self.reduction == 'none':
            return loss_per_sample
        elif self.reduction == 'sum':
            return np.sum(loss_per_sample)
        else:  # mean
            valid_count = np.sum(self._mask)
            if valid_count == 0:
                return 0.0
            return np.sum(loss_per_sample) / valid_count

    def backward(self) -> np.ndarray:
        """
        Compute gradient with respect to logits.

        The beautiful result: dL/dz_i = softmax(z)_i - y_i

        Returns:
            Gradient with respect to logits
        """
        if self._logits is None:
            raise RuntimeError("Must call forward before backward")

        batch_size, num_classes = self._logits.shape

        # Start with softmax output
        grad = self._softmax_output.copy()

        # Subtract 1 from true class (equivalent to subtracting one-hot)
        safe_targets = np.where(self._mask, self._targets, 0)
        grad[np.arange(batch_size), safe_targets] -= 1

        # Handle label smoothing gradient
        if self.label_smoothing > 0:
            # Adjust gradient for smoothing
            grad = (1 - self.label_smoothing) * grad
            grad += self.label_smoothing * (
                self._softmax_output - 1.0 / num_classes
            )

        # Apply mask
        grad = grad * self._mask.reshape(-1, 1)

        # Apply reduction scaling
        if self.reduction == 'mean':
            valid_count = np.sum(self._mask)
            if valid_count > 0:
                grad = grad / valid_count

        return grad

    def __call__(
        self,
        logits: np.ndarray,
        targets: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Allow using instance as a function."""
        return self.forward(logits, targets)


def softmax_cross_entropy_with_logits(
    logits: np.ndarray,
    targets: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute softmax cross-entropy loss and gradient in one pass.

    This is the most efficient way to compute both.

    Args:
        logits: Raw scores, shape (N, C)
        targets: Target class indices, shape (N,)

    Returns:
        Tuple of (loss, gradient)
    """
    batch_size, num_classes = logits.shape

    # Compute stable softmax
    max_logits = np.max(logits, axis=1, keepdims=True)
    shifted = logits - max_logits
    exp_shifted = np.exp(shifted)
    softmax_probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    # Compute loss: -log(p_y) = log_sum_exp - z_y
    log_sum_exp = max_logits.squeeze() + np.log(
        np.sum(exp_shifted, axis=1)
    )
    logits_true = logits[np.arange(batch_size), targets]
    loss = np.mean(log_sum_exp - logits_true)

    # Compute gradient: softmax - one_hot
    grad = softmax_probs.copy()
    grad[np.arange(batch_size), targets] -= 1
    grad = grad / batch_size  # For mean reduction

    return loss, grad
```

### Gradient/Backward Pass

```python
class SoftmaxCrossEntropyWithGradients:
    """
    Complete implementation with full gradient computation.

    Demonstrates the full forward and backward pass including:
    - Softmax forward
    - Cross-entropy forward
    - Combined backward pass
    """

    def __init__(self):
        self.cache = {}

    def forward(
        self,
        logits: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Forward pass computing loss.

        Args:
            logits: Shape (N, C)
            targets: Shape (N,) with class indices

        Returns:
            Scalar loss value
        """
        batch_size, num_classes = logits.shape

        # Step 1: Compute stable softmax
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
        probs = exp_shifted / sum_exp

        # Step 2: Compute cross-entropy loss
        # Using log probabilities directly for stability
        log_probs = shifted - np.log(sum_exp)

        # Get log probability of true class
        true_log_probs = log_probs[np.arange(batch_size), targets]
        loss = -np.mean(true_log_probs)

        # Cache values needed for backward pass
        self.cache = {
            'probs': probs,
            'targets': targets,
            'batch_size': batch_size,
            'num_classes': num_classes
        }

        return loss

    def backward(self) -> np.ndarray:
        """
        Backward pass computing gradient w.r.t. logits.

        Mathematical derivation:
        L = -1/N * sum_n(log(p_y_n))

        For softmax: p_i = exp(z_i) / sum(exp(z_j))

        Chain rule:
        dL/dz_i = dL/dp * dp/dz_i

        After simplification:
        dL/dz_i = (1/N) * (p_i - 1{i=y})

        Returns:
            Gradient with respect to input logits
        """
        probs = self.cache['probs']
        targets = self.cache['targets']
        batch_size = self.cache['batch_size']

        # Gradient is simply: (probs - one_hot) / batch_size
        grad = probs.copy()
        grad[np.arange(batch_size), targets] -= 1.0
        grad /= batch_size

        return grad

    def numerical_gradient(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        epsilon: float = 1e-5
    ) -> np.ndarray:
        """
        Compute numerical gradient for verification.

        Uses finite differences: (f(x+h) - f(x-h)) / (2h)

        Args:
            logits: Input logits
            targets: Target labels
            epsilon: Finite difference step size

        Returns:
            Numerical gradient approximation
        """
        grad = np.zeros_like(logits)

        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                # f(x + h)
                logits_plus = logits.copy()
                logits_plus[i, j] += epsilon
                loss_plus = self.forward(logits_plus, targets)

                # f(x - h)
                logits_minus = logits.copy()
                logits_minus[i, j] -= epsilon
                loss_minus = self.forward(logits_minus, targets)

                # Central difference
                grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        return grad

    def verify_gradient(
        self,
        logits: np.ndarray,
        targets: np.ndarray
    ) -> dict:
        """
        Verify analytical gradient against numerical gradient.

        Returns:
            Dictionary with comparison metrics
        """
        # Compute analytical gradient
        self.forward(logits, targets)
        analytical_grad = self.backward()

        # Compute numerical gradient
        numerical_grad = self.numerical_gradient(logits, targets)

        # Compute relative error
        diff = np.abs(analytical_grad - numerical_grad)
        rel_error = diff / (np.abs(analytical_grad) + np.abs(numerical_grad) + 1e-8)

        return {
            'analytical_grad': analytical_grad,
            'numerical_grad': numerical_grad,
            'max_absolute_error': np.max(diff),
            'max_relative_error': np.max(rel_error),
            'mean_relative_error': np.mean(rel_error),
            'gradient_check_passed': np.max(rel_error) < 1e-5
        }


def compute_jacobian_softmax(probs: np.ndarray) -> np.ndarray:
    """
    Compute the full Jacobian matrix of softmax.

    J[i,j] = d(softmax_i) / d(z_j) = p_i * (delta_ij - p_j)

    Args:
        probs: Softmax output, shape (C,)

    Returns:
        Jacobian matrix, shape (C, C)
    """
    n = len(probs)
    jacobian = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                jacobian[i, j] = probs[i] * (1 - probs[i])
            else:
                jacobian[i, j] = -probs[i] * probs[j]

    # Equivalent vectorized form:
    # jacobian = np.diag(probs) - np.outer(probs, probs)

    return jacobian
```

---

## PyTorch Comparison

```python
import torch
import torch.nn.functional as F


def compare_implementations():
    """
    Compare our implementation with PyTorch.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    # Test data
    batch_size, num_classes = 4, 5
    logits_np = np.random.randn(batch_size, num_classes).astype(np.float32)
    targets_np = np.random.randint(0, num_classes, batch_size)

    # Convert to PyTorch
    logits_torch = torch.tensor(logits_np, requires_grad=True)
    targets_torch = torch.tensor(targets_np, dtype=torch.long)

    print("=" * 60)
    print("COMPARISON: Custom Implementation vs PyTorch")
    print("=" * 60)

    # 1. Softmax comparison
    print("\n1. SOFTMAX COMPARISON")
    print("-" * 40)

    # Our implementation
    softmax_custom = softmax_stable(logits_np)

    # PyTorch
    softmax_pytorch = F.softmax(logits_torch, dim=1).detach().numpy()

    print(f"Max absolute difference: {np.max(np.abs(softmax_custom - softmax_pytorch)):.2e}")
    print(f"Match: {np.allclose(softmax_custom, softmax_pytorch)}")

    # 2. Log-softmax comparison
    print("\n2. LOG-SOFTMAX COMPARISON")
    print("-" * 40)

    log_softmax_custom = log_softmax(logits_np)
    log_softmax_pytorch = F.log_softmax(logits_torch, dim=1).detach().numpy()

    print(f"Max absolute difference: {np.max(np.abs(log_softmax_custom - log_softmax_pytorch)):.2e}")
    print(f"Match: {np.allclose(log_softmax_custom, log_softmax_pytorch)}")

    # 3. Cross-entropy loss comparison
    print("\n3. CROSS-ENTROPY LOSS COMPARISON")
    print("-" * 40)

    # Our implementation
    loss_fn = SoftmaxCrossEntropyLoss(reduction='mean')
    loss_custom = loss_fn.forward(logits_np, targets_np)

    # PyTorch
    loss_pytorch = F.cross_entropy(logits_torch, targets_torch).item()

    print(f"Custom loss: {loss_custom:.6f}")
    print(f"PyTorch loss: {loss_pytorch:.6f}")
    print(f"Difference: {abs(loss_custom - loss_pytorch):.2e}")
    print(f"Match: {np.isclose(loss_custom, loss_pytorch)}")

    # 4. Gradient comparison
    print("\n4. GRADIENT COMPARISON")
    print("-" * 40)

    # Our gradient
    grad_custom = loss_fn.backward()

    # PyTorch gradient
    logits_torch_grad = torch.tensor(logits_np, requires_grad=True)
    loss_torch = F.cross_entropy(logits_torch_grad, targets_torch)
    loss_torch.backward()
    grad_pytorch = logits_torch_grad.grad.numpy()

    print(f"Max gradient difference: {np.max(np.abs(grad_custom - grad_pytorch)):.2e}")
    print(f"Gradients match: {np.allclose(grad_custom, grad_pytorch, rtol=1e-5)}")

    # 5. Label smoothing comparison
    print("\n5. LABEL SMOOTHING COMPARISON")
    print("-" * 40)

    smoothing = 0.1

    # Our implementation
    loss_fn_smooth = SoftmaxCrossEntropyLoss(
        reduction='mean',
        label_smoothing=smoothing
    )
    loss_smooth_custom = loss_fn_smooth.forward(logits_np, targets_np)

    # PyTorch
    loss_smooth_pytorch = F.cross_entropy(
        torch.tensor(logits_np),
        torch.tensor(targets_np, dtype=torch.long),
        label_smoothing=smoothing
    ).item()

    print(f"Custom loss (smoothed): {loss_smooth_custom:.6f}")
    print(f"PyTorch loss (smoothed): {loss_smooth_pytorch:.6f}")
    print(f"Difference: {abs(loss_smooth_custom - loss_smooth_pytorch):.2e}")

    # 6. Numerical stability test
    print("\n6. NUMERICAL STABILITY TEST")
    print("-" * 40)

    # Large logits that would overflow naive implementation
    large_logits = np.array([[1000.0, 1001.0, 1002.0]])

    try:
        naive_result = softmax_naive(large_logits)
        print(f"Naive softmax: {naive_result}")
        print(f"Contains NaN/Inf: {np.any(np.isnan(naive_result) | np.isinf(naive_result))}")
    except:
        print("Naive softmax: FAILED (overflow)")

    stable_result = softmax_stable(large_logits)
    print(f"Stable softmax: {stable_result}")
    print(f"Sum equals 1: {np.isclose(np.sum(stable_result), 1.0)}")

    print("\n" + "=" * 60)
    print("All comparisons completed!")
    print("=" * 60)


def benchmark_implementations():
    """
    Benchmark custom vs PyTorch implementations.
    """
    import time

    # Test sizes
    sizes = [(32, 10), (128, 100), (256, 1000), (512, 10000)]

    print("\nBENCHMARK: Custom vs PyTorch")
    print("=" * 70)
    print(f"{'Size':<20} {'Custom (ms)':<15} {'PyTorch (ms)':<15} {'Ratio':<10}")
    print("-" * 70)

    for batch_size, num_classes in sizes:
        logits_np = np.random.randn(batch_size, num_classes).astype(np.float32)
        targets_np = np.random.randint(0, num_classes, batch_size)

        logits_torch = torch.tensor(logits_np, requires_grad=True)
        targets_torch = torch.tensor(targets_np, dtype=torch.long)

        # Warm up
        loss_fn = SoftmaxCrossEntropyLoss()
        _ = loss_fn.forward(logits_np, targets_np)
        _ = F.cross_entropy(logits_torch, targets_torch)

        # Benchmark custom
        n_runs = 100
        start = time.perf_counter()
        for _ in range(n_runs):
            loss = loss_fn.forward(logits_np, targets_np)
            grad = loss_fn.backward()
        custom_time = (time.perf_counter() - start) * 1000 / n_runs

        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(n_runs):
            logits_torch = torch.tensor(logits_np, requires_grad=True)
            loss = F.cross_entropy(logits_torch, targets_torch)
            loss.backward()
        pytorch_time = (time.perf_counter() - start) * 1000 / n_runs

        size_str = f"({batch_size}, {num_classes})"
        ratio = custom_time / pytorch_time
        print(f"{size_str:<20} {custom_time:<15.3f} {pytorch_time:<15.3f} {ratio:<10.2f}x")


if __name__ == "__main__":
    compare_implementations()
    benchmark_implementations()
```

---

## Complexity Analysis

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Softmax forward | O(N * C) | N = batch size, C = classes |
| Softmax backward | O(N * C) | Element-wise operations |
| Cross-entropy forward | O(N * C) | Log and multiply |
| Cross-entropy backward | O(N * C) | Simple subtraction |
| Combined forward | O(N * C) | Single pass |
| Combined backward | O(N * C) | Most efficient |

### Space Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Softmax | O(N * C) | Store probabilities |
| Cross-entropy | O(N * C) | Store log probabilities |
| Combined | O(N * C) | Only store softmax output |
| Gradients | O(N * C) | Same as input size |

### Numerical Considerations

```python
def analyze_numerical_precision():
    """
    Analyze numerical precision characteristics.
    """
    print("NUMERICAL PRECISION ANALYSIS")
    print("=" * 60)

    # 1. Overflow threshold
    print("\n1. Overflow Threshold")
    print("-" * 40)

    # float32 max exponent
    max_float32 = np.finfo(np.float32).max
    max_exp = np.log(max_float32)
    print(f"Max float32: {max_float32:.2e}")
    print(f"Max safe exponent: {max_exp:.1f}")
    print("=> Logits > ~88 will overflow in naive softmax")

    # 2. Underflow considerations
    print("\n2. Underflow Threshold")
    print("-" * 40)

    min_float32 = np.finfo(np.float32).tiny
    min_exp = np.log(min_float32)
    print(f"Min positive float32: {min_float32:.2e}")
    print(f"Min safe exponent: {min_exp:.1f}")
    print("=> Very negative logits can underflow to 0")

    # 3. Log(0) issue
    print("\n3. Log(0) Protection")
    print("-" * 40)

    probs = np.array([0.9, 0.1, 0.0])
    print(f"Probabilities: {probs}")
    print(f"log(probs) without clipping: {np.log(probs + 1e-12)}")
    print("=> Always clip or add epsilon before log")

    # 4. Precision loss in subtraction
    print("\n4. Catastrophic Cancellation")
    print("-" * 40)

    # When probabilities are very close to 1
    logits = np.array([[100.0, 0.0, 0.0]])
    probs = softmax_stable(logits)
    print(f"Logits: {logits[0]}")
    print(f"Softmax: {probs[0]}")
    print(f"1 - p[0] = {1 - probs[0, 0]}")
    print("=> Stable implementation handles extreme cases")


analyze_numerical_precision()
```

---

## Common Interview Follow-ups

### 1. Why Combine Softmax and Cross-Entropy?

```python
def demonstrate_combined_advantages():
    """
    Show why combined implementation is better.
    """
    print("ADVANTAGES OF COMBINED IMPLEMENTATION")
    print("=" * 60)

    # Reason 1: Numerical stability
    print("\n1. NUMERICAL STABILITY")
    print("-" * 40)

    logits = np.array([[100.0, 101.0, 102.0]])
    targets = np.array([2])

    # Separate: might have issues
    probs = softmax_stable(logits)
    # If any prob is 0 or 1, log will be -inf or 0
    print(f"Softmax output: {probs}")
    print(f"Log probs: {np.log(probs + 1e-12)}")

    # Combined: directly computes log-softmax
    loss_fn = SoftmaxCrossEntropyLoss()
    loss = loss_fn.forward(logits, targets)
    print(f"Combined loss: {loss:.6f}")

    # Reason 2: Simpler gradients
    print("\n2. SIMPLER GRADIENTS")
    print("-" * 40)

    print("Separate gradient computation:")
    print("  dL/dz = dL/dp * dp/dz")
    print("  dL/dp = -y/p (can be very large if p small)")
    print("  dp/dz = p(1-p) or -p_i*p_j (requires Jacobian)")
    print("")
    print("Combined gradient:")
    print("  dL/dz = p - y (simple subtraction!)")

    # Reason 3: Computational efficiency
    print("\n3. COMPUTATIONAL EFFICIENCY")
    print("-" * 40)

    print("Separate:")
    print("  - Compute exp(z) for softmax")
    print("  - Compute log(p) for cross-entropy")
    print("  - Compute Jacobian for backward")
    print("")
    print("Combined:")
    print("  - Compute log-sum-exp directly")
    print("  - No explicit log needed")
    print("  - No Jacobian needed")


demonstrate_combined_advantages()
```

### 2. How to Handle Numerical Overflow?

```python
def overflow_handling_strategies():
    """
    Demonstrate overflow handling techniques.
    """
    print("OVERFLOW HANDLING STRATEGIES")
    print("=" * 60)

    # Problem case
    large_logits = np.array([1000.0, 1001.0, 999.0])

    # Strategy 1: Max subtraction (standard)
    print("\n1. MAX SUBTRACTION (Standard)")
    print("-" * 40)
    shifted = large_logits - np.max(large_logits)
    print(f"Original: {large_logits}")
    print(f"Shifted: {shifted}")
    print(f"Max value after shift: {np.max(shifted)}")
    result = np.exp(shifted) / np.sum(np.exp(shifted))
    print(f"Result: {result}")

    # Strategy 2: Log-domain computation
    print("\n2. LOG-DOMAIN COMPUTATION")
    print("-" * 40)
    print("Compute everything in log space:")
    print("  log(softmax) = z - log(sum(exp(z)))")
    print("  Use log-sum-exp trick for the sum")
    log_probs = log_softmax(large_logits)
    print(f"Log probabilities: {log_probs}")
    print(f"Probabilities: {np.exp(log_probs)}")

    # Strategy 3: Chunked computation for very long sequences
    print("\n3. CHUNKED COMPUTATION")
    print("-" * 40)
    print("For very large C, compute in chunks to manage memory:")

    def chunked_log_sum_exp(logits, chunk_size=1000):
        """Compute log-sum-exp in chunks."""
        n = len(logits)
        max_val = np.max(logits)

        total = 0.0
        for i in range(0, n, chunk_size):
            chunk = logits[i:i + chunk_size]
            total += np.sum(np.exp(chunk - max_val))

        return max_val + np.log(total)

    test_logits = np.random.randn(10000)
    result_normal = np.log(np.sum(np.exp(test_logits - np.max(test_logits)))) + np.max(test_logits)
    result_chunked = chunked_log_sum_exp(test_logits)
    print(f"Normal log-sum-exp: {result_normal:.6f}")
    print(f"Chunked log-sum-exp: {result_chunked:.6f}")
    print(f"Match: {np.isclose(result_normal, result_chunked)}")


overflow_handling_strategies()
```

### 3. Binary vs Categorical Cross-Entropy

```python
def compare_binary_categorical():
    """
    Compare binary and categorical cross-entropy.
    """
    print("BINARY vs CATEGORICAL CROSS-ENTROPY")
    print("=" * 60)

    # Setup equivalent problems
    np.random.seed(42)

    # Binary classification
    print("\n1. BINARY CLASSIFICATION")
    print("-" * 40)

    # Using sigmoid (binary)
    logits_binary = np.array([2.0, -1.0, 0.5, -0.5])
    targets_binary = np.array([1, 0, 1, 0])

    # Sigmoid
    probs_binary = 1 / (1 + np.exp(-logits_binary))
    bce_loss = binary_cross_entropy(probs_binary, targets_binary)

    print(f"Logits: {logits_binary}")
    print(f"Sigmoid probs: {probs_binary}")
    print(f"Targets: {targets_binary}")
    print(f"BCE Loss: {bce_loss:.6f}")

    # Equivalent as 2-class problem
    print("\n2. EQUIVALENT 2-CLASS CATEGORICAL")
    print("-" * 40)

    # Convert to 2-class logits
    logits_2class = np.stack([np.zeros_like(logits_binary), logits_binary], axis=1)

    # Softmax
    probs_2class = softmax_stable(logits_2class)
    ce_loss = cross_entropy_loss(probs_2class, targets_binary)

    print(f"2-class logits:\n{logits_2class}")
    print(f"Softmax probs:\n{probs_2class}")
    print(f"CE Loss: {ce_loss:.6f}")

    # They should be equal!
    print(f"\nBCE == CE: {np.isclose(bce_loss, ce_loss)}")

    # When to use which
    print("\n3. WHEN TO USE WHICH")
    print("-" * 40)

    print("Use BINARY Cross-Entropy when:")
    print("  - 2-class classification")
    print("  - Multi-label classification (multiple labels can be true)")
    print("  - Each output is independent")
    print("")
    print("Use CATEGORICAL Cross-Entropy when:")
    print("  - Multi-class classification (exactly one class is true)")
    print("  - Classes are mutually exclusive")
    print("  - Outputs should sum to 1")

    # Multi-label example
    print("\n4. MULTI-LABEL EXAMPLE")
    print("-" * 40)

    # Image can have multiple tags: [cat, dog, outdoor]
    logits_multilabel = np.array([2.0, -1.0, 1.5])  # [cat, dog, outdoor]
    targets_multilabel = np.array([1, 0, 1])  # Has cat and outdoor, no dog

    # Must use sigmoid + BCE for each label
    probs_multilabel = 1 / (1 + np.exp(-logits_multilabel))
    print(f"Probs (independent): {probs_multilabel}")
    print(f"Sum of probs: {np.sum(probs_multilabel):.2f} (not 1!)")
    print("=> Softmax would be wrong here!")


compare_binary_categorical()
```

---

## Edge Cases and Pitfalls

### Large Logits Causing Overflow

```python
def test_large_logits():
    """
    Test handling of large logits.
    """
    print("EDGE CASE: Large Logits")
    print("=" * 60)

    # Progressively larger logits
    scales = [10, 100, 500, 1000, 10000]

    for scale in scales:
        logits = np.array([[0.0, 1.0, 2.0]]) * scale

        # Naive (will fail for large scale)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                naive = softmax_naive(logits)
                naive_ok = not (np.any(np.isnan(naive)) or np.any(np.isinf(naive)))
        except:
            naive_ok = False

        # Stable
        stable = softmax_stable(logits)
        stable_ok = not (np.any(np.isnan(stable)) or np.any(np.isinf(stable)))
        stable_sum = np.sum(stable)

        print(f"Scale {scale:>5}: Naive={'OK' if naive_ok else 'FAIL':>4}, "
              f"Stable={'OK' if stable_ok else 'FAIL':>4}, Sum={stable_sum:.6f}")


test_large_logits()
```

### Zero Probabilities

```python
def test_zero_probabilities():
    """
    Test handling of zero or near-zero probabilities.
    """
    print("\nEDGE CASE: Zero Probabilities")
    print("=" * 60)

    # Extreme logits leading to near-zero probs
    logits = np.array([[100.0, 0.0, 0.0]])
    targets = np.array([1])  # Target is the low-probability class

    probs = softmax_stable(logits)
    print(f"Logits: {logits[0]}")
    print(f"Probs: {probs[0]}")
    print(f"Target class prob: {probs[0, 1]:.2e}")

    # Loss computation
    loss_fn = SoftmaxCrossEntropyLoss()
    loss = loss_fn.forward(logits, targets)
    print(f"Loss: {loss:.6f}")

    # This should be large but not inf
    print(f"Is finite: {np.isfinite(loss)}")

    # Gradient should still be valid
    grad = loss_fn.backward()
    print(f"Gradient: {grad[0]}")
    print(f"Gradient is finite: {np.all(np.isfinite(grad))}")


test_zero_probabilities()
```

### Label Smoothing

```python
def test_label_smoothing():
    """
    Test label smoothing implementation.
    """
    print("\nEDGE CASE: Label Smoothing")
    print("=" * 60)

    logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    targets = np.array([4])  # True class is 4

    smoothing_values = [0.0, 0.1, 0.2, 0.5]

    for alpha in smoothing_values:
        loss_fn = SoftmaxCrossEntropyLoss(label_smoothing=alpha)
        loss = loss_fn.forward(logits, targets)

        # Expected target distribution
        num_classes = logits.shape[1]
        if alpha > 0:
            smooth_target = np.zeros(num_classes)
            smooth_target[targets[0]] = 1 - alpha
            smooth_target += alpha / num_classes
        else:
            smooth_target = np.zeros(num_classes)
            smooth_target[targets[0]] = 1.0

        print(f"Smoothing={alpha}: Loss={loss:.4f}, Target dist={smooth_target}")

    print("\nBenefits of label smoothing:")
    print("  1. Prevents overconfident predictions")
    print("  2. Improves generalization")
    print("  3. Provides regularization effect")


test_label_smoothing()
```

### Batch with Mixed Valid/Invalid Indices

```python
def test_ignore_index():
    """
    Test ignore_index functionality.
    """
    print("\nEDGE CASE: Ignore Index")
    print("=" * 60)

    logits = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0]
    ])

    # Some targets are padding (-100)
    targets = np.array([0, 1, -100, 2])

    loss_fn = SoftmaxCrossEntropyLoss(ignore_index=-100)
    loss = loss_fn.forward(logits, targets)
    grad = loss_fn.backward()

    print(f"Targets: {targets}")
    print(f"Loss (mean over valid): {loss:.6f}")
    print(f"Gradient:\n{grad}")
    print(f"Gradient for ignored sample (row 2): {grad[2]}")
    print(f"Is zero: {np.allclose(grad[2], 0)}")


test_ignore_index()
```

---

## Complete Test Suite

```python
def run_all_tests():
    """
    Run comprehensive test suite.
    """
    print("=" * 70)
    print("COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Basic softmax
    print("\nTest 1: Basic Softmax")
    try:
        logits = np.array([[1.0, 2.0, 3.0]])
        result = softmax_stable(logits)
        assert np.isclose(np.sum(result), 1.0), "Softmax should sum to 1"
        assert np.all(result > 0), "Softmax should be positive"
        assert np.all(result < 1), "Softmax should be less than 1"
        print("  PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"  FAILED: {e}")
        tests_failed += 1

    # Test 2: Softmax numerical stability
    print("\nTest 2: Softmax Numerical Stability")
    try:
        large_logits = np.array([[1000.0, 1001.0, 1002.0]])
        result = softmax_stable(large_logits)
        assert np.isclose(np.sum(result), 1.0), "Should sum to 1"
        assert np.all(np.isfinite(result)), "Should be finite"
        print("  PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"  FAILED: {e}")
        tests_failed += 1

    # Test 3: Cross-entropy with probabilities
    print("\nTest 3: Cross-Entropy Loss")
    try:
        probs = np.array([[0.7, 0.2, 0.1]])
        targets = np.array([0])
        loss = cross_entropy_loss(probs, targets)
        expected = -np.log(0.7)
        assert np.isclose(loss, expected), f"Expected {expected}, got {loss}"
        print("  PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"  FAILED: {e}")
        tests_failed += 1

    # Test 4: Combined loss
    print("\nTest 4: Combined Softmax + Cross-Entropy")
    try:
        logits = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([2])

        # Method 1: Separate
        probs = softmax_stable(logits)
        loss1 = -np.log(probs[0, targets[0]])

        # Method 2: Combined
        loss_fn = SoftmaxCrossEntropyLoss(reduction='none')
        loss2 = loss_fn.forward(logits, targets)[0]

        assert np.isclose(loss1, loss2), f"Losses should match: {loss1} vs {loss2}"
        print("  PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"  FAILED: {e}")
        tests_failed += 1

    # Test 5: Gradient correctness
    print("\nTest 5: Gradient Correctness")
    try:
        logits = np.random.randn(2, 4)
        targets = np.array([1, 3])

        grad_checker = SoftmaxCrossEntropyWithGradients()
        result = grad_checker.verify_gradient(logits, targets)

        assert result['gradient_check_passed'], \
            f"Max rel error: {result['max_relative_error']}"
        print("  PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"  FAILED: {e}")
        tests_failed += 1

    # Test 6: Batch processing
    print("\nTest 6: Batch Processing")
    try:
        batch_size = 32
        num_classes = 10

        logits = np.random.randn(batch_size, num_classes)
        targets = np.random.randint(0, num_classes, batch_size)

        loss_fn = SoftmaxCrossEntropyLoss()
        loss = loss_fn.forward(logits, targets)
        grad = loss_fn.backward()

        assert np.isfinite(loss), "Loss should be finite"
        assert grad.shape == logits.shape, "Gradient shape should match input"
        assert np.all(np.isfinite(grad)), "Gradient should be finite"
        print("  PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"  FAILED: {e}")
        tests_failed += 1

    # Test 7: Label smoothing
    print("\nTest 7: Label Smoothing")
    try:
        logits = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([2])

        loss_no_smooth = SoftmaxCrossEntropyLoss(label_smoothing=0.0)
        loss_smooth = SoftmaxCrossEntropyLoss(label_smoothing=0.1)

        l1 = loss_no_smooth.forward(logits, targets)
        l2 = loss_smooth.forward(logits, targets)

        # Smoothed loss should be higher (more uncertain)
        assert l2 > l1, "Smoothed loss should be higher"
        print("  PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"  FAILED: {e}")
        tests_failed += 1

    # Test 8: Ignore index
    print("\nTest 8: Ignore Index")
    try:
        logits = np.array([[1.0, 2.0], [1.0, 2.0]])
        targets = np.array([0, -100])

        loss_fn = SoftmaxCrossEntropyLoss(ignore_index=-100)
        loss = loss_fn.forward(logits, targets)
        grad = loss_fn.backward()

        # Second sample should have zero gradient
        assert np.allclose(grad[1], 0), "Ignored sample should have zero gradient"
        print("  PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"  FAILED: {e}")
        tests_failed += 1

    # Test 9: Binary cross-entropy
    print("\nTest 9: Binary Cross-Entropy")
    try:
        probs = np.array([0.9, 0.1, 0.8])
        targets = np.array([1, 0, 1])

        loss = binary_cross_entropy(probs, targets)

        # Manual calculation
        expected = -np.mean([
            np.log(0.9),      # target=1, pred=0.9
            np.log(0.9),      # target=0, pred=0.1 -> log(1-0.1)
            np.log(0.8)       # target=1, pred=0.8
        ])

        assert np.isclose(loss, expected), f"Expected {expected}, got {loss}"
        print("  PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"  FAILED: {e}")
        tests_failed += 1

    # Test 10: PyTorch comparison
    print("\nTest 10: PyTorch Comparison")
    try:
        logits_np = np.random.randn(8, 5).astype(np.float32)
        targets_np = np.random.randint(0, 5, 8)

        # Our implementation
        loss_fn = SoftmaxCrossEntropyLoss()
        our_loss = loss_fn.forward(logits_np, targets_np)

        # PyTorch
        logits_torch = torch.tensor(logits_np)
        targets_torch = torch.tensor(targets_np, dtype=torch.long)
        pytorch_loss = F.cross_entropy(logits_torch, targets_torch).item()

        assert np.isclose(our_loss, pytorch_loss, rtol=1e-5), \
            f"Our: {our_loss}, PyTorch: {pytorch_loss}"
        print("  PASSED")
        tests_passed += 1
    except AssertionError as e:
        print(f"  FAILED: {e}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
    print("=" * 70)

    return tests_passed, tests_failed


# Run tests
if __name__ == "__main__":
    run_all_tests()
```

---

## Key Takeaways

1. **Always use numerically stable softmax** - Subtract max before exponentiating
2. **Combine softmax and cross-entropy** - Simpler gradients, better stability
3. **Use log-sum-exp trick** - Prevents overflow in log computations
4. **Understand the gradient** - dL/dz = softmax(z) - y is beautifully simple
5. **Handle edge cases** - Zero probabilities, ignore indices, label smoothing
6. **Choose the right loss** - Binary for multi-label, categorical for multi-class

---

## Interview Tips

- Start by explaining the math before coding
- Mention numerical stability immediately
- Know why combined implementation is preferred
- Be ready to derive the gradient by hand
- Understand the relationship between softmax temperature and confidence
- Know when to use binary vs categorical cross-entropy
