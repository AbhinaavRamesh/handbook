---
title: SVM Implementation
description: Implement Support Vector Machine from scratch with kernel trick and soft margin.
---

# SVM Implementation

> **Maximum margin classifier with kernel methods**

Support Vector Machines find the optimal hyperplane that maximizes the margin between classes. This guide covers the mathematical foundations and provides complete implementations.

## Mathematical Foundation

### The Classification Problem

Given training data $\{(x_i, y_i)\}_{i=1}^{n}$ where $x_i \in \mathbb{R}^d$ and $y_i \in \{-1, +1\}$, find a hyperplane that separates the classes.

**Hyperplane equation**: $w^T x + b = 0$

**Decision function**: $f(x) = \text{sign}(w^T x + b)$

### Geometric Margin

The distance from point $x_i$ to the hyperplane:

$$\gamma_i = y_i \cdot \frac{w^T x_i + b}{\|w\|}$$

The margin of the classifier is the minimum distance:

$$\gamma = \min_{i} \gamma_i$$

---

## Hard Margin SVM

### Optimization Problem

For linearly separable data, maximize the margin:

$$\max_{w,b} \frac{2}{\|w\|}$$

Subject to: $y_i(w^T x_i + b) \geq 1$ for all $i$

Equivalently (minimization form):

$$\min_{w,b} \frac{1}{2}\|w\|^2$$

Subject to: $y_i(w^T x_i + b) \geq 1$ for all $i$

### Lagrangian Formulation

$$L(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^{n} \alpha_i [y_i(w^T x_i + b) - 1]$$

**KKT Conditions**:
1. $\frac{\partial L}{\partial w} = 0 \Rightarrow w = \sum_{i=1}^{n} \alpha_i y_i x_i$
2. $\frac{\partial L}{\partial b} = 0 \Rightarrow \sum_{i=1}^{n} \alpha_i y_i = 0$
3. $\alpha_i \geq 0$
4. $\alpha_i [y_i(w^T x_i + b) - 1] = 0$ (complementary slackness)

### Dual Problem

Substituting the KKT conditions:

$$\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j x_i^T x_j$$

Subject to:
- $\alpha_i \geq 0$ for all $i$
- $\sum_{i=1}^{n} \alpha_i y_i = 0$

```python
import numpy as np
from typing import Tuple, List, Callable, Optional
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HardMarginSVM:
    """
    Hard Margin SVM using quadratic programming.
    Only works for linearly separable data.
    """

    def __init__(self, tol: float = 1e-5):
        self.tol = tol
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_labels = None
        self.alphas = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HardMarginSVM':
        """
        Fit hard margin SVM using dual formulation.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels in {-1, +1}
        """
        n_samples, n_features = X.shape

        # Compute Gram matrix
        K = np.dot(X, X.T)

        # Quadratic programming: minimize 0.5 * alpha^T @ P @ alpha - q^T @ alpha
        # P[i,j] = y_i * y_j * K[i,j]
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)

        # Constraints: sum(alpha_i * y_i) = 0, alpha_i >= 0
        def objective(alpha):
            return 0.5 * np.dot(alpha, np.dot(P, alpha)) + np.dot(q, alpha)

        def gradient(alpha):
            return np.dot(P, alpha) + q

        # Equality constraint: sum(alpha_i * y_i) = 0
        constraints = {'type': 'eq', 'fun': lambda a: np.dot(a, y)}

        # Bounds: alpha_i >= 0
        bounds = [(0, None) for _ in range(n_samples)]

        # Initial guess
        alpha0 = np.zeros(n_samples)

        # Solve
        result = minimize(objective, alpha0, jac=gradient,
                         constraints=constraints, bounds=bounds,
                         method='SLSQP')

        alphas = result.x

        # Find support vectors (alpha > tol)
        sv_mask = alphas > self.tol
        self.alphas = alphas[sv_mask]
        self.support_vectors = X[sv_mask]
        self.support_labels = y[sv_mask]

        # Compute w = sum(alpha_i * y_i * x_i)
        self.w = np.sum(self.alphas[:, np.newaxis] *
                        self.support_labels[:, np.newaxis] *
                        self.support_vectors, axis=0)

        # Compute b using any support vector
        # y_s(w^T x_s + b) = 1 => b = y_s - w^T x_s
        self.b = np.mean(self.support_labels -
                        np.dot(self.support_vectors, self.w))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.sign(np.dot(X, self.w) + self.b)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values."""
        return np.dot(X, self.w) + self.b
```

---

## Soft Margin SVM

### Handling Non-Separable Data

Introduce slack variables $\xi_i \geq 0$ to allow misclassification:

$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \xi_i$$

Subject to:
- $y_i(w^T x_i + b) \geq 1 - \xi_i$
- $\xi_i \geq 0$

### The C Parameter

| C Value | Effect |
|---------|--------|
| Large C | Hard margin (few violations) |
| Small C | Soft margin (more violations allowed) |
| C → ∞ | Equivalent to hard margin |

### Dual Formulation

$$\max_{\alpha} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j x_i^T x_j$$

Subject to:
- $0 \leq \alpha_i \leq C$ for all $i$
- $\sum_{i=1}^{n} \alpha_i y_i = 0$

The only difference from hard margin is the upper bound C on $\alpha_i$.

---

## Hinge Loss Formulation

### Unconstrained Optimization

The soft margin SVM can be written as:

$$\min_{w,b} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(w^T x_i + b))$$

### Hinge Loss Function

$$L_{\text{hinge}}(y, f(x)) = \max(0, 1 - y \cdot f(x))$$

Properties:
- Zero loss when $y \cdot f(x) \geq 1$ (correct with margin)
- Linear penalty for violations
- Non-differentiable at $y \cdot f(x) = 1$

### Subgradient

$$\frac{\partial L_{\text{hinge}}}{\partial f} = \begin{cases} 0 & \text{if } y \cdot f(x) > 1 \\ -y & \text{if } y \cdot f(x) < 1 \\ [-y, 0] & \text{if } y \cdot f(x) = 1 \end{cases}$$

```python
class SoftMarginSVM:
    """
    Soft Margin SVM using hinge loss and gradient descent.
    """

    def __init__(self, C: float = 1.0, learning_rate: float = 0.001,
                 n_iterations: int = 1000, tol: float = 1e-5):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.w = None
        self.b = None
        self.loss_history = []

    def _hinge_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute total hinge loss."""
        margins = y * (np.dot(X, self.w) + self.b)
        hinge = np.maximum(0, 1 - margins)
        return 0.5 * np.dot(self.w, self.w) + self.C * np.sum(hinge)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SoftMarginSVM':
        """
        Fit soft margin SVM using subgradient descent.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels in {-1, +1}
        """
        n_samples, n_features = X.shape

        # Initialize weights
        self.w = np.zeros(n_features)
        self.b = 0

        for iteration in range(self.n_iterations):
            # Compute margins
            margins = y * (np.dot(X, self.w) + self.b)

            # Find misclassified or margin violations
            violations = margins < 1

            # Compute gradients
            # dL/dw = w - C * sum(y_i * x_i) for violations
            dw = self.w - self.C * np.sum(
                y[violations, np.newaxis] * X[violations], axis=0
            )

            # dL/db = -C * sum(y_i) for violations
            db = -self.C * np.sum(y[violations])

            # Update with learning rate decay
            lr = self.learning_rate / (1 + iteration * 0.001)
            self.w -= lr * dw
            self.b -= lr * db

            # Record loss
            loss = self._hinge_loss(X, y)
            self.loss_history.append(loss)

            # Check convergence
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.sign(np.dot(X, self.w) + self.b)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values."""
        return np.dot(X, self.w) + self.b

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

---

## The Kernel Trick

### Motivation

Many problems are not linearly separable. Map data to higher dimensions where it becomes separable.

### Feature Mapping

Map $x \in \mathbb{R}^d$ to $\phi(x) \in \mathbb{R}^D$ where $D >> d$.

The dual formulation only uses dot products $x_i^T x_j$, which becomes $\phi(x_i)^T \phi(x_j)$.

### Kernel Function

A kernel computes the inner product in feature space without explicit mapping:

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

### Common Kernels

| Kernel | Formula | Parameters |
|--------|---------|------------|
| Linear | $K(x,y) = x^T y$ | None |
| Polynomial | $K(x,y) = (\gamma x^T y + r)^d$ | $\gamma, r, d$ |
| RBF (Gaussian) | $K(x,y) = \exp(-\gamma \|x-y\|^2)$ | $\gamma$ |
| Sigmoid | $K(x,y) = \tanh(\gamma x^T y + r)$ | $\gamma, r$ |

### RBF Kernel Deep Dive

The RBF kernel maps to infinite-dimensional space:

$$K(x, y) = \exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)$$

where $\gamma = \frac{1}{2\sigma^2}$.

**Interpretation**: Points closer in input space have kernel values closer to 1.

```python
class KernelFunctions:
    """Collection of kernel functions for SVM."""

    @staticmethod
    def linear(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Linear kernel: K(x, y) = x^T y"""
        return np.dot(x1, x2.T)

    @staticmethod
    def polynomial(x1: np.ndarray, x2: np.ndarray,
                   degree: int = 3, gamma: float = 1.0,
                   coef0: float = 1.0) -> np.ndarray:
        """Polynomial kernel: K(x, y) = (gamma * x^T y + coef0)^degree"""
        return (gamma * np.dot(x1, x2.T) + coef0) ** degree

    @staticmethod
    def rbf(x1: np.ndarray, x2: np.ndarray,
            gamma: float = 1.0) -> np.ndarray:
        """
        RBF (Gaussian) kernel: K(x, y) = exp(-gamma * ||x - y||^2)

        Handles both vector and matrix inputs.
        """
        if x1.ndim == 1:
            x1 = x1.reshape(1, -1)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)

        # Compute squared Euclidean distances
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T y
        x1_sq = np.sum(x1 ** 2, axis=1).reshape(-1, 1)
        x2_sq = np.sum(x2 ** 2, axis=1).reshape(1, -1)
        distances = x1_sq + x2_sq - 2 * np.dot(x1, x2.T)

        return np.exp(-gamma * distances)

    @staticmethod
    def sigmoid(x1: np.ndarray, x2: np.ndarray,
                gamma: float = 0.01, coef0: float = 0.0) -> np.ndarray:
        """Sigmoid kernel: K(x, y) = tanh(gamma * x^T y + coef0)"""
        return np.tanh(gamma * np.dot(x1, x2.T) + coef0)


class KernelSVM:
    """
    Kernelized SVM using dual formulation.
    """

    def __init__(self, C: float = 1.0, kernel: str = 'rbf',
                 gamma: float = 1.0, degree: int = 3, coef0: float = 1.0,
                 tol: float = 1e-5, max_iter: int = 1000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

        # Model parameters
        self.alphas = None
        self.support_vectors = None
        self.support_labels = None
        self.b = None

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between X1 and X2."""
        if self.kernel == 'linear':
            return KernelFunctions.linear(X1, X2)
        elif self.kernel == 'polynomial':
            return KernelFunctions.polynomial(X1, X2, self.degree,
                                              self.gamma, self.coef0)
        elif self.kernel == 'rbf':
            return KernelFunctions.rbf(X1, X2, self.gamma)
        elif self.kernel == 'sigmoid':
            return KernelFunctions.sigmoid(X1, X2, self.gamma, self.coef0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KernelSVM':
        """
        Fit kernelized SVM using quadratic programming.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels in {-1, +1}
        """
        n_samples = X.shape[0]

        # Compute kernel matrix
        K = self._compute_kernel(X, X)

        # Setup QP problem
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)

        def objective(alpha):
            return 0.5 * np.dot(alpha, np.dot(P, alpha)) + np.dot(q, alpha)

        def gradient(alpha):
            return np.dot(P, alpha) + q

        constraints = {'type': 'eq', 'fun': lambda a: np.dot(a, y)}
        bounds = [(0, self.C) for _ in range(n_samples)]
        alpha0 = np.zeros(n_samples)

        result = minimize(objective, alpha0, jac=gradient,
                         constraints=constraints, bounds=bounds,
                         method='SLSQP', options={'maxiter': self.max_iter})

        alphas = result.x

        # Find support vectors
        sv_mask = alphas > self.tol
        self.alphas = alphas[sv_mask]
        self.support_vectors = X[sv_mask]
        self.support_labels = y[sv_mask]

        # Compute bias using support vectors on margin
        # These have 0 < alpha < C
        margin_sv_mask = (alphas > self.tol) & (alphas < self.C - self.tol)

        if np.sum(margin_sv_mask) > 0:
            margin_sv = X[margin_sv_mask]
            margin_labels = y[margin_sv_mask]

            # b = y_s - sum(alpha_i * y_i * K(x_i, x_s))
            K_sv = self._compute_kernel(self.support_vectors, margin_sv)
            predictions = np.dot(self.alphas * self.support_labels, K_sv)
            self.b = np.mean(margin_labels - predictions)
        else:
            # Fallback: use all support vectors
            K_sv = self._compute_kernel(self.support_vectors,
                                        self.support_vectors)
            predictions = np.dot(self.alphas * self.support_labels, K_sv)
            self.b = np.mean(self.support_labels - predictions)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values."""
        K = self._compute_kernel(self.support_vectors, X)
        return np.dot(self.alphas * self.support_labels, K) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.sign(self.decision_function(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        return np.mean(self.predict(X) == y)
```

---

## Sequential Minimal Optimization (SMO)

### Overview

SMO solves the SVM optimization by breaking it into smallest possible subproblems: optimizing two $\alpha$ values at a time.

### Why Two at a Time?

The constraint $\sum_i \alpha_i y_i = 0$ means we cannot change just one $\alpha$.

With two variables $\alpha_1, \alpha_2$:
$$\alpha_1 y_1 + \alpha_2 y_2 = \text{constant}$$

### The Two-Variable Subproblem

Given old values $\alpha_1^{\text{old}}, \alpha_2^{\text{old}}$, find new values.

**Bounds on $\alpha_2$**:
- If $y_1 \neq y_2$: $L = \max(0, \alpha_2^{\text{old}} - \alpha_1^{\text{old}})$, $H = \min(C, C + \alpha_2^{\text{old}} - \alpha_1^{\text{old}})$
- If $y_1 = y_2$: $L = \max(0, \alpha_1^{\text{old}} + \alpha_2^{\text{old}} - C)$, $H = \min(C, \alpha_1^{\text{old}} + \alpha_2^{\text{old}})$

**Update formula**:
$$\alpha_2^{\text{new}} = \alpha_2^{\text{old}} + \frac{y_2(E_1 - E_2)}{\eta}$$

where:
- $E_i = f(x_i) - y_i$ (prediction error)
- $\eta = 2K(x_1, x_2) - K(x_1, x_1) - K(x_2, x_2)$

Clip to bounds: $\alpha_2^{\text{new}} = \text{clip}(\alpha_2^{\text{new}}, L, H)$

Update $\alpha_1$:
$$\alpha_1^{\text{new}} = \alpha_1^{\text{old}} + y_1 y_2 (\alpha_2^{\text{old}} - \alpha_2^{\text{new}})$$

```python
class SMO_SVM:
    """
    Support Vector Machine using Sequential Minimal Optimization.

    SMO breaks the large QP problem into small 2-variable subproblems.
    """

    def __init__(self, C: float = 1.0, kernel: str = 'rbf',
                 gamma: float = 1.0, degree: int = 3, coef0: float = 1.0,
                 tol: float = 1e-3, max_passes: int = 100):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes

        # Model state
        self.alphas = None
        self.b = 0
        self.X = None
        self.y = None
        self.K = None  # Cached kernel matrix
        self.errors = None  # Cached errors

    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute full kernel matrix."""
        if self.kernel == 'linear':
            return KernelFunctions.linear(X, X)
        elif self.kernel == 'rbf':
            return KernelFunctions.rbf(X, X, self.gamma)
        elif self.kernel == 'polynomial':
            return KernelFunctions.polynomial(X, X, self.degree,
                                              self.gamma, self.coef0)

    def _decision_function_single(self, i: int) -> float:
        """Compute f(x_i) = sum(alpha_j * y_j * K(x_j, x_i)) + b"""
        return np.dot(self.alphas * self.y, self.K[:, i]) + self.b

    def _compute_error(self, i: int) -> float:
        """Compute E_i = f(x_i) - y_i"""
        return self._decision_function_single(i) - self.y[i]

    def _take_step(self, i1: int, i2: int) -> bool:
        """
        Optimize alpha[i1] and alpha[i2].
        Returns True if optimization made progress.
        """
        if i1 == i2:
            return False

        alpha1_old = self.alphas[i1]
        alpha2_old = self.alphas[i2]
        y1, y2 = self.y[i1], self.y[i2]
        E1 = self.errors[i1]
        E2 = self.errors[i2]

        s = y1 * y2

        # Compute bounds L and H
        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha1_old + alpha2_old - self.C)
            H = min(self.C, alpha1_old + alpha2_old)

        if L >= H:
            return False

        # Compute eta
        k11 = self.K[i1, i1]
        k12 = self.K[i1, i2]
        k22 = self.K[i2, i2]
        eta = 2 * k12 - k11 - k22

        if eta < 0:
            # Standard case: compute new alpha2
            alpha2_new = alpha2_old - y2 * (E1 - E2) / eta

            # Clip to bounds
            if alpha2_new > H:
                alpha2_new = H
            elif alpha2_new < L:
                alpha2_new = L
        else:
            # Rare case: evaluate objective at bounds
            # (handling edge cases in numerical stability)
            return False

        # Check if change is significant
        if abs(alpha2_new - alpha2_old) < self.tol * (alpha2_new + alpha2_old + self.tol):
            return False

        # Compute new alpha1
        alpha1_new = alpha1_old + s * (alpha2_old - alpha2_new)

        # Update bias
        b1 = E1 + y1 * (alpha1_new - alpha1_old) * k11 + \
             y2 * (alpha2_new - alpha2_old) * k12 + self.b
        b2 = E2 + y1 * (alpha1_new - alpha1_old) * k12 + \
             y2 * (alpha2_new - alpha2_old) * k22 + self.b

        if 0 < alpha1_new < self.C:
            self.b = b1
        elif 0 < alpha2_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        # Update alphas
        self.alphas[i1] = alpha1_new
        self.alphas[i2] = alpha2_new

        # Update error cache
        self.errors[i1] = self._compute_error(i1)
        self.errors[i2] = self._compute_error(i2)

        return True

    def _examine_example(self, i2: int) -> int:
        """
        Try to find a good i1 to pair with i2.
        Returns 1 if optimization succeeded, 0 otherwise.
        """
        y2 = self.y[i2]
        alpha2 = self.alphas[i2]
        E2 = self.errors[i2]
        r2 = E2 * y2

        # Check if alpha2 violates KKT conditions
        if (r2 < -self.tol and alpha2 < self.C) or \
           (r2 > self.tol and alpha2 > 0):

            # Get indices of non-bound alphas
            non_bound = np.where((self.alphas > 0) & (self.alphas < self.C))[0]

            if len(non_bound) > 1:
                # Second choice heuristic: maximize |E1 - E2|
                if E2 > 0:
                    i1 = np.argmin(self.errors)
                else:
                    i1 = np.argmax(self.errors)

                if self._take_step(i1, i2):
                    return 1

            # Loop over non-bound alphas
            for i1 in np.random.permutation(non_bound):
                if self._take_step(i1, i2):
                    return 1

            # Loop over all alphas
            for i1 in np.random.permutation(len(self.alphas)):
                if self._take_step(i1, i2):
                    return 1

        return 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SMO_SVM':
        """
        Fit SVM using SMO algorithm.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels in {-1, +1}
        """
        n_samples = X.shape[0]

        self.X = X
        self.y = y
        self.alphas = np.zeros(n_samples)
        self.b = 0

        # Cache kernel matrix (for small datasets)
        self.K = self._compute_kernel_matrix(X)

        # Initialize error cache
        self.errors = -y.astype(float)  # E_i = f(x_i) - y_i = -y_i initially

        num_changed = 0
        examine_all = True
        passes = 0

        while (num_changed > 0 or examine_all) and passes < self.max_passes:
            num_changed = 0

            if examine_all:
                # Loop over all training examples
                for i in range(n_samples):
                    num_changed += self._examine_example(i)
            else:
                # Loop over non-bound examples
                non_bound = np.where((self.alphas > 0) &
                                    (self.alphas < self.C))[0]
                for i in non_bound:
                    num_changed += self._examine_example(i)

            passes += 1

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

        # Store support vectors
        sv_mask = self.alphas > 1e-7
        self.support_vectors = X[sv_mask]
        self.support_labels = y[sv_mask]
        self.support_alphas = self.alphas[sv_mask]

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function for new samples."""
        if self.kernel == 'rbf':
            K = KernelFunctions.rbf(self.support_vectors, X, self.gamma)
        elif self.kernel == 'linear':
            K = KernelFunctions.linear(self.support_vectors, X)
        elif self.kernel == 'polynomial':
            K = KernelFunctions.polynomial(self.support_vectors, X,
                                           self.degree, self.gamma, self.coef0)

        return np.dot(self.support_alphas * self.support_labels, K) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.sign(self.decision_function(X))
```

---

## Multi-Class Strategies

SVMs are inherently binary classifiers. For multi-class problems:

### One-vs-Rest (OvR)

Train K binary classifiers, one for each class vs. all others.

**Prediction**: Choose class with highest decision value.

### One-vs-One (OvO)

Train $\binom{K}{2} = \frac{K(K-1)}{2}$ binary classifiers for each pair.

**Prediction**: Voting - each classifier votes for one class.

```python
class MultiClassSVM:
    """
    Multi-class SVM using One-vs-Rest or One-vs-One strategies.
    """

    def __init__(self, strategy: str = 'ovr', C: float = 1.0,
                 kernel: str = 'rbf', gamma: float = 1.0):
        self.strategy = strategy
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.classifiers = {}
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiClassSVM':
        """
        Fit multi-class SVM.

        Args:
            X: Training features
            y: Training labels (any integer labels)
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        if self.strategy == 'ovr':
            # One-vs-Rest: K classifiers
            for cls in self.classes:
                # Create binary labels: +1 for this class, -1 for others
                y_binary = np.where(y == cls, 1, -1)

                clf = KernelSVM(C=self.C, kernel=self.kernel, gamma=self.gamma)
                clf.fit(X, y_binary)
                self.classifiers[cls] = clf

        elif self.strategy == 'ovo':
            # One-vs-One: K(K-1)/2 classifiers
            for i, cls1 in enumerate(self.classes):
                for cls2 in self.classes[i+1:]:
                    # Get samples from both classes
                    mask = (y == cls1) | (y == cls2)
                    X_pair = X[mask]
                    y_pair = np.where(y[mask] == cls1, 1, -1)

                    clf = KernelSVM(C=self.C, kernel=self.kernel, gamma=self.gamma)
                    clf.fit(X_pair, y_pair)
                    self.classifiers[(cls1, cls2)] = clf

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        n_samples = X.shape[0]

        if self.strategy == 'ovr':
            # Get decision values from each classifier
            decision_values = np.zeros((n_samples, len(self.classes)))

            for i, cls in enumerate(self.classes):
                decision_values[:, i] = self.classifiers[cls].decision_function(X)

            # Choose class with highest decision value
            return self.classes[np.argmax(decision_values, axis=1)]

        elif self.strategy == 'ovo':
            # Voting
            votes = np.zeros((n_samples, len(self.classes)))

            for (cls1, cls2), clf in self.classifiers.items():
                predictions = clf.predict(X)

                cls1_idx = np.where(self.classes == cls1)[0][0]
                cls2_idx = np.where(self.classes == cls2)[0][0]

                votes[predictions == 1, cls1_idx] += 1
                votes[predictions == -1, cls2_idx] += 1

            return self.classes[np.argmax(votes, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        return np.mean(self.predict(X) == y)
```

---

## Complete Implementation Example

```python
def create_classification_data(n_samples: int = 200,
                               noise: float = 0.1,
                               pattern: str = 'circles') -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic classification data."""
    np.random.seed(42)

    if pattern == 'linear':
        # Linearly separable data
        X = np.random.randn(n_samples, 2)
        y = np.sign(X[:, 0] + X[:, 1] + noise * np.random.randn(n_samples))

    elif pattern == 'circles':
        # Concentric circles (not linearly separable)
        t = np.random.uniform(0, 2 * np.pi, n_samples)
        inner = n_samples // 2

        r_inner = 1 + noise * np.random.randn(inner)
        r_outer = 3 + noise * np.random.randn(n_samples - inner)

        X = np.vstack([
            np.column_stack([r_inner * np.cos(t[:inner]),
                            r_inner * np.sin(t[:inner])]),
            np.column_stack([r_outer * np.cos(t[inner:]),
                            r_outer * np.sin(t[inner:])])
        ])
        y = np.array([-1] * inner + [1] * (n_samples - inner))

    elif pattern == 'moons':
        # Two moons
        n_each = n_samples // 2
        t = np.linspace(0, np.pi, n_each)

        moon1 = np.column_stack([np.cos(t), np.sin(t)])
        moon2 = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)])

        X = np.vstack([moon1, moon2]) + noise * np.random.randn(n_samples, 2)
        y = np.array([-1] * n_each + [1] * n_each)

    # Shuffle
    perm = np.random.permutation(n_samples)
    return X[perm], y[perm]


def plot_decision_boundary(clf, X: np.ndarray, y: np.ndarray,
                          title: str = "SVM Decision Boundary"):
    """Visualize the decision boundary and support vectors."""
    plt.figure(figsize=(10, 8))

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict on mesh
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 20),
                 cmap='RdBu', alpha=0.6)
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'],
                linestyles=['--', '-', '--'], linewidths=[1.5, 2, 1.5])

    # Plot data points
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', marker='o',
                edgecolors='k', s=50, label='Class -1')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='s',
                edgecolors='k', s=50, label='Class +1')

    # Highlight support vectors
    if hasattr(clf, 'support_vectors') and clf.support_vectors is not None:
        plt.scatter(clf.support_vectors[:, 0], clf.support_vectors[:, 1],
                   s=200, facecolors='none', edgecolors='green', linewidths=2,
                   label=f'Support Vectors ({len(clf.support_vectors)})')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.colorbar(label='Decision Function')
    plt.tight_layout()
    plt.show()


def compare_kernels(X: np.ndarray, y: np.ndarray):
    """Compare different kernel functions."""
    kernels = ['linear', 'polynomial', 'rbf']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, kernel in zip(axes, kernels):
        clf = KernelSVM(C=1.0, kernel=kernel, gamma=1.0, degree=3)
        clf.fit(X, y)

        # Create mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
        ax.scatter(X[y == -1, 0], X[y == -1, 1], c='blue', edgecolors='k')
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', edgecolors='k')
        ax.set_title(f'{kernel.upper()} Kernel\nAccuracy: {clf.score(X, y):.2%}')

    plt.tight_layout()
    plt.show()


# Main demonstration
if __name__ == "__main__":
    # Generate data
    X, y = create_classification_data(n_samples=200, noise=0.3, pattern='circles')

    # Split data
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print("=" * 60)
    print("SVM IMPLEMENTATION DEMONSTRATION")
    print("=" * 60)

    # Test Soft Margin SVM with gradient descent
    print("\n1. Soft Margin SVM (Gradient Descent):")
    svm_soft = SoftMarginSVM(C=1.0, learning_rate=0.01, n_iterations=1000)
    svm_soft.fit(X_train, y_train)
    print(f"   Training Accuracy: {svm_soft.score(X_train, y_train):.2%}")
    print(f"   Test Accuracy: {svm_soft.score(X_test, y_test):.2%}")

    # Test Kernel SVM
    print("\n2. Kernel SVM (RBF):")
    svm_kernel = KernelSVM(C=1.0, kernel='rbf', gamma=0.5)
    svm_kernel.fit(X_train, y_train)
    print(f"   Training Accuracy: {svm_kernel.score(X_train, y_train):.2%}")
    print(f"   Test Accuracy: {svm_kernel.score(X_test, y_test):.2%}")
    print(f"   Number of Support Vectors: {len(svm_kernel.support_vectors)}")

    # Test SMO
    print("\n3. SMO Algorithm (RBF):")
    svm_smo = SMO_SVM(C=1.0, kernel='rbf', gamma=0.5)
    svm_smo.fit(X_train, y_train)
    print(f"   Training Accuracy: {np.mean(svm_smo.predict(X_train) == y_train):.2%}")
    print(f"   Test Accuracy: {np.mean(svm_smo.predict(X_test) == y_test):.2%}")

    # Multi-class example
    print("\n4. Multi-Class SVM:")
    # Create 3-class problem
    X_multi = np.vstack([
        np.random.randn(50, 2) + [0, 0],
        np.random.randn(50, 2) + [3, 0],
        np.random.randn(50, 2) + [1.5, 2.5]
    ])
    y_multi = np.array([0] * 50 + [1] * 50 + [2] * 50)

    mc_svm = MultiClassSVM(strategy='ovr', C=1.0, kernel='rbf', gamma=0.5)
    mc_svm.fit(X_multi, y_multi)
    print(f"   OvR Accuracy: {mc_svm.score(X_multi, y_multi):.2%}")

    mc_svm_ovo = MultiClassSVM(strategy='ovo', C=1.0, kernel='rbf', gamma=0.5)
    mc_svm_ovo.fit(X_multi, y_multi)
    print(f"   OvO Accuracy: {mc_svm_ovo.score(X_multi, y_multi):.2%}")
```

---

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Effect |
|-----------|--------|
| C | Trade-off between margin and violations |
| gamma (RBF) | Kernel bandwidth - locality of influence |
| degree (Poly) | Complexity of decision boundary |

### Grid Search

```python
def grid_search_svm(X: np.ndarray, y: np.ndarray,
                    C_values: List[float], gamma_values: List[float],
                    n_folds: int = 5) -> Tuple[float, float, float]:
    """Find best hyperparameters using cross-validation."""
    n_samples = len(X)
    fold_size = n_samples // n_folds

    best_score = 0
    best_C, best_gamma = C_values[0], gamma_values[0]

    for C in C_values:
        for gamma in gamma_values:
            scores = []

            for fold in range(n_folds):
                # Create validation fold
                val_start = fold * fold_size
                val_end = val_start + fold_size

                X_val = X[val_start:val_end]
                y_val = y[val_start:val_end]
                X_train = np.vstack([X[:val_start], X[val_end:]])
                y_train = np.concatenate([y[:val_start], y[val_end:]])

                # Train and evaluate
                clf = KernelSVM(C=C, kernel='rbf', gamma=gamma)
                clf.fit(X_train, y_train)
                scores.append(clf.score(X_val, y_val))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_C = C
                best_gamma = gamma

    return best_C, best_gamma, best_score
```

---

## Summary

| Aspect | Key Points |
|--------|------------|
| **Hard Margin** | For linearly separable data, finds maximum margin hyperplane |
| **Soft Margin** | Introduces slack variables $\xi$ and regularization $C$ |
| **Hinge Loss** | $\max(0, 1 - y \cdot f(x))$ - enables gradient-based optimization |
| **Kernel Trick** | Computes inner products in high-dimensional space efficiently |
| **SMO** | Decomposes QP into 2-variable subproblems |
| **Multi-class** | OvR (K classifiers) or OvO (K(K-1)/2 classifiers) |

**Computational Complexity**:
- Training: $O(n^2)$ to $O(n^3)$ depending on algorithm
- Prediction: $O(n_{sv} \cdot d)$ where $n_{sv}$ is number of support vectors

**When to Use SVM**:
- High-dimensional data (text classification)
- Clear margin of separation
- When you need sparse solutions (only support vectors matter)
- Small to medium-sized datasets
