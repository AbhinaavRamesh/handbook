# ML Coding Quick Reference

> **Code templates, complexity analysis, and common patterns**

---

## Algorithm Complexity Summary

| Algorithm | Training | Prediction | Space |
|-----------|----------|------------|-------|
| KNN | O(1) | O(n*d) | O(n*d) |
| K-Means | O(i*n*k*d) | O(k*d) | O(n*d + k*d) |
| Linear Regression | O(n*d^2) | O(d) | O(d^2) |
| Logistic Regression | O(i*n*d) | O(d) | O(d) |
| Decision Tree | O(n*d*log n) | O(log n) | O(nodes) |
| Random Forest | O(t*n*d*log n) | O(t*log n) | O(t*nodes) |
| Naive Bayes | O(n*d) | O(d) | O(d) |
| SVM (kernel) | O(n^2*d) to O(n^3) | O(sv*d) | O(sv*d) |

**Legend**: n=samples, d=features, k=clusters, i=iterations, t=trees, sv=support vectors

---

## Distance Metrics

```python
import numpy as np

# Euclidean (L2)
def euclidean(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Manhattan (L1)
def manhattan(x1, x2):
    return np.sum(np.abs(x1 - x2))

# Cosine Distance
def cosine_distance(x1, x2):
    similarity = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-8)
    return 1 - similarity

# Minkowski (general)
def minkowski(x1, x2, p=2):
    return np.power(np.sum(np.abs(x1 - x2) ** p), 1/p)

# Vectorized pairwise Euclidean
def pairwise_euclidean(X, Y):
    """Distance matrix between all pairs in X and Y."""
    X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_sq = np.sum(Y ** 2, axis=1).reshape(1, -1)
    return np.sqrt(np.maximum(X_sq + Y_sq - 2 * np.dot(X, Y.T), 0))
```

---

## Evaluation Metrics

```python
import numpy as np

# === Classification ===

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred, positive_class=1):
    tp = np.sum((y_pred == positive_class) & (y_true == positive_class))
    fp = np.sum((y_pred == positive_class) & (y_true != positive_class))
    return tp / (tp + fp + 1e-8)

def recall(y_true, y_pred, positive_class=1):
    tp = np.sum((y_pred == positive_class) & (y_true == positive_class))
    fn = np.sum((y_pred != positive_class) & (y_true == positive_class))
    return tp / (tp + fn + 1e-8)

def f1_score(y_true, y_pred, positive_class=1):
    p = precision(y_true, y_pred, positive_class)
    r = recall(y_true, y_pred, positive_class)
    return 2 * p * r / (p + r + 1e-8)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_cls) & (y_pred == pred_cls))
    return matrix

# === Regression ===

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)

# === Ranking ===

def ndcg_at_k(relevance_scores, k):
    """Normalized Discounted Cumulative Gain."""
    def dcg(scores, k):
        scores = np.array(scores)[:k]
        gains = 2 ** scores - 1
        discounts = np.log2(np.arange(len(scores)) + 2)
        return np.sum(gains / discounts)

    actual_dcg = dcg(relevance_scores, k)
    ideal_dcg = dcg(sorted(relevance_scores, reverse=True), k)
    return actual_dcg / (ideal_dcg + 1e-8)
```

---

## Activation Functions

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2
```

---

## Loss Functions

```python
import numpy as np

def binary_cross_entropy(y_true, y_pred, epsilon=1e-8):
    """Binary classification loss."""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred, epsilon=1e-8):
    """Multi-class classification loss (y_true is one-hot)."""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

def mean_squared_error(y_true, y_pred):
    """Regression loss."""
    return np.mean((y_true - y_pred) ** 2)

def hinge_loss(y_true, y_pred):
    """SVM loss (y_true should be -1 or 1)."""
    return np.mean(np.maximum(0, 1 - y_true * y_pred))
```

---

## Data Preprocessing

```python
import numpy as np

# === Scaling ===

def standardize(X, mean=None, std=None):
    """Z-score normalization: (x - mean) / std"""
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0) + 1e-8
    return (X - mean) / std, mean, std

def min_max_scale(X, min_val=None, max_val=None, feature_range=(0, 1)):
    """Scale to [0, 1] or custom range."""
    if min_val is None:
        min_val = np.min(X, axis=0)
    if max_val is None:
        max_val = np.max(X, axis=0)
    X_scaled = (X - min_val) / (max_val - min_val + 1e-8)
    return X_scaled * (feature_range[1] - feature_range[0]) + feature_range[0]

def normalize_l2(X):
    """L2 normalize each sample."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norms

# === Encoding ===

def one_hot_encode(y, n_classes=None):
    """Convert integer labels to one-hot vectors."""
    if n_classes is None:
        n_classes = np.max(y) + 1
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def label_encode(categories):
    """Convert categories to integer labels."""
    unique = np.unique(categories)
    mapping = {cat: i for i, cat in enumerate(unique)}
    return np.array([mapping[cat] for cat in categories]), mapping
```

---

## Gradient Descent Variants

```python
import numpy as np

class SGD:
    """Stochastic Gradient Descent with momentum."""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        self.velocity = self.momentum * self.velocity - self.lr * grads
        return params + self.velocity


class Adam:
    """Adam optimizer."""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

---

## Linear Models from Scratch

```python
import numpy as np

class LinearRegressionGD:
    """Linear Regression with Gradient Descent."""
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


class LogisticRegressionGD:
    """Logistic Regression with Gradient Descent."""
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = 1 / (1 + np.exp(-linear))

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return 1 / (1 + np.exp(-linear))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

---

## NumPy Essentials

```python
import numpy as np

# === Array Creation ===
np.zeros((m, n))          # m x n zeros
np.ones((m, n))           # m x n ones
np.eye(n)                 # n x n identity
np.random.rand(m, n)      # Uniform [0, 1)
np.random.randn(m, n)     # Normal(0, 1)
np.arange(start, stop, step)
np.linspace(start, stop, num)

# === Shape Manipulation ===
x.reshape(new_shape)      # Reshape
x.flatten()               # To 1D
x.T                       # Transpose
x[:, np.newaxis]          # Add axis
x.squeeze()               # Remove size-1 axes

# === Aggregation ===
np.sum(x, axis=0)         # Sum along axis
np.mean(x, axis=1)        # Mean along axis
np.max(x, axis=0)         # Max along axis
np.argmax(x, axis=1)      # Index of max
np.cumsum(x)              # Cumulative sum

# === Broadcasting ===
# (m, n) + (n,) -> (m, n)   # Row-wise
# (m, 1) + (n,) -> (m, n)   # Outer product
# (m, n) + (1, n) -> (m, n) # Column-wise

# === Selection ===
x[x > 0]                  # Boolean indexing
np.where(x > 0, x, 0)     # Conditional select
np.argsort(x)[:k]         # Top-k indices
np.argpartition(x, k)[:k] # Faster top-k (unsorted)

# === Linear Algebra ===
np.dot(A, B)              # Matrix multiply
A @ B                     # Matrix multiply (Python 3.5+)
np.linalg.norm(x)         # L2 norm
np.linalg.inv(A)          # Inverse
np.linalg.eig(A)          # Eigenvalues
```

---

## Interview Coding Checklist

### Before Coding
- [ ] Clarify input/output format
- [ ] Ask about edge cases
- [ ] Confirm library constraints
- [ ] State your approach verbally

### While Coding
- [ ] Use meaningful variable names
- [ ] Add brief comments for complex logic
- [ ] Handle basic edge cases
- [ ] Test with simple example mentally

### After Coding
- [ ] Walk through an example
- [ ] State time and space complexity
- [ ] Discuss potential optimizations
- [ ] Mention what you'd add in production

### Common Mistakes to Avoid
- [ ] Division by zero (add epsilon)
- [ ] Index out of bounds
- [ ] Modifying input in place
- [ ] Forgetting edge cases (empty input, k > n)
- [ ] Broadcasting errors (check shapes)
