# Implement K-Nearest Neighbors (KNN)

> **From-scratch implementation** with variations and optimizations

---

## Problem Statement

Implement the K-Nearest Neighbors algorithm for classification. Given training data `X_train` with labels `y_train`, predict the label for a query point `x_query` based on the k nearest training points.

---

## Clarifying Questions to Ask

1. **Classification or regression?** (Start with classification, extend to regression)
2. **Distance metric?** (Euclidean is default, mention Manhattan, cosine as alternatives)
3. **How to handle ties?** (When k neighbors have equal votes)
4. **Libraries allowed?** (NumPy yes, sklearn no for implementation questions)
5. **k validation?** (What if k > n_samples?)

---

## Solution: Basic KNN Classifier

### Step-by-Step Implementation

```python
import numpy as np
from collections import Counter

def knn_classify(X_train, y_train, x_query, k=3):
    """
    K-Nearest Neighbors classification.

    Args:
        X_train: Training features, shape (n_samples, n_features)
        y_train: Training labels, shape (n_samples,)
        x_query: Query point, shape (n_features,)
        k: Number of neighbors

    Returns:
        Predicted class label
    """
    # Step 1: Validate inputs
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(X_train):
        raise ValueError(f"k={k} exceeds training set size={len(X_train)}")

    # Step 2: Compute distances from query to all training points
    # Using Euclidean distance: sqrt(sum((a - b)^2))
    distances = np.sqrt(np.sum((X_train - x_query) ** 2, axis=1))

    # Step 3: Find indices of k nearest neighbors
    k_nearest_indices = np.argsort(distances)[:k]

    # Step 4: Get labels of k nearest neighbors
    k_nearest_labels = y_train[k_nearest_indices]

    # Step 5: Majority vote
    label_counts = Counter(k_nearest_labels)
    prediction = label_counts.most_common(1)[0][0]

    return prediction


def knn_classify_batch(X_train, y_train, X_query, k=3):
    """
    Classify multiple query points.

    Args:
        X_train: Training features, shape (n_train, n_features)
        y_train: Training labels, shape (n_train,)
        X_query: Query points, shape (n_query, n_features)
        k: Number of neighbors

    Returns:
        Predictions, shape (n_query,)
    """
    predictions = []
    for x_query in X_query:
        pred = knn_classify(X_train, y_train, x_query, k)
        predictions.append(pred)
    return np.array(predictions)
```

---

## Walkthrough Example

```python
# Example data
X_train = np.array([
    [1, 1],   # Class 0
    [1, 2],   # Class 0
    [2, 1],   # Class 0
    [5, 5],   # Class 1
    [5, 6],   # Class 1
    [6, 5],   # Class 1
])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Query point
x_query = np.array([3, 3])
k = 3

# Step-by-step trace:
# 1. Distances: [2.83, 2.24, 2.24, 2.83, 3.61, 3.61]
# 2. Sorted indices: [1, 2, 0, 3, 4, 5]
# 3. Top 3: [1, 2, 0]
# 4. Labels: [0, 0, 0]
# 5. Majority: 0

prediction = knn_classify(X_train, y_train, x_query, k=3)
print(f"Prediction: {prediction}")  # Output: 0
```

---

## Variations

### Variation 1: KNN Regression

```python
def knn_regress(X_train, y_train, x_query, k=3):
    """
    K-Nearest Neighbors regression (average of k nearest values).
    """
    # Compute distances
    distances = np.sqrt(np.sum((X_train - x_query) ** 2, axis=1))

    # Find k nearest
    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_values = y_train[k_nearest_indices]

    # Return mean (regression) instead of mode (classification)
    return np.mean(k_nearest_values)
```

### Variation 2: Weighted KNN

```python
def knn_classify_weighted(X_train, y_train, x_query, k=3):
    """
    KNN with distance-based weighting (closer neighbors have more influence).
    """
    distances = np.sqrt(np.sum((X_train - x_query) ** 2, axis=1))
    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_labels = y_train[k_nearest_indices]
    k_nearest_distances = distances[k_nearest_indices]

    # Weight = 1 / distance (add small epsilon to avoid division by zero)
    weights = 1.0 / (k_nearest_distances + 1e-8)

    # Weighted vote
    unique_labels = np.unique(y_train)
    weighted_votes = {}

    for label in unique_labels:
        mask = k_nearest_labels == label
        weighted_votes[label] = np.sum(weights[mask])

    return max(weighted_votes, key=weighted_votes.get)
```

### Variation 3: Different Distance Metrics

```python
def compute_distance(x1, x2, metric='euclidean'):
    """
    Compute distance between two points.
    """
    if metric == 'euclidean':
        return np.sqrt(np.sum((x1 - x2) ** 2))
    elif metric == 'manhattan':
        return np.sum(np.abs(x1 - x2))
    elif metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        similarity = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return 1 - similarity
    else:
        raise ValueError(f"Unknown metric: {metric}")


def knn_classify_flexible(X_train, y_train, x_query, k=3, metric='euclidean'):
    """
    KNN with configurable distance metric.
    """
    if metric == 'euclidean':
        distances = np.sqrt(np.sum((X_train - x_query) ** 2, axis=1))
    elif metric == 'manhattan':
        distances = np.sum(np.abs(X_train - x_query), axis=1)
    elif metric == 'cosine':
        # Cosine distance
        dot_products = np.dot(X_train, x_query)
        norms_train = np.linalg.norm(X_train, axis=1)
        norm_query = np.linalg.norm(x_query)
        similarities = dot_products / (norms_train * norm_query + 1e-8)
        distances = 1 - similarities
    else:
        raise ValueError(f"Unknown metric: {metric}")

    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_labels = y_train[k_nearest_indices]

    return Counter(k_nearest_labels).most_common(1)[0][0]
```

---

## Vectorized Batch Prediction

For efficiency when predicting many points:

```python
def knn_classify_batch_vectorized(X_train, y_train, X_query, k=3):
    """
    Vectorized KNN for batch prediction.

    Uses pairwise distance matrix for efficiency.
    """
    n_train = len(X_train)
    n_query = len(X_query)

    # Compute pairwise distance matrix: shape (n_query, n_train)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    X_train_sq = np.sum(X_train ** 2, axis=1).reshape(1, -1)  # (1, n_train)
    X_query_sq = np.sum(X_query ** 2, axis=1).reshape(-1, 1)  # (n_query, 1)
    cross_term = np.dot(X_query, X_train.T)  # (n_query, n_train)

    dist_sq = X_query_sq + X_train_sq - 2 * cross_term
    distances = np.sqrt(np.maximum(dist_sq, 0))  # Numerical stability

    # Find k nearest for each query
    k_nearest_indices = np.argpartition(distances, k, axis=1)[:, :k]

    # Predict for each query
    predictions = []
    for i in range(n_query):
        k_labels = y_train[k_nearest_indices[i]]
        pred = Counter(k_labels).most_common(1)[0][0]
        predictions.append(pred)

    return np.array(predictions)
```

---

## Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Single prediction | O(n * d + n log n) | O(n) |
| Batch prediction (naive) | O(m * (n * d + n log n)) | O(n) |
| Batch prediction (vectorized) | O(m * n * d + m * n) | O(m * n) |

Where:
- n = number of training samples
- m = number of query points
- d = number of features

---

## Optimizations to Discuss

### 1. KD-Tree for Efficient Nearest Neighbor Search

```python
from scipy.spatial import KDTree

def knn_with_kdtree(X_train, y_train, X_query, k=3):
    """
    KNN using KD-Tree for O(log n) nearest neighbor queries.
    """
    tree = KDTree(X_train)

    # Query k nearest neighbors
    distances, indices = tree.query(X_query, k=k)

    # Predict for each query
    predictions = []
    for idx_list in indices:
        k_labels = y_train[idx_list]
        pred = Counter(k_labels).most_common(1)[0][0]
        predictions.append(pred)

    return np.array(predictions)
```

**When to use**: n > 10,000 and d < 20 (KD-tree degrades in high dimensions)

### 2. Ball Tree for High Dimensions

Mention as alternative when d > 20.

### 3. Approximate Nearest Neighbors

For very large datasets, mention libraries like Faiss or Annoy that trade accuracy for speed.

---

## Edge Cases to Handle

```python
def knn_classify_robust(X_train, y_train, x_query, k=3):
    """
    Robust KNN with edge case handling.
    """
    # Edge case 1: Empty training set
    if len(X_train) == 0:
        raise ValueError("Training set is empty")

    # Edge case 2: k larger than training set
    if k > len(X_train):
        print(f"Warning: k={k} > n_samples={len(X_train)}, using k={len(X_train)}")
        k = len(X_train)

    # Edge case 3: Mismatched dimensions
    if X_train.shape[1] != len(x_query):
        raise ValueError(f"Feature dimension mismatch: {X_train.shape[1]} vs {len(x_query)}")

    # Edge case 4: Handle ties (take first if tie)
    distances = np.sqrt(np.sum((X_train - x_query) ** 2, axis=1))
    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_labels = y_train[k_nearest_indices]

    label_counts = Counter(k_nearest_labels)
    # most_common handles ties by returning first
    return label_counts.most_common(1)[0][0]
```

---

## Complete Test Suite

```python
def test_knn():
    """Test cases for KNN implementation."""

    # Test 1: Basic classification
    X_train = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11], [12, 12]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    x_query = np.array([1.5, 1.5])
    assert knn_classify(X_train, y_train, x_query, k=3) == 0
    print("Test 1 passed: Basic classification")

    # Test 2: Query near class boundary
    x_query = np.array([5, 5])
    pred = knn_classify(X_train, y_train, x_query, k=3)
    print(f"Test 2: Boundary query predicted as class {pred}")

    # Test 3: k=1 (nearest neighbor)
    x_query = np.array([10.1, 10.1])
    assert knn_classify(X_train, y_train, x_query, k=1) == 1
    print("Test 3 passed: k=1 nearest neighbor")

    # Test 4: k equals dataset size
    pred = knn_classify(X_train, y_train, x_query, k=6)
    print(f"Test 4: k=n prediction: {pred}")

    print("\nAll tests passed!")

# Run tests
test_knn()
```

---

## Interview Tips

1. **Start with clarifying questions** — don't assume Euclidean distance
2. **Write clean code first** — optimize later if time permits
3. **Trace through an example** — shows you understand the algorithm
4. **Discuss complexity** — mention KD-tree optimization for large n
5. **Handle edge cases** — shows production mindset

---

**Previous**: [← 01_Coding_Overview](./01_Coding_Overview.md) | **Next**: [03_KMeans_Implementation →](./03_KMeans_Implementation.md)
