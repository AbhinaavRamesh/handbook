# K-Nearest Neighbors (KNN)

> **Instance-based learning** - distance metrics, k selection, curse of dimensionality

---

## Overview

KNN predicts by finding the $k$ training points closest to the query point and aggregating their labels (majority vote for classification, mean for regression).

### Algorithm Steps

1. **Store** all training data (lazy learning - no training phase)
2. **Compute** distance to all training points for new query
3. **Find** $k$ nearest neighbors
4. **Aggregate**: majority vote (classification) or mean (regression)

---

## How K Affects Predictions

![Effect of k on KNN predictions](/images/knn-k-effect.svg)

| k Value | Behavior | Trade-off |
|---------|----------|-----------|
| $k = 1$ | Highly flexible, captures noise | High variance, overfitting |
| $k$ small | Complex decision boundary | May overfit |
| $k$ large | Smooth decision boundary | May underfit |
| $k = n$ | Always predicts majority class | No learning |

---

## Decision Boundaries

![KNN Decision Boundaries for Different k Values](/images/knn-decision-boundary.svg)

- **Small k**: Jagged boundaries, sensitive to individual points
- **Large k**: Smoother boundaries, more robust to noise
- **Optimal k**: Found via cross-validation

---

## Distance Metrics

![Comparison of Euclidean vs Manhattan Distance](/images/knn-distance-metrics.svg)

### Common Metrics

| Metric | Formula | Best For |
|--------|---------|----------|
| **Euclidean (L2)** | $d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$ | Continuous features (default) |
| **Manhattan (L1)** | $d(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n}\|x_i - y_i\|$ | High-dimensional, robust to outliers |
| **Cosine** | $d(\mathbf{x}, \mathbf{y}) = 1 - \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$ | Text/documents, sparse vectors |
| **Minkowski** | $d(\mathbf{x}, \mathbf{y}) = \left(\sum_{i=1}^{n}\|x_i - y_i\|^p\right)^{1/p}$ | Generalized ($p=1$: Manhattan, $p=2$: Euclidean) |

> **Important**: Always normalize/scale features before using Euclidean distance!

---

## Choosing K

### Selection Methods

| Method | Description |
|--------|-------------|
| **Cross-validation** | Most reliable - test multiple k values |
| **Rule of thumb** | $k = \sqrt{n}$ as starting point |
| **Odd k for binary** | Avoids ties in voting |

### Weighted KNN

Weight neighbors by inverse distance to reduce sensitivity to k choice:

$$w_i = \frac{1}{d_i}$$

```python
# Weighted KNN in sklearn
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
```

---

## Curse of Dimensionality

### The Problem

In high dimensions ($d > 20$):
- All points become approximately **equidistant**
- "Nearest" neighbor isn't meaningfully closer
- Volume concentrates at hypercube edges

### Volume Example

For a unit hypercube with inner cube of side 0.9:

| Dimensions | Inner Volume |
|------------|--------------|
| $d = 1$ | 90% |
| $d = 10$ | 35% |
| $d = 100$ | ~0% |

### Solutions

- **Dimensionality reduction**: PCA, t-SNE before KNN
- **Feature selection**: Remove irrelevant features
- **Manhattan distance**: More robust in high-d
- **Approximate NN**: LSH, Annoy, Faiss

---

## Complexity

| Operation | Brute Force | With KD-tree |
|-----------|-------------|--------------|
| Training | $O(1)$ | $O(n \log n)$ |
| Prediction | $O(nd)$ | $O(d \log n)$ avg |
| Space | $O(nd)$ | $O(nd)$ |

### Speed Optimizations

| Method | When to Use |
|--------|-------------|
| **KD-Tree** | $d < 20$, exact NN |
| **Ball Tree** | Higher dimensions than KD-tree |
| **LSH** | Approximate NN, very large datasets |
| **Annoy/Faiss** | Production systems, millions of points |

---

## Strengths vs Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Simple, no training phase | Slow prediction $O(nd)$ |
| No distribution assumptions | Memory intensive |
| Naturally handles multi-class | Curse of dimensionality |
| Easy to add new data | Sensitive to feature scale |
| Adapts locally | All features affect distance |

---

## Code Reference

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ALWAYS scale features for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Basic KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)

# Weighted KNN (recommended)
knn_weighted = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='euclidean'
)

# Find optimal k via cross-validation
k_range = range(1, 31)
cv_scores = [
    cross_val_score(
        KNeighborsClassifier(n_neighbors=k),
        X_train_scaled, y_train, cv=5
    ).mean()
    for k in k_range
]
best_k = k_range[np.argmax(cv_scores)]

# Fast queries with KD-tree
knn_fast = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='kd_tree',  # 'ball_tree', 'auto'
    leaf_size=30
)

# Get neighbors for interpretability
distances, indices = knn.kneighbors(X_test_scaled[:1], n_neighbors=5)
```

---

## Interview Questions

### Q1: How do you choose k?

> Use **cross-validation** to find optimal k. Start with $k = \sqrt{n}$, use odd k for binary classification. Consider **weighted KNN** to reduce sensitivity to k choice.

### Q2: What's the curse of dimensionality?

> In high dimensions, all points become equidistant, making "nearest" meaningless. **Solutions**: reduce dimensions (PCA), select features, use Manhattan distance, or approximate NN methods.

### Q3: KNN vs model-based approaches?

> **Use KNN**: low dimensions ($d < 20$), complex boundaries, online learning, interpretability needed.
> **Use model-based**: high dimensions, fast prediction required, parametric assumptions hold, limited memory.

### Q4: How to speed up KNN?

> 1. **KD-tree/Ball-tree** for low-d exact NN
> 2. **LSH/Annoy/Faiss** for approximate NN at scale
> 3. **Prototype selection** to reduce dataset
> 4. **Dimensionality reduction** (PCA)

---

## Quick Reference Card

```
K-NEAREST NEIGHBORS
---------------------------------------------
Algorithm: Store data, find k nearest at prediction
Classify:  Majority vote among k neighbors
Regress:   Mean of k neighbors' values

DISTANCE METRICS
---------------------------------------------
Euclidean: sqrt(sum((xi-yi)^2))  - default
Manhattan: sum(|xi-yi|)          - robust, high-d
Cosine:    1 - cos(theta)        - text, sparse

CHOOSING K
---------------------------------------------
Small k: Complex boundary, may overfit
Large k: Smooth boundary, may underfit
Method:  Cross-validation (best), sqrt(n) (rule)

COMPLEXITY
---------------------------------------------
Training:   O(1)
Prediction: O(n*d), or O(log n) with KD-tree
Space:      O(n*d)

CURSE OF DIMENSIONALITY
---------------------------------------------
Problem: In high-d, all points equidistant
Fix:     Reduce dimensions, feature selection

USE WHEN
---------------------------------------------
+ Low-dimensional data (d < 20)
+ Complex/irregular boundaries
+ Online learning / easy updates
- High-dimensional data
- Fast prediction needed
- Limited memory
```
