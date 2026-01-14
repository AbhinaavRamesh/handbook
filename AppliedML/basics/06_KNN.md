# K-Nearest Neighbors (KNN)

> **Instance-based learning** — distance metrics, k selection, curse of dimensionality

---

## One-Sentence Description

KNN predicts by finding the k training points closest to the query point and aggregating their labels (majority vote for classification, mean for regression).

---

## Core Concept

### How It Works

1. Store all training data (lazy learning)
2. For new query point:
   - Compute distance to all training points
   - Find k nearest neighbors
   - Classification: majority vote
   - Regression: mean of neighbors' values

### Visual Example

```
Query: ?

    ●        ●
        ○
    ○   ?   ●    k=3: 2 circles, 1 triangle → predict circle
        ●        k=5: 2 circles, 3 triangles → predict triangle
    ○
```

**k matters!**

---

## Distance Metrics

### Euclidean Distance (L2)

```
d(x, y) = √(Σ(xᵢ - yᵢ)²)
```

- Most common default
- Sensitive to scale → always normalize features
- Works well for continuous features

### Manhattan Distance (L1)

```
d(x, y) = Σ|xᵢ - yᵢ|
```

- More robust to outliers
- Better for high-dimensional sparse data
- Preferred when features are on different scales

### Cosine Distance

```
d(x, y) = 1 - (x·y)/(||x|| × ||y||)
```

- Measures angle, not magnitude
- Good for text/document similarity
- Invariant to vector length

### Minkowski Distance (General)

```
d(x, y) = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)
```

- p=1: Manhattan
- p=2: Euclidean
- p→∞: Chebyshev (max difference)

### When to Use Each

| Metric | Use Case |
|--------|----------|
| Euclidean | Default for continuous features |
| Manhattan | High-dimensional, mixed scales |
| Cosine | Text, documents, sparse vectors |
| Hamming | Categorical features |

---

## Choosing K

### Effect of K

| K Value | Characteristics |
|---------|-----------------|
| K = 1 | Most flexible, high variance, sensitive to noise |
| K small | Complex decision boundary, may overfit |
| K large | Smooth decision boundary, may underfit |
| K = n | Always predict majority class |

### Selection Methods

1. **Cross-validation** — Most reliable
   ```python
   from sklearn.model_selection import cross_val_score

   k_range = range(1, 31)
   scores = []
   for k in k_range:
       knn = KNeighborsClassifier(n_neighbors=k)
       cv_scores = cross_val_score(knn, X, y, cv=5)
       scores.append(cv_scores.mean())

   best_k = k_range[np.argmax(scores)]
   ```

2. **Rule of thumb** — k = √n (for classification)

3. **Odd k for binary classification** — Avoids ties

### Weighted KNN

Instead of equal votes, weight by inverse distance:

```
weight_i = 1 / distance_i
```

Closer neighbors have more influence. Reduces sensitivity to k choice.

---

## Curse of Dimensionality

### The Problem

In high dimensions, distances become meaningless:
- All points are approximately equidistant
- "Nearest" neighbor isn't much closer than others
- Volume concentrates at edges, not center

### Why It Happens

In d dimensions, most of a hypercube's volume is near the corners/edges.

**Example**: Unit hypercube, inner cube with side 0.9
- d=1: Inner contains 90% of volume
- d=10: Inner contains 35% of volume
- d=100: Inner contains ~0% of volume

### Solutions

1. **Dimensionality reduction** — PCA, t-SNE before KNN
2. **Feature selection** — Remove irrelevant features
3. **Use Manhattan distance** — More robust in high-d
4. **Approximate nearest neighbors** — LSH, Annoy, Faiss

---

## Key Properties

### Strengths

1. **Simple** — No training phase, easy to understand
2. **No assumptions** — Works with any distribution
3. **Naturally multi-class** — No modification needed
4. **Adapts locally** — Different behavior in different regions
5. **Online learning** — Easy to add new data

### Weaknesses

1. **Slow prediction** — O(n × d) per query
2. **Memory intensive** — Stores all training data
3. **Curse of dimensionality** — Fails in high dimensions
4. **Sensitive to scale** — Must normalize features
5. **Sensitive to irrelevant features** — All features contribute to distance

---

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Training | O(1) | O(n × d) |
| Prediction | O(n × d) | O(n) |
| Prediction with KD-tree | O(d × log n) average | O(n × d) |

### Optimizations

1. **KD-Tree** — O(log n) nearest neighbor for low d
   - Works well for d < 20
   - Degrades to O(n) for high d

2. **Ball Tree** — Better for high dimensions
   - More robust than KD-tree
   - Still degrades, but later

3. **Approximate methods**
   - **LSH** (Locality Sensitive Hashing)
   - **Annoy** (Approximate Nearest Neighbors Oh Yeah)
   - **Faiss** (Facebook AI Similarity Search)

---

## Interview Questions

### Q1: "How do you choose k in KNN?"

**Strong answer**:
> "I'd use cross-validation to find the k that minimizes validation error.
>
> In practice:
> - Start with k = √n as a baseline
> - Use odd k for binary classification to avoid ties
> - Search over a range (say 1-30) and pick the best
>
> The optimal k depends on the dataset:
> - Noisy data → larger k smooths out noise
> - Complex boundaries → smaller k captures detail
>
> I'd also consider weighted KNN where neighbors are weighted by inverse distance — this makes the choice of k less critical."

### Q2: "What's the curse of dimensionality and how does it affect KNN?"

**Strong answer**:
> "In high dimensions, distances become meaningless — all points are approximately equidistant. This is the curse of dimensionality.
>
> For KNN specifically:
> - The 'nearest' neighbor isn't meaningfully closer than others
> - The concept of locality breaks down
> - More data is needed to cover the space (exponential in d)
>
> To mitigate:
> 1. **Reduce dimensions** — PCA, feature selection before KNN
> 2. **Use Manhattan distance** — More robust than Euclidean in high-d
> 3. **Feature engineering** — Create meaningful combined features
> 4. **Consider other algorithms** — Trees, linear models less affected"

### Q3: "When would you use KNN vs. a model-based approach?"

**Strong answer**:
> "KNN is good when:
> - **No clear model assumption** — Complex, irregular decision boundaries
> - **Data is low-dimensional** — d < 20 typically
> - **Online learning** — Easy to add new points
> - **Interpretability** — 'Similar to cases X, Y, Z'
>
> Model-based (like logistic regression) is better when:
> - **High-dimensional data** — Curse of dimensionality hurts KNN
> - **Fast prediction needed** — KNN is O(n) per query
> - **Parametric assumptions hold** — Linear boundaries work
> - **Limited memory** — KNN stores all training data
>
> KNN is often a good baseline to try first, but for production systems with latency requirements, I'd usually prefer a model-based approach."

### Q4: "How would you speed up KNN for large datasets?"

**Strong answer**:
> "Several approaches:
>
> 1. **Data structures for search**
>    - KD-tree: O(log n) for low dimensions (d < 20)
>    - Ball tree: Better for higher dimensions
>
> 2. **Approximate nearest neighbors**
>    - LSH: Hash similar points to same buckets
>    - Annoy/Faiss: Trade accuracy for speed
>
> 3. **Reduce dataset size**
>    - Prototype selection: Keep only representative points
>    - Condensed nearest neighbors: Remove redundant points
>
> 4. **Reduce dimensions**
>    - PCA before KNN
>    - Random projections
>
> For a production system, I'd likely use Faiss or a similar library for approximate nearest neighbors — they can handle millions of points with sub-millisecond queries."

---

## Code Reference

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Always scale features for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Basic KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)

# Weighted KNN
knn_weighted = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',  # Weight by inverse distance
    metric='euclidean'
)

# Find optimal k with cross-validation
k_range = range(1, 31)
cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())

best_k = k_range[np.argmax(cv_scores)]
print(f"Best k: {best_k}")

# With KD-tree for faster queries
knn_fast = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='kd_tree',  # or 'ball_tree', 'auto'
    leaf_size=30
)

# Get neighbors (for explanation)
distances, indices = knn.kneighbors(X_test_scaled[:1], n_neighbors=5)
print(f"Nearest neighbors: {indices}")
print(f"Distances: {distances}")
```

---

## Quick Reference Card

```
K-NEAREST NEIGHBORS
─────────────────────────────────────────────────
Algorithm: Store data, find k nearest at prediction
Classify:  Majority vote among k neighbors
Regress:   Mean of k neighbors' values

DISTANCE METRICS
─────────────────────────────────────────────────
Euclidean: √Σ(xᵢ-yᵢ)²  — default, continuous features
Manhattan: Σ|xᵢ-yᵢ|    — robust, high-dimensional
Cosine:    1 - cos(θ)   — text, sparse data

CHOOSING K
─────────────────────────────────────────────────
Small k: Complex boundary, may overfit
Large k: Smooth boundary, may underfit
Method:  Cross-validation (best), √n (rule of thumb)

COMPLEXITY
─────────────────────────────────────────────────
Training:   O(1)
Prediction: O(n × d), or O(log n) with KD-tree
Space:      O(n × d)

CURSE OF DIMENSIONALITY
─────────────────────────────────────────────────
Problem: In high-d, all points equidistant
Fix:     Reduce dimensions, feature selection

USE WHEN
─────────────────────────────────────────────────
✓ Low-dimensional data (d < 20)
✓ Complex/irregular boundaries
✓ Online learning / easy updates
✗ High-dimensional data
✗ Fast prediction needed
✗ Limited memory
```

---

**Previous**: [← 05_SVM](./05_SVM.md) | **Next**: [07_Neural_Networks →](./07_Neural_Networks.md)
