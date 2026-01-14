# ML Coding Interview Overview

> **How to approach coding questions** in Google ML interviews

---

## What to Expect

### Types of ML Coding Questions

| Type | Description | Example |
|------|-------------|---------|
| **Algorithm Implementation** | Code an ML algorithm from scratch | Implement KNN, K-Means, gradient descent |
| **Data Processing** | Transform, clean, or prepare data | Split dataset, handle missing values |
| **Model Evaluation** | Compute metrics or validation | Cross-validation, custom metrics |
| **Feature Engineering** | Extract or transform features | Text tokenization, numerical encoding |
| **Debugging** | Find and fix issues in ML code | Memory leaks, numerical instability |

### Interview Format

- **Duration**: 45-60 minutes
- **Environment**: Colab notebook, whiteboard, or shared IDE
- **Languages**: Python preferred (NumPy/Pandas allowed)
- **Goal**: Working code + explanation of choices

---

## The 5-Step Approach

### Step 1: Clarify Requirements (3-5 min)

**Questions to ask**:
- Input format and size? (numpy array, list, DataFrame?)
- Output format? (labels, probabilities, indices?)
- Edge cases to handle? (empty input, single sample?)
- Performance requirements? (time/space complexity?)
- Libraries allowed? (pure Python vs NumPy vs sklearn?)

**Example dialogue**:
> Interviewer: "Implement K-Nearest Neighbors"
>
> You: "A few clarifying questions: Should I implement for classification or regression? What distance metric - Euclidean? Should the function handle the case where k is larger than the dataset? Can I use NumPy for vectorized operations?"

### Step 2: Outline Approach (2-3 min)

Before coding, verbally outline your approach:

```
"For KNN, my approach will be:
1. Compute distances from query point to all training points
2. Find the k indices with smallest distances
3. For classification: majority vote among k neighbors
4. Return the predicted class

I'll use NumPy for vectorized distance computation to avoid O(n*d) loops."
```

### Step 3: Code Core Logic First (15-20 min)

- Start with the happy path
- Skip edge cases initially
- Use meaningful variable names
- Add brief comments for complex logic

**Good practice**:
```python
def knn_predict(X_train, y_train, x_query, k):
    # Step 1: Compute distances
    distances = np.sqrt(np.sum((X_train - x_query) ** 2, axis=1))

    # Step 2: Find k nearest indices
    k_indices = np.argsort(distances)[:k]

    # Step 3: Majority vote
    k_labels = y_train[k_indices]
    prediction = np.bincount(k_labels).argmax()

    return prediction
```

### Step 4: Handle Edge Cases (5-10 min)

Add validation and edge case handling:

```python
def knn_predict(X_train, y_train, x_query, k):
    # Validation
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(X_train):
        k = len(X_train)  # or raise error, ask interviewer

    # Core logic...
```

### Step 5: Test and Analyze (5-10 min)

- Walk through a simple example
- State time and space complexity
- Discuss potential optimizations

```
"Let me trace through with X_train = [[0,0], [1,1], [2,2]], y_train = [0, 0, 1],
x_query = [1.1, 1.1], k = 2.

Distances: [1.56, 0.14, 1.27]
Top 2 indices: [1, 2]
Labels: [0, 1]
Majority: 0 (tie broken by bincount behavior)

Time complexity: O(n*d) for distance computation + O(n log n) for sorting = O(n log n)
Space complexity: O(n) for storing distances

To optimize for large n, we could use a KD-tree for O(log n) nearest neighbor queries."
```

---

## Common Patterns

### Pattern 1: Vectorized Distance Computation

**Euclidean distance between query and all points**:
```python
# Avoid loops:
# Bad: for i in range(len(X_train)): dist[i] = np.sqrt(sum((X_train[i] - x_query)**2))

# Good: vectorized
distances = np.linalg.norm(X_train - x_query, axis=1)

# Or manually:
distances = np.sqrt(np.sum((X_train - x_query) ** 2, axis=1))
```

### Pattern 2: Pairwise Distance Matrix

**All pairwise distances between two sets**:
```python
from scipy.spatial.distance import cdist

# Shape: (n_samples_X, n_samples_Y)
dist_matrix = cdist(X, Y, metric='euclidean')

# Pure NumPy (squared Euclidean):
# ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
Y_sq = np.sum(Y ** 2, axis=1).reshape(1, -1)
dist_sq = X_sq + Y_sq - 2 * np.dot(X, Y.T)
dist_matrix = np.sqrt(np.maximum(dist_sq, 0))  # Numerical stability
```

### Pattern 3: Efficient Top-K Selection

**Get k smallest/largest without full sort**:
```python
# Full sort: O(n log n)
top_k_indices = np.argsort(distances)[:k]

# Partial sort: O(n + k log k) using argpartition
top_k_indices = np.argpartition(distances, k)[:k]
# Note: argpartition doesn't sort within top k, so add:
top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]
```

### Pattern 4: Convergence Checking

**For iterative algorithms**:
```python
def has_converged(old_centroids, new_centroids, tolerance=1e-6):
    """Check if centroids have stopped moving"""
    shift = np.sqrt(np.sum((old_centroids - new_centroids) ** 2))
    return shift < tolerance
```

### Pattern 5: Batch Processing

**Process data in batches for memory efficiency**:
```python
def predict_in_batches(X, model, batch_size=1000):
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        batch_preds = model.predict(batch)
        predictions.extend(batch_preds)
    return np.array(predictions)
```

---

## Complexity Analysis Cheat Sheet

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Distance to n points (d dims) | O(n * d) | O(n) |
| Full sort | O(n log n) | O(n) |
| Partial sort (top k) | O(n + k log k) | O(k) |
| Matrix multiply (n x d) @ (d x m) | O(n * d * m) | O(n * m) |
| K-Means (n points, k clusters, i iterations) | O(i * n * k * d) | O(n * d + k * d) |

---

## Common Pitfalls

### Pitfall 1: Numerical Instability

**Problem**: Floating point errors in distance computation
```python
# Can produce negative values due to floating point errors
dist_sq = X_sq + Y_sq - 2 * np.dot(X, Y.T)
dist = np.sqrt(dist_sq)  # NaN for negative values!

# Fix:
dist = np.sqrt(np.maximum(dist_sq, 0))
```

### Pitfall 2: In-Place Modifications

**Problem**: Modifying input arrays unexpectedly
```python
# Bad: modifies input
def normalize(X):
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

# Good: creates copy
def normalize(X):
    X = X.copy()
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X
```

### Pitfall 3: Integer Division

**Problem**: Python 2 vs 3 behavior
```python
# Always use explicit float division for means
centroid = np.sum(cluster_points, axis=0) / float(len(cluster_points))
```

### Pitfall 4: Empty Clusters

**Problem**: K-Means can have empty clusters
```python
# Handle empty cluster case
if len(cluster_points) == 0:
    # Option 1: Reinitialize randomly
    new_centroid = X[np.random.randint(len(X))]
    # Option 2: Keep old centroid
    new_centroid = old_centroids[cluster_idx]
```

### Pitfall 5: Broadcasting Errors

**Problem**: Shape mismatches in vectorized operations
```python
# Ensure shapes are explicit
X_train.shape  # (n_samples, n_features)
x_query.shape  # Should be (n_features,) or (1, n_features)

# Reshape if needed
x_query = x_query.reshape(1, -1)  # (1, n_features)
```

---

## What Interviewers Look For

| Criterion | Strong Signal | Weak Signal |
|-----------|---------------|-------------|
| **Correctness** | Code works for edge cases | Only happy path works |
| **Efficiency** | Uses vectorization, analyzes complexity | Nested loops for everything |
| **Communication** | Explains approach before coding | Silent coding |
| **Code Quality** | Clean, readable, well-named variables | Dense, cryptic code |
| **Testing** | Traces through example, considers edge cases | "I think it works" |
| **Optimization** | Suggests improvements, knows trade-offs | Can't improve beyond first solution |

---

## Practice Strategy

### Week 1: Core Algorithms
- [ ] KNN from scratch (classification + regression)
- [ ] K-Means from scratch
- [ ] Linear regression with gradient descent

### Week 2: Neural Network Basics
- [ ] 2D Convolution from scratch
- [ ] Softmax implementation
- [ ] Cross-entropy loss

### Week 3: Practical Problems
- [ ] Train/validation/test split with stratification
- [ ] Cross-validation implementation
- [ ] Feature scaling and encoding

### Week 4: Speed and Polish
- [ ] Time yourself (aim for 30 min per problem)
- [ ] Practice explaining while coding
- [ ] Review complexity analysis
