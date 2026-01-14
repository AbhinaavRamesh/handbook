# Implement K-Means Clustering

> **From-scratch implementation** with initialization strategies and optimal K selection

---

## Problem Statement

Implement the K-Means clustering algorithm. Given unlabeled data `X`, partition it into `k` clusters by iteratively updating cluster centroids until convergence.

---

## Clarifying Questions to Ask

1. **How to initialize centroids?** (Random, K-means++, or provided?)
2. **Convergence criterion?** (Max iterations, centroid movement threshold?)
3. **How to handle empty clusters?** (Can happen with bad initialization)
4. **Return values?** (Labels only, or centroids too?)
5. **Libraries allowed?** (NumPy yes, sklearn no)

---

## Solution: Basic K-Means

### Step-by-Step Implementation

::: code-group

```python [Python]
from typing import List, Tuple
import random
import math

def kmeans(X: List[List[float]], k: int, max_iters: int = 100,
           tolerance: float = 1e-6, random_state: int = None) -> Tuple[List[int], List[List[float]]]:
    """
    K-Means clustering algorithm.

    Args:
        X: Data points, list of feature vectors
        k: Number of clusters
        max_iters: Maximum iterations
        tolerance: Convergence threshold for centroid movement
        random_state: Random seed for reproducibility

    Returns:
        labels: Cluster assignment for each point
        centroids: Final centroid positions
    """
    if random_state is not None:
        random.seed(random_state)

    n_samples = len(X)
    n_features = len(X[0])

    # Step 1: Initialize centroids (random selection from data points)
    random_indices = random.sample(range(n_samples), k)
    centroids = [X[i][:] for i in random_indices]  # Deep copy

    for iteration in range(max_iters):
        # Step 2: Assign each point to nearest centroid
        labels = assign_clusters(X, centroids)

        # Step 3: Update centroids as mean of assigned points
        new_centroids = update_centroids(X, labels, k, n_features)

        # Step 4: Check for convergence
        centroid_shift = math.sqrt(sum(
            sum((new_centroids[i][j] - centroids[i][j]) ** 2
                for j in range(n_features))
            for i in range(k)
        ))

        if centroid_shift < tolerance:
            print(f"Converged at iteration {iteration}")
            break

        centroids = new_centroids

    return labels, centroids


def assign_clusters(X: List[List[float]], centroids: List[List[float]]) -> List[int]:
    """
    Assign each point to the nearest centroid.
    """
    labels = []
    for x in X:
        min_dist = float('inf')
        min_idx = 0
        for i, centroid in enumerate(centroids):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(x, centroid)))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        labels.append(min_idx)
    return labels


def update_centroids(X: List[List[float]], labels: List[int], k: int,
                     n_features: int) -> List[List[float]]:
    """
    Update centroids as mean of assigned points.
    """
    new_centroids = [[0.0] * n_features for _ in range(k)]
    counts = [0] * k

    for i, x in enumerate(X):
        cluster_idx = labels[i]
        counts[cluster_idx] += 1
        for j in range(n_features):
            new_centroids[cluster_idx][j] += x[j]

    for cluster_idx in range(k):
        if counts[cluster_idx] > 0:
            for j in range(n_features):
                new_centroids[cluster_idx][j] /= counts[cluster_idx]
        else:
            # Handle empty cluster: reinitialize randomly
            random_idx = random.randint(0, len(X) - 1)
            new_centroids[cluster_idx] = X[random_idx][:]

    return new_centroids
```

```python [NumPy]
import numpy as np

def kmeans(X, k, max_iters=100, tolerance=1e-6, random_state=None):
    """
    K-Means clustering algorithm.

    Args:
        X: Data points, shape (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum iterations
        tolerance: Convergence threshold for centroid movement
        random_state: Random seed for reproducibility

    Returns:
        labels: Cluster assignment for each point, shape (n_samples,)
        centroids: Final centroid positions, shape (k, n_features)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape

    # Step 1: Initialize centroids (random selection from data points)
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices].copy()

    for iteration in range(max_iters):
        # Step 2: Assign each point to nearest centroid
        labels = assign_clusters(X, centroids)

        # Step 3: Update centroids as mean of assigned points
        new_centroids = update_centroids(X, labels, k)

        # Step 4: Check for convergence
        centroid_shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))

        if centroid_shift < tolerance:
            print(f"Converged at iteration {iteration}")
            break

        centroids = new_centroids

    return labels, centroids


def assign_clusters(X, centroids):
    """
    Assign each point to the nearest centroid.

    Args:
        X: Data points, shape (n_samples, n_features)
        centroids: Current centroids, shape (k, n_features)

    Returns:
        labels: Cluster assignment for each point, shape (n_samples,)
    """
    # Compute distance from each point to each centroid
    # Using broadcasting: (n_samples, 1, n_features) - (k, n_features)
    distances = np.sqrt(np.sum((X[:, np.newaxis, :] - centroids) ** 2, axis=2))
    # distances shape: (n_samples, k)

    # Assign to nearest centroid
    labels = np.argmin(distances, axis=1)

    return labels


def update_centroids(X, labels, k):
    """
    Update centroids as mean of assigned points.

    Args:
        X: Data points, shape (n_samples, n_features)
        labels: Current cluster assignments, shape (n_samples,)
        k: Number of clusters

    Returns:
        new_centroids: Updated centroids, shape (k, n_features)
    """
    n_features = X.shape[1]
    new_centroids = np.zeros((k, n_features))

    for cluster_idx in range(k):
        # Get points assigned to this cluster
        cluster_mask = labels == cluster_idx
        cluster_points = X[cluster_mask]

        if len(cluster_points) > 0:
            # Centroid is mean of assigned points
            new_centroids[cluster_idx] = np.mean(cluster_points, axis=0)
        else:
            # Handle empty cluster: reinitialize randomly
            new_centroids[cluster_idx] = X[np.random.randint(len(X))]

    return new_centroids
```

```python [PyTorch]
import torch
from typing import Tuple

def kmeans(X: torch.Tensor, k: int, max_iters: int = 100,
           tolerance: float = 1e-6, random_state: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K-Means clustering algorithm.

    Args:
        X: Data points, shape (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum iterations
        tolerance: Convergence threshold for centroid movement
        random_state: Random seed for reproducibility

    Returns:
        labels: Cluster assignment for each point, shape (n_samples,)
        centroids: Final centroid positions, shape (k, n_features)
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    n_samples, n_features = X.shape

    # Step 1: Initialize centroids (random selection from data points)
    random_indices = torch.randperm(n_samples)[:k]
    centroids = X[random_indices].clone()

    for iteration in range(max_iters):
        # Step 2: Assign each point to nearest centroid
        labels = assign_clusters(X, centroids)

        # Step 3: Update centroids as mean of assigned points
        new_centroids = update_centroids(X, labels, k, n_features)

        # Step 4: Check for convergence
        centroid_shift = torch.sqrt(torch.sum((new_centroids - centroids) ** 2))

        if centroid_shift < tolerance:
            print(f"Converged at iteration {iteration}")
            break

        centroids = new_centroids

    return labels, centroids


def assign_clusters(X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    Assign each point to the nearest centroid.
    """
    # Compute distance from each point to each centroid
    # Using broadcasting: (n_samples, 1, n_features) - (k, n_features)
    distances = torch.sqrt(torch.sum((X.unsqueeze(1) - centroids) ** 2, dim=2))
    # distances shape: (n_samples, k)

    # Assign to nearest centroid
    labels = torch.argmin(distances, dim=1)

    return labels


def update_centroids(X: torch.Tensor, labels: torch.Tensor, k: int,
                     n_features: int) -> torch.Tensor:
    """
    Update centroids as mean of assigned points.
    """
    new_centroids = torch.zeros((k, n_features), dtype=X.dtype, device=X.device)

    for cluster_idx in range(k):
        # Get points assigned to this cluster
        cluster_mask = labels == cluster_idx
        cluster_points = X[cluster_mask]

        if len(cluster_points) > 0:
            # Centroid is mean of assigned points
            new_centroids[cluster_idx] = torch.mean(cluster_points, dim=0)
        else:
            # Handle empty cluster: reinitialize randomly
            random_idx = torch.randint(len(X), (1,)).item()
            new_centroids[cluster_idx] = X[random_idx]

    return new_centroids
```

:::

---

## Walkthrough Example

```python
# Example data: two clear clusters
X = np.array([
    [1, 1], [1.5, 1.5], [1, 2],  # Cluster A
    [5, 5], [5.5, 5.5], [5, 6],  # Cluster B
])

# Run K-Means with k=2
labels, centroids = kmeans(X, k=2, random_state=42)

# Trace (depends on random initialization):
# Iteration 0:
#   Initial centroids: [[1, 1], [5.5, 5.5]] (random selection)
#   Assign: [0, 0, 0, 1, 1, 1]
#   Update: [[1.17, 1.5], [5.17, 5.5]]
# Iteration 1:
#   Assign: [0, 0, 0, 1, 1, 1] (same)
#   Update: (small change)
# ...converges

print(f"Labels: {labels}")       # [0, 0, 0, 1, 1, 1] or [1, 1, 1, 0, 0, 0]
print(f"Centroids: {centroids}")
```

---

## K-Means++ Initialization

Better initialization for faster convergence and better results:

::: code-group

```python [Python]
from typing import List
import random
import math

def kmeans_plusplus_init(X: List[List[float]], k: int,
                         random_state: int = None) -> List[List[float]]:
    """
    K-Means++ initialization: choose centroids spread apart.

    Algorithm:
    1. Choose first centroid randomly
    2. For each subsequent centroid:
       - Compute distance from each point to nearest existing centroid
       - Choose next centroid with probability proportional to distance squared
    """
    if random_state is not None:
        random.seed(random_state)

    n_samples = len(X)
    n_features = len(X[0])
    centroids = []

    # Step 1: Choose first centroid randomly
    first_idx = random.randint(0, n_samples - 1)
    centroids.append(X[first_idx][:])

    # Step 2: Choose remaining centroids
    for c in range(1, k):
        # Compute distance to nearest existing centroid
        distances = []
        for i in range(n_samples):
            min_dist = float('inf')
            for centroid in centroids:
                dist = math.sqrt(sum((X[i][j] - centroid[j]) ** 2
                                     for j in range(n_features)))
                min_dist = min(min_dist, dist)
            distances.append(min_dist)

        # Choose next centroid with probability proportional to distance squared
        probabilities = [d ** 2 for d in distances]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        # Weighted random choice
        r = random.random()
        cumsum = 0
        for i, p in enumerate(probabilities):
            cumsum += p
            if r <= cumsum:
                centroids.append(X[i][:])
                break

    return centroids


def kmeans_plusplus(X: List[List[float]], k: int, max_iters: int = 100,
                    tolerance: float = 1e-6, random_state: int = None):
    """
    K-Means with K-Means++ initialization.
    """
    if random_state is not None:
        random.seed(random_state)

    n_features = len(X[0])

    # Initialize with K-Means++
    centroids = kmeans_plusplus_init(X, k, random_state)

    # Run standard K-Means iterations
    for iteration in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k, n_features)

        centroid_shift = math.sqrt(sum(
            sum((new_centroids[i][j] - centroids[i][j]) ** 2
                for j in range(n_features))
            for i in range(k)
        ))
        if centroid_shift < tolerance:
            break

        centroids = new_centroids

    return labels, centroids
```

```python [NumPy]
import numpy as np

def kmeans_plusplus_init(X, k, random_state=None):
    """
    K-Means++ initialization: choose centroids spread apart.

    Algorithm:
    1. Choose first centroid randomly
    2. For each subsequent centroid:
       - Compute distance from each point to nearest existing centroid
       - Choose next centroid with probability proportional to distance squared
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    centroids = np.zeros((k, n_features))

    # Step 1: Choose first centroid randomly
    first_idx = np.random.randint(n_samples)
    centroids[0] = X[first_idx]

    # Step 2: Choose remaining centroids
    for c in range(1, k):
        # Compute distance to nearest existing centroid
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            min_dist = np.inf
            for j in range(c):
                dist = np.sqrt(np.sum((X[i] - centroids[j]) ** 2))
                min_dist = min(min_dist, dist)
            distances[i] = min_dist

        # Choose next centroid with probability proportional to distance squared
        probabilities = distances ** 2
        probabilities /= np.sum(probabilities)
        next_idx = np.random.choice(n_samples, p=probabilities)
        centroids[c] = X[next_idx]

    return centroids


def kmeans_plusplus(X, k, max_iters=100, tolerance=1e-6, random_state=None):
    """
    K-Means with K-Means++ initialization.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize with K-Means++
    centroids = kmeans_plusplus_init(X, k, random_state)

    # Run standard K-Means iterations
    for iteration in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        centroid_shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
        if centroid_shift < tolerance:
            break

        centroids = new_centroids

    return labels, centroids
```

```python [PyTorch]
import torch
from typing import Tuple

def kmeans_plusplus_init(X: torch.Tensor, k: int,
                         random_state: int = None) -> torch.Tensor:
    """
    K-Means++ initialization: choose centroids spread apart.
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    n_samples, n_features = X.shape
    centroids = torch.zeros((k, n_features), dtype=X.dtype, device=X.device)

    # Step 1: Choose first centroid randomly
    first_idx = torch.randint(n_samples, (1,)).item()
    centroids[0] = X[first_idx]

    # Step 2: Choose remaining centroids
    for c in range(1, k):
        # Compute distance to nearest existing centroid
        # Using broadcasting for efficiency
        dists_to_centroids = torch.sqrt(
            torch.sum((X.unsqueeze(1) - centroids[:c].unsqueeze(0)) ** 2, dim=2)
        )  # Shape: (n_samples, c)
        distances = torch.min(dists_to_centroids, dim=1).values

        # Choose next centroid with probability proportional to distance squared
        probabilities = distances ** 2
        probabilities = probabilities / torch.sum(probabilities)
        next_idx = torch.multinomial(probabilities, 1).item()
        centroids[c] = X[next_idx]

    return centroids


def kmeans_plusplus(X: torch.Tensor, k: int, max_iters: int = 100,
                    tolerance: float = 1e-6, random_state: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K-Means with K-Means++ initialization.
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    n_features = X.shape[1]

    # Initialize with K-Means++
    centroids = kmeans_plusplus_init(X, k, random_state)

    # Run standard K-Means iterations
    for iteration in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k, n_features)

        centroid_shift = torch.sqrt(torch.sum((new_centroids - centroids) ** 2))
        if centroid_shift < tolerance:
            break

        centroids = new_centroids

    return labels, centroids
```

:::

### Why K-Means++ is Better

| Initialization | Time to Converge | Result Quality |
|---------------|------------------|----------------|
| Random | More iterations | Can get stuck in bad local minimum |
| K-Means++ | Fewer iterations | Better spread, usually better results |

---

## Finding Optimal K: The Elbow Method

::: code-group

```python [Python]
from typing import List, Tuple

def compute_inertia(X: List[List[float]], labels: List[int],
                    centroids: List[List[float]]) -> float:
    """
    Compute inertia (sum of squared distances to centroids).
    Also called "within-cluster sum of squares" (WCSS).
    """
    inertia = 0.0
    for i, x in enumerate(X):
        centroid = centroids[labels[i]]
        inertia += sum((a - b) ** 2 for a, b in zip(x, centroid))
    return inertia


def find_optimal_k(X: List[List[float]], k_range: range = range(1, 11),
                   random_state: int = None) -> Tuple[List[int], List[float]]:
    """
    Find optimal k using the elbow method.

    Returns:
        k_values: List of k values tested
        inertias: Corresponding inertia values
    """
    inertias = []

    for k in k_range:
        labels, centroids = kmeans(X, k, random_state=random_state)
        inertia = compute_inertia(X, labels, centroids)
        inertias.append(inertia)
        print(f"k={k}, Inertia={inertia:.2f}")

    return list(k_range), inertias
```

```python [NumPy]
import numpy as np

def compute_inertia(X, labels, centroids):
    """
    Compute inertia (sum of squared distances to centroids).
    Also called "within-cluster sum of squares" (WCSS).
    """
    inertia = 0
    for i, x in enumerate(X):
        centroid = centroids[labels[i]]
        inertia += np.sum((x - centroid) ** 2)
    return inertia


def find_optimal_k(X, k_range=range(1, 11), random_state=None):
    """
    Find optimal k using the elbow method.

    Returns:
        k_values: List of k values tested
        inertias: Corresponding inertia values
    """
    inertias = []

    for k in k_range:
        labels, centroids = kmeans(X, k, random_state=random_state)
        inertia = compute_inertia(X, labels, centroids)
        inertias.append(inertia)
        print(f"k={k}, Inertia={inertia:.2f}")

    return list(k_range), inertias


def plot_elbow(k_values, inertias):
    """
    Plot the elbow curve (would use matplotlib in practice).
    """
    print("\nElbow Plot:")
    print("k\tInertia\t\tPlot")
    max_inertia = max(inertias)
    for k, inertia in zip(k_values, inertias):
        bar_len = int(50 * inertia / max_inertia)
        print(f"{k}\t{inertia:.2f}\t\t{'#' * bar_len}")
```

```python [PyTorch]
import torch
from typing import List, Tuple

def compute_inertia(X: torch.Tensor, labels: torch.Tensor,
                    centroids: torch.Tensor) -> float:
    """
    Compute inertia (sum of squared distances to centroids).
    Also called "within-cluster sum of squares" (WCSS).
    """
    inertia = 0.0
    for i in range(len(X)):
        centroid = centroids[labels[i]]
        inertia += torch.sum((X[i] - centroid) ** 2).item()
    return inertia


def find_optimal_k(X: torch.Tensor, k_range: range = range(1, 11),
                   random_state: int = None) -> Tuple[List[int], List[float]]:
    """
    Find optimal k using the elbow method.

    Returns:
        k_values: List of k values tested
        inertias: Corresponding inertia values
    """
    inertias = []

    for k in k_range:
        labels, centroids = kmeans(X, k, random_state=random_state)
        inertia = compute_inertia(X, labels, centroids)
        inertias.append(inertia)
        print(f"k={k}, Inertia={inertia:.2f}")

    return list(k_range), inertias
```

:::

### Finding the Elbow Programmatically

```python
def find_elbow_point(k_values, inertias):
    """
    Find the elbow point using the maximum curvature method.

    The elbow is where adding more clusters doesn't significantly reduce inertia.
    """
    # Compute second derivative (rate of change of slope)
    # Elbow is where second derivative is maximum

    if len(k_values) < 3:
        return k_values[0]

    # Compute differences
    first_diff = np.diff(inertias)
    second_diff = np.diff(first_diff)

    # Elbow is at maximum second derivative
    # (where rate of decrease slows the most)
    elbow_idx = np.argmax(second_diff) + 1  # +1 because diff reduces length

    return k_values[elbow_idx]
```

---

## Silhouette Score for K Selection

Alternative to elbow method:

::: code-group

```python [Python]
from typing import List, Tuple
import math

def compute_silhouette_score(X: List[List[float]], labels: List[int]) -> float:
    """
    Compute silhouette score for clustering quality.

    For each point i:
    - a(i) = average distance to points in same cluster
    - b(i) = average distance to points in nearest other cluster
    - s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Score range: [-1, 1], higher is better
    """
    n_samples = len(X)
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        return 0  # Can't compute silhouette with single cluster

    silhouette_scores = []

    for i in range(n_samples):
        # a(i): average distance to same-cluster points
        same_cluster_dists = []
        for j in range(n_samples):
            if j != i and labels[j] == labels[i]:
                dist = math.sqrt(sum((X[i][d] - X[j][d]) ** 2
                                     for d in range(len(X[0]))))
                same_cluster_dists.append(dist)

        a_i = sum(same_cluster_dists) / len(same_cluster_dists) if same_cluster_dists else 0

        # b(i): minimum average distance to other clusters
        b_i = float('inf')
        for cluster_label in unique_labels:
            if cluster_label != labels[i]:
                other_cluster_dists = []
                for j in range(n_samples):
                    if labels[j] == cluster_label:
                        dist = math.sqrt(sum((X[i][d] - X[j][d]) ** 2
                                             for d in range(len(X[0]))))
                        other_cluster_dists.append(dist)
                if other_cluster_dists:
                    avg_dist = sum(other_cluster_dists) / len(other_cluster_dists)
                    b_i = min(b_i, avg_dist)

        # Silhouette score for point i
        if max(a_i, b_i) > 0:
            silhouette_scores.append((b_i - a_i) / max(a_i, b_i))
        else:
            silhouette_scores.append(0)

    return sum(silhouette_scores) / len(silhouette_scores)
```

```python [NumPy]
import numpy as np

def compute_silhouette_score(X, labels):
    """
    Compute silhouette score for clustering quality.

    For each point i:
    - a(i) = average distance to points in same cluster
    - b(i) = average distance to points in nearest other cluster
    - s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Score range: [-1, 1], higher is better
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        return 0  # Can't compute silhouette with single cluster

    silhouette_scores = np.zeros(n_samples)

    for i in range(n_samples):
        # a(i): average distance to same-cluster points
        same_cluster = labels == labels[i]
        same_cluster[i] = False  # Exclude self
        if np.sum(same_cluster) > 0:
            a_i = np.mean(np.sqrt(np.sum((X[same_cluster] - X[i]) ** 2, axis=1)))
        else:
            a_i = 0

        # b(i): minimum average distance to other clusters
        b_i = np.inf
        for cluster_label in unique_labels:
            if cluster_label != labels[i]:
                other_cluster = labels == cluster_label
                if np.sum(other_cluster) > 0:
                    avg_dist = np.mean(np.sqrt(np.sum((X[other_cluster] - X[i]) ** 2, axis=1)))
                    b_i = min(b_i, avg_dist)

        # Silhouette score for point i
        if max(a_i, b_i) > 0:
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_scores[i] = 0

    return np.mean(silhouette_scores)


def find_optimal_k_silhouette(X, k_range=range(2, 11), random_state=None):
    """
    Find optimal k using silhouette score (higher is better).
    """
    scores = []

    for k in k_range:
        labels, centroids = kmeans(X, k, random_state=random_state)
        score = compute_silhouette_score(X, labels)
        scores.append(score)
        print(f"k={k}, Silhouette={score:.4f}")

    best_k = k_range[np.argmax(scores)]
    print(f"\nBest k by silhouette score: {best_k}")

    return best_k, list(k_range), scores
```

```python [PyTorch]
import torch

def compute_silhouette_score(X: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute silhouette score for clustering quality.
    """
    n_samples = len(X)
    unique_labels = torch.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        return 0  # Can't compute silhouette with single cluster

    silhouette_scores = torch.zeros(n_samples)

    for i in range(n_samples):
        # a(i): average distance to same-cluster points
        same_cluster = labels == labels[i]
        same_cluster[i] = False  # Exclude self
        if torch.sum(same_cluster) > 0:
            a_i = torch.mean(torch.sqrt(torch.sum((X[same_cluster] - X[i]) ** 2, dim=1)))
        else:
            a_i = 0

        # b(i): minimum average distance to other clusters
        b_i = float('inf')
        for cluster_label in unique_labels:
            if cluster_label != labels[i]:
                other_cluster = labels == cluster_label
                if torch.sum(other_cluster) > 0:
                    avg_dist = torch.mean(torch.sqrt(torch.sum((X[other_cluster] - X[i]) ** 2, dim=1)))
                    b_i = min(b_i, avg_dist.item())

        # Silhouette score for point i
        a_i_val = a_i.item() if torch.is_tensor(a_i) else a_i
        if max(a_i_val, b_i) > 0:
            silhouette_scores[i] = (b_i - a_i_val) / max(a_i_val, b_i)
        else:
            silhouette_scores[i] = 0

    return torch.mean(silhouette_scores).item()
```

:::

---

## Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| One iteration | O(n * k * d) | O(n + k * d) |
| Full algorithm (i iterations) | O(i * n * k * d) | O(n + k * d) |
| K-Means++ init | O(n * k * d) | O(k * d) |
| Silhouette score | O(n^2 * d) | O(n) |

Where:
- n = number of samples
- k = number of clusters
- d = number of features
- i = number of iterations

---

## Edge Cases and Robustness

::: code-group

```python [Python]
from typing import List, Tuple
import random

def kmeans_robust(X: List[List[float]], k: int, max_iters: int = 100,
                  n_init: int = 10, random_state: int = None) -> Tuple[List[int], List[List[float]], float]:
    """
    Robust K-Means with multiple random initializations.

    Runs K-Means n_init times and returns the best result (lowest inertia).
    """
    best_labels = None
    best_centroids = None
    best_inertia = float('inf')

    for init in range(n_init):
        seed = random_state + init if random_state else None
        labels, centroids = kmeans(X, k, max_iters, random_state=seed)
        inertia = compute_inertia(X, labels, centroids)

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    return best_labels, best_centroids, best_inertia
```

```python [NumPy]
import numpy as np

def kmeans_robust(X, k, max_iters=100, n_init=10, random_state=None):
    """
    Robust K-Means with multiple random initializations.

    Runs K-Means n_init times and returns the best result (lowest inertia).
    """
    best_labels = None
    best_centroids = None
    best_inertia = np.inf

    for init in range(n_init):
        seed = random_state + init if random_state else None
        labels, centroids = kmeans(X, k, max_iters, random_state=seed)
        inertia = compute_inertia(X, labels, centroids)

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    return best_labels, best_centroids, best_inertia
```

```python [PyTorch]
import torch
from typing import Tuple, List

def kmeans_robust(X: torch.Tensor, k: int, max_iters: int = 100,
                  n_init: int = 10, random_state: int = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Robust K-Means with multiple random initializations.

    Runs K-Means n_init times and returns the best result (lowest inertia).
    """
    best_labels = None
    best_centroids = None
    best_inertia = float('inf')

    for init in range(n_init):
        seed = random_state + init if random_state else None
        labels, centroids = kmeans(X, k, max_iters, random_state=seed)
        inertia = compute_inertia(X, labels, centroids)

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids

    return best_labels, best_centroids, best_inertia
```

:::

### Handling Edge Cases

```python
def kmeans_with_checks(X, k, max_iters=100):
    """K-Means with comprehensive edge case handling."""

    # Check 1: Valid k
    if k <= 0:
        raise ValueError("k must be positive")
    if k > len(X):
        raise ValueError(f"k={k} exceeds number of samples={len(X)}")

    # Check 2: Non-empty input
    if len(X) == 0:
        raise ValueError("Input array is empty")

    # Check 3: k=1 special case
    if k == 1:
        labels = np.zeros(len(X), dtype=int)
        centroids = np.mean(X, axis=0, keepdims=True)
        return labels, centroids

    # Run standard K-Means
    return kmeans(X, k, max_iters)
```

---

## Variations to Mention

### 1. Mini-Batch K-Means
For large datasets, update centroids using small random batches:
```python
# Conceptually:
batch = X[np.random.choice(len(X), batch_size, replace=False)]
# Update centroids using only batch instead of full dataset
```

### 2. K-Medoids
Use actual data points as centroids (more robust to outliers).

### 3. Fuzzy C-Means
Soft clustering where points belong to multiple clusters with probabilities.

---

## Interview Tips

1. **Ask about initialization** - K-Means++ shows you know best practices
2. **Discuss convergence** - mention both max iterations AND tolerance
3. **Handle empty clusters** - shows production awareness
4. **Know the elbow method** - but mention it's subjective
5. **Mention silhouette score** - shows you know alternatives
6. **Discuss limitations** - K-Means assumes spherical clusters, sensitive to scale
