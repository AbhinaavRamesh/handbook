# Clustering Algorithms

> **Unsupervised learning** — K-Means, DBSCAN, Hierarchical clustering

---

## Overview

Clustering groups similar data points without labels. Use cases:
- Customer segmentation
- Document organization
- Anomaly detection
- Feature compression
- Data exploration

---

## K-Means Clustering

### One-Sentence Description

K-Means partitions data into k clusters by iteratively assigning points to the nearest centroid and updating centroids as cluster means.

### Algorithm

```
1. Initialize k centroids (randomly or K-Means++)
2. Repeat until convergence:
   a. Assign each point to nearest centroid
   b. Update centroids as mean of assigned points
3. Return cluster assignments
```

### Mathematical Formulation

**Objective**: Minimize within-cluster sum of squares (WCSS)

```
J = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
```

Where μᵢ is centroid of cluster Cᵢ.

### K-Means++ Initialization

Better initialization for faster convergence:
1. Choose first centroid randomly
2. For each subsequent centroid:
   - Compute distance to nearest existing centroid for each point
   - Choose next centroid with probability proportional to distance²

### Choosing K

| Method | How It Works |
|--------|--------------|
| **Elbow** | Plot WCSS vs k, find "elbow" |
| **Silhouette** | Maximize silhouette score |
| **Gap statistic** | Compare to null distribution |
| **Domain knowledge** | Business requirements |

### Strengths and Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Simple, fast | Must specify k |
| Scales well | Assumes spherical clusters |
| Guaranteed convergence | Sensitive to initialization |
| Interpretable centroids | Sensitive to outliers |

---

## DBSCAN

### One-Sentence Description

DBSCAN finds dense regions separated by sparse regions, automatically determining cluster count and identifying outliers.

### Key Concepts

- **Core point**: Has ≥ minPts neighbors within radius ε
- **Border point**: Within ε of a core point but not itself core
- **Noise point**: Neither core nor border

### Algorithm

```
For each unvisited point p:
1. Find all points within ε of p
2. If |neighbors| ≥ minPts:
   - p is core point, start new cluster
   - Add all neighbors to cluster
   - Recursively expand cluster from each core neighbor
3. Else: mark as noise (may later become border point)
```

### Parameters

| Parameter | Effect |
|-----------|--------|
| **ε (eps)** | Neighborhood radius |
| **minPts** | Minimum points for core status |

**Rule of thumb for minPts**: ≥ d + 1 where d is dimensions, commonly 4-5.

### Choosing ε

**K-distance graph method**:
1. Compute distance to kth nearest neighbor for all points
2. Sort distances and plot
3. ε is at the "elbow" (sharp increase)

### Strengths and Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Finds arbitrary shapes | Sensitive to ε and minPts |
| Doesn't need k | Struggles with varying densities |
| Identifies outliers | High-dimensional challenges |
| Robust to noise | Can merge close clusters |

---

## Hierarchical Clustering

### One-Sentence Description

Hierarchical clustering builds a tree (dendrogram) of clusters by either merging (agglomerative) or splitting (divisive) iteratively.

### Agglomerative (Bottom-Up)

```
1. Start with each point as its own cluster
2. Repeat until one cluster remains:
   a. Find two closest clusters
   b. Merge them
3. Cut dendrogram at desired level
```

### Linkage Methods

| Linkage | Distance Between Clusters | Properties |
|---------|---------------------------|------------|
| **Single** | Min distance between points | Chaining effect |
| **Complete** | Max distance between points | Compact clusters |
| **Average** | Mean distance between points | Balanced |
| **Ward** | Minimize variance increase | Most common |

### Strengths and Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| No k needed (dendrogram) | O(n³) time, O(n²) space |
| Hierarchical structure | Not scalable |
| Any linkage criterion | Can't undo merges |
| Deterministic | Sensitive to noise |

---

## Algorithm Comparison

| Aspect | K-Means | DBSCAN | Hierarchical |
|--------|---------|--------|--------------|
| **Cluster shape** | Spherical | Arbitrary | Depends on linkage |
| **Need k?** | Yes | No | Optional (cut level) |
| **Outlier handling** | Poor | Good | Poor |
| **Scalability** | O(nki) | O(n log n) | O(n³) |
| **Deterministic** | No | Yes | Yes |

---

## Evaluation Metrics

### With Ground Truth (External)

| Metric | Description |
|--------|-------------|
| **Adjusted Rand Index (ARI)** | Agreement adjusted for chance |
| **Normalized Mutual Information (NMI)** | Shared information |
| **Homogeneity** | All cluster members same class |
| **Completeness** | All class members same cluster |

### Without Ground Truth (Internal)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Silhouette** | (b-a)/max(a,b) | [-1, 1], higher better |
| **Calinski-Harabasz** | Between/Within variance | Higher better |
| **Davies-Bouldin** | Avg similarity ratio | Lower better |

### Silhouette Score

For point i:
- a(i) = avg distance to points in same cluster
- b(i) = avg distance to points in nearest other cluster
- s(i) = (b(i) - a(i)) / max(a(i), b(i))

---

## Interview Questions

### Q1: "How do you choose between K-Means and DBSCAN?"

**Strong answer**:
> "Depends on the data and requirements:
>
> **K-Means** if:
> - Clusters are roughly spherical and similar size
> - You know or can estimate k
> - Need fast, scalable solution
> - Outliers are not important
>
> **DBSCAN** if:
> - Clusters have arbitrary shapes
> - Don't know k
> - Need outlier detection
> - Clusters have similar density
>
> I'd visualize the data first (2D projection if high-dimensional). If clusters look spherical and separate, K-Means. If they're irregular or have noise, DBSCAN. Often I'd try both and compare using silhouette score."

### Q2: "What happens if K-Means is initialized badly?"

**Strong answer**:
> "Bad initialization can cause:
> 1. **Convergence to local minimum** — Not global optimum WCSS
> 2. **Empty clusters** — Centroid far from all points
> 3. **Slow convergence** — More iterations needed
>
> Solutions:
> 1. **K-Means++** — Smart initialization, spreads initial centroids
> 2. **Multiple runs** — Run n times, keep best by WCSS
> 3. **Warm start** — Initialize from previous solution
>
> In sklearn, K-Means++ is default (`init='k-means++'`) and multiple runs (`n_init=10`). This usually solves initialization issues."

### Q3: "How does DBSCAN handle varying density clusters?"

**Strong answer**:
> "Honestly, it doesn't handle it well. A single ε value can't capture clusters with different densities:
> - Too small ε: Dense cluster found, sparse cluster becomes noise
> - Too large ε: Sparse cluster found, dense clusters merge
>
> Solutions:
> 1. **OPTICS** — Creates ordering, can extract clusters at multiple densities
> 2. **HDBSCAN** — Hierarchical version, handles varying densities
> 3. **Local scaling** — Adapt ε based on local density
>
> For varying density data, I'd use HDBSCAN — it's a drop-in replacement that handles this case well."

### Q4: "What's the silhouette score and how do you interpret it?"

**Strong answer**:
> "Silhouette score measures how well each point fits its cluster vs other clusters.
>
> For point i:
> - a(i) = average distance to same-cluster points
> - b(i) = average distance to nearest other cluster
> - s(i) = (b(i) - a(i)) / max(a(i), b(i))
>
> Interpretation:
> - **+1**: Point is well-matched to its cluster, far from others
> - **0**: Point is on boundary between clusters
> - **-1**: Point is probably in wrong cluster
>
> Average silhouette score across all points gives overall clustering quality:
> - **> 0.7**: Strong structure
> - **0.5-0.7**: Reasonable structure
> - **0.25-0.5**: Weak structure
> - **< 0.25**: No substantial structure
>
> I'd also look at per-cluster silhouette to identify problematic clusters."

---

## Code Reference

```python
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Always scale for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
print(f"K-Means silhouette: {silhouette_score(X_scaled, labels_kmeans):.3f}")

# Find optimal k
silhouettes = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    silhouettes.append(silhouette_score(X_scaled, labels))

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)
n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)
print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")

# Hierarchical
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg = agg.fit_predict(X_scaled)

# HDBSCAN (better for varying density)
# pip install hdbscan
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
labels_hdbscan = clusterer.fit_predict(X_scaled)
```

---

## Quick Reference Card

```
CLUSTERING COMPARISON
─────────────────────────────────────────────────
K-Means:      Fast, spherical clusters, needs k
DBSCAN:       Arbitrary shapes, finds outliers, needs ε
Hierarchical: Dendrogram, no k needed, slow

K-MEANS
─────────────────────────────────────────────────
Objective: min Σ||x - μ||² (WCSS)
Init:      K-Means++ (default)
Choose k:  Elbow, silhouette, domain knowledge

DBSCAN
─────────────────────────────────────────────────
Parameters: ε (radius), minPts (density threshold)
Core point: ≥ minPts neighbors in ε
Noise:     Not reachable from any core point

EVALUATION
─────────────────────────────────────────────────
Silhouette: (b-a)/max(a,b), [-1, 1], higher better
ARI:        Agreement with ground truth
Inertia:    WCSS (K-Means objective)

USE WHEN
─────────────────────────────────────────────────
K-Means:      Know k, spherical, scalable
DBSCAN:       Unknown k, arbitrary shape, outliers
Hierarchical: Need dendrogram, small data
HDBSCAN:      Varying densities, robust
```
