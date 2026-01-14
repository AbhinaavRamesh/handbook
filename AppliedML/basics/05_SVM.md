# Support Vector Machines (SVM)

> **Maximum margin classifiers** — margins, kernels, soft margin

---

## One-Sentence Description

SVM finds the hyperplane that maximizes the margin between classes, with kernel tricks enabling non-linear decision boundaries.

---

## Core Concept: Maximum Margin

### What is Margin?

The margin is the distance between the decision boundary and the nearest data points from either class.

```
  Class +1:  ●    ●
                    ← margin
        ─────────── decision boundary (w·x + b = 0)
                    ← margin
  Class -1:  ○    ○
```

### Why Maximize Margin?

- **Better generalization** — More room for error on new data
- **Unique solution** — Maximum margin hyperplane is unique
- **Robust** — Less sensitive to small perturbations

### Support Vectors

The data points closest to the decision boundary are called **support vectors**. They "support" (define) the boundary.

Key insight: Only support vectors matter. Other points can be removed without changing the solution.

---

## Mathematical Formulation

### Hard Margin SVM

For linearly separable data:

**Goal**: Find w, b that maximize margin = 2/||w||

**Constraints**: yᵢ(w·xᵢ + b) ≥ 1 for all i

**Optimization problem**:
```
minimize:   (1/2)||w||²
subject to: yᵢ(w·xᵢ + b) ≥ 1
```

This is a convex quadratic programming problem with a unique solution.

### Soft Margin SVM

For non-separable data, allow some misclassification:

```
minimize:   (1/2)||w||² + C Σξᵢ
subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
            ξᵢ ≥ 0
```

Where:
- ξᵢ = slack variable (amount of violation)
- C = trade-off between margin and violations

**C parameter**:
- Large C: Less tolerant of violations → smaller margin, may overfit
- Small C: More tolerant → larger margin, may underfit

---

## Kernel Trick

### The Problem

Linear SVM can only find linear decision boundaries. What about this?

```
  ○ ○ ○
○ ● ● ● ○
○ ● ● ● ○
  ○ ○ ○
```

No linear boundary separates these.

### The Solution

1. Map data to higher-dimensional space: φ(x)
2. In higher dimensions, data may be linearly separable
3. Find linear boundary in high-dimensional space
4. Use kernel trick to avoid computing φ(x) explicitly

### Kernel Function

```
K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ)
```

Kernel computes dot product in high-dimensional space without explicitly transforming data.

### Common Kernels

| Kernel | Formula | Use Case |
|--------|---------|----------|
| **Linear** | K(x,y) = x·y | Linearly separable data |
| **Polynomial** | K(x,y) = (γx·y + r)^d | Polynomial boundaries |
| **RBF/Gaussian** | K(x,y) = exp(-γ||x-y||²) | Non-linear, most common |
| **Sigmoid** | K(x,y) = tanh(γx·y + r) | Neural network-like |

### RBF Kernel

```
K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)
```

- γ controls the "reach" of each training example
- Large γ: Each point has narrow influence → complex boundary, may overfit
- Small γ: Each point has wide influence → smooth boundary, may underfit

---

## Hyperparameter Tuning

### Key Parameters

| Parameter | Effect |
|-----------|--------|
| **C** | Regularization: larger C = less regularization |
| **kernel** | Type of decision boundary |
| **γ (gamma)** | RBF kernel width |
| **degree** | Polynomial kernel degree |

### Tuning Strategy

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'kernel': ['rbf', 'poly']
}

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
```

### Common Pitfalls

1. **Always scale features** — SVM uses distances, sensitive to scale
2. **Start with RBF kernel** — Works well in most cases
3. **Use validation for C and γ** — Default values rarely optimal

---

## Key Properties

### Strengths

1. **Effective in high dimensions** — Works well when d > n
2. **Memory efficient** — Only stores support vectors
3. **Versatile** — Different kernels for different problems
4. **Robust to outliers** — Soft margin handles noise
5. **Strong theoretical foundation** — PAC learning guarantees

### Weaknesses

1. **Slow for large n** — Training is O(n²) to O(n³)
2. **Requires scaling** — Features must be normalized
3. **No probability** — Native SVM gives only labels (sklearn can calibrate)
4. **Kernel selection** — Choosing right kernel is hard
5. **Hard to interpret** — Kernel makes coefficients meaningless

---

## Complexity Analysis

| Operation | Complexity |
|-----------|------------|
| Training | O(n²) to O(n³) time, O(n²) space |
| Prediction | O(n_sv × d) where n_sv = support vectors |

For large datasets, consider:
- Linear SVM (sklearn's `LinearSVC`) — O(n × d)
- SGD classifier with hinge loss
- Approximations (Nyström, random features)

---

## SVM vs Other Classifiers

### SVM vs Logistic Regression

| Aspect | SVM | Logistic Regression |
|--------|-----|---------------------|
| Objective | Maximize margin | Maximize likelihood |
| Decision boundary | Maximum margin | Maximum likelihood |
| Probabilities | Not native | Natural probabilities |
| Kernels | Easy to apply | Requires explicit features |
| Scaling | Critical | Less sensitive |
| Large datasets | Slower | Faster |

### When to Use SVM

- **High-dimensional data** (text, genomics)
- **n_features > n_samples**
- **Non-linear boundaries needed**
- **Medium-sized datasets** (< 100K samples)

### When to Avoid SVM

- **Large datasets** (use linear SVM or SGD)
- **Need probabilities** (use logistic regression)
- **Interpretability required** (use linear models or trees)

---

## Interview Questions

### Q1: "Explain what the kernel trick does."

**Strong answer**:
> "The kernel trick lets us compute dot products in a high-dimensional feature space without explicitly transforming the data there.
>
> The insight is that SVM only needs dot products between data points, never the transformed features themselves. A kernel function K(xᵢ, xⱼ) computes φ(xᵢ)·φ(xⱼ) directly, even when φ maps to infinite dimensions (like RBF).
>
> This means we can find non-linear decision boundaries in the original space that correspond to linear boundaries in the transformed space, without the computational cost of working in high dimensions.
>
> For example, the RBF kernel K(x,y) = exp(-γ||x-y||²) implicitly maps to infinite dimensions but is computed in O(d) time."

### Q2: "What's the role of C in SVM?"

**Strong answer**:
> "C is the regularization parameter that controls the trade-off between two goals:
> 1. Maximizing the margin
> 2. Minimizing classification errors on training data
>
> **Large C**: Prioritizes correct classification, allows smaller margin. The model tries hard to classify all training points correctly, which can lead to overfitting.
>
> **Small C**: Prioritizes large margin, tolerates some misclassification. The model is more regularized and generalizes better but may underfit.
>
> I'd tune C using cross-validation, typically searching over powers of 10 (0.01, 0.1, 1, 10, 100)."

### Q3: "What are support vectors and why do they matter?"

**Strong answer**:
> "Support vectors are the training points that lie on or within the margin boundary — they're the points closest to the decision boundary.
>
> They matter because:
> 1. **They define the model** — The decision boundary depends only on support vectors, not other points
> 2. **Sparse solution** — We only need to store support vectors for prediction, not all training data
> 3. **Model complexity indicator** — Many support vectors suggests complex boundary or noisy data
>
> This sparsity is a key advantage of SVM: if we have 10,000 training points but only 500 support vectors, prediction only involves those 500 points."

### Q4: "How would you handle a large dataset with SVM?"

**Strong answer**:
> "Standard SVM doesn't scale well — it's O(n²) to O(n³). For large datasets:
>
> 1. **Linear SVM** — If a linear kernel works, use `LinearSVC` which is O(n×d) using coordinate descent.
>
> 2. **SGD with hinge loss** — `SGDClassifier(loss='hinge')` is essentially online SVM, scales to millions of samples.
>
> 3. **Kernel approximations** — Random Fourier features or Nyström approximation let you approximate RBF kernel with linear SVM.
>
> 4. **Subsampling** — Train on a subset, though you lose information.
>
> I'd start with linear SVM or SGD, then try kernel approximations only if non-linear boundaries are necessary."

---

## Code Reference

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Always scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Basic SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_scaled, y_train)

# With probability estimates
svm_prob = SVC(kernel='rbf', probability=True)
svm_prob.fit(X_train_scaled, y_train)
probs = svm_prob.predict_proba(X_test_scaled)

# Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 'scale'],
}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Access support vectors
print(f"Number of support vectors: {len(svm.support_vectors_)}")
print(f"Support vector indices: {svm.support_}")
```

---

## Quick Reference Card

```
SUPPORT VECTOR MACHINE
─────────────────────────────────────────────────
Goal:     Maximize margin between classes
Boundary: w·x + b = 0
Margin:   2/||w||

SOFT MARGIN
─────────────────────────────────────────────────
min: (1/2)||w||² + C Σξᵢ
C large → fit training data, small margin
C small → large margin, allow errors

KERNELS
─────────────────────────────────────────────────
Linear:     K(x,y) = x·y
Polynomial: K(x,y) = (γx·y + r)^d
RBF:        K(x,y) = exp(-γ||x-y||²)

HYPERPARAMETERS
─────────────────────────────────────────────────
C:     Regularization (larger = less)
gamma: RBF width (larger = more complex)

COMPLEXITY
─────────────────────────────────────────────────
Training:   O(n²) to O(n³)
Prediction: O(n_sv × d)

USE WHEN
─────────────────────────────────────────────────
✓ High-dimensional data (d > n)
✓ Non-linear boundaries needed
✓ Medium-sized datasets
✗ Large datasets (use linear SVM)
✗ Need probability estimates
```

---

**Previous**: [← 04_Decision_Trees](./04_Decision_Trees.md) | **Next**: [06_KNN →](./06_KNN.md)
