# Decision Trees

> **Rule-based learning** — splits, impurity, pruning, CART

---

## One-Sentence Description

Decision trees recursively partition the feature space using axis-aligned splits, creating a flowchart-like structure where each leaf represents a prediction.

---

## Core Concept

### How It Works

1. Start with all data at root node
2. Find the best feature and split point
3. Partition data into child nodes
4. Recursively repeat until stopping criterion
5. Each leaf predicts majority class (classification) or mean (regression)

### Visual Example

```
                    [Income > 50K?]
                    /             \
                 Yes               No
                  |                 |
          [Age > 30?]        [Education = College?]
           /       \              /            \
        Yes         No          Yes            No
         |           |           |              |
     [Approve]   [Deny]     [Approve]       [Deny]
```

---

## Splitting Criteria

### For Classification

#### Gini Impurity
```
Gini(S) = 1 - Σᵢ pᵢ²
```

Where pᵢ is the proportion of class i in set S.

- Gini = 0: Pure node (all same class)
- Gini = 0.5: Maximum impurity (binary, 50-50)

#### Entropy (Information Gain)
```
Entropy(S) = -Σᵢ pᵢ log₂(pᵢ)
```

Information Gain = Entropy(parent) - weighted Entropy(children)

**Gini vs Entropy**: Very similar in practice. Gini is faster (no log). Entropy tends to create more balanced trees.

### For Regression

#### Mean Squared Error
```
MSE(S) = (1/|S|) Σ (yᵢ - ȳ)²
```

Find split that minimizes weighted MSE of children.

#### Mean Absolute Error
```
MAE(S) = (1/|S|) Σ |yᵢ - median(y)|
```

More robust to outliers.

---

## Split Selection Algorithm

```python
def find_best_split(X, y, criterion='gini'):
    """Find the best feature and threshold to split on."""
    best_gain = -float('inf')
    best_feature = None
    best_threshold = None

    current_impurity = compute_impurity(y, criterion)

    for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])

        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_impurity = compute_impurity(y[left_mask], criterion)
            right_impurity = compute_impurity(y[right_mask], criterion)

            # Weighted average of children impurities
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            n_total = n_left + n_right

            weighted_child_impurity = (
                (n_left / n_total) * left_impurity +
                (n_right / n_total) * right_impurity
            )

            gain = current_impurity - weighted_child_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


def compute_impurity(y, criterion='gini'):
    """Compute impurity of a set."""
    if len(y) == 0:
        return 0

    if criterion == 'gini':
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    elif criterion == 'entropy':
        proportions = np.bincount(y) / len(y)
        proportions = proportions[proportions > 0]  # Avoid log(0)
        return -np.sum(proportions * np.log2(proportions))
```

---

## Controlling Overfitting

Decision trees are prone to overfitting. Control with:

### Pre-Pruning (Early Stopping)

| Parameter | Effect |
|-----------|--------|
| `max_depth` | Limit tree depth |
| `min_samples_split` | Minimum samples to split a node |
| `min_samples_leaf` | Minimum samples in leaf |
| `max_features` | Number of features to consider per split |
| `max_leaf_nodes` | Maximum number of leaves |

### Post-Pruning

After growing full tree, remove nodes that don't improve validation performance:
- **Reduced Error Pruning**: Remove nodes if it doesn't increase validation error
- **Cost-Complexity Pruning**: Add penalty for number of leaves

```
Cost = Error + α × (number of leaves)
```

---

## CART Algorithm

**Classification And Regression Trees** (CART):
- Binary splits only (left vs right)
- Gini for classification, MSE for regression
- Most common implementation (sklearn uses this)

### Full CART Implementation

```python
import numpy as np

class DecisionTreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Feature to split on
        self.threshold = threshold       # Split threshold
        self.left = left                 # Left child
        self.right = right               # Right child
        self.value = value               # Leaf prediction (if leaf)

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            return DecisionTreeNode(value=self._compute_leaf_value(y))

        # Find best split
        best_feature, best_threshold, best_gain = find_best_split(X, y)

        if best_gain <= 0:
            return DecisionTreeNode(value=self._compute_leaf_value(y))

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Check min_samples_leaf
        if np.sum(left_mask) < self.min_samples_leaf or \
           np.sum(right_mask) < self.min_samples_leaf:
            return DecisionTreeNode(value=self._compute_leaf_value(y))

        # Recursively build children
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )

    def _compute_leaf_value(self, y):
        # For classification: majority class
        return np.argmax(np.bincount(y))

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
```

---

## Key Properties

### Strengths

1. **Interpretable** — Can visualize and explain decisions
2. **No scaling needed** — Invariant to feature scales
3. **Handles mixed types** — Numerical and categorical
4. **Captures interactions** — Tree structure models feature interactions
5. **Fast prediction** — O(log n) for balanced tree

### Weaknesses

1. **Overfitting** — Deep trees memorize training data
2. **Instability** — Small data changes → different tree
3. **Axis-aligned splits** — Can't efficiently capture diagonal boundaries
4. **Greedy** — Local optimal splits may not be globally optimal
5. **Imbalanced data** — Biased toward majority class

---

## Interview Questions

### Q1: "How do you prevent overfitting in decision trees?"

**Strong answer**:
> "Two approaches:
> 1. **Pre-pruning** — Stop growth early with hyperparameters:
>    - `max_depth`: Limit how deep the tree grows
>    - `min_samples_split`: Require minimum samples to split
>    - `min_samples_leaf`: Require minimum samples in leaves
>
> 2. **Post-pruning** — Grow full tree, then remove nodes:
>    - Cost-complexity pruning: Add penalty for number of leaves
>    - Cross-validate to find optimal complexity
>
> In practice, I'd tune these hyperparameters using cross-validation. Or better, use an ensemble like Random Forest which averages many trees to reduce variance."

### Q2: "What's the difference between Gini and Entropy?"

**Strong answer**:
> "Both measure impurity — how mixed the classes are in a node.
>
> **Gini**: 1 - Σpᵢ² — ranges from 0 to 0.5 (binary)
> **Entropy**: -Σpᵢ log pᵢ — ranges from 0 to 1 (binary)
>
> Mathematically, Gini is a quadratic approximation of Entropy. In practice:
> - Results are very similar — studies show <2% difference in accuracy
> - Gini is slightly faster (no logarithm)
> - Entropy tends to produce more balanced splits
>
> I'd default to Gini (sklearn default) and not worry about it unless there's a specific reason."

### Q3: "When would you use a decision tree vs. logistic regression?"

**Strong answer**:
> "Decision tree if:
> - **Non-linear relationships** — Trees capture them naturally
> - **Feature interactions** — Trees model X₁ AND X₂ conditions
> - **Interpretability via rules** — 'If income > 50K AND age > 30...'
> - **Mixed feature types** — Handles categorical without encoding
>
> Logistic regression if:
> - **Linear relationship** — Simpler, more stable
> - **Probabilistic output** — Well-calibrated probabilities
> - **Coefficient interpretation** — 'Each year of experience increases odds by X%'
> - **High-dimensional** — Trees struggle with many features
>
> I'd often try both. For complex patterns with interpretability needs, I might use a shallow tree (depth 3-5) as a middle ground."

### Q4: "Why are decision trees unstable?"

**Strong answer**:
> "Small changes in training data can produce completely different trees. This happens because:
> 1. **Hierarchical structure** — One different split at the root cascades to all children
> 2. **Greedy optimization** — Picks locally optimal splits that may not be globally optimal
> 3. **Discrete splits** — Changing one data point can flip which side of a threshold it falls on
>
> This variance is a problem for single trees but becomes a feature in ensembles:
> - **Random Forest**: Train many different trees on bootstrap samples, average predictions
> - **Bagging**: Reduces variance by averaging
>
> The instability that hurts single trees becomes the diversity that helps ensembles."

---

## Comparison: Tree vs Linear

| Aspect | Decision Tree | Linear Model |
|--------|---------------|--------------|
| Decision boundary | Axis-aligned rectangles | Single hyperplane |
| Feature interactions | Captures naturally | Need explicit terms |
| Feature scaling | Not needed | Critical |
| Interpretability | Rules ("if X then Y") | Coefficients |
| Stability | Unstable (high variance) | Stable |
| Extrapolation | Constant prediction | Linear extrapolation |

---

## Quick Reference Card

```
DECISION TREE
─────────────────────────────────────────────────
Algorithm: Recursively partition feature space
Splits:    Axis-aligned (one feature at a time)
Leaf:      Majority class (classification) or mean (regression)

SPLITTING CRITERIA
─────────────────────────────────────────────────
Gini:    1 - Σpᵢ²        (faster)
Entropy: -Σpᵢ log₂(pᵢ)   (more balanced)
MSE:     Σ(yᵢ - ȳ)²      (regression)

HYPERPARAMETERS
─────────────────────────────────────────────────
max_depth:         Limit tree depth
min_samples_split: Min samples to split
min_samples_leaf:  Min samples in leaf
max_features:      Features per split

COMPLEXITY
─────────────────────────────────────────────────
Training:   O(n·d·log n) average
Prediction: O(log n) to O(n) worst case
Space:      O(nodes)

USE WHEN
─────────────────────────────────────────────────
✓ Interpretable rules needed
✓ Feature interactions important
✓ Mixed feature types
✗ High-dimensional data (prefer ensemble)
✗ Need stable predictions
```

---

**Previous**: [← 03_Logistic_Regression](./03_Logistic_Regression.md) | **Next**: [05_SVM →](./05_SVM.md)
