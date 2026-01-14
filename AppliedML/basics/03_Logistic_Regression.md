# Logistic Regression

> **Binary classification** — sigmoid, cross-entropy, decision boundaries

---

## One-Sentence Description

Logistic regression models the probability of a binary outcome using a linear combination of features passed through a sigmoid function.

---

## Core Formulation

### Model
```
P(y=1|x) = σ(w·x + b) = 1 / (1 + e^(-(w·x + b)))
```

Where σ is the sigmoid function:
```
σ(z) = 1 / (1 + e^(-z))
```

### Why Sigmoid?

1. **Maps any real number to (0, 1)** — interpretable as probability
2. **S-shaped curve** — captures "threshold" behavior
3. **Nice derivative** — σ'(z) = σ(z)(1 - σ(z))

### Decision Boundary

Predict y=1 if P(y=1|x) > 0.5, which means w·x + b > 0

The decision boundary is the hyperplane **w·x + b = 0**

---

## Loss Function

### Why Not MSE?

MSE loss with sigmoid creates a **non-convex** optimization problem with many local minima.

### Cross-Entropy Loss (Log Loss)

```
L(w) = -(1/n) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

Where ŷᵢ = σ(w·xᵢ + b)

**Intuition**:
- If y=1 and ŷ→1: loss → 0 (correct, confident)
- If y=1 and ŷ→0: loss → ∞ (wrong, confident)
- Penalizes confident wrong predictions heavily

### Maximum Likelihood Interpretation

Cross-entropy is equivalent to maximum likelihood estimation assuming Bernoulli distribution:
```
P(y|x) = ŷʸ(1-ŷ)^(1-y)
Log-likelihood = y log(ŷ) + (1-y) log(1-ŷ)
```

Minimizing cross-entropy = maximizing likelihood.

---

## Optimization

### Gradient Descent

```
∂L/∂w = (1/n) Σ (ŷᵢ - yᵢ)xᵢ
```

Remarkably similar to linear regression gradient!

```python
def logistic_regression_gd(X, y, lr=0.01, n_iters=1000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0

    for _ in range(n_iters):
        z = X @ w + b
        y_pred = 1 / (1 + np.exp(-z))

        dw = (1/n) * X.T @ (y_pred - y)
        db = (1/n) * np.sum(y_pred - y)

        w -= lr * dw
        b -= lr * db

    return w, b
```

### Newton's Method

For faster convergence, use second-order optimization:
```
w_new = w - H⁻¹∇L
```

Where H is the Hessian matrix. Converges in fewer iterations but each iteration is O(d³).

---

## Regularization

### L2 Regularization (Ridge)
```
L(w) = CrossEntropy + λ||w||²
```
- Prevents large weights
- Standard in most implementations

### L1 Regularization (Lasso)
```
L(w) = CrossEntropy + λ|w|
```
- Produces sparse weights
- Feature selection

### Regularization Parameter (C in sklearn)

sklearn uses C = 1/λ, so:
- **Large C**: Less regularization, more complex model
- **Small C**: More regularization, simpler model

---

## Multiclass Extension

### One-vs-Rest (OvR)

Train k binary classifiers, one per class:
- Classifier 1: Class 1 vs (Classes 2, 3, ..., k)
- Classifier 2: Class 2 vs (Classes 1, 3, ..., k)
- ...

Predict: class with highest probability

### Softmax Regression (Multinomial)

Generalize sigmoid to k classes:
```
P(y=j|x) = e^(wⱼ·x) / Σₖ e^(wₖ·x)
```

Loss: categorical cross-entropy
```
L = -Σᵢ Σⱼ yᵢⱼ log(P(y=j|xᵢ))
```

---

## Key Properties

### Assumptions

| Assumption | Description |
|------------|-------------|
| **Linearity in log-odds** | log(p/(1-p)) = w·x is linear |
| **Independence** | Observations are independent |
| **No multicollinearity** | Features not perfectly correlated |
| **Large sample size** | ML estimation needs sufficient data |

### Strengths

1. **Probabilistic output** — get confidence, not just labels
2. **Interpretable coefficients** — odds ratios have meaning
3. **Efficient** — O(nd) training per iteration
4. **Regularization built-in** — handles high dimensions
5. **No distributional assumptions on X** — flexible

### Weaknesses

1. **Linear decision boundary** — can't model XOR without features
2. **Requires feature engineering** — for non-linear relationships
3. **Sensitive to outliers** — extreme points affect decision boundary
4. **Class imbalance issues** — may predict majority class

---

## Coefficient Interpretation

### Log-Odds Interpretation

```
log(p/(1-p)) = w₀ + w₁x₁ + w₂x₂ + ...
```

**wⱼ = change in log-odds for one-unit increase in xⱼ**

### Odds Ratio Interpretation

```
Odds Ratio for xⱼ = e^(wⱼ)
```

- OR > 1: Feature increases odds of positive class
- OR < 1: Feature decreases odds
- OR = 1: No effect

**Example**: If w₁ = 0.5 for "years_experience":
- Odds ratio = e^0.5 ≈ 1.65
- "Each additional year of experience increases odds of success by 65%"

---

## Interview Questions

### Q1: "Why use logistic regression over linear regression for classification?"

**Strong answer**:
> "Three reasons:
> 1. **Bounded output** — Linear regression can predict values outside [0, 1], which don't make sense as probabilities. Logistic regression is bounded.
> 2. **Probabilistic interpretation** — Logistic regression outputs calibrated probabilities, useful for ranking or setting thresholds.
> 3. **Proper loss function** — MSE with classification is inappropriate. Cross-entropy correctly penalizes confident wrong predictions.
>
> That said, linear regression can work for classification (linear discriminant analysis), but logistic regression is more principled."

### Q2: "How do you handle class imbalance in logistic regression?"

**Strong answer**:
> "Several approaches:
> 1. **Class weights** — Upweight minority class in loss function. sklearn has `class_weight='balanced'`.
> 2. **Threshold adjustment** — Don't use 0.5 cutoff. Choose threshold based on precision-recall trade-off.
> 3. **Resampling** — Oversample minority (SMOTE) or undersample majority. Be careful about evaluation — do this after train/test split.
> 4. **Different metrics** — Don't use accuracy. Use precision, recall, F1, or PR-AUC.
>
> I'd start with class weights since it's simple and doesn't require generating synthetic data."

### Q3: "What if the relationship isn't linear in log-odds?"

**Strong answer**:
> "Options:
> 1. **Feature engineering** — Add polynomial features, interaction terms, log transforms. Keep interpretability.
> 2. **Generalized additive models** — Allow smooth non-linear effects while staying interpretable.
> 3. **Non-linear models** — Decision trees, neural networks. Better performance but less interpretable.
>
> I'd start with feature engineering. If the decision boundary needs to be non-linear, splines or GAMs are good middle ground before jumping to black-box models."

### Q4: "What's the difference between the sigmoid and softmax?"

**Strong answer**:
> "Sigmoid maps a single value to (0, 1) — used for binary classification.
>
> Softmax maps a vector of k values to k probabilities that sum to 1 — used for multiclass classification.
>
> For binary classification, softmax with k=2 is equivalent to sigmoid. Specifically, softmax([z, 0]) gives the same output as [sigmoid(z), 1-sigmoid(z)].
>
> Softmax is the natural generalization of sigmoid to multiple classes, both derived from maximizing likelihood under the assumption of exponential family distributions."

---

## Code Reference

```python
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, regularization='l2', lambda_=0.1):
        self.lr = lr
        self.n_iters = n_iters
        self.regularization = regularization
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n, d = X.shape
        self.weights = np.zeros(d)
        self.bias = 0

        for _ in range(self.n_iters):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            # Gradients
            dw = (1/n) * X.T @ (y_pred - y)
            db = (1/n) * np.sum(y_pred - y)

            # Add regularization
            if self.regularization == 'l2':
                dw += (self.lambda_ / n) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_ / n) * np.sign(self.weights)

            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class SoftmaxRegression:
    """Multiclass logistic regression."""
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        n, d = X.shape
        k = len(np.unique(y))

        # One-hot encode y
        y_onehot = np.zeros((n, k))
        y_onehot[np.arange(n), y] = 1

        self.weights = np.zeros((d, k))

        for _ in range(self.n_iters):
            z = X @ self.weights
            y_pred = self.softmax(z)
            grad = (1/n) * X.T @ (y_pred - y_onehot)
            self.weights -= self.lr * grad

    def predict(self, X):
        z = X @ self.weights
        return np.argmax(self.softmax(z), axis=1)
```

---

## Quick Reference Card

```
LOGISTIC REGRESSION
─────────────────────────────────────────────────
Model:    P(y=1|x) = σ(w·x + b)
Sigmoid:  σ(z) = 1/(1 + e^(-z))
Loss:     Cross-entropy = -[y log(ŷ) + (1-y) log(1-ŷ)]
Gradient: ∂L/∂w = (1/n) Σ (ŷ - y)x

DECISION BOUNDARY
─────────────────────────────────────────────────
Linear: w·x + b = 0
Predict y=1 if w·x + b > 0

MULTICLASS
─────────────────────────────────────────────────
OvR:     Train k binary classifiers
Softmax: P(y=j|x) = e^(wⱼ·x) / Σₖ e^(wₖ·x)

INTERPRETATION
─────────────────────────────────────────────────
wⱼ = change in log-odds per unit increase in xⱼ
e^(wⱼ) = odds ratio for xⱼ

USE WHEN
─────────────────────────────────────────────────
✓ Binary/multiclass classification
✓ Need probability estimates
✓ Want interpretable coefficients
✓ Linear decision boundary is sufficient
```

---

**Previous**: [← 02_Linear_Regression](./02_Linear_Regression.md) | **Next**: [04_Decision_Trees →](./04_Decision_Trees.md)
