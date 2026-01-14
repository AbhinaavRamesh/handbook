# Linear Regression

> **Foundation of supervised learning** — regression, OLS, gradient descent

---

## One-Sentence Description

Linear regression models the relationship between features and a continuous target as a linear combination, finding weights that minimize prediction error.

---

## Core Formulation

### Model
```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = w·x + b
```

In matrix form: **ŷ = Xw**

### Loss Function (Mean Squared Error)
```
L(w) = (1/n) Σ(yᵢ - ŷᵢ)² = (1/n) ||y - Xw||²
```

### Goal
Find weights **w** that minimize MSE.

---

## Two Solution Methods

### Method 1: Normal Equation (Closed-Form)

```
w = (XᵀX)⁻¹Xᵀy
```

**Derivation**:
1. Take derivative of L(w) with respect to w
2. Set to zero: ∂L/∂w = -2Xᵀ(y - Xw) = 0
3. Solve for w: Xᵀy = XᵀXw → w = (XᵀX)⁻¹Xᵀy

**When to use**:
- n_features < 10,000 (matrix inversion is O(d³))
- XᵀX is invertible (no perfect multicollinearity)

**Complexity**: O(nd² + d³) time, O(d²) space

### Method 2: Gradient Descent

```python
def gradient_descent_linear(X, y, lr=0.01, n_iters=1000):
    n, d = X.shape
    w = np.zeros(d)

    for _ in range(n_iters):
        predictions = X @ w
        gradient = (2/n) * X.T @ (predictions - y)
        w = w - lr * gradient

    return w
```

**When to use**:
- Large n_features (can't invert large matrix)
- Online learning (streaming data)
- Want regularization (add penalty to gradient)

**Complexity**: O(nd) per iteration

---

## Regularization

### Ridge Regression (L2)
```
L(w) = ||y - Xw||² + λ||w||²
```
- Shrinks weights toward zero
- Keeps all features
- Closed-form: w = (XᵀX + λI)⁻¹Xᵀy

### Lasso Regression (L1)
```
L(w) = ||y - Xw||² + λ|w|
```
- Can drive weights exactly to zero
- Performs feature selection
- No closed-form; use coordinate descent

### Elastic Net
```
L(w) = ||y - Xw||² + λ₁|w| + λ₂||w||²
```
- Combines L1 and L2
- Gets sparsity of Lasso with stability of Ridge

---

## Assumptions

### The 5 Key Assumptions

| Assumption | What It Means | How to Check | If Violated |
|------------|---------------|--------------|-------------|
| **Linearity** | y is linear in x | Residual vs fitted plot | Add polynomial features |
| **Independence** | Errors are independent | Durbin-Watson test | Use time-series models |
| **Homoscedasticity** | Constant error variance | Residuals vs fitted plot | Use weighted least squares |
| **Normality** | Errors are normally distributed | Q-Q plot | Less critical for large n |
| **No multicollinearity** | Features not highly correlated | VIF (Variance Inflation Factor) | Remove correlated features |

### What Happens When Assumptions Fail

**Non-linearity**:
- Predictions systematically wrong in some regions
- Fix: Add polynomial features, use non-linear model

**Heteroscedasticity** (non-constant variance):
- Predictions less reliable where variance is high
- Fix: Log transform target, weighted least squares

**Multicollinearity**:
- Coefficients unstable (high variance)
- Can't interpret individual feature importance
- Fix: Remove correlated features, use regularization

---

## Interpretation

### Coefficient Meaning

For standardized features:
> "A one standard deviation increase in x₁ is associated with a w₁ standard deviation change in y, holding other features constant."

For raw features:
> "A one unit increase in x₁ is associated with a w₁ unit change in y."

### R² (Coefficient of Determination)

```
R² = 1 - (SS_res / SS_tot) = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
```

- R² = 1: Perfect fit
- R² = 0: No better than predicting mean
- R² < 0: Worse than predicting mean

**Adjusted R²** penalizes for adding features:
```
Adjusted R² = 1 - (1 - R²)(n - 1)/(n - d - 1)
```

---

## Interview Questions

### Q1: "When would you use linear regression over more complex models?"

**Strong answer**:
> "I'd use linear regression when:
> 1. **Interpretability matters** — stakeholders need to understand feature contributions
> 2. **Relationship is approximately linear** — or can be made linear with feature engineering
> 3. **Limited data** — complex models overfit with small n
> 4. **Baseline model** — always start with a simple model before adding complexity
>
> Even if I use a complex model later, linear regression tells me if the relationship is learnable with a linear combination."

### Q2: "How do you choose between L1 and L2 regularization?"

**Strong answer**:
> "L1 (Lasso) drives weights to exactly zero, performing feature selection. I'd use it when I suspect many features are irrelevant and want a sparse model.
>
> L2 (Ridge) shrinks weights but keeps all features. I'd use it when features are correlated (L1 arbitrarily picks one) or when I want to keep all features but prevent overfitting.
>
> If unsure, Elastic Net combines both. In practice, I'd try all three with cross-validation."

### Q3: "What does multicollinearity do and how do you detect it?"

**Strong answer**:
> "Multicollinearity means features are highly correlated. It doesn't hurt prediction accuracy, but it makes coefficients unstable and uninterpretable — small data changes cause large coefficient swings.
>
> I'd detect it using:
> 1. Correlation matrix between features
> 2. VIF (Variance Inflation Factor) — VIF > 10 is concerning
>
> To fix: remove one of the correlated features, combine them, or use regularization which handles it gracefully."

### Q4: "What if the relationship isn't linear?"

**Strong answer**:
> "Three approaches:
> 1. **Feature engineering** — add polynomial features (x², x³), interaction terms (x₁·x₂), or log transforms. Still interpretable.
> 2. **Splines** — piecewise polynomial fitting. Good for smooth non-linear relationships.
> 3. **Non-linear models** — decision trees, neural networks. Better performance but less interpretability.
>
> I'd start with feature engineering to keep interpretability, then move to complex models if needed."

---

## Code Reference

```python
import numpy as np

class LinearRegression:
    def __init__(self, method='normal', lr=0.01, n_iters=1000, regularization=None, lambda_=0.1):
        self.method = method
        self.lr = lr
        self.n_iters = n_iters
        self.regularization = regularization
        self.lambda_ = lambda_
        self.weights = None

    def fit(self, X, y):
        # Add bias term
        X = np.c_[np.ones(len(X)), X]

        if self.method == 'normal':
            if self.regularization == 'l2':
                # Ridge: (XᵀX + λI)⁻¹Xᵀy
                I = np.eye(X.shape[1])
                I[0, 0] = 0  # Don't regularize bias
                self.weights = np.linalg.inv(X.T @ X + self.lambda_ * I) @ X.T @ y
            else:
                # OLS: (XᵀX)⁻¹Xᵀy
                self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
        else:
            # Gradient descent
            self.weights = np.zeros(X.shape[1])
            for _ in range(self.n_iters):
                grad = (2/len(X)) * X.T @ (X @ self.weights - y)
                if self.regularization == 'l2':
                    grad[1:] += 2 * self.lambda_ * self.weights[1:]
                self.weights -= self.lr * grad

    def predict(self, X):
        X = np.c_[np.ones(len(X)), X]
        return X @ self.weights

    def score(self, X, y):
        """R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
```

---

## Quick Reference Card

```
LINEAR REGRESSION
─────────────────────────────────────────────────
Model:    ŷ = Xw
Loss:     MSE = (1/n)||y - Xw||²
Solution: w = (XᵀX)⁻¹Xᵀy  OR  gradient descent

REGULARIZATION
─────────────────────────────────────────────────
L2 (Ridge): + λ||w||²  → shrinks weights
L1 (Lasso): + λ|w|     → sparse weights

ASSUMPTIONS
─────────────────────────────────────────────────
1. Linearity
2. Independence of errors
3. Homoscedasticity (constant variance)
4. Normality of errors
5. No multicollinearity

COMPLEXITY
─────────────────────────────────────────────────
Normal equation: O(nd² + d³) time, O(d²) space
Gradient descent: O(nd) per iteration

USE WHEN
─────────────────────────────────────────────────
✓ Interpretability needed
✓ Linear relationship (or engineerable)
✓ Baseline model
✓ Small-medium dataset
```

---

**Previous**: [← 01_Concepts_Overview](./01_Concepts_Overview.md) | **Next**: [03_Logistic_Regression →](./03_Logistic_Regression.md)
