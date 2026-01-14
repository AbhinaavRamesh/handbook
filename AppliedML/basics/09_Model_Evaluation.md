# Model Evaluation

> **Metrics, cross-validation, and bias-variance** — the complete evaluation toolkit

---

## Overview

Model evaluation answers: "How well does this model generalize to unseen data?"

Key aspects:
1. **Metrics** — What to measure
2. **Validation** — How to estimate generalization
3. **Diagnostics** — Understanding model behavior

---

## Classification Metrics

### Confusion Matrix

```
                 Predicted
              Positive  Negative
Actual  Positive   TP       FN
        Negative   FP       TN
```

### Core Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
| **Precision** | TP/(TP+FP) | Cost of FP high (spam) |
| **Recall/Sensitivity** | TP/(TP+FN) | Cost of FN high (cancer) |
| **Specificity** | TN/(TN+FP) | Need to identify negatives |
| **F1 Score** | 2·P·R/(P+R) | Balance P and R |

### Precision-Recall Trade-off

- **High threshold** → High precision, low recall (few positives, but confident)
- **Low threshold** → High recall, low precision (catch all positives, many false alarms)

### ROC and PR Curves

**ROC Curve**: TPR (recall) vs FPR (1-specificity)
- AUC = 0.5: Random classifier
- AUC = 1.0: Perfect classifier
- Robust to class imbalance? **No**

**PR Curve**: Precision vs Recall
- Better for imbalanced data
- Focus on positive class performance

### When to Use What

| Scenario | Best Metric |
|----------|-------------|
| Balanced classes | Accuracy, F1, ROC-AUC |
| Imbalanced classes | PR-AUC, F1, Recall |
| FP is costly | Precision |
| FN is costly | Recall |
| Ranking | ROC-AUC, PR-AUC |

---

## Regression Metrics

| Metric | Formula | Properties |
|--------|---------|------------|
| **MSE** | Σ(y-ŷ)²/n | Penalizes large errors |
| **RMSE** | √MSE | Same units as target |
| **MAE** | Σ\|y-ŷ\|/n | Robust to outliers |
| **MAPE** | Σ\|y-ŷ\|/y·100 | Percentage error |
| **R²** | 1 - SS_res/SS_tot | Variance explained |

### Choosing Regression Metrics

- **RMSE**: Default, penalizes large errors
- **MAE**: When outliers exist
- **MAPE**: When relative error matters
- **R²**: For interpretability (0-1 scale)

---

## Cross-Validation

### Why Cross-Validate?

Single train/test split:
- Wastes data (test set not used for training)
- High variance estimate (depends on split)

Cross-validation:
- Uses all data for training and validation
- More robust estimate of generalization

### K-Fold Cross-Validation

```
Data: [1][2][3][4][5]

Fold 1: Train on [2][3][4][5], Validate on [1]
Fold 2: Train on [1][3][4][5], Validate on [2]
Fold 3: Train on [1][2][4][5], Validate on [3]
Fold 4: Train on [1][2][3][5], Validate on [4]
Fold 5: Train on [1][2][3][4], Validate on [5]

Final estimate = mean of 5 validation scores
```

### CV Variants

| Variant | Use Case |
|---------|----------|
| **K-Fold** | Default (k=5 or 10) |
| **Stratified K-Fold** | Imbalanced classification |
| **Leave-One-Out (LOO)** | Small datasets |
| **Time Series Split** | Temporal data |
| **Group K-Fold** | Multiple samples per entity |

### Nested Cross-Validation

For hyperparameter tuning + evaluation:
```
Outer loop: Evaluate final performance
  Inner loop: Tune hyperparameters
```

Prevents overfitting to test set during tuning.

---

## Bias-Variance Trade-off

### Decomposition

```
Total Error = Bias² + Variance + Irreducible Error
```

- **Bias**: Error from wrong assumptions (underfitting)
- **Variance**: Error from sensitivity to training data (overfitting)
- **Irreducible**: Noise in the data

### Diagnosis

| Training Error | Validation Error | Problem |
|----------------|------------------|---------|
| High | High | High bias (underfit) |
| Low | High | High variance (overfit) |
| Low | Low | Good fit |

### Fixing High Bias (Underfitting)

- More features
- More complex model
- Less regularization
- Longer training

### Fixing High Variance (Overfitting)

- More training data
- Fewer features
- More regularization
- Simpler model
- Ensemble methods

---

## Learning Curves

### Training Size Curve

Plot training and validation error vs training set size:

```
Error
  ^
  |  ______ Validation error
  | /
  |/_______ Training error
  +-----------------> Training size
```

**High bias**: Both curves plateau high, more data won't help
**High variance**: Gap between curves, more data helps

### Complexity Curve

Plot error vs model complexity:

```
Error
  ^
  |\___ Validation error
  |    \____/
  |      _____ Training error
  +-----------------> Complexity
```

**Optimal**: Where validation error is minimum

---

## Practical Evaluation Pipeline

### Step 1: Train/Val/Test Split

```python
# 60/20/20 split
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)
```

### Step 2: Cross-Validate on Training

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print(f"CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
```

### Step 3: Tune Hyperparameters

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
grid = GridSearchCV(model, param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

### Step 4: Final Evaluation on Test

```python
# Train best model on full training data
best_model = grid.best_estimator_
best_model.fit(X_train, y_train)

# Evaluate on held-out test set
from sklearn.metrics import classification_report
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## Interview Questions

### Q1: "Your model has 95% accuracy but stakeholders say it doesn't work. What's wrong?"

**Strong answer**:
> "95% accuracy can be misleading. I'd investigate:
>
> 1. **Class imbalance** — If 95% of data is class A, predicting 'always A' achieves 95% accuracy while being useless. I'd check precision/recall for the minority class.
>
> 2. **Wrong metric** — Maybe the business cares about recall (catching all positives) but high accuracy comes from good precision. I'd align metrics with business goals.
>
> 3. **Slice performance** — Overall accuracy might hide poor performance on important segments. I'd evaluate on key user groups.
>
> 4. **Training-serving skew** — Production data might differ from training data. I'd compare distributions.
>
> 5. **Business vs ML metrics** — 95% accuracy might not translate to business outcomes. I'd check downstream metrics."

### Q2: "How do you choose between precision and recall?"

**Strong answer**:
> "Depends on the cost of errors:
>
> **Prioritize Precision** (avoid false positives):
> - Spam detection: Users hate legitimate email marked as spam
> - Fraud alerts: Too many false alarms cause alert fatigue
> - Drug approval: False claims of efficacy are dangerous
>
> **Prioritize Recall** (avoid false negatives):
> - Cancer screening: Missing cancer is worse than extra tests
> - Security threats: Missing an attack is catastrophic
> - Search results: Better to include irrelevant than miss relevant
>
> In practice, I'd:
> 1. Quantify costs with stakeholders (cost of FP vs FN)
> 2. Plot precision-recall curve
> 3. Choose threshold that minimizes expected cost
> 4. If unsure, start with F1 (harmonic mean) as a balanced approach"

### Q3: "What's the difference between ROC-AUC and PR-AUC?"

**Strong answer**:
> "Both measure ranking quality but behave differently with class imbalance:
>
> **ROC-AUC** (TPR vs FPR):
> - Not affected by class imbalance mathematically
> - But can be misleading — high AUC while minority class poorly predicted
> - Use when classes are roughly balanced
>
> **PR-AUC** (Precision vs Recall):
> - Directly measures performance on positive class
> - More informative for imbalanced data
> - Use when positive class is rare or more important
>
> Example: 1% positive class, model predicts all negative:
> - ROC-AUC: ~0.5 (correctly shows randomness)
> - PR-AUC: Very low (shows failure on positives)
>
> I'd use PR-AUC for imbalanced classification problems like fraud detection or disease diagnosis."

### Q4: "Why use cross-validation instead of a single train/test split?"

**Strong answer**:
> "Single split has two problems:
>
> 1. **Wasted data** — Test set isn't used for training. With limited data, this hurts.
>
> 2. **High variance estimate** — Performance depends heavily on which samples end up in test. Get a different split, get a different estimate.
>
> Cross-validation solves both:
> - Every sample is used for training (in k-1 folds) and validation (in 1 fold)
> - Average over k folds gives more stable estimate
> - Standard deviation shows estimate uncertainty
>
> I'd use:
> - **K-fold (k=5 or 10)** for most cases
> - **Stratified K-fold** for imbalanced classification
> - **Time series split** for temporal data
> - **Group K-fold** when samples from same entity must stay together"

---

## Quick Reference Card

```
CLASSIFICATION METRICS
─────────────────────────────────────────────────
Precision: TP/(TP+FP) — "Of predicted positive, how many correct?"
Recall:    TP/(TP+FN) — "Of actual positive, how many found?"
F1:        2·P·R/(P+R) — Harmonic mean
ROC-AUC:   Area under TPR vs FPR — Ranking quality
PR-AUC:    Area under Precision vs Recall — Imbalanced data

REGRESSION METRICS
─────────────────────────────────────────────────
RMSE: √(Σ(y-ŷ)²/n) — Same units, penalizes large errors
MAE:  Σ|y-ŷ|/n     — Robust to outliers
R²:   1 - SS_res/SS_tot — Variance explained [0, 1]

BIAS-VARIANCE
─────────────────────────────────────────────────
High bias:     Train high, Val high → More complexity
High variance: Train low, Val high → More regularization

CROSS-VALIDATION
─────────────────────────────────────────────────
K-Fold:     Default (k=5 or 10)
Stratified: Imbalanced classes
Time Split: Temporal data
Nested:     Tune + evaluate

PIPELINE
─────────────────────────────────────────────────
1. Split: Train/Val/Test (60/20/20)
2. CV: Estimate performance on training
3. Tune: GridSearch with CV
4. Evaluate: Final test set (once!)
```

---

**Previous**: [← 08_Clustering](./08_Clustering.md) | **Back to Index**: [00_INDEX](./00_INDEX.md)
