# Feature Engineering

> **Transforming raw data into model-ready inputs** - encoding, scaling, selection, and handling missing data

---

## Overview

Feature engineering is often the highest-leverage activity in applied ML. Good features can make a simple model outperform a complex one on raw data. Interviewers test your ability to think about data transformations and their implications.

---

## Numeric Transformations

### Scaling

| Method | Formula | When to Use | Sensitive to Outliers? |
|--------|---------|-------------|----------------------|
| **Standard Scaling** | $z = \frac{x - \mu}{\sigma}$ | SVM, logistic regression, neural nets | Yes |
| **Min-Max Scaling** | $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | When bounded range needed [0,1] | Yes |
| **Robust Scaling** | $x' = \frac{x - \text{median}}{\text{IQR}}$ | Data with outliers | No |
| **Log Transform** | $x' = \log(x + 1)$ | Right-skewed distributions | Reduces impact |
| **Power Transform** | Box-Cox / Yeo-Johnson | Make distribution more normal | Reduces impact |

### Binning

| Type | Method | Example |
|------|--------|--------|
| **Equal-width** | Fixed interval size | Age: 0-20, 20-40, 40-60, 60+ |
| **Equal-frequency** | Same count per bin | Quartile-based |
| **Custom** | Domain knowledge | Income brackets |

---

## Categorical Encoding

| Method | Cardinality | Preserves Order? | Output Dims | Use Case |
|--------|-------------|-------------------|-------------|----------|
| **Label Encoding** | Any | Yes (ordinal only) | 1 | Tree-based models, ordinal features |
| **One-Hot Encoding** | Low (< 20) | No | $k$ | Linear models, neural nets |
| **Binary Encoding** | Medium | No | $\lceil\log_2 k\rceil$ | Moderate cardinality |
| **Target Encoding** | High | No | 1 | High cardinality, with regularization |
| **Frequency Encoding** | Any | No | 1 | When category frequency matters |
| **Hash Encoding** | Very High | No | Fixed $d$ | Text features, very high cardinality |

### Target Encoding Pitfalls

- **Data leakage**: target information bleeds into features
- **Mitigation**: use leave-one-out or K-fold encoding within cross-validation
- **Smoothing**: blend category mean with global mean: $\text{enc} = \frac{n \cdot \bar{y}_{\text{cat}} + m \cdot \bar{y}_{\text{global}}}{n + m}$

---

## Missing Data Handling

| Strategy | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **Drop rows** | < 5% missing, MCAR | Simple | Loses data |
| **Drop columns** | > 50% missing | Removes noise | Loses information |
| **Mean/Median impute** | Numeric, MCAR | Simple, preserves mean | Reduces variance |
| **Mode impute** | Categorical | Simple | Can bias toward majority |
| **KNN impute** | MAR, moderate size | Uses relationships | Slow for large data |
| **Model-based (MICE)** | MAR, complex patterns | Handles multivariate | Complex, iterative |
| **Missing indicator** | When missingness is informative | Captures signal | Adds features |

---

## Feature Selection

### Filter Methods

| Method | Measures | For |
|--------|----------|-----|
| **Variance threshold** | Feature variance | Remove near-constant features |
| **Correlation** | Pairwise correlation | Remove redundant features |
| **Mutual information** | Non-linear dependency | Rank features by relevance |
| **Chi-squared** | Categorical independence | Categorical features vs. target |
| **ANOVA F-test** | Group mean differences | Numeric features vs. categorical target |

### Wrapper Methods

| Method | Approach | Trade-off |
|--------|----------|----------|
| **Forward selection** | Add best feature iteratively | Slow, may miss interactions |
| **Backward elimination** | Remove worst feature iteratively | Expensive for many features |
| **Recursive Feature Elimination (RFE)** | Train model, remove least important, repeat | Good with model feature importances |

### Embedded Methods

| Method | Mechanism |
|--------|----------|
| **Lasso (L1)** | Drives coefficients to zero |
| **Tree importance** | Split-based or permutation importance |
| **Elastic Net** | Combines L1 and L2 |

---

## Feature Interaction & Creation

| Technique | Example | When Useful |
|-----------|---------|-------------|
| **Polynomial features** | $x_1^2, x_1 \cdot x_2$ | Non-linear relationships |
| **Ratio features** | revenue / users = ARPU | Business metrics |
| **Aggregation** | Mean purchase per user | Entity-level features |
| **Time-based** | Day of week, hour, is_weekend | Temporal patterns |
| **Window features** | 7-day rolling average | Trend and momentum |
| **Lag features** | Value at $t-1$, $t-7$ | Time series |

---

## Dimensionality Reduction

| Method | Type | Preserves |
|--------|------|----------|
| **PCA** | Linear | Maximum variance |
| **t-SNE** | Non-linear | Local structure |
| **UMAP** | Non-linear | Local + global structure |
| **Autoencoders** | Non-linear (learned) | Learned representations |

---

## Interview Questions

1. **"How would you handle a categorical feature with 50,000 unique values?"**
   - Options: target encoding with smoothing, frequency encoding, hash encoding, or embedding layer (if neural net). Never one-hot — creates 50K sparse columns.

2. **"When should you scale features?"**
   - Required for distance-based (KNN, SVM, K-means), gradient-based (neural nets, logistic regression). Not needed for tree-based models (Random Forest, XGBoost).

3. **"How do you detect and handle data leakage in features?"**
   - Leakage: feature contains information from the future or from the target. Detect: check feature importance (suspiciously strong), check timestamps, check if feature exists at prediction time. Fix: remove or lag.

4. **"What's the curse of dimensionality?"**
   - As dimensions increase, data becomes sparse, distances become meaningless, and models need exponentially more data. Combat with feature selection, PCA, or regularization.
