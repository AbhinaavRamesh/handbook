# ML Fundamentals & Concepts

> **Core ML theory** for SDE-ML, MLE, and AI Engineer interviews

---

## Overview

This section covers the theoretical foundations of ML that top tech companies evaluate in concept-based interviews. You'll need to explain these concepts clearly and know when to apply them.

---

## Document Structure

| Document | Focus |
|----------|-------|
| [concepts-overview](./concepts-overview) | How to approach ML concept questions |
| [linear-regression](./linear-regression) | OLS, gradient descent, assumptions |
| [logistic-regression](./logistic-regression) | Binary classification, MLE, decision boundary |
| [decision-trees](./decision-trees) | Splitting criteria, pruning, CART |
| [svm](./svm) | Margins, kernels, soft margin |
| [knn](./knn) | Distance metrics, k selection, curse of dimensionality |
| [neural-networks](./neural-networks/) | Backprop, activations, architectures |
| [clustering](./clustering) | K-Means, DBSCAN, hierarchical |
| [model-evaluation](./model-evaluation) | Metrics, cross-validation, bias-variance |

---

## Interview FAQ Section

Comprehensive Q&A guides with visualizations for the most commonly asked ML interview questions.

| Topic | Key Questions |
|-------|---------------|
| [Interview FAQ Index](./interview-faq/) | Complete FAQ navigation |
| [Bias-Variance Tradeoff](./interview-faq/core-concepts/bias-variance) | Error decomposition, learning curves |
| [Regularization](./interview-faq/core-concepts/regularization) | L1 vs L2, dropout, early stopping |
| [Gradient Descent](./interview-faq/optimization/gradient-descent) | SGD, Adam, momentum, saddle points |
| [Cross-Validation](./interview-faq/optimization/cross-validation) | K-fold, data leakage, stratified CV |
| [Ensemble Methods](./interview-faq/models/ensemble-methods) | Bagging vs boosting, XGBoost |
| [Hypothesis Testing](./interview-faq/statistics/hypothesis-testing) | P-values, A/B testing, Type I/II errors |

---

## Interview Question Types

| Type | What They Ask | Example |
|------|--------------|---------|
| **Algorithm Mechanics** | How does X work? | "Explain how gradient descent works" |
| **Trade-offs** | When would you use X vs Y? | "Decision Tree vs Random Forest?" |
| **Assumptions** | What assumptions does X make? | "Linear regression assumptions?" |
| **Failure Modes** | When does X fail? | "When does KNN perform poorly?" |
| **Hyperparameters** | How do you tune X? | "How do you choose k in K-Means?" |

---

## Study Order

**Week 1**: Supervised Learning Basics
1. Linear Regression
2. Logistic Regression
3. Model Evaluation

**Week 2**: Tree-Based & Instance-Based
4. Decision Trees
5. KNN
6. SVM

**Week 3**: Deep Learning & Clustering
7. Neural Networks
8. Clustering (K-Means, DBSCAN)

---

## Quick Reference: Algorithm Selection

| Problem Type | First Choice | When to Use |
|--------------|--------------|-------------|
| Regression | Linear Regression | Linear relationship, interpretability |
| Binary Classification | Logistic Regression | Interpretable probabilities |
| Multi-class | Random Forest | Non-linear, robust |
| High-dimensional | SVM with RBF | n_features > n_samples |
| Clustering | K-Means | Spherical clusters, known k |
| Density-based clustering | DBSCAN | Arbitrary shapes, outliers |
