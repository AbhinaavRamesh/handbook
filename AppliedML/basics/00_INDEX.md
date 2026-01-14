# ML Fundamentals & Concepts

> **Core ML theory** for Google ML Engineer interviews

---

## Overview

This section covers the theoretical foundations of ML that Google evaluates in concept-based interviews. You'll need to explain these concepts clearly and know when to apply them.

---

## Document Structure

| Document | Focus |
|----------|-------|
| [01_Concepts_Overview.md](./01_Concepts_Overview.md) | How to approach ML concept questions |
| [02_Linear_Regression.md](./02_Linear_Regression.md) | OLS, gradient descent, assumptions |
| [03_Logistic_Regression.md](./03_Logistic_Regression.md) | Binary classification, MLE, decision boundary |
| [04_Decision_Trees.md](./04_Decision_Trees.md) | Splitting criteria, pruning, CART |
| [05_SVM.md](./05_SVM.md) | Margins, kernels, soft margin |
| [06_KNN.md](./06_KNN.md) | Distance metrics, k selection, curse of dimensionality |
| [07_Neural_Networks.md](./07_Neural_Networks.md) | Backprop, activations, architectures |
| [08_Clustering.md](./08_Clustering.md) | K-Means, DBSCAN, hierarchical |
| [09_Model_Evaluation.md](./09_Model_Evaluation.md) | Metrics, cross-validation, bias-variance |

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

---

**Back to Main Index**: [../00_INDEX.md](../00_INDEX.md)
