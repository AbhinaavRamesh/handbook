# Machine Learning Interview FAQ

A comprehensive collection of frequently asked ML interview questions with detailed answers and visualizations. This guide covers core concepts tested at top tech companies (Google, Meta, Amazon, etc.) for MLE, SDE-ML, and Data Science roles.

---

## Quick Navigation

| Section | Topics Covered | Difficulty |
|---------|---------------|------------|
| [Core Concepts](#core-concepts) | Bias-Variance, Regularization, Overfitting | Fundamental |
| [Optimization](#optimization) | Gradient Descent, Cross-Validation | Intermediate |
| [Models](#models) | Ensemble Methods, Bagging vs Boosting | Intermediate |
| [Statistics](#statistics) | Hypothesis Testing, A/B Testing | Intermediate |

---

## Core Concepts

### [Bias-Variance Tradeoff](./core-concepts/bias-variance.md)

Understanding the fundamental tension between model simplicity and complexity.

**Key Topics:**
- What is bias? What is variance?
- The bias-variance decomposition formula
- How to diagnose high bias vs high variance
- Strategies to fix underfitting and overfitting
- Learning curves interpretation

**Visual Concepts:**
- Dartboard analogy
- Model complexity curves
- Polynomial fitting examples

---

### [Regularization Techniques](./core-concepts/regularization.md)

Techniques to prevent overfitting and improve model generalization.

**Key Topics:**
- L1 (Lasso) vs L2 (Ridge) regularization
- Why L1 produces sparse solutions
- Elastic Net regularization
- Dropout in neural networks
- Early stopping

**Visual Concepts:**
- Geometric interpretation of L1 vs L2
- Coefficient shrinkage paths
- Dropout visualization

---

## Optimization

### [Gradient Descent & Optimizers](./optimization/gradient-descent.md)

How neural networks and ML models learn through optimization.

**Key Topics:**
- Batch GD vs SGD vs Mini-batch GD
- Learning rate selection
- Momentum and its benefits
- Adam optimizer explained
- Saddle points and local minima
- Vanishing/exploding gradients

**Visual Concepts:**
- Learning rate effects
- Optimizer comparison paths
- Loss landscape visualization

---

### [Cross-Validation](./optimization/cross-validation.md)

Robust model evaluation and preventing data leakage.

**Key Topics:**
- K-Fold cross-validation
- Stratified vs regular K-Fold
- Leave-One-Out CV (LOOCV)
- Data leakage prevention
- Time series cross-validation
- Choosing the right K value

**Visual Concepts:**
- K-Fold split diagrams
- Data leakage illustration
- CV types comparison

---

## Models

### [Ensemble Methods](./models/ensemble-methods.md)

Combining multiple models for better predictions.

**Key Topics:**
- Bagging vs Boosting
- Random Forest explained
- AdaBoost algorithm
- Gradient Boosting
- XGBoost vs LightGBM vs CatBoost
- When to use which algorithm

**Visual Concepts:**
- Bagging vs Boosting architecture
- Bootstrap sampling
- Feature importance

---

## Statistics

### [Hypothesis Testing & A/B Testing](./statistics/hypothesis-testing.md)

Statistical foundations for making data-driven decisions.

**Key Topics:**
- Null and alternative hypotheses
- P-values and their interpretation
- Type I and Type II errors
- Statistical power
- A/B test design
- Multiple testing problem

**Visual Concepts:**
- Error type visualization
- P-value distributions
- Power analysis curves

---

## Interview Preparation Tips

### Before the Interview

1. **Master the fundamentals**: Bias-variance, regularization, and gradient descent appear in almost every ML interview
2. **Practice explaining visually**: Be ready to draw diagrams on a whiteboard
3. **Know the tradeoffs**: Every technique has pros and cons - understand when to use what
4. **Prepare practical examples**: Real-world scenarios where you applied these concepts

### During the Interview

1. **Start with the intuition**: Explain concepts simply before diving into math
2. **Use analogies**: The dartboard analogy for bias-variance is a classic example
3. **Discuss edge cases**: What happens when learning rate is too high? When K is too small?
4. **Connect concepts**: Show how regularization relates to bias-variance tradeoff

### Common Follow-up Questions

| Topic | Likely Follow-ups |
|-------|-------------------|
| Bias-Variance | "How would you detect this in practice?" |
| Regularization | "When would you use L1 vs L2?" |
| Gradient Descent | "Why does Adam work better for Transformers?" |
| Cross-Validation | "How do you handle time series data?" |
| Ensemble | "Random Forest vs XGBoost - how do you choose?" |
| A/B Testing | "How do you handle multiple comparisons?" |

---

## Quick Reference Tables

### Regularization Summary

| Technique | Sparsity | Feature Selection | Best For |
|-----------|----------|-------------------|----------|
| L1 (Lasso) | Yes | Yes | High-dimensional data |
| L2 (Ridge) | No | No | Multicollinearity |
| Elastic Net | Partial | Partial | Correlated features |
| Dropout | No | No | Neural networks |

### Optimizer Comparison

| Optimizer | Convergence | Memory | Best For |
|-----------|-------------|--------|----------|
| SGD | Slow | Low | CNNs, final fine-tuning |
| Adam | Fast | High | Transformers, quick prototyping |
| AdamW | Fast | High | Large language models |

### Ensemble Algorithm Selection

| Algorithm | Training | Interpretability | Best For |
|-----------|----------|------------------|----------|
| Random Forest | Parallel | High | Noisy data, feature importance |
| XGBoost | Sequential | Medium | Competitions, tabular data |
| LightGBM | Sequential | Medium | Large datasets |
| CatBoost | Sequential | Medium | Categorical features |

---

## Formula Cheat Sheet

### Bias-Variance Decomposition
```
Total Error = Bias² + Variance + Irreducible Error
```

### Regularization Terms
```
L1: λ Σ|wᵢ|
L2: λ Σwᵢ²
Elastic Net: λ₁ Σ|wᵢ| + λ₂ Σwᵢ²
```

### Gradient Descent Update
```
θ = θ - α ∇L(θ)
```

### Statistical Power
```
Power = 1 - β = P(Reject H₀ | H₀ is false)
```

---

## Additional Resources

- **Practice Problems**: LeetCode ML problems, Interview Query
- **Books**: "Hands-On ML" by Aurélien Géron, "ISLR" by James et al.
- **Courses**: Andrew Ng's ML course, fast.ai

---

*Last updated: January 2026*
