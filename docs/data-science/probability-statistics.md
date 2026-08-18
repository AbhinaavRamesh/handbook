# Probability & Statistics

> **The mathematical backbone of data science** - distributions, Bayes, hypothesis testing, confidence intervals

---

## Overview

Statistical reasoning is the most frequently tested skill in data science interviews. You need both the intuition and the math.

---

## Probability Foundations

### Rules

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

$$P(A \cap B) = P(A) \cdot P(B|A)$$

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}  \quad \text{(Bayes' Theorem)}$$

### Conditional Probability

| Concept | Formula | Intuition |
|---------|---------|----------|
| **Joint** | $P(A, B) = P(A) \cdot P(B \mid A)$ | Both events happen |
| **Marginal** | $P(A) = \sum_b P(A, B=b)$ | Sum over all possibilities of B |
| **Conditional** | $P(A \mid B) = P(A, B) / P(B)$ | A given B already happened |
| **Independence** | $P(A, B) = P(A) \cdot P(B)$ | Knowing B tells nothing about A |

### Bayes' Theorem - Interview Pattern

> "1% of the population has disease X. A test is 95% sensitive and 90% specific. A person tests positive. What's the probability they have the disease?"

$$P(\text{disease} \mid +) = \frac{P(+ \mid \text{disease}) \cdot P(\text{disease})}{P(+)} = \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.10 \times 0.99} = \frac{0.0095}{0.1085} \approx 8.8\%$$

**Key insight**: With low base rates, even good tests produce mostly false positives.

---

## Common Distributions

| Distribution | Type | Use Case | Parameters | Mean | Variance |
|-------------|------|----------|------------|------|----------|
| **Bernoulli** | Discrete | Single yes/no trial | $p$ | $p$ | $p(1-p)$ |
| **Binomial** | Discrete | Number of successes in $n$ trials | $n, p$ | $np$ | $np(1-p)$ |
| **Poisson** | Discrete | Events per time interval | $\lambda$ | $\lambda$ | $\lambda$ |
| **Geometric** | Discrete | Trials until first success | $p$ | $1/p$ | $(1-p)/p^2$ |
| **Uniform** | Continuous | Equal probability in range | $a, b$ | $(a+b)/2$ | $(b-a)^2/12$ |
| **Normal** | Continuous | Natural measurements | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ |
| **Exponential** | Continuous | Time between events | $\lambda$ | $1/\lambda$ | $1/\lambda^2$ |

### Central Limit Theorem

For $n$ i.i.d. samples with mean $\mu$ and variance $\sigma^2$:

$$\bar{X} \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{as } n \to \infty$$

**Interview implication**: Regardless of the population distribution, sample means are approximately normal for large $n$ (typically $n \geq 30$).

---

## Hypothesis Testing

### Framework

| Step | Action |
|------|--------|
| 1 | State $H_0$ (null) and $H_1$ (alternative) |
| 2 | Choose significance level $\alpha$ (typically 0.05) |
| 3 | Compute test statistic |
| 4 | Find p-value or compare to critical value |
| 5 | Reject $H_0$ if p-value $< \alpha$ |

### Error Types

| | $H_0$ True | $H_0$ False |
|---|---|---|
| **Fail to reject** | Correct ($1 - \alpha$) | Type II error ($\beta$) |
| **Reject** | Type I error ($\alpha$) | Correct (Power = $1 - \beta$) |

### Common Tests

| Test | When to Use | Assumptions |
|------|------------|-------------|
| **z-test** | Compare mean to known value, $n \geq 30$ | Normal or large $n$, known $\sigma$ |
| **t-test** | Compare means, small $n$ | Normal, unknown $\sigma$ |
| **Paired t-test** | Before/after measurements | Paired observations, normal differences |
| **Chi-squared** | Categorical data independence | Expected counts $\geq 5$ |
| **Mann-Whitney U** | Non-parametric comparison of two groups | Ordinal data, no normality needed |
| **ANOVA** | Compare 3+ group means | Normal, equal variances |

### P-Value

The probability of observing a result at least as extreme as the test statistic, assuming $H_0$ is true.

**Common misconception**: A p-value is NOT the probability that $H_0$ is true.

---

## Confidence Intervals

$$\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

| Confidence Level | $z_{\alpha/2}$ |
|-----------------|----------------|
| 90% | 1.645 |
| 95% | 1.96 |
| 99% | 2.576 |

**Interpretation**: If we repeated the experiment many times, 95% of the intervals would contain the true parameter.

**NOT**: "There is a 95% probability the true value is in this interval."

---

## Expectation & Variance

| Property | Expected Value | Variance |
|----------|---------------|----------|
| Constant | $E[c] = c$ | $\text{Var}(c) = 0$ |
| Linearity | $E[aX + b] = aE[X] + b$ | $\text{Var}(aX + b) = a^2 \text{Var}(X)$ |
| Sum | $E[X + Y] = E[X] + E[Y]$ | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ if independent |
| Product | $E[XY] = E[X]E[Y]$ if independent | — |

### Covariance & Correlation

$$\text{Cov}(X, Y) = E[XY] - E[X]E[Y]$$

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$$

| Correlation Type | Measures | When to Use |
|-----------------|----------|-------------|
| **Pearson** | Linear relationship | Continuous, normally distributed |
| **Spearman** | Monotonic relationship | Ordinal, non-normal |
| **Kendall's Tau** | Concordance of pairs | Small samples, ordinal |

---

## Interview Questions

1. **"What is the difference between a population and a sample?"**
   - Population: entire group. Sample: subset used for inference. Statistics estimate parameters.

2. **"Explain the law of large numbers vs. CLT."**
   - LLN: sample mean converges to population mean. CLT: distribution of sample means becomes normal.

3. **"A coin is flipped 100 times and lands heads 60 times. Is it fair?"**
   - $z = \frac{0.6 - 0.5}{\sqrt{0.5 \times 0.5 / 100}} = 2.0$, p-value $\approx 0.046$. Reject at $\alpha = 0.05$.

4. **"What's the difference between correlation and causation?"**
   - Correlation measures association. Causation requires: temporal precedence, association, and no confounders. Only experiments (RCTs) establish causation.
