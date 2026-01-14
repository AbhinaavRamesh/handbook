# ML Concepts Interview Overview

> **How to answer** ML theory and concepts questions at Google

---

## What Google Evaluates

ML Concepts interviews test your understanding of:
1. **Algorithm mechanics** — How does it actually work?
2. **Assumptions & limitations** — When does it fail?
3. **Trade-offs** — Why choose X over Y?
4. **Practical considerations** — How would you deploy it?

---

## The 4-Part Answer Framework

For any "Explain algorithm X" question, cover:

### 1. Core Intuition (30 seconds)
> "In one sentence, what does this algorithm do?"

**Example for Linear Regression**:
> "Linear regression finds the best-fit line through data by minimizing the squared distance between predictions and actual values."

### 2. How It Works (1-2 minutes)
Explain the mechanism at the appropriate level:
- **Math-light**: Conceptual description
- **Math-heavy**: Formulas, derivations

**Example for Linear Regression**:
> "We model y = wx + b, where w is the weight and b is the bias. We find w and b by minimizing the mean squared error. This can be solved analytically using the normal equation, or iteratively using gradient descent."

### 3. Key Properties (1 minute)
Cover assumptions, strengths, weaknesses:

| Property | Details |
|----------|---------|
| **Assumptions** | Linear relationship, independent errors, homoscedasticity |
| **Strengths** | Interpretable, fast, closed-form solution |
| **Weaknesses** | Can't model non-linear patterns |

### 4. Practical Considerations (30 seconds)
- When to use it
- How to tune it
- Production concerns

> "Use when you need interpretability, expect linear relationships, or as a baseline. Scale features for gradient descent. Monitor for feature drift in production."

---

## Question Type: Data Handling

### "How would you handle missing values?"

**Framework**:
1. **Understand the missingness** — MCAR, MAR, or MNAR?
2. **Simple approaches** — Drop, impute with mean/median/mode
3. **Advanced approaches** — KNN imputation, model-based imputation
4. **What to avoid** — Don't impute target variable

**Strong answer**:
> "First, I'd analyze the pattern of missingness. If it's random (MCAR), dropping rows might be fine for small amounts. For larger amounts, I'd impute — mean/median for numerical features, mode for categorical. For tree-based models, I might use a 'missing' indicator feature. I'd be careful not to impute using information that would leak the target."

---

## Question Type: Model Selection

### "How would you choose between Logistic Regression and Random Forest?"

**Framework**:
1. **Acknowledge both are valid** — No "wrong" answer
2. **List trade-offs** — Interpretability, performance, data requirements
3. **Connect to context** — What matters for this problem?
4. **Make a recommendation** — With justification

**Strong answer**:
> "It depends on the requirements. Logistic Regression gives interpretable coefficients and works well when the relationship is roughly linear. I'd start here if stakeholders need to understand the model. Random Forest handles non-linear patterns and interactions automatically, often with better accuracy, but it's a black box. If accuracy is paramount and interpretability less critical, I'd try Random Forest. In practice, I'd train both and compare performance on a validation set."

---

## Question Type: Evaluation

### "Your model has 95% accuracy but stakeholders say it's not working. What's happening?"

**Framework**:
1. **Question the metric** — Is accuracy the right metric?
2. **Consider class imbalance** — 95% could be bad
3. **Slice the data** — Performance might vary by segment
4. **Check the deployment** — Training-serving skew

**Strong answer**:
> "95% accuracy sounds good, but I'd first check class distribution. If 95% of samples are class A, a model predicting 'always A' would achieve 95% accuracy while being useless. I'd look at precision and recall for the minority class. I'd also slice performance by user segments — maybe it works for most users but fails for a critical segment. Finally, I'd check for training-serving skew — is production data different from training data?"

---

## Question Type: Production

### "How would you monitor this model in production?"

**Framework**:
1. **Performance metrics** — Accuracy, latency, throughput
2. **Input monitoring** — Feature drift, data quality
3. **Output monitoring** — Prediction distribution shifts
4. **Business metrics** — Downstream impact

**Strong answer**:
> "I'd monitor at four levels. First, system metrics — latency, error rate, throughput. Second, input monitoring — track feature distributions and alert if they drift significantly from training. Third, output monitoring — track prediction distribution; a sudden shift might indicate model degradation. Fourth, business metrics — if this model drives decisions, track the downstream impact like conversion rate or user engagement. I'd also set up automated retraining triggers based on performance degradation."

---

## Interview Rubric: What Gets High Scores

| Level | Data Handling | Model Questions | Evaluation | Production |
|-------|---------------|-----------------|------------|------------|
| **Weak** | Only knows one approach | Can't explain trade-offs | Uses wrong metrics | No production awareness |
| **Average** | Knows multiple approaches | Explains algorithms | Knows common metrics | Mentions monitoring |
| **Strong** | Discusses context-dependent choices | Explains when to use each | Segment-level evaluation | Full monitoring strategy |

---

## ML Glossary: Key Terms

| Term | Definition | Example |
|------|------------|---------|
| **Bias** | Error from wrong assumptions | Linear model on curved data |
| **Variance** | Error from sensitivity to training data | Deep tree overfits |
| **Regularization** | Penalty for model complexity | L1, L2, dropout |
| **Cross-validation** | Estimate generalization error | 5-fold CV |
| **Hyperparameter** | Not learned from data | Learning rate, k in KNN |
| **Feature engineering** | Create inputs from raw data | TF-IDF from text |
| **Data leakage** | Target information in features | Future data in training |
| **Class imbalance** | Unequal class frequencies | 1% fraud in transactions |
| **Training-serving skew** | Train/production data differs | Context persistence bug |
| **Concept drift** | Relationship changes over time | User behavior shifts |

---

## Common Mistakes to Avoid

| Mistake | Why It's Bad | Better Approach |
|---------|--------------|-----------------|
| "It depends" (alone) | Evasive | "It depends on X, Y, Z — here's how I'd decide" |
| Memorized definitions | Shows no understanding | Explain with intuition + examples |
| Ignoring assumptions | Missing critical failures | Always state assumptions |
| Only discussing accuracy | Incomplete evaluation | Discuss precision, recall, business metrics |
| No production thinking | Theoretical only | Mention monitoring, scaling, maintenance |

---

## How to Study

### For Each Algorithm, Know:

1. **One-sentence description** — What does it do?
2. **Mathematical formulation** — Loss function, optimization
3. **Training complexity** — Time and space
4. **Hyperparameters** — What to tune, how to tune
5. **Assumptions** — When it works, when it fails
6. **Comparison to alternatives** — Trade-offs

### Practice Method:

1. Read about algorithm
2. Implement from scratch (coding folder)
3. Explain to someone (or record yourself)
4. Answer practice questions out loud
5. Review and refine explanations

---

**Next**: [02_Linear_Regression →](./02_Linear_Regression.md)
