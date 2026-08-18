# Data Science Fundamentals

> **Core data science skills** for Data Scientist, Applied Scientist, and Analytics Engineer interviews

---

## Overview

This section covers the foundational data science skills that companies evaluate beyond ML theory. These topics bridge the gap between knowing algorithms and applying them effectively: understanding data, designing experiments, writing SQL, and communicating results.

::: tip Who This Is For
If you're targeting Data Scientist, Applied Scientist, or Analytics-heavy ML Engineer roles, this section is essential. Many candidates over-index on modeling and under-prepare for these fundamentals.
:::

---

## Document Structure

| Document | Focus |
|----------|-------|
| [Probability & Statistics](./probability-statistics) | Distributions, Bayes' theorem, hypothesis testing |
| [Exploratory Data Analysis](./exploratory-data-analysis) | EDA workflow, visualization, data quality |
| [Feature Engineering](./feature-engineering) | Encoding, scaling, selection, missing data |
| [A/B Testing & Experiment Design](./ab-testing) | Randomization, power analysis, pitfalls |
| [SQL for Data Science](./sql-fundamentals) | Joins, window functions, interview patterns |
| [Time Series Analysis](./time-series) | Decomposition, forecasting, stationarity |

---

## Interview Question Types

| Type | What They Ask | Example |
|------|--------------|--------|
| **Statistical Reasoning** | Explain a concept and when to use it | "When would you use a t-test vs. a z-test?" |
| **Experiment Design** | Design an A/B test for a product change | "How would you test a new ranking algorithm?" |
| **Data Manipulation** | Write SQL or pandas to answer a question | "Find the top 5 users by 7-day retention" |
| **EDA & Communication** | Walk through how you'd explore a dataset | "You're given a table of user events. What do you do first?" |
| **Feature Engineering** | Transform raw data into model-ready features | "How would you encode categorical variables with 10K levels?" |
| **Time Series** | Forecast or detect anomalies in sequential data | "How would you forecast daily revenue?" |

---

## Study Order

**Week 1**: Foundations
1. Probability & Statistics
2. Exploratory Data Analysis

**Week 2**: Applied Skills
3. Feature Engineering
4. SQL for Data Science

**Week 3**: Experimentation & Sequences
5. A/B Testing & Experiment Design
6. Time Series Analysis

---

## Quick Reference: When to Use What

| Situation | Technique | Why |
|-----------|-----------|-----|
| Comparing two groups | t-test / Mann-Whitney | Test if difference is statistically significant |
| Comparing 3+ groups | ANOVA / Kruskal-Wallis | Avoid multiple comparison problems |
| Testing proportions | Chi-squared / Fisher's exact | Categorical outcome data |
| Measuring relationships | Pearson / Spearman correlation | Linear vs. monotonic association |
| Feature has outliers | Robust scaling or log transform | Reduce influence of extremes |
| High cardinality categorical | Target encoding or hashing | Avoid dimensionality explosion |
| Sequential data with trend | Differencing or decomposition | Remove non-stationarity |
| Designing an experiment | Power analysis first | Determine required sample size |

---

## Related Sections

| Section | Connection |
|---------|------------|
| [ML Fundamentals](/ml-fundamentals/) | Algorithms that operate on prepared data |
| [ML System Design](/ml-design/) | End-to-end systems that include data pipelines |
| [ML Coding](/ml-coding/) | From-scratch implementations of core algorithms |
| [Interview FAQ](/ml-fundamentals/interview-faq/) | Common ML theory questions that overlap with statistics |

---

*Estimated completion time: 2-3 weeks*
