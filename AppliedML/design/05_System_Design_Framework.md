# 5. ML System Design Framework

[← Back to Index](./00_INDEX.md) | [Previous: Real Interview Questions](./04_Real_Interview_Questions.md) | [Next: Key Phrases →](./06_Key_Phrases.md)

---

Use this 6-step framework for any ML system design question. It provides structure while allowing flexibility for different problem types.

## Overview: The 6 Steps

| Step | Time | Focus |
|------|------|-------|
| 1. Problem Clarification | 5 min | Requirements, constraints, success metrics |
| 2. High-Level Architecture | 5-10 min | End-to-end system diagram |
| 3. Feature Pipeline | 5-10 min | Data sources, feature engineering, feature store |
| 4. Model Training | 3-5 min | Algorithm choice, training infrastructure |
| 5. Serving & Inference | 5 min | Latency, throughput, deployment strategy |
| 6. Monitoring & Feedback | 5 min | Metrics, drift detection, retraining |

---

## Step 1: Problem Clarification (5 min)

**Goal:** Ensure you understand the problem before diving into solutions.

### Questions to Ask

| Category | Example Questions |
|----------|-------------------|
| **Business** | What defines success? What's the business metric? |
| **Scale** | How many users? Queries per second? Data volume? |
| **Latency** | Real-time or batch? What latency is acceptable? |
| **Accuracy** | How much error is tolerable? Cost of mistakes? |
| **Constraints** | Privacy? Regional compliance? Infrastructure limits? |

### Example Clarification Dialogue

> **Interviewer:** "Design a recommendation system for an e-commerce site."

> **You:** "Before I dive in, I'd like to clarify a few things:
> 1. What's our primary metric—revenue, engagement, or user satisfaction?
> 2. How many products and users are we dealing with?
> 3. What's the latency requirement—real-time or precomputed?
> 4. Are there cold start concerns—new users or new products?
> 5. Any constraints like fairness across sellers or diversity requirements?"

### What You Should Know After Step 1

- Primary success metric
- Scale (users, items, QPS)
- Latency requirements
- Key constraints
- Scope boundaries

---

## Step 2: High-Level Architecture (5-10 min)

**Goal:** Draw the end-to-end system before diving into components.

### Standard ML System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│                                                                     │
│   Batch Data ──────┐    Stream Data ──────┐    External APIs ───┐   │
│   (Data Lake)      │    (Kafka/PubSub)    │                     │   │
└────────────────────┼──────────────────────┼─────────────────────┼───┘
                     │                      │                     │
                     ▼                      ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FEATURE LAYER                                 │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    FEATURE STORE                            │   │
│   │    Offline Store (training)    │    Online Store (serving)  │   │
│   └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         │                                           │
         ▼                                           ▼
┌─────────────────────────────────┐    ┌─────────────────────────────┐
│       TRAINING LAYER            │    │       SERVING LAYER         │
│                                 │    │                             │
│  Training Pipeline              │    │  Model Server               │
│  ├── Data validation            │    │  ├── Load balancer          │
│  ├── Feature transform          │    │  ├── Inference workers      │
│  ├── Model training             │    │  ├── Prediction cache       │
│  ├── Evaluation                 │    │  └── Fallback logic         │
│  └── Model registry             │    │                             │
└─────────────────────────────────┘    └─────────────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────┐
                               │       MONITORING LAYER              │
                               │                                     │
                               │  Metrics │ Alerts │ Drift Detection │
                               │            │                        │
                               │            ▼                        │
                               │     Retraining Trigger              │
                               └─────────────────────────────────────┘
```

### Tips for Drawing

- Start with user request, end with response
- Show data flow with arrows
- Identify offline (training) vs online (serving) paths
- Highlight where you'll dive deeper
- Keep it clean—details come later

---

## Step 3: Feature Pipeline (5-10 min)

**Goal:** Define what data you need and how to transform it.

### Feature Categories

| Category | Examples | Compute Time | Storage |
|----------|----------|--------------|---------|
| **Static** | User demographics, item metadata | Batch | Cache |
| **Dynamic** | Recent clicks, session history | Real-time | Online store |
| **Aggregate** | 7-day average, rolling counts | Batch | Feature store |
| **Contextual** | Time of day, device, location | Real-time | Compute at request |
| **Cross** | User-item similarity | Varies | Depends on freshness |

### Feature Store Design Principles

1. **Single source of truth** for feature definitions
2. **Point-in-time correctness** for training
3. **Low latency** for online serving
4. **Versioning** for reproducibility
5. **Monitoring** for data quality

### Critical: Training-Serving Skew

> **Most ML production failures come from training-serving skew.**

| Cause | Problem | Solution |
|-------|---------|----------|
| Different code paths | Features differ | Shared transformation code |
| Time leakage | Training uses future data | Point-in-time joins |
| Missing features | Unavailable at serving | Graceful defaults |
| Different defaults | NaN handled differently | Centralized imputation |

---

## Step 4: Model Training (3-5 min)

**Goal:** Justify your algorithm choice and training approach.

### Algorithm Selection

| Data Type | Small Data | Large Data |
|-----------|------------|------------|
| Tabular | Logistic Regression, XGBoost | XGBoost, Neural Nets |
| Text | TF-IDF + Classifier | BERT, Fine-tuned LLMs |
| Images | Pre-trained CNNs | Custom CNN/ViT |
| Sequences | RNNs, Traditional | Transformers |

### Trade-off Discussion Template

> "I'm considering [Option A] vs [Option B]:
> 
> **Option A:** [Pros] / [Cons]
> **Option B:** [Pros] / [Cons]
> 
> Given [constraint], I'd recommend [choice] because [reason]."

### Evaluation Strategy

```
Training Data → K-Fold CV → Offline Metrics
                                ↓
                        Holdout Test Set
                                ↓
                         A/B Test (Online)
                                ↓
                        Long-term Metrics
```

---

## Step 5: Serving & Inference (5 min)

**Goal:** Design for production requirements.

### Latency Budget Example

```
Total budget: 100ms

Feature fetch:        20ms  ████
Feature transform:    10ms  ██
Model inference:      50ms  ██████████
Post-processing:      10ms  ██
Network overhead:     10ms  ██
                      ─────
Total:               100ms
```

### Optimization Techniques

| Technique | Speedup | Trade-off |
|-----------|---------|-----------|
| Distillation | 2-10x | Slight accuracy loss |
| Quantization (INT8) | 2-4x | Minimal accuracy loss |
| Pruning | 2-5x | Depends on sparsity |
| Caching | 10-100x | Staleness |
| Batching | Better throughput | Higher latency |

### Deployment Strategy

```
Current Model ────────────────────────────► Production Traffic
                                               │
New Model ─► Shadow Test ─► Canary (1%) ─► Gradual Rollout
                │               │
                ▼               ▼
         Compare metrics  Compare metrics
```

### Fallback Strategies

| Scenario | Fallback |
|----------|----------|
| Model timeout | Return cached prediction |
| Model error | Use rule-based default |
| Feature missing | Use default value |
| High load | Degrade to simpler model |

---

## Step 6: Monitoring & Feedback (5 min)

**This step is often missed by junior candidates—always cover it!**

### What to Monitor

| Category | Metrics | Alert Threshold |
|----------|---------|-----------------|
| **Performance** | Accuracy, precision, recall | Drop >5% |
| **Latency** | P50, P95, P99 | P95 >SLA |
| **Throughput** | QPS, error rate | Error >1% |
| **Data** | Input distribution, null rates | Drift detected |
| **Business** | CTR, conversion, revenue | Drop >3% |

### Drift Detection

| Drift Type | Detection Method |
|------------|------------------|
| Covariate | Compare input distributions (KL divergence) |
| Concept | Monitor prediction accuracy over time |
| Label | Compare label distributions |

### Retraining Triggers

| Trigger | When to Use |
|---------|-------------|
| Scheduled | Retrain weekly/monthly |
| Performance-based | When metrics drop below threshold |
| Drift-based | When distribution shift detected |
| Manual | Ad-hoc for specific issues |

### Feedback Loop

```
User Interactions → Label Collection → Data Warehouse
                                            ↓
                                    Training Pipeline
                                            ↓
                                       New Model
                                            ↓
                                        A/B Test
                                            ↓
                                    Deploy if better
```

---

## Time Management Summary

| Step | Minutes | Key Output |
|------|---------|------------|
| Clarify | 5 | Requirements confirmed |
| Architecture | 5-10 | System diagram |
| Features | 5-10 | Feature list + store design |
| Training | 3-5 | Algorithm + trade-offs |
| Serving | 5 | Latency + deployment |
| Monitoring | 5 | Metrics + retraining |
| Buffer | 5 | Follow-ups |
| **Total** | **~45 min** | |

---

## Quick Checklist

Before finishing, verify you've covered:

- [ ] Clarified business metric and success criteria
- [ ] Defined scale: users, QPS, data volume
- [ ] Drew high-level architecture diagram
- [ ] Discussed feature engineering and training-serving parity
- [ ] Explained model choice with trade-offs
- [ ] Covered serving: latency, caching, fallbacks
- [ ] Addressed monitoring and drift detection
- [ ] Discussed retraining triggers and feedback loops

---

## Common Mistakes to Avoid

| Mistake | Why It's Bad | What to Do Instead |
|---------|--------------|---------------------|
| Jumping to model | Misses requirements | Start with clarification |
| Ignoring scale | Solution won't work | Ask about QPS, data size |
| No monitoring | Incomplete design | Always cover observability |
| Single solution | Shows narrow thinking | Discuss alternatives |
| No trade-offs | Seems junior | Compare options explicitly |
| Forgetting fallbacks | Brittle system | Design for failure |

---

[← Back to Index](./00_INDEX.md) | [Previous: Real Interview Questions](./04_Real_Interview_Questions.md) | [Next: Key Phrases →](./06_Key_Phrases.md)
