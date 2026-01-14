# 8. Quick Reference Cards

[← Back to Index](./00_INDEX.md) | [Previous: Experience Mapping](./07_Experience_Mapping.md)

---

Use these cards for quick review before your interview. Print them or save them on your phone.

---

## Card 1: Bias-Variance Trade-off

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BIAS-VARIANCE TRADE-OFF                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  HIGH BIAS (Underfitting)                                           │
│  • Training AND validation error both high                          │
│  • Model too simple                                                 │
│  • Fix: More complexity, more features, less regularization         │
│                                                                     │
│  HIGH VARIANCE (Overfitting)                                        │
│  • Training error low, validation error high                        │
│  • Model too complex                                                │
│  • Fix: Regularization, dropout, more data, simpler model           │
│                                                                     │
│  REGULARIZATION                                                     │
│  • L1 (Lasso): Drives weights to ZERO → Feature selection           │
│  • L2 (Ridge): Shrinks weights toward zero → Keeps all features     │
│  • λ controls bias-variance: Higher λ = more bias, less variance    │
│                                                                     │
│  CORE PRINCIPLE                                                     │
│  Total Error = Bias² + Variance + Irreducible Error                 │
│  Goal: Find the sweet spot that minimizes total error               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Card 2: Evaluation Metrics

```
┌─────────────────────────────────────────────────────────────────────┐
│                      EVALUATION METRICS                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CLASSIFICATION                                                     │
│  • Precision = TP / (TP + FP)     [When FP is costly]               │
│  • Recall    = TP / (TP + FN)     [When FN is costly]               │
│  • F1        = 2PR / (P + R)      [Balance P and R]                 │
│  • ROC-AUC   = Area under curve   [Overall discrimination]         │
│  • PR-AUC    = Precision-Recall   [Imbalanced data]                │
│                                                                     │
│  RANKING                                                            │
│  • NDCG@K    = Normalized DCG     [Order + relevance matter]        │
│  • MRR       = Mean Reciprocal Rank [First relevant item]           │
│  • Recall@K  = Relevant in top K  [Coverage]                        │
│                                                                     │
│  REGRESSION                                                         │
│  • MSE       = Mean squared error [Penalizes large errors]          │
│  • MAE       = Mean absolute error [Robust to outliers]             │
│  • R²        = Variance explained [Overall fit quality]             │
│                                                                     │
│  REMEMBER                                                           │
│  • Imbalanced data → Don't use accuracy!                            │
│  • Always do slice-based evaluation (by user group, time, etc.)     │
│  • Offline metrics ≠ Online metrics (A/B test!)                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Card 3: Data Drift Types

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA DRIFT TYPES                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SUDDEN DRIFT                                                       │
│  • Abrupt change in data distribution                               │
│  • Examples: COVID pandemic, competitor launch, viral event         │
│  • Detection: Spike in distribution metrics                         │
│                                                                     │
│  GRADUAL DRIFT                                                      │
│  • Slow erosion over time                                           │
│  • Examples: Evolving fraud patterns, demographic shifts            │
│  • Detection: Trend in rolling metrics                              │
│                                                                     │
│  CONCEPT DRIFT                                                      │
│  • P(Y|X) changes - relationship between features and labels        │
│  • Examples: What users want from search evolves                    │
│  • Detection: Model accuracy degrades, recalibration needed         │
│                                                                     │
│  COVARIATE DRIFT                                                    │
│  • P(X) changes - input distribution shifts                         │
│  • Examples: Different device types, new user demographics          │
│  • Detection: Input distribution monitoring (KL divergence)         │
│                                                                     │
│  HANDLING DRIFT                                                     │
│  • Monitor: Performance metrics, input distributions                │
│  • Alert: Automated thresholds                                      │
│  • Retrain: Scheduled, triggered, or continuous                     │
│  • Rollback: If new model degrades                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Card 4: System Design Checklist

```
┌─────────────────────────────────────────────────────────────────────┐
│                   ML SYSTEM DESIGN CHECKLIST                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  □ STEP 1: CLARIFY (5 min)                                          │
│    □ Business metric / success criteria                             │
│    □ Scale: users, QPS, data volume                                 │
│    □ Latency requirements                                           │
│    □ Constraints: privacy, compliance                               │
│                                                                     │
│  □ STEP 2: ARCHITECTURE (5-10 min)                                  │
│    □ Draw end-to-end system                                         │
│    □ Data → Features → Model → Serving → Monitoring                 │
│    □ Identify offline vs online paths                               │
│                                                                     │
│  □ STEP 3: FEATURES (5-10 min)                                      │
│    □ Key features and why they're predictive                        │
│    □ Feature store design                                           │
│    □ Training-serving parity                                        │
│                                                                     │
│  □ STEP 4: MODEL (3-5 min)                                          │
│    □ Algorithm choice with trade-offs                               │
│    □ Training infrastructure                                        │
│    □ Evaluation strategy                                            │
│                                                                     │
│  □ STEP 5: SERVING (5 min)                                          │
│    □ Latency budget breakdown                                       │
│    □ Deployment strategy (canary, A/B)                              │
│    □ Fallback / degradation                                         │
│                                                                     │
│  □ STEP 6: MONITORING (5 min)                                       │
│    □ Metrics to track                                               │
│    □ Drift detection                                                │
│    □ Retraining triggers                                            │
│    □ Feedback loops                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Card 5: Hyperparameter Tuning

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HYPERPARAMETER TUNING                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  METHODS                                                            │
│                                                                     │
│  Grid Search                                                        │
│  • Tries all combinations                                           │
│  • Simple but expensive                                             │
│  • Best for: Small parameter spaces                                 │
│                                                                     │
│  Random Search                                                      │
│  • Random sampling from distributions                               │
│  • Often outperforms grid search                                    │
│  • Best for: Large/continuous parameter spaces                      │
│                                                                     │
│  Bayesian Optimization                                              │
│  • Models performance as function of hyperparameters                │
│  • Intelligent exploration-exploitation                             │
│  • Best for: Expensive model training                               │
│                                                                     │
│  CRITICAL RULE                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Tune on VALIDATION data, NOT test data                      │   │
│  │ Otherwise you overfit hyperparameters to test set!          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  COMMON HYPERPARAMETERS                                             │
│  • Neural Nets: learning rate, batch size, dropout, architecture    │
│  • Trees: max depth, min samples, n_estimators, learning rate       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Card 6: Common Interview Questions

```
┌─────────────────────────────────────────────────────────────────────┐
│                  COMMON INTERVIEW QUESTIONS                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  FUNDAMENTALS                                                       │
│  Q: Explain bias-variance trade-off                                 │
│  Q: When to use L1 vs L2 regularization?                            │
│  Q: How does dropout prevent overfitting?                           │
│  Q: What's the difference between bagging and boosting?             │
│                                                                     │
│  SYSTEM DESIGN                                                      │
│  Q: Design a recommendation system for [X]                          │
│  Q: Design a fraud detection system                                 │
│  Q: Design a search ranking system                                  │
│  Q: How would you handle 1M QPS?                                    │
│                                                                     │
│  PRODUCTION ML                                                      │
│  Q: Model works offline but fails in production. How to debug?      │
│  Q: How do you detect data drift?                                   │
│  Q: How do you handle training-serving skew?                        │
│  Q: When would you retrain a model?                                 │
│                                                                     │
│  BEHAVIORAL                                                         │
│  Q: Tell me about a time you improved model performance             │
│  Q: Describe a complex system you designed                          │
│  Q: Tell me about a time you worked with ambiguous requirements     │
│  Q: How do you handle disagreements with teammates?                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Card 7: Key Numbers to Know

```
┌─────────────────────────────────────────────────────────────────────┐
│                     KEY NUMBERS TO KNOW                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  LATENCY EXPECTATIONS                                               │
│  • Real-time ML serving: <100ms                                     │
│  • User-facing recommendations: <200ms                              │
│  • Search results: <500ms total                                     │
│  • Feature store lookup: <10ms                                      │
│  • Database query: <50ms                                            │
│                                                                     │
│  SCALE REFERENCES                                                   │
│  • "Large scale": >1M QPS                                           │
│  • "Big data": >1TB data                                            │
│  • "Many users": >100M users                                        │
│  • "High cardinality": >1M unique values                            │
│                                                                     │
│  MODEL SIZES                                                        │
│  • BERT-base: 110M parameters                                       │
│  • GPT-3: 175B parameters                                           │
│  • Typical prod model: <1B parameters (latency constraints)         │
│                                                                     │
│  RULE OF THUMB                                                      │
│  • 10x data → ~1% accuracy improvement (diminishing returns)        │
│  • 2x model complexity → ~2x latency                                │
│  • P99 latency often 3-5x P50                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Card 8: Your Key Stats

```
┌─────────────────────────────────────────────────────────────────────┐
│                       YOUR KEY STATS                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ARUBA NETWORKING COPILOT                                           │
│                                                                     │
│  Scale                                                              │
│  • 6M+ network devices                                              │
│  • 100K+ customers                                                  │
│  • Real-time inference                                              │
│                                                                     │
│  Architecture                                                       │
│  • Hierarchical multi-agent system                                  │
│  • Built with LangGraph                                             │
│  • Orchestrator + specialized agents                                │
│                                                                     │
│  NER System                                                         │
│  • 98.5% accuracy (up from 94%)                                     │
│  • Multi-task DeBERTa architecture                                  │
│  • Contrastive learning                                             │
│  • Patent pending                                                   │
│                                                                     │
│  Business Impact                                                    │
│  • 50% reduction in Mean Time to Resolution                         │
│  • Direct customer satisfaction improvement                         │
│                                                                     │
│  Technical Highlights                                               │
│  • Context-aware training pipeline v2.5                             │
│  • Synthetic data generation                                        │
│  • Training-serving skew debugging                                  │
│  • ClickHouse time-series monitoring                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Card 9: Interview Day Reminders

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INTERVIEW DAY REMINDERS                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BEFORE THE INTERVIEW                                               │
│  □ Review these quick reference cards                               │
│  □ Prepare your STAR stories                                        │
│  □ Test your video/audio setup                                      │
│  □ Have water nearby                                                │
│  □ Have pen and paper for notes                                     │
│                                                                     │
│  DURING THE INTERVIEW                                               │
│  □ Ask clarifying questions first                                   │
│  □ Think out loud                                                   │
│  □ Discuss trade-offs explicitly                                    │
│  □ Connect to business impact                                       │
│  □ Don't forget monitoring!                                         │
│                                                                     │
│  KEY PHRASES TO USE                                                 │
│  • "Let me think about the trade-offs..."                           │
│  • "What metrics matter most for this problem?"                     │
│  • "How would we monitor this in production?"                       │
│  • "What would cause this to fail?"                                 │
│                                                                     │
│  IF YOU GET STUCK                                                   │
│  • "Let me step back and think about this differently..."           │
│  • "I'm not certain, but my intuition is... because..."             │
│  • "Can you help me understand [specific aspect] better?"           │
│                                                                     │
│  REMEMBER                                                           │
│  You've built production ML systems at scale.                       │
│  You've shipped features serving millions of devices.               │
│  You've debugged complex distributed systems.                       │
│                                                                     │
│  Design → Build → Deploy → Repeat                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Final Motivation

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                                                                     │
│            You've built production ML systems at scale.             │
│                                                                     │
│         You've shipped features serving millions of devices.        │
│                                                                     │
│           You've debugged complex distributed systems.              │
│                                                                     │
│                                                                     │
│                  ╔═══════════════════════════════╗                  │
│                  ║                               ║                  │
│                  ║   Design → Build → Deploy →   ║                  │
│                  ║           Repeat              ║                  │
│                  ║                               ║                  │
│                  ╚═══════════════════════════════╝                  │
│                                                                     │
│                                                                     │
│                       Go show them what you've got.                 │
│                                                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

[← Back to Index](./00_INDEX.md) | [Previous: Experience Mapping](./07_Experience_Mapping.md)
