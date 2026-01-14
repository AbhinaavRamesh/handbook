# 2. ML Domain Round

---

The ML Domain round evaluates your **practical application of ML knowledge** rather than pure theory. This is where your production experience with the Aruba Networking Copilot becomes highly relevant.

## What Gets Tested

At L4, interviewers evaluate your grasp and practical application of machine learning. This is fundamentally different from theoretical depth.

### Core Assessment Areas

| Area | What They're Looking For |
|------|--------------------------|
| **Algorithm Selection** | Why this algorithm over alternatives? Trade-offs? |
| **Data Preprocessing** | How do you handle real-world messy data? |
| **Feature Engineering** | What features matter? How to compute at scale? |
| **Evaluation Metrics** | Which metrics align with business goals? |
| **Debugging Strategies** | How do you diagnose and fix ML issues? |
| **Systems Thinking** | Beyond the model—pipelines, serving, monitoring |

---

## Real Recent Interview Questions

These are actual questions from Google L4 ML interviews in 2024-2025:

### Question 1: Email Ranking System

> "Given a user query, email content, and user profile, design a system that retrieves and ranks relevant emails."

**Follow-up:** Add personalization based on user profile.

**What they tested:**
- Transformer embeddings
- Cosine similarity for ranking
- PyTorch implementation discussion
- Cold start handling

---

### Question 2: NLP + Clustering

> "Design an open-ended modeling approach for [specific text classification problem]"

**What they tested:**
- Data preparation strategies
- Model selection rationale
- Evaluation methodology
- Scaling considerations

---

### Question 3: Content Moderation

> "Design a system that detects if multimedia/ad content violates terms or contains offensive materials"

**What they tested:**
- Multi-modal understanding
- Classification with high-stakes decisions
- False positive/negative trade-offs
- Human-in-the-loop considerations

---

### Question 4: Real-time Anomaly Detection

> "How would you design a real-time ML system for detecting user engagement anomalies on Google Search?"

**What they tested:**
- Streaming architecture
- Time-series anomaly detection
- Alert fatigue management
- Explainability

---

## Key Differentiators for L4

Strong candidates explicitly demonstrate these qualities:

### 1. Trade-off Thinking

**Weak:** "I would use a neural network because they're powerful"

**Strong:** "For this problem, I'd consider the trade-off between a neural network and gradient boosting. Given the structured tabular data and need for interpretability, I'd start with XGBoost. If we need to capture complex feature interactions and have sufficient data, we could explore neural approaches."

### 2. Bias-Variance Awareness

**Weak:** "We need to tune the model to get better accuracy"

**Strong:** "Looking at the gap between training and validation metrics, we seem to be overfitting. I'd address this through regularization, reducing model complexity, or gathering more training data. Specifically, for this deep learning model, I'd try dropout and early stopping first."

### 3. Production Constraints

**Weak:** "We can use BERT for this classification task"

**Strong:** "BERT gives us strong accuracy, but with a 500ms latency requirement and 10K QPS, we need to consider distillation to a smaller model, or use a simpler baseline with cached embeddings. Let me walk through the latency breakdown..."

### 4. Monitoring Strategy

**Weak:** "We deploy the model and monitor accuracy"

**Strong:** "Beyond accuracy, I'd monitor input distribution changes using KL divergence, prediction confidence distributions, and segment-level performance. For this recommendation system, we'd also track engagement metrics like CTR and watch time with a 7-day moving average to catch gradual drift."

### 5. End-to-End Thinking

**Weak:** "I'd train a model on this data"

**Strong:** "Let me walk through the full pipeline: data collection and validation, feature engineering with training-serving parity, model training with proper cross-validation, A/B testing strategy for deployment, and ongoing monitoring for drift detection."

---

## Question Archetypes and Frameworks

### Archetype 1: Classification System

**Examples:** Spam detection, content moderation, fraud detection

**Framework:**
```
1. Clarify: What's the cost of FP vs FN?
2. Data: Labels, imbalance, quality
3. Features: Domain-specific signals
4. Model: Start simple, justify complexity
5. Evaluation: Precision-recall based on business needs
6. Deployment: Threshold selection, A/B testing
7. Monitoring: Drift detection, feedback loops
```

**Your experience connection:** Your NER system for intent/entity extraction is fundamentally a classification problem. Discuss your multi-task DeBERTa architecture and how you achieved 98.5% accuracy.

---

### Archetype 2: Ranking/Recommendation System

**Examples:** Search ranking, content recommendations, email prioritization

**Framework:**
```
1. Clarify: Objective metric (engagement, satisfaction, revenue)
2. Architecture: Two-stage (retrieval → ranking)
3. Features: User, item, context, interactions
4. Model: Point-wise, pair-wise, or list-wise
5. Evaluation: NDCG, MRR, online A/B
6. Serving: Latency constraints, caching
7. Feedback: Implicit signals, exploration-exploitation
```

**Your experience connection:** Your Copilot's intent routing is essentially a ranking problem—which agent should handle this query?

---

### Archetype 3: NLP/Text Understanding

**Examples:** Query understanding, entity extraction, text classification

**Framework:**
```
1. Clarify: Task definition, languages, domains
2. Data: Annotation quality, coverage
3. Preprocessing: Tokenization, normalization
4. Model: Pretrained (BERT/etc) → fine-tune
5. Evaluation: Entity-level F1, exact match
6. Serving: Latency (distillation, quantization)
7. Edge cases: OOV, domain shift
```

**Your experience connection:** This is your wheelhouse. Your NER system with context-aware training pipeline and 98.5% accuracy demonstrates exactly this expertise.

---

### Archetype 4: Anomaly/Outlier Detection

**Examples:** Fraud detection, system health, engagement anomalies

**Framework:**
```
1. Clarify: What counts as "anomaly"?
2. Data: Labeled anomalies? Mostly normal?
3. Approach: Supervised if labels, unsupervised if not
4. Methods: Statistical, clustering, or deep learning
5. Evaluation: Precision at low FPR
6. Alerting: Explainability, actionability
7. Adaptation: New anomaly types over time
```

---

## How to Structure Your Response

### The 5-Minute Framework

**Minutes 0-1: Clarify**
- Restate the problem
- Ask 2-3 clarifying questions
- Confirm success metrics

**Minutes 1-2: High-Level Approach**
- State your overall strategy
- Mention 2-3 key considerations
- Get buy-in before diving deep

**Minutes 2-8: Deep Dive**
- Walk through your solution systematically
- Discuss trade-offs at each decision point
- Connect to business impact

**Minutes 8-10: Wrap-Up**
- Summarize key decisions
- Acknowledge limitations
- Suggest extensions or improvements

---

## Common Pitfalls to Avoid

### Pitfall 1: Jumping to the Model

**Wrong:** "I'd use a transformer for this..."

**Right:** "Let me first understand the data characteristics and constraints. What's our latency requirement? How much labeled data do we have?"

### Pitfall 2: Ignoring Scale

**Wrong:** "We compute embeddings for all users and items..."

**Right:** "With 100M users and 10M items, storing all embeddings would require... Instead, we could use approximate nearest neighbors or a two-stage retrieval approach."

### Pitfall 3: Forgetting Evaluation

**Wrong:** "We train the model and deploy it"

**Right:** "Before deployment, we'd evaluate using offline metrics like NDCG@10, then run an A/B test measuring engagement lift with appropriate sample size for statistical significance."

### Pitfall 4: No Monitoring Story

**Wrong:** [Silence about what happens after deployment]

**Right:** "Post-deployment, we monitor prediction distribution shifts, segment-level performance degradation, and set up automated alerts if accuracy drops below our threshold."

### Pitfall 5: Overcomplicating

**Wrong:** "We need a multi-task learning setup with contrastive pre-training and attention mechanisms..."

**Right:** "Let me start with a simple baseline—logistic regression with well-crafted features. Once we understand its limitations, we can justify more complexity."

---

## Practice Questions

Try these with a 45-minute timer:

1. **Spam Detection at Scale**
   > Design a spam detection system for Gmail that processes billions of emails daily.

2. **Query Understanding**
   > Design a system to understand user intent from search queries for Google Shopping.

3. **Video Recommendations**
   > Design a recommendation system for YouTube Shorts that maximizes engagement while maintaining diversity.

4. **Ad Click Prediction**
   > Design a system to predict whether a user will click on an ad in Google Search results.

5. **Document Summarization**
   > Design a system that automatically generates summaries for Google Docs.

---

## Your Competitive Advantage

Use these experiences in your responses:

| When They Ask About... | Mention Your Experience With... |
|------------------------|--------------------------------|
| NLP/NLU systems | Your patent-pending NER with 98.5% accuracy |
| Multi-model systems | Hierarchical multi-agent with LangGraph |
| Production scale | 6M+ devices, 100K+ customers |
| Business impact | 50% MTTR reduction |
| Feature engineering | Context-aware training pipeline v2.5 |
| Debugging ML systems | NER context persistence issues |
