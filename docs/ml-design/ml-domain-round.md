# ML Domain Round

---

The ML Domain round evaluates your **practical application of ML knowledge** rather than pure theory. This is where your production experience becomes highly relevant.

## What Gets Tested

Interviewers evaluate your grasp and practical application of machine learning. This is fundamentally different from theoretical depth.

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

## Real Interview Questions

These are common ML domain interview questions:

### Question 1: Email Ranking System

> "Given a user query, email content, and user profile, design a system that retrieves and ranks relevant emails."

**Follow-up:** Add personalization based on user profile.

**What they test:**
- Transformer embeddings
- Cosine similarity for ranking
- Implementation discussion
- Cold start handling

---

### Question 2: NLP + Clustering

> "Design an open-ended modeling approach for [specific text classification problem]"

**What they test:**
- Data preparation strategies
- Model selection rationale
- Evaluation methodology
- Scaling considerations

---

### Question 3: Content Moderation

> "Design a system that detects if multimedia/ad content violates terms or contains offensive materials"

**What they test:**
- Multi-modal understanding
- Classification with high-stakes decisions
- False positive/negative trade-offs
- Human-in-the-loop considerations

---

### Question 4: Real-time Anomaly Detection

> "Design a real-time ML system for detecting user engagement anomalies"

**What they test:**
- Streaming architecture
- Time-series anomaly detection
- Alert fatigue management
- Explainability

---

## Key Differentiators

Strong candidates explicitly demonstrate these qualities:

### 1. Production Awareness
- Consider latency, throughput, cost
- Discuss monitoring and alerting
- Think about failure modes

### 2. Trade-off Analysis
- Accuracy vs. latency
- Complexity vs. maintainability
- Batch vs. real-time

### 3. Business Alignment
- Connect technical choices to business goals
- Discuss metrics that matter
- Consider user experience

### 4. Systematic Debugging
- Structured approach to ML debugging
- Data quality checks
- Model vs. system issues

---

## How to Answer ML Domain Questions

### Step 1: Clarify (2-3 min)
- What's the business goal?
- What data is available?
- What are the constraints (latency, scale)?

### Step 2: High-Level Design (5 min)
- End-to-end system architecture
- Key components and their interactions
- Data flow diagram

### Step 3: Deep Dive (15-20 min)
- Model architecture choices
- Feature engineering approach
- Training and evaluation strategy
- Serving considerations

### Step 4: Trade-offs (5 min)
- Alternatives considered
- Why this approach vs. others
- Known limitations

---

## Common ML Domain Topics

### Ranking & Recommendation
- Collaborative filtering vs. content-based
- Two-tower architectures
- Handling cold start
- Diversity and exploration

### NLP/NLU
- Intent classification
- Named entity recognition
- Semantic search
- Question answering

### Computer Vision
- Object detection
- Image classification
- Multi-modal systems
- Video understanding

### Time Series
- Anomaly detection
- Forecasting
- Trend analysis
- Seasonality handling

---

## Practice Questions

### Design Questions
1. Design a recommendation system for an e-commerce platform
2. Design a spam detection system for email
3. Design a search ranking system
4. Design a content moderation system

### Debugging Questions
1. Your model's production accuracy is lower than offline—why?
2. Users report recommendations are stale—how do you diagnose?
3. Model latency suddenly increased—what do you check?

### Trade-off Questions
1. When would you use a simpler model vs. deep learning?
2. Batch vs. real-time inference—how do you decide?
3. How do you balance precision vs. recall for your use case?

---

## Resources

- [ML System Design Primer](https://github.com/chiphuyen/machine-learning-systems-design)
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) by Chip Huyen
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) by Google
