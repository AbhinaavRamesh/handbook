# 4. Real Interview Questions

---

These are detailed walkthroughs of actual Google L4 ML interview questions. Practice explaining your thought process out loud.

## Question 1: Email Autocomplete/Suggestions

> **Problem:** "Design a system that provides autocomplete suggestions for email composition. Given the email context (subject, recipient, partial body), suggest completions."

### Clarifying Questions to Ask

1. "What's the latency requirement for suggestions?"
2. "Are we personalizing to individual users or general suggestions?"
3. "What languages do we need to support?"
4. "How do we handle sensitive/private information?"
5. "Is this mobile, desktop, or both?"

### High-Level Approach

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  User Input │────►│  Candidate  │────►│   Ranking   │────► Suggestions
│  (Context)  │     │  Generation │     │   Model     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    [LLM / N-gram]      [Personalization]
                                       [Relevance Score]
```

### Detailed Solution

#### 1. Data and Features

**Input features:**
- Subject line (tokenized, embedded)
- Recipient(s) (relationship, past interaction patterns)
- Partial email body (last N tokens)
- User's writing style profile (if personalized)
- Time of day (professional vs casual language)
- Email thread context (if reply)

**Training data:**
- Historical sent emails (anonymized)
- User typing patterns and corrections
- Which suggestions were accepted vs ignored

#### 2. Model Architecture

**Option A: Fine-tuned Language Model**
```
Approach: Fine-tune GPT/T5 on email completion task
Pros: High quality, context-aware completions
Cons: High latency, compute cost
Use when: Quality is paramount, can afford compute
```

**Option B: Transformer Encoder-Decoder**
```
Approach: Seq2seq with attention over context
Pros: Balance of quality and speed
Cons: Requires more training data
Use when: Medium latency requirements (200-500ms)
```

**Option C: N-gram with Neural Reranking**
```
Approach: Fast n-gram generation, neural rerank
Pros: Very fast, predictable latency
Cons: Lower quality for complex completions
Use when: Strict latency (<100ms)
```

**Recommended:** Start with Option B, with Option C fallback for timeout cases.

#### 3. Inference Optimization

```
User types → Local prefix cache check
                    │
         ┌──────────┴──────────┐
         │                     │
    Cache hit              Cache miss
         │                     │
         ▼                     ▼
  Return cached           Model inference
  suggestions             (batched, async)
                               │
                               ▼
                         Update cache
```

**Latency optimizations:**
- Speculative decoding for faster generation
- Knowledge distillation to smaller model
- Quantization (INT8) for faster inference
- Prefix caching for common patterns
- Async prefetching as user types

#### 4. Privacy Considerations

**Critical for this problem:**
- Never train on identifiable user data
- Differential privacy in training
- No PII in suggestions (filter before serving)
- User opt-out mechanism
- Data retention policies

#### 5. Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Acceptance rate | % of suggestions clicked | >20% |
| Keystroke savings | Chars saved / total chars | >15% |
| Latency P95 | 95th percentile response time | <150ms |
| User satisfaction | Survey/feedback score | >4.0/5.0 |

### Follow-up Questions and Answers

**Q: "How do you handle users with few emails (cold start)?"**

A: "For cold start, I'd use a tiered approach:
1. Start with a general model trained on aggregate patterns
2. Leverage signals from similar users (collaborative filtering)
3. Quickly adapt using first few emails (meta-learning)
4. Weight suggestions toward safer, generic options initially"

**Q: "How do you prevent repetitive or generic suggestions?"**

A: "Multiple approaches:
1. Diverse beam search to generate varied candidates
2. Penalize recently shown suggestions
3. Mix specific and general completions
4. Temperature sampling in generation
5. Business rules for minimum diversity"

**Q: "How do you measure quality without explicit feedback?"**

A: "Implicit signals:
- Suggestion acceptance rate
- Time spent before accepting/rejecting
- Post-acceptance editing (did they modify it?)
- Full message completion rate
- A/B test engagement metrics"

---

## Question 2: YouTube Recommendation System

> **Problem:** "Design a recommendation system for YouTube that suggests videos to users. The goal is to maximize user engagement while maintaining content diversity."

### Clarifying Questions to Ask

1. "What's our primary metric—watch time, clicks, or user satisfaction?"
2. "What's the latency requirement for recommendations?"
3. "How important is content diversity vs relevance?"
4. "Do we need to consider content safety/quality?"
5. "What scale are we talking about—users, videos, requests per second?"

### High-Level Architecture

```
                                YOUTUBE RECOMMENDATIONS

┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  ┌─────────────┐                                                      │
│  │ User Query  │                                                      │
│  │ + Context   │                                                      │
│  └──────┬──────┘                                                      │
│         │                                                             │
│         ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    CANDIDATE GENERATION                         │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │  │
│  │  │ Collaborative│  │ Content-   │  │ Knowledge Graph-based   │  │  │
│  │  │ Filtering   │  │ Based       │  │ (Topics, Creators)      │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │  │
│  │                                                                 │  │
│  │         100M videos → 10,000 candidates                         │  │
│  └─────────────────────────────────┬───────────────────────────────┘  │
│                                    │                                  │
│                                    ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                         RANKING                                 │  │
│  │                                                                 │  │
│  │   Deep Neural Network                                           │  │
│  │   ├── User features (history, demographics, device)             │  │
│  │   ├── Video features (embeddings, metadata, quality)            │  │
│  │   ├── Context features (time, location, session)                │  │
│  │   └── Interaction features (user-video similarities)            │  │
│  │                                                                 │  │
│  │         10,000 candidates → 100 ranked videos                   │  │
│  └─────────────────────────────────┬───────────────────────────────┘  │
│                                    │                                  │
│                                    ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                       RE-RANKING                                │  │
│  │                                                                 │  │
│  │   ├── Diversity injection (topic, creator diversity)            │  │
│  │   ├── Freshness boost (recent uploads)                          │  │
│  │   ├── Business rules (promotions, demotion of low quality)      │  │
│  │   └── Safety filters (age-appropriate, no violations)           │  │
│  │                                                                 │  │
│  │         100 ranked → Final 20 recommendations                   │  │
│  └─────────────────────────────────┬───────────────────────────────┘  │
│                                    │                                  │
│                                    ▼                                  │
│                         ┌─────────────────┐                           │
│                         │   User Feed     │                           │
│                         └─────────────────┘                           │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Candidate Generation

**Goal:** Reduce 100M+ videos to ~10,000 candidates quickly

**Approaches:**

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| Collaborative Filtering | Users who watched X also watched Y | Captures behavioral patterns | Cold start for new videos |
| Content-Based | Similar to what user watched | Works for new videos | Filter bubble |
| Knowledge Graph | Based on topics, creators, entities | Explainable | Requires maintained graph |

**Implementation:**
```
Two-tower model:
- User tower: Encodes user history → user embedding
- Video tower: Encodes video features → video embedding

At serving time:
1. Compute user embedding
2. ANN search over video embeddings (FAISS, ScaNN)
3. Return top-K similar videos
```

### Stage 2: Ranking

**Goal:** Score candidates for relevance to this user in this context

**Model architecture:**
```
Input Features:
├── User features
│   ├── Watch history (sequence of video embeddings)
│   ├── Search history
│   ├── Demographics (age, country)
│   └── Account age, subscription status
│
├── Video features
│   ├── Title/description embeddings
│   ├── Visual embeddings (thumbnail, frames)
│   ├── Upload age, view count, engagement rate
│   └── Creator features
│
├── Context features
│   ├── Time of day, day of week
│   ├── Device type
│   ├── Session history (what they just watched)
│   └── How they arrived (search, browse, notification)
│
└── Cross features
    ├── User-video similarity
    ├── Historical engagement with this creator
    └── Topic affinity scores

                    ↓
            Deep Neural Network
            (Multi-layer with attention)
                    ↓

Output: P(engagement) for each candidate
```

### Stage 3: Re-Ranking

**Goal:** Optimize for objectives beyond pure relevance

**Diversity injection:**
```python
# Ensure topic diversity in final list
selected = []
topic_counts = defaultdict(int)

for video in ranked_candidates:
    if topic_counts[video.topic] < MAX_PER_TOPIC:
        selected.append(video)
        topic_counts[video.topic] += 1
    if len(selected) >= NUM_RECOMMENDATIONS:
        break
```

**Other re-ranking factors:**
- Freshness: Boost recently uploaded videos
- Quality: Demote clickbait, low-quality content
- Safety: Filter age-inappropriate content
- Business rules: Promote YouTube Premium content

### Key Trade-offs to Discuss

| Trade-off | Option A | Option B | How to Decide |
|-----------|----------|----------|---------------|
| Relevance vs Diversity | Highly relevant but repetitive | Diverse but less relevant | A/B test user retention |
| Freshness vs Quality | New but unproven | Proven but stale | Balance with time decay |
| Exploration vs Exploitation | Try new content | Stick to known preferences | Multi-armed bandit |
| Complexity vs Latency | Rich features, better predictions | Fast but simpler | Latency budget |

### Evaluation

**Offline metrics:**
- NDCG@K: Ranking quality
- Recall@K: Coverage of relevant items
- Diversity metrics: Topic/creator spread

**Online metrics:**
- Watch time per session
- Click-through rate
- User retention (return visits)
- Explicit feedback (likes, not interested)

### Monitoring & Feedback Loop

```
User interactions (clicks, watch time, likes)
                    │
                    ▼
            Log to data warehouse
                    │
                    ▼
            Daily model retraining
                    │
                    ▼
            A/B test new model
                    │
                    ▼
            Gradual rollout if better
```

---

## Question 3: Fraud Detection with Text Alerts

> **Problem:** "Design a system that detects fraudulent credit card transactions in real-time and sends text alerts to users for verification."

### System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     FRAUD DETECTION SYSTEM                           │
│                                                                      │
│  Transaction                                                         │
│  Stream          ┌─────────────────────────────────────────────────┐ │
│       │          │              FEATURE ENGINE                     │ │
│       │          │                                                 │ │
│       │          │  ┌─────────────┐  ┌─────────────────────────┐  │ │
│       ▼          │  │   Real-time │  │     Batch Features      │  │ │
│  ┌─────────┐     │  │   Features  │  │   (from Feature Store)  │  │ │
│  │ Kafka/  │────►│  │             │  │                         │  │ │
│  │ PubSub  │     │  │ • Amount    │  │ • 30-day avg spend      │  │ │
│  └─────────┘     │  │ • Merchant  │  │ • Usual merchants       │  │ │
│                  │  │ • Location  │  │ • Travel patterns       │  │ │
│                  │  │ • Time      │  │ • Device history        │  │ │
│                  │  └─────────────┘  └─────────────────────────┘  │ │
│                  └───────────────────────┬─────────────────────────┘ │
│                                          │                           │
│                                          ▼                           │
│                  ┌─────────────────────────────────────────────────┐ │
│                  │                 ML MODELS                       │ │
│                  │                                                 │ │
│                  │   ┌──────────────────┐  ┌───────────────────┐  │ │
│                  │   │  Classification  │  │ Anomaly Detection │  │ │
│                  │   │  (Known Fraud)   │  │ (Novel Patterns)  │  │ │
│                  │   └────────┬─────────┘  └─────────┬─────────┘  │ │
│                  │            │                      │            │ │
│                  │            └──────────┬───────────┘            │ │
│                  │                       ▼                        │ │
│                  │              Ensemble Score                    │ │
│                  └───────────────────────┬─────────────────────────┘ │
│                                          │                           │
│                                          ▼                           │
│                  ┌─────────────────────────────────────────────────┐ │
│                  │              DECISION ENGINE                    │ │
│                  │                                                 │ │
│                  │   Score < 0.3  →  APPROVE                       │ │
│                  │   Score 0.3-0.7 → ALERT USER (text verification)│ │
│                  │   Score > 0.7  →  BLOCK + ALERT                 │ │
│                  └───────────────────────┬─────────────────────────┘ │
│                                          │                           │
│                                          ▼                           │
│                                   ┌─────────────┐                    │
│                                   │ Alert System│──► SMS Gateway     │
│                                   └─────────────┘                    │
│                                          │                           │
│                                          ▼                           │
│                          User Response (Yes/No fraud)                │
│                                          │                           │
│                                          ▼                           │
│                              ┌─────────────────────┐                 │
│                              │ Feedback → Retrain  │                 │
│                              └─────────────────────┘                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Feature Engineering

**Real-time features (computed on transaction):**
| Feature | Description |
|---------|-------------|
| Transaction amount | Raw and normalized by user history |
| Merchant category | One-hot or embedding |
| Time features | Hour, day of week, time since last txn |
| Location | Distance from home, new country? |
| Device fingerprint | Known device? New device? |

**Batch features (pre-computed):**
| Feature | Description |
|---------|-------------|
| Spending patterns | 7/30/90-day averages by category |
| Typical merchants | Frequently visited merchants |
| Travel history | Countries visited, patterns |
| Account age | Risk correlation |
| Previous fraud | History of chargebacks |

### Handling Class Imbalance

Fraud is rare (~0.1% of transactions), so imbalance handling is critical:

| Technique | Description | When to Use |
|-----------|-------------|-------------|
| Class weights | Higher loss for fraud class | Always as baseline |
| SMOTE | Synthetic minority oversampling | Moderate imbalance |
| Undersampling | Reduce majority class | Very large datasets |
| Focal loss | Down-weight easy examples | Neural networks |
| Threshold tuning | Adjust decision boundary | After training |

### Threshold Selection

```
                 Precision-Recall Trade-off

Precision ▲
   1.0    │   ●
          │    ●
   0.8    │     ●●
          │       ●●
   0.6    │         ●●●
          │            ●●●●
   0.4    │                ●●●●●
          │                     ●●●●●
   0.2    │                          ●●●
          │
   0.0    └───────────────────────────────────► Recall
          0.0   0.2   0.4   0.6   0.8   1.0

Choose threshold based on cost:
- High FP cost (user friction) → Higher precision
- High FN cost (fraud loss) → Higher recall
```

### Evaluation Metrics

| Metric | Why It Matters | Target |
|--------|----------------|--------|
| Precision | False alerts annoy users | >90% |
| Recall | Missing fraud = loss | >80% |
| Latency | Real-time requirement | <100ms |
| False positive rate | User experience | <1% |

---

## Question 4: Real-Time Engagement Anomaly Detection

> **Problem:** "Design a real-time ML system for detecting user engagement anomalies on Google Search."

### Problem Framing

**What are we detecting?**
- Sudden drop in click-through rate
- Unusual query patterns (potential bug or attack)
- Regional outages
- Feature launch issues
- Bot traffic spikes

### Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    ANOMALY DETECTION SYSTEM                        │
│                                                                    │
│   Search Events                                                    │
│        │                                                           │
│        ▼                                                           │
│   ┌─────────────────┐                                              │
│   │  Streaming      │                                              │
│   │  Aggregation    │                                              │
│   │  (1-min, 5-min) │                                              │
│   └────────┬────────┘                                              │
│            │                                                       │
│            ▼                                                       │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │              TIME-SERIES PROCESSING                         │  │
│   │                                                             │  │
│   │   For each metric (CTR, queries, latency):                  │  │
│   │                                                             │  │
│   │   ┌───────────────┐     ┌───────────────────────────────┐  │  │
│   │   │ Decomposition │────►│ Trend + Seasonality + Residual│  │  │
│   │   └───────────────┘     └───────────────────────────────┘  │  │
│   │                                    │                        │  │
│   │                                    ▼                        │  │
│   │   ┌───────────────────────────────────────────────────────┐│  │
│   │   │     Compare Residual to Expected Bounds               ││  │
│   │   │     (Statistical: >3σ, or ML: Isolation Forest)       ││  │
│   │   └───────────────────────────────────────────────────────┘│  │
│   └─────────────────────────────────┬───────────────────────────┘  │
│                                     │                              │
│                                     ▼                              │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                  ALERT GENERATION                           │  │
│   │                                                             │  │
│   │  Severity = f(magnitude, duration, affected scope)          │  │
│   │                                                             │  │
│   │  ┌─────────────────────────────────────────────────────┐   │  │
│   │  │ Low: Log for investigation                          │   │  │
│   │  │ Medium: Page on-call engineer                       │   │  │
│   │  │ High: Auto-rollback, exec notification              │   │  │
│   │  └─────────────────────────────────────────────────────┘   │  │
│   └─────────────────────────────────┬───────────────────────────┘  │
│                                     │                              │
│                                     ▼                              │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                  ALERT ENRICHMENT                           │  │
│   │                                                             │  │
│   │  • Which regions affected?                                  │  │
│   │  • Which query types?                                       │  │
│   │  • Correlated with recent deployments?                      │  │
│   │  • Similar past incidents?                                  │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Anomaly Detection Methods

| Method | Description | Best For |
|--------|-------------|----------|
| Statistical (Z-score) | Flag points > N std devs | Simple, interpretable |
| Prophet | Time-series with seasonality | Regular patterns |
| Isolation Forest | Tree-based outlier detection | Complex, multi-feature |
| LSTM Autoencoder | Learn normal patterns | Complex temporal patterns |
| Heuristic rules | Domain-specific thresholds | Known failure modes |

### Key Considerations

**1. Handling Seasonality**
```
Search traffic has strong patterns:
- Time of day (peak hours)
- Day of week (weekdays vs weekends)
- Holidays and events
- Geographic patterns

Solution: Decompose signal, only alert on residuals
```

**2. Alert Fatigue Prevention**
- Aggregate related anomalies
- Escalation only if anomaly persists
- Root cause grouping (one alert for related issues)
- Feedback loop to tune thresholds

**3. Explainability**
Every alert should include:
- What: Which metric, how abnormal
- When: Start time, duration
- Where: Affected regions/segments
- Why: Potential causes, correlated events
- Action: Recommended next steps

---

## Practice These Questions

Time yourself (45 minutes each):

1. **Google Maps Review Spam Detection**
   > Design a system to detect and filter spam reviews on Google Maps.

2. **Gmail Priority Inbox**
   > Design the ML system behind Gmail's priority inbox feature.

3. **Google Translate Quality Scoring**
   > Design a system that scores translation quality in real-time.

4. **YouTube Thumbnail Selection**
   > Design a system that recommends optimal thumbnails for video creators.

5. **Google Ads Click Fraud Detection**
   > Design a real-time system to detect click fraud on Google Ads.
