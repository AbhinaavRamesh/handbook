# Multi-Head Attention

> Multi-head attention allows a model to attend to information from different representation subspaces at different positions simultaneously, dramatically increasing the expressiveness of the attention mechanism.

## Why Multiple Heads?

### The Limitation of Single-Head Attention

From Module 1, we learned that a single attention head computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

This single head performs attention in a fixed representation space determined by the dimensionality of $Q$, $K$, and $V$. However, a critical limitation emerges: **a single head forces all attention patterns through one bottleneck**.

Consider what happens in practice. When the model attends to a specific input position, it uses a single set of weights to combine all value vectors. This means:
- The same linear transformation applies to all positions
- One attention distribution must capture all types of relationships
- Short-range syntactic patterns and long-range semantic relationships compete for the same weights

Imagine a machine translation system translating "The president met with the ambassador, who discussed trade policy." A single attention head must simultaneously learn:
- Which words relate to nearby words (syntactic patterns like subject-verb agreement)
- Which nouns and pronouns refer to the same entity ("the ambassador" and "who")
- Which words affect a specific output position

These are fundamentally different types of relationships, yet a single head must balance all of them.

### The Insight: Multiple Specialized Experts

The key insight driving multi-head attention is elegant: **what if we ran multiple attention mechanisms in parallel, each learning different types of relationships?**

By splitting the attention computation across multiple heads, we allow:

1. **Different representation subspaces**: Each head operates on a different linear projection of the input, learning to focus on different aspects of the data.

2. **Diverse attention patterns**: Some heads might specialize in short-range syntactic relationships, others in long-range semantic dependencies, others in positional patterns.

3. **Richer information aggregation**: The final output combines information from all heads, creating a much richer representation than any single head could produce.

This is analogous to having a panel of experts, each specializing in a different domain. A medical diagnosis benefits not from one generalist, but from a cardiologist, a neurologist, and an oncologist all contributing their specialized knowledge.

### Empirical Evidence from Pre-trained Models

Research on BERT and GPT has demonstrated that this intuition is correct. By analyzing attention patterns in trained models, researchers have discovered:

- **Head 1**: Focuses on adjacent word positions (next token, previous token), likely learning local syntax
- **Head 2**: Concentrates attention on rare words and proper nouns
- **Head 3**: Attends to words in specific grammatical categories (verbs to nouns, subjects to verbs)
- **Head 4-8**: Learn increasingly abstract patterns and long-range dependencies

This specialization emerges naturally during training without explicit supervision—the model automatically discovers which attention patterns are useful for prediction and assigns different heads to different tasks.

## Multi-Head Attention Mathematical Formulation

### The Complete Formula

Multi-head attention is formally defined as:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

where each head is computed as:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Breaking this down:

**For each head $i$ (where $i = 1, 2, \ldots, h$):**

1. **Project inputs to subspace**: Apply learned linear transformations
   - $Q_i = QW_i^Q$ where $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
   - $K_i = KW_i^K$ where $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
   - $V_i = VW_i^V$ where $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$

2. **Apply attention in subspace**:
   $$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

3. **Concatenate all heads**:
   $$\text{Concat} = [\text{head}_1; \text{head}_2; \cdots; \text{head}_h]$$

   Output shape: $[seq\_len, h \times d_v]$

4. **Final linear projection**:
   $$\text{MultiHead}(Q,K,V) = \text{Concat} \cdot W^O$$

   where $W^O \in \mathbb{R}^{h \times d_v \times d_{model}}$

### Understanding the Weight Matrices

The weight matrices are crucial to understanding how multi-head attention works:

| Matrix | Shape | Purpose | Learned |
|--------|-------|---------|---------|
| $W_i^Q$ | $d_{model} \times d_k$ | Project input queries to $i$-th head's query space | Yes (per head) |
| $W_i^K$ | $d_{model} \times d_k$ | Project input keys to $i$-th head's key space | Yes (per head) |
| $W_i^V$ | $d_{model} \times d_v$ | Project input values to $i$-th head's value space | Yes (per head) |
| $W^O$ | $h \times d_v \times d_{model}$ | Combine and project all heads back to model dimension | Yes (shared) |

**Key insight**: Rather than using a single large projection of dimension $d_{model}$, each head uses a smaller projection of dimension $d_k$. This reduces computation while allowing multiple parallel attention mechanisms.

## Head Specialization Patterns

### What Do Different Heads Learn?

One of the most compelling aspects of multi-head attention is that different heads naturally specialize in different types of relationships. This specialization emerges without any explicit labels or guidance—it's a natural consequence of the optimization process.

### Common Specialization Patterns

**Positional Heads**: Some heads develop strong preferences for specific relative positions.

- Attend primarily to the next token (position $t+1$)
- Attend primarily to the previous token (position $t-1$)
- Attend to the beginning of the sequence (CLS token in BERT)
- These patterns emerge because they're predictive of local syntax

**Content-Based Heads**: Other heads focus on token content regardless of position.

- Attend to all instances of a particular word type (verbs, nouns, named entities)
- Attend to rare tokens that carry high information content
- Attend to semantic roles (subject, object, predicate)

**Long-Range Dependency Heads**: Some heads specialize in capturing distant relationships.

- One head might attend from verbs to their objects across a clause
- Another might track pronoun-antecedent relationships
- These patterns reveal the model's understanding of semantic roles

### Evidence from Literature

In the paper "Attention is Not Explanation" and related work analyzing BERT and GPT:

- **BERT Layer 1**: Early layers focus heavily on local context and position-adjacent tokens
- **BERT Layer 6**: Middle layers show increased diversity, with some heads tracking named entities and others tracking syntactic relationships
- **BERT Layer 12**: Later layers display highly abstract patterns, including semantic role labeling and discourse relationships

This layered specialization (easy patterns in early layers, complex patterns in deep layers) mirrors the information hierarchy in convolutional neural networks and provides insights into how transformers learn hierarchical representations.

## Parameter and Dimension Relationships

### The Dimensional Constraint

In practice, multi-head attention is organized so that the total dimensional complexity remains manageable. The key relationship is:

$$d_{model} = h \times d_k = h \times d_v$$

where:
- $h$ = number of heads
- $d_k$ = dimension of query/key in each head
- $d_v$ = dimension of value in each head
- $d_{model}$ = total model dimension

**Standard configurations:**

| Model Size | $d_{model}$ | Heads $(h)$ | $d_k = d_v$ | Params/Head |
|------------|-------------|------------|------------|------------|
| Small | 256 | 4 | 64 | 65,536 |
| Medium | 512 | 8 | 64 | 262,144 |
| Large (BERT) | 768 | 12 | 64 | 589,824 |
| XLarge (GPT-3) | 12,288 | 96 | 128 | 150 Million |

### Why Divisibility Matters

The requirement that $d_{model}$ divides evenly by $h$ is not arbitrary—it has important implications:

1. **Efficient computation**: When $d_{model} = h \times d_k$, we can reshape tensors cleanly without padding or truncation.

2. **Balanced capacity**: Each head operates on identical dimensionality, ensuring fair representation learning.

3. **Implementation efficiency**: Matrix operations on modern hardware (GPUs, TPUs) are optimized for certain tensor shapes. Divisible dimensions enable optimal kernel utilization.

4. **Gradient flow**: Symmetric head capacity ensures gradients flow evenly through different heads during backpropagation.

### Parameter Count Comparison

An important observation: **multi-head attention has roughly the same number of parameters as a single large attention head.**

**Single large head:**
- $Q, K, V$ projections: $3 \times d_{model}^2$
- Output projection: $d_{model}^2$
- **Total**: $4 \times d_{model}^2$

**Multi-head (8 heads, $d_k = d_{model}/8$):**
- Per-head projections: $h \times 3 \times d_{model} \times d_{model}/h = 3 \times d_{model}^2$
- Output projection: $d_{model}^2$
- **Total**: $4 \times d_{model}^2$

The parameter count is identical, yet multi-head attention is substantially more expressive because the projections are learned separately per head, creating implicit feature extraction at multiple scales.

## Computational Considerations

### Parallelization Benefits

Multi-head attention's greatest computational advantage is its parallel structure. All heads can be computed simultaneously:

```
Single-Head Computation (Sequential):
Input → Project Q,K,V → Attention → Output
         Time: O(T²)

Multi-Head Computation (Parallel):
Input → [Project for Head 1] ─→ Attention ─→ ┐
         [Project for Head 2] ─→ Attention ─→ ├→ Concatenate → Output
         [Project for Head 3] ─→ Attention ─→ ┤
         [Project for Head 4] ─→ Attention ─→ ┘
         Time: O(T²) for all heads combined
```

The time complexity remains $O(T^2)$ (where $T$ is sequence length), but the constant factor is reduced through parallel execution.

### Memory Footprint Analysis

Consider memory usage with 8 parallel heads:

**Naive approach (8 sequential passes):**
- Load input: $seq\_len \times d_{model}$
- 8 separate attention computations
- Memory: $\text{High due to sequential passes}$

**Optimized approach (batch processing all heads):**
- Reshape query/key/value into [batch, num\_heads, seq\_len, d_k]
- Single batched matrix multiplication
- Softmax on all heads simultaneously
- Memory: $\text{Actually lower than sequential!}$

```
Memory comparison for seq_len=512, d_model=512, batch=32:

Single head (d_model=512):
  Attention matrix: 32 × 512 × 512 = 8.4 MB (float32)

8 heads (d_k=64):
  Attention matrices: 32 × 8 × 512 × 512 = 67.1 MB total
  But: Computed in single batched operation
       With flash attention: ~20% memory reduction
```

Modern implementations use **FlashAttention** (by Dao et al., 2022) and similar techniques to compute multi-head attention with dramatically reduced memory usage through careful IO-aware algorithms.

### Why Multiple Heads Trump One Large Head

While parameter counts are similar, multi-head attention achieves better results than increasing head dimensionality:

| Configuration | $d_k$ | Computational Efficiency | Information Mixing | Typical Performance |
|---|---|---|---|---|
| 1 head, $d_k=512$ | 512 | Lower (less parallelizable) | Complete mixing | Baseline |
| 8 heads, $d_k=64$ | 64 | Higher (embarrassingly parallel) | Selective (richer) | +3-5% typically |

The multiple heads force the model to learn different projections, creating implicit regularization that prevents redundant feature learning.

## Interview Questions

### Question 1: Why use multiple attention heads instead of one larger head with the same total dimension?

**Sample Answer:**

This is an excellent question that touches on both expressiveness and optimization.

**Parameter count equivalence**: As mentioned, 8 heads with $d_k=64$ requires roughly the same parameters as 1 head with $d_k=512$. So why prefer multiple heads?

**The key reasons:**

1. **Forced feature diversity**: With multiple heads, the model must learn different projections of the input. Each $W_i^Q$, $W_i^K$, $W_i^V$ is trained independently, forcing each head to extract different features from the same input. This acts as an implicit regularizer, similar to ensemble learning.

   With a single large head, there's no such constraint—the model could learn redundant representations in different dimensions. Multiple heads prevent this.

2. **Richer attention patterns**: A single large attention matrix learns a single softmax distribution over positions. With multiple heads, we learn $h$ different distributions simultaneously. This is far more expressive.

   For example, on position $t$:
   - Head 1 might attend to position $t-1$ with weight 0.9
   - Head 2 might attend to position 3 (a noun from earlier) with weight 0.8
   - Head 3 might attend uniformly across all positions

   A single head must find a compromise distribution.

3. **Gradient flow during backpropagation**: With multiple heads, gradients during backpropagation flow through many diverse paths. This creates implicit gradient amplification and prevents dead zones in the parameter space.

4. **Computational parallelism**: Modern GPUs and TPUs excel at batch operations. Computing 8 heads in parallel is significantly faster than a single sequential operation on a much larger matrix, despite similar FLOPs.

5. **Empirical evidence**: BERT, GPT, and every modern transformer uses multiple heads. Extensive ablation studies show that removing heads uniformly degrades performance more than reducing dimension uniformly, suggesting each head contributes unique information.

**The intuition**: Think of it like asking for feedback from 8 different experts versus one expert who thinks about 8 different aspects in sequence. The 8 experts can think in parallel and often arrive at more diverse conclusions.

### Question 2: How many heads should you use? What are the trade-offs?

**Sample Answer:**

The number of heads is a hyperparameter that requires careful consideration. There's no universally optimal value, but we can analyze the trade-offs:

**Factors favoring more heads:**

1. **Increased expressiveness**: More heads mean more diverse attention patterns. Each head can specialize in different relationship types.

2. **Better learned representations**: With more heads, the model must distribute the feature learning across more diverse projections, reducing redundancy.

3. **Robustness**: Empirical evidence suggests 8-12 heads is typically optimal for tasks like language understanding. Benchmarks like GLUE consistently favor this range.

4. **Scaling patterns**: OpenAI's GPT-3 uses 96 heads for its 12,288-dimensional model. As model size increases, head count tends to increase proportionally.

**Factors favoring fewer heads:**

1. **Computational cost**: While the total FLOPs might be similar, maintaining 16+ heads creates overhead in tensor operations, memory management, and synchronization.

2. **Training stability**: More heads mean more independently learned attention patterns. This can increase variance in gradient updates, making training noisier.

3. **Interpretability**: With fewer heads, attention patterns are more concentrated and easier to analyze. Some researchers prefer 4-6 heads for interpretability.

4. **Resource constraints**: In mobile or edge deployment, fewer heads reduce memory footprint and latency.

**Empirical guidance from literature:**

- **8 heads**: Sweet spot for most NLP tasks (BERT, standard transformers)
- **12 heads**: Used for larger models (BERT-large, RoBERTa)
- **16+ heads**: Used for very large models (GPT-2, GPT-3 variants)
- **1-4 heads**: Only for toy models or resource-constrained environments

**Recommendation heuristic:**

$$h_{\text{typical}} = \frac{d_{model}}{64} \text{ (capped between 1 and 16)}$$

For $d_{model} = 512$: $h = 8$ heads
For $d_{model} = 768$: $h = 12$ heads
For $d_{model} = 1024$: $h = 16$ heads

**Trade-off analysis table:**

| Aspect | Few Heads (2-4) | Medium Heads (8-12) | Many Heads (16+) |
|--------|---|---|---|
| Expressiveness | Lower | Balanced | Higher |
| Compute speed | Slightly faster | Baseline | Slightly slower |
| Memory efficient | Better | Baseline | Slightly higher |
| Training stability | More stable | Baseline | More noisy |
| Interpretability | Easier | Moderate | Harder |
| Typical performance | Suboptimal | Optimal | Slightly worse |

**The key insight**: Diminishing returns kick in beyond 12-16 heads for most datasets. The optimal number depends on your problem size, computational budget, and interpretability needs.

### Question 3: What happens if you set num_heads = 1? What about num_heads = seq_length?

**Sample Answer:**

These extreme cases are instructive for understanding multi-head attention's design space.

**Case 1: num_heads = 1**

If $h = 1$, we revert to standard single-head attention:

$$\text{MultiHead}(Q,K,V) = \text{Attention}(QW_1^Q, KW_1^K, VW_1^V) \cdot W^O$$

Since there's only one set of projections, this is mathematically equivalent to learning three projection matrices ($W^Q$, $W^K$, $W^V$) followed by attention. The only difference from the original attention mechanism is the addition of the output projection $W^O$.

**Why this doesn't work well:**

1. Single attention distribution: Only one softmax distribution, creating a bottleneck
2. No head specialization: All relationship types compressed into one attention pattern
3. Empirically: Ablation studies show 1-head transformers underperform 8-head equivalents by 5-10%

**However**: A single head can work reasonably well with a much larger projection dimension. The point is that multiple smaller heads outperform one large head (same parameters), suggesting the multi-head structure itself provides benefit.

**Case 2: num_heads = seq_length**

Suppose $h = T$ (sequence length) and we keep $d_{model} = T \times d_k$, so $d_k = d_{model}/T$.

**What happens:**

- As sequence length grows, $d_k$ shrinks. For a sequence of length 512, $d_k$ becomes tiny (e.g., 1-2 dimensions for a 512-dimensional model).
- Each head operates on extremely low-dimensional projections
- Attention matrices become increasingly sparse (512×512)

**Why this fails:**

1. **Low-rank expressiveness**: With $d_k = 1$, each head's query/key projections are essentially scalars. The attention score is just a dot product of scalars, losing almost all expressive power.

2. **Too much specialization**: With hundreds of heads, there's no meaningful role differentiation. Most heads would learn near-identical patterns out of necessity.

3. **Gradient pathology**: With extremely sparse high-dimensional tensors, gradient flow becomes unstable. Many heads contribute near-zero gradients.

4. **Computational overhead**: Managing hundreds of separate projections and softmax operations creates severe overhead.

**Empirical result**: This approach performs terribly. Researchers have tried variable head counts, and extreme numbers (too many or too few) consistently underperform moderate values (8-16).

**The sweet spot principle:**

$$\text{Optimal: } 8 \leq h \leq 16$$
$$\text{Constraint: } d_k = d_v = d_{model} / h \text{ should be reasonable (e.g., 32-128)}$$

This ensures each head has sufficient dimensionality to learn meaningful transformations while maintaining diversity across heads.

## Quick Reference Card

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                    MULTI-HEAD ATTENTION CHEAT SHEET                         ║
╚═════════════════════════════════════════════════════════════════════════════╝

┌─ CORE FORMULA ───────────────────────────────────────────────────────────┐
│                                                                            │
│  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O                    │
│                                                                            │
│  where each head is:                                                      │
│  head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)                           │
│         = softmax((Q W_i^Q)(K W_i^K)^T / √d_k) V W_i^V                   │
│                                                                            │
│  Input dimensions:  Q, K ∈ ℝ^[batch, seq_len, d_model]                   │
│                     V ∈ ℝ^[batch, seq_len, d_model]                      │
│                                                                            │
│  Output dimension:  ℝ^[batch, seq_len, d_model]                          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌─ KEY DIMENSIONAL RELATIONSHIPS ──────────────────────────────────────────┐
│                                                                            │
│  Constraint:  d_model = h × d_k = h × d_v                                │
│                                                                            │
│  Typical values:  d_k = d_v = 64  (per head)                             │
│                   h = 8 to 16      (number of heads)                      │
│                   d_model = 512 to 768                                     │
│                                                                            │
│  BERT:         d_model = 768, h = 12, d_k = 64                           │
│  GPT-2:        d_model = 1600, h = 25, d_k = 64                          │
│  GPT-3 (175B): d_model = 12288, h = 96, d_k = 128                        │
│                                                                            │
│  Why divisibility matters:                                                │
│   • Clean tensor reshaping (no padding needed)                            │
│   • Balanced capacity across heads                                        │
│   • Efficient GPU/TPU utilization                                         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌─ WHY MULTIPLE HEADS? ─────────────────────────────────────────────────────┐
│                                                                            │
│  Problem with single head:                                                │
│   • One softmax distribution bottleneck                                   │
│   • Single learned attention pattern                                       │
│   • All relationship types compressed into one vector                     │
│                                                                            │
│  Solution: Multiple parallel attention mechanisms                          │
│   • Each head learns from different linear projections                    │
│   • Different heads specialize in different patterns                      │
│   • Combined output contains richer information                           │
│                                                                            │
│  Empirical evidence:                                                       │
│   ✓ BERT layer analysis: Different heads learn different tasks            │
│   ✓ Ablation studies: Removing heads reduces performance                  │
│   ✓ All modern transformers use 8+ heads                                  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌─ HEAD SPECIALIZATION PATTERNS ───────────────────────────────────────────┐
│                                                                            │
│  Positional Pattern Heads:                                                │
│   • Attend strongly to adjacent tokens (syntax)                           │
│   • Focus on specific relative positions                                  │
│                                                                            │
│  Content-Based Heads:                                                     │
│   • Attend to specific word types (verbs, nouns)                          │
│   • Track rare/important tokens                                           │
│                                                                            │
│  Long-Range Heads:                                                        │
│   • Capture semantic roles across clauses                                 │
│   • Track anaphoric relationships                                         │
│                                                                            │
│  Abstract Pattern Heads (deeper layers):                                  │
│   • Complex discourse relationships                                       │
│   • Semantic understanding                                                │
│                                                                            │
│  Note: Specialization emerges naturally during training without explicit   │
│       supervision. Different layers show different patterns (hierarchical).│
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌─ PARAMETER COUNT ANALYSIS ────────────────────────────────────────────────┐
│                                                                            │
│  Single-Head (d_model = 512):                                             │
│   W_q, W_k, W_v, W_o: 4 × 512² = 1,048,576 params                        │
│                                                                            │
│  Multi-Head (8 heads, d_k = 64, d_model = 512):                           │
│   Per-head projections: 8 × 3 × 64 × 512 = 786,432 params                │
│   Output projection: 512 × 512 = 262,144 params                           │
│   Total: 1,048,576 params (SAME!)                                         │
│                                                                            │
│  Key insight: Same parameters, but HIGHER EXPRESSIVENESS                  │
│   • Multiple independently-learned projections                            │
│   • Forces feature diversity                                              │
│   • Better gradient flow during training                                   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌─ COMPUTATIONAL CONSIDERATIONS ───────────────────────────────────────────┐
│                                                                            │
│  Parallelization:                                                         │
│   All h heads computed simultaneously → O(1) depth (despite O(h) width)   │
│   Modern GPUs naturally handle this parallelism                           │
│                                                                            │
│  Memory efficiency:                                                       │
│   Per-head dimension (d_k=64) smaller than full model                    │
│   Attention matrices more memory-efficient                                │
│   FlashAttention further reduces memory by 50%+                           │
│                                                                            │
│  Speed comparison (512 seq_len, d_model=512, batch=32):                  │
│   1 head, d_k=512:   ~X ms (slower, hard to parallelize)                 │
│   8 heads, d_k=64:   ~X/4 ms (faster, naturally parallel)                │
│                                                                            │
│  Trade-off: More heads = slightly more overhead, significantly better    │
│            performance. Typically 8-12 heads optimal.                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌─ OPTIMAL HEAD COUNT GUIDANCE ─────────────────────────────────────────────┐
│                                                                            │
│  Rule of thumb:                                                           │
│   h = d_model / 64  (typical for NLP)                                     │
│                                                                            │
│  Empirical sweet spots:                                                   │
│   d_model=256  →  h=4  heads                                              │
│   d_model=512  →  h=8  heads                                              │
│   d_model=768  →  h=12 heads                                              │
│   d_model=1024 →  h=16 heads                                              │
│                                                                            │
│  Why not extremes?                                                        │
│   h=1:           Single bottleneck, no specialization                     │
│   h>>16:         Redundant patterns, gradient instability                  │
│   h=seq_length: Extremely low-rank projections, poor learning             │
│                                                                            │
│  Practice: 8-12 heads works for most NLP tasks. Scale with model size.   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌─ CONNECTING TO OTHER MODULES ────────────────────────────────────────────┐
│                                                                            │
│  Module 1: Attention Fundamentals                                         │
│           Single attention mechanism, dot-product scoring                 │
│                                                                            │
│  Module 2: Self-Attention Mechanics (prerequisite)                        │
│           Q, K, V all from same source, attending to self                │
│                                                                            │
│  Module 3: Multi-Head Attention (CURRENT)                                │
│           Multiple parallel self-attention mechanisms                     │
│           Enables richer representations                                  │
│                                                                            │
│  Module 4: Positional Encoding (next)                                     │
│           Injects sequence position information                           │
│           Used with multi-head attention                                  │
│                                                                            │
│  Module 5: Full Transformer Architecture                                  │
│           Combines multi-head attention + FFN + residuals                │
│           Uses multiple layers of multi-head attention                    │
│                                                                            │
│  Modules 6+: Advanced Architectures                                       │
│            BERT, GPT, etc. use multi-head attention extensively           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌─ IMPLEMENTATION CHECKLIST ────────────────────────────────────────────────┐
│                                                                            │
│  When building multi-head attention:                                      │
│                                                                            │
│  ✓ Ensure d_model is divisible by num_heads                              │
│  ✓ Initialize W_i^Q, W_i^K, W_i^V, W^O with proper variance              │
│  ✓ Compute all heads in batch (reshape, not loop)                         │
│  ✓ Apply scaling factor 1/√d_k before softmax                            │
│  ✓ Concatenate outputs along embedding dimension                          │
│  ✓ Apply final projection W^O                                             │
│  ✓ Consider masking (padding, causal) before softmax                      │
│  ✓ For autoregressive: apply causal mask                                  │
│                                                                            │
│  Optional optimizations:                                                  │
│  • Use FlashAttention for memory efficiency                               │
│  • Implement in optimized backends (CUDA, Metal)                          │
│  • Cache Q, K, V projections if reused                                    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Key Takeaways

1. **Multiple heads are essential for expressiveness**: A single attention head is a bottleneck. Multiple independent heads learning different projections dramatically increase the model's expressiveness without adding parameters.

2. **Heads naturally specialize**: Different heads learn to focus on different types of relationships—syntactic patterns, semantic roles, positional relationships, and abstract patterns. This specialization emerges naturally during training.

3. **Dimension matters**: The standard constraint $d_{model} = h \times d_k$ ensures clean implementation and balanced capacity. Typical values are $h = 8$ heads with $d_k = 64$ for a 512-dimensional model.

4. **Parameter efficiency**: Multi-head attention has approximately the same parameter count as a single large head but is substantially more expressive. This is one of the key innovations of the Transformer architecture.

5. **Computational efficiency**: Despite parallel computation, multi-head attention is highly efficient on modern hardware. GPUs and TPUs naturally parallelize across heads, making computation faster than single large heads.

6. **Empirical validation**: Analysis of trained BERT and GPT models confirms that different heads learn meaningfully different patterns. Ablation studies show that removing heads uniformly degrades performance, validating the multi-head approach.

7. **Optimal head count**: 8-16 heads is typically optimal for most NLP tasks. Too few heads creates a bottleneck, while too many heads leads to redundancy and training instability.

This module is foundational for understanding modern transformers. Multi-head attention, combined with self-attention mechanics (Module 2) and positional encoding (Module 4), forms the core of the Transformer architecture (Module 5) that powers BERT, GPT, and state-of-the-art language models.

[VISUALIZATION NOTE: Images of head specialization patterns from BERT would be helpful here, showing different attention distributions for different heads. Heatmaps of attention matrices comparing single vs. multi-head would illustrate the concept effectively.]

---

**Image Placeholders:**
- **Figure 1**: Multi-head attention architecture diagram (input splits to multiple heads)
- **Figure 2**: Head specialization heatmaps (different attention patterns per head)
- **Figure 3**: BERT layer analysis (how head patterns change across layers)
- **Figure 4**: Computational parallelization diagram (sequential vs. parallel)
