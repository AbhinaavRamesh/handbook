# Deep Reinforcement Learning

> **Scaling RL with neural networks** — DQN, experience replay, and stability tricks

---

## One-Sentence Summary

Deep RL uses neural networks as function approximators to handle high-dimensional state spaces, with DQN innovations like experience replay and target networks enabling stable training on complex tasks like Atari games.

---

## Why Deep Learning + RL?

### The Curse of Dimensionality

Traditional tabular RL methods (Q-tables) work well when state and action spaces are small and discrete. Consider these scenarios:

| Problem | State Space Size | Feasible for Tables? |
|---------|------------------|---------------------|
| Tic-Tac-Toe | ~5,000 states | Yes |
| Chess | ~$10^{47}$ states | No |
| Atari (84x84 pixels) | $256^{84 \times 84}$ states | No |
| Robot control | Continuous | No |

For high-dimensional or continuous spaces, we cannot store a separate Q-value for every state-action pair.

### Function Approximation

Instead of memorizing Q-values, we **approximate** them with a parameterized function:

$$Q(s, a) \approx Q(s, a; \theta)$$

where $\theta$ are learnable parameters. Neural networks are universal function approximators, making them ideal for this task.

**Benefits:**
- **Generalization**: Similar states get similar Q-values
- **Scalability**: Handle any input dimensionality
- **Feature learning**: Network learns relevant features automatically

![DQN Architecture](/images/deep-rl/dqn-architecture.svg)

---

## DQN: The Breakthrough

DeepMind's DQN (2013-2015) achieved human-level performance on 49 Atari games using a single architecture. The key insight: raw pixels go in, optimal actions come out.

### Architecture

```
Input: 84x84x4 stacked grayscale frames
       ↓
Conv Layer 1: 32 filters, 8x8, stride 4, ReLU
       ↓
Conv Layer 2: 64 filters, 4x4, stride 2, ReLU
       ↓
Conv Layer 3: 64 filters, 3x3, stride 1, ReLU
       ↓
Flatten → FC Layer: 512 units, ReLU
       ↓
Output: Q-values for each action (e.g., 18 for Atari)
```

The network outputs Q-values for **all actions simultaneously**, enabling efficient action selection via $\arg\max_a Q(s, a; \theta)$.

### The Deadly Triad Problem

Naive deep RL combining (1) function approximation, (2) bootstrapping (TD learning), and (3) off-policy learning often **diverges**. DQN introduced two critical stabilization techniques.

---

## Innovation 1: Experience Replay

### The Problem with Online Learning

In standard RL, the agent learns from consecutive experiences $(s_t, a_t, r_t, s_{t+1})$. This causes two issues:

1. **Correlation**: Sequential samples are highly correlated, violating the i.i.d. assumption of SGD
2. **Catastrophic forgetting**: Recent experiences overwrite knowledge of earlier states

### The Solution: Replay Buffer

Store transitions in a **circular buffer** of fixed capacity (e.g., 1 million transitions):

$$\mathcal{D} = \{(s_i, a_i, r_i, s_{i+1})\}_{i=1}^{N}$$

During training, sample **random mini-batches** from this buffer.

![Experience Replay Buffer Animation](/images/deep-rl/experience-replay.gif)

**Benefits:**
- **Decorrelation**: Random sampling breaks temporal correlations
- **Data efficiency**: Each experience can be used multiple times
- **Stable gradients**: Mini-batch averaging reduces variance

### Implementation Insight

```
# Simplified replay buffer logic
buffer.append((state, action, reward, next_state, done))
if len(buffer) >= batch_size:
    batch = random.sample(buffer, batch_size)
    train_on_batch(batch)
```

---

## Innovation 2: Target Network

### The Moving Target Problem

In Q-learning, we minimize the TD error:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta)\right)^2\right]$$

Both the prediction $Q(s, a; \theta)$ and target $r + \gamma \max_{a'} Q(s', a'; \theta)$ depend on the **same** network. As $\theta$ updates, the target shifts, causing instability.

**Analogy**: Imagine trying to hit a target that moves every time you adjust your aim.

### The Solution: Frozen Target Network

Maintain **two networks**:
- **Online network** $Q(s, a; \theta)$: Updated every step
- **Target network** $Q(s, a; \theta^-)$: Updated periodically (every $C$ steps)

The loss becomes:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

![Target Network Update Animation](/images/deep-rl/target-network-update.gif)

**Update strategies:**
- **Hard update**: Copy weights every $C$ steps: $\theta^- \leftarrow \theta$
- **Soft update (Polyak averaging)**: $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ where $\tau \ll 1$

---

## Double DQN: Reducing Overestimation

### The Overestimation Problem

Standard Q-learning uses $\max$ to select actions:

$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

The $\max$ operator introduces **upward bias**: noise in Q-value estimates causes the maximum to be systematically overestimated.

![Q-Value Overestimation Problem](/images/deep-rl/overestimation.svg)

**Why it matters**: Overestimation propagates through bootstrapping, leading to unrealistically high Q-values and suboptimal policies.

### The Solution: Decouple Selection and Evaluation

**Double DQN** uses:
- **Online network** to **select** the best action
- **Target network** to **evaluate** that action

$$y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

This decoupling prevents the same noise from both selecting and evaluating actions, reducing overestimation significantly.

---

## Dueling DQN: Value and Advantage Streams

### Architectural Insight

Not all states require knowing the value of each action. In some states, any action leads to similar outcomes (e.g., empty road in driving). In others, action choice matters critically (e.g., approaching obstacle).

**Dueling DQN** separates the Q-function into two streams:

$$Q(s, a) = V(s) + A(s, a)$$

where:
- $V(s)$ = **State value**: How good is it to be in state $s$?
- $A(s, a)$ = **Advantage**: How much better is action $a$ than average?

### Architecture

```
      Shared Convolutional Layers
                ↓
          ┌─────┴─────┐
          ↓           ↓
    Value Stream   Advantage Stream
      FC → 1       FC → |A| values
          ↓           ↓
          └─────┬─────┘
                ↓
    Q(s,a) = V(s) + (A(s,a) - mean(A))
```

The subtraction of $\text{mean}(A)$ ensures identifiability (forces advantages to have zero mean).

**Benefits:**
- **Efficient learning**: V(s) updated by all actions, not just the one taken
- **Better generalization**: Learns state value independently of action-specific nuances

---

## Rainbow DQN: The Full Package

Rainbow (2017) combined six DQN improvements, achieving state-of-the-art Atari performance:

| Component | Purpose |
|-----------|---------|
| **Double DQN** | Reduce overestimation bias |
| **Prioritized Replay** | Sample important transitions more often |
| **Dueling Architecture** | Separate value and advantage |
| **Multi-step Learning** | Use n-step returns for faster propagation |
| **Distributional RL** | Model full return distribution, not just mean |
| **Noisy Nets** | Learned exploration via noisy network layers |

**Key insight from ablation studies**: Prioritized replay and multi-step learning provided the largest individual gains.

---

## Stability Challenges in Deep RL

### Why Deep RL is Hard

| Challenge | Cause | Mitigation |
|-----------|-------|------------|
| **Non-stationarity** | Policy changes as training progresses | Target networks, slower updates |
| **Sample correlation** | Sequential experience collection | Experience replay |
| **Overestimation** | Max operator bias | Double DQN |
| **Sparse rewards** | Infrequent feedback signals | Reward shaping, curiosity |
| **Credit assignment** | Long delays between action and reward | Multi-step returns, eligibility traces |
| **Hyperparameter sensitivity** | Many interacting components | Extensive tuning, Rainbow defaults |

### Practical Tips

1. **Start with known-good hyperparameters** (e.g., Rainbow defaults)
2. **Monitor Q-value magnitudes** — rapid growth indicates instability
3. **Use learning rate warmup** for initial stability
4. **Clip rewards to [-1, 1]** for consistent scale across games
5. **Frame stacking** (typically 4 frames) provides velocity information

---

## Interview Questions

### Q1: "Why do we need experience replay in DQN?"

> "Experience replay serves three critical purposes:
>
> **1. Decorrelation**: In online RL, consecutive samples are highly correlated (frame $t$ looks like frame $t+1$). This violates SGD's i.i.d. assumption and causes the network to overfit to recent experiences. Random sampling from a large buffer breaks these correlations.
>
> **2. Data efficiency**: Each transition can be sampled multiple times. Since collecting experience is often expensive (especially in real-world robotics), reusing data dramatically improves sample efficiency.
>
> **3. Stability**: By maintaining a buffer of diverse experiences, we prevent catastrophic forgetting of earlier learned behaviors. The network sees a stable distribution of training data.
>
> Without replay, DQN training diverges or learns suboptimal policies on most complex tasks."

### Q2: "Explain the target network and why it helps stabilize training."

> "In Q-learning, we minimize: $L = (r + \gamma \max Q(s', a'; \theta) - Q(s, a; \theta))^2$
>
> **The problem**: Both the prediction and target depend on the same weights $\theta$. Each gradient step changes $\theta$, which changes the target we're trying to reach. This creates a 'moving target' problem — like trying to hit something that moves every time you adjust your aim.
>
> **The solution**: Maintain a separate target network $\theta^-$ that's updated slowly:
> - Every $C$ steps (e.g., 10,000), copy: $\theta^- \leftarrow \theta$
> - Or use soft updates: $\theta^- \leftarrow 0.001\theta + 0.999\theta^-$
>
> Now the target $r + \gamma \max Q(s', a'; \theta^-)$ stays fixed for many updates, giving the online network a stable objective to optimize. This dramatically reduces variance and prevents divergence."

### Q3: "What is the overestimation problem in Q-learning and how does Double DQN address it?"

> "**The problem**: Standard Q-learning uses $y = r + \gamma \max_a Q(s', a)$. The max operator is biased upward when there's estimation noise.
>
> Consider: if true Q-values are all 0, but estimates have noise $\pm\epsilon$, then $\max$ will pick the positive noise, giving expected value $> 0$. This bias compounds through bootstrapping across many states.
>
> **Why it matters**: Overestimated Q-values lead to overconfident, suboptimal policies. The agent thinks bad actions are good because their Q-values are inflated.
>
> **Double DQN solution**: Decouple action selection from evaluation:
>
> $$y = r + \gamma Q(s', \arg\max_a Q(s', a; \theta); \theta^-)$$
>
> The online network ($\theta$) selects the best action, but the target network ($\theta^-$) evaluates it. Since different networks have different noise patterns, the selection bias doesn't compound with evaluation bias. Empirically, this significantly reduces overestimation and improves final performance."

---

## Quick Reference Card

```
DEEP RL (DQN) FUNDAMENTALS
═══════════════════════════════════════════════════════════════

WHY DEEP RL?
───────────────────────────────────────────────────────────────
- Tabular methods fail in high-dimensional state spaces
- Neural networks approximate Q(s,a) → generalization
- Learn features automatically from raw inputs (pixels)

DQN INNOVATIONS
───────────────────────────────────────────────────────────────
Experience Replay:  Store transitions, sample randomly
                    Breaks correlations, improves data efficiency

Target Network:     Separate network for computing targets
                    Hard update (copy every C steps) or
                    Soft update (Polyak: τθ + (1-τ)θ⁻)

DOUBLE DQN
───────────────────────────────────────────────────────────────
Problem:    max operator overestimates Q-values
Solution:   Select action with online net, evaluate with target
            y = r + γ Q(s', argmax_a Q(s',a;θ); θ⁻)

DUELING DQN
───────────────────────────────────────────────────────────────
Architecture: Q(s,a) = V(s) + (A(s,a) - mean(A))
Benefit:      Learns state value from all actions

RAINBOW COMPONENTS
───────────────────────────────────────────────────────────────
Double DQN          | Prioritized Replay    | Dueling
Multi-step Returns  | Distributional RL     | Noisy Nets

STABILITY TIPS
───────────────────────────────────────────────────────────────
- Monitor Q-value magnitudes (shouldn't explode)
- Clip rewards to [-1, 1]
- Use frame stacking (4 frames typical)
- Start with proven hyperparameters
- Soft target updates often more stable than hard
```

---

## Key Takeaways

1. **Function approximation** enables RL to scale to high-dimensional state spaces where tabular methods fail.

2. **Experience replay** breaks temporal correlations and improves sample efficiency by reusing past experiences.

3. **Target networks** provide stable optimization targets, preventing the "moving target" problem that causes divergence.

4. **Double DQN** reduces overestimation bias by decoupling action selection from evaluation.

5. **Dueling architectures** separate state value from action advantages, enabling more efficient learning.

6. **Deep RL remains challenging** due to non-stationarity, sample correlation, and hyperparameter sensitivity — always start with proven configurations.
