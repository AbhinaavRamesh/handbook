# Reinforcement Learning

> **Learning through interaction** — from bandits to RLHF for LLMs

---

## Overview

Reinforcement Learning (RL) is the branch of machine learning where agents learn optimal behavior through trial-and-error interactions with an environment. Unlike supervised learning (which learns from labeled examples) or unsupervised learning (which finds patterns in data), RL learns from rewards and penalties received after taking actions.

At its core, RL addresses the fundamental question: **How can an agent learn to make good decisions in uncertain environments?**

This framework has produced remarkable achievements — from AlphaGo defeating world champions to robots learning complex manipulation tasks to ChatGPT being fine-tuned with human feedback. Understanding RL is increasingly essential for ML interviews, particularly as RLHF (Reinforcement Learning from Human Feedback) has become the standard technique for aligning large language models.

---

## Why RL Matters for ML Interviews

### The RLHF Revolution

Modern LLMs like ChatGPT, Claude, and Gemini all use RLHF to align model outputs with human preferences. Interviewers expect candidates to understand:

- How reward models are trained from human preference data
- Why PPO (Proximal Policy Optimization) is the go-to algorithm for RLHF
- The exploration-exploitation tradeoffs in policy optimization
- Alternatives like DPO (Direct Preference Optimization) and their tradeoffs

### Interview Question Types

| Type | Example Questions |
|------|-------------------|
| **Fundamentals** | "What is the Bellman equation?", "Explain the exploration-exploitation tradeoff" |
| **Algorithm Mechanics** | "Walk me through Q-learning", "How does PPO prevent destructive policy updates?" |
| **RLHF Specific** | "How would you train a reward model?", "Why use RL instead of supervised fine-tuning?" |
| **System Design** | "Design a recommendation system using bandits", "How would you implement RLHF at scale?" |

---

## Learning Paths

Choose your path based on available time and depth requirements:

### Quick Path (4-5 hours)

For time-constrained preparation focusing on interview essentials:

1. **MDP Foundations** — Core formalism (45 min)
2. **TD Learning (Q-Learning, SARSA)** — Most asked algorithm (60 min)
3. **PPO & Modern Methods** — Industry standard (60 min)
4. **RLHF & Alignment** — LLM-critical knowledge (75 min)

*Outcome: Solid foundation for most RL interview questions*

### Standard Path (8-10 hours)

Complete coverage for thorough preparation:

1. MDP Foundations
2. Value Functions & Bellman
3. Exploration vs Exploitation
4. TD Learning (Q-Learning, SARSA)
5. Policy Gradients
6. Actor-Critic Methods
7. Deep RL (DQN)
8. PPO & Modern Methods
9. RLHF & Alignment
10. Advanced Topics

*Outcome: Comprehensive understanding for senior-level interviews*

### Deep Dive Path (12+ hours)

For those targeting ML research roles or wanting expert-level knowledge:

- Complete all 10 modules in order
- Implement key algorithms from scratch (Q-learning, DQN, REINFORCE, PPO)
- Work through coding exercises in each module
- Read referenced papers (especially Schulman et al. on PPO, Christiano et al. on RLHF)

*Outcome: Research-grade understanding with implementation experience*

---

## Module Overview

| Module | Focus | Key Topics | Time |
|--------|-------|------------|------|
| [**MDP Foundations**](./mdp-foundations) | Core formalism | States, actions, transitions, rewards, discount factor | 45 min |
| [**Value Functions & Bellman**](./value-functions) | Evaluation | V(s), Q(s,a), Bellman equations, policy evaluation | 50 min |
| [**Exploration vs Exploitation**](./exploration-exploitation) | Fundamental tradeoff | Epsilon-greedy, UCB, Thompson sampling, bandits | 45 min |
| [**TD Learning**](./temporal-difference) | Model-free learning | Q-Learning, SARSA, TD(0), eligibility traces | 60 min |
| [**Policy Gradients**](./policy-gradients) | Direct policy optimization | REINFORCE, baseline, variance reduction | 55 min |
| [**Actor-Critic Methods**](./actor-critic) | Combining approaches | A2C, advantage function, GAE | 50 min |
| [**Deep RL (DQN)**](./deep-rl) | Neural network integration | Experience replay, target networks, Double DQN | 60 min |
| [**PPO & Modern Methods**](./ppo-modern-methods) | State-of-the-art | Clipped objective, TRPO comparison, hyperparameters | 60 min |
| [**RLHF & Alignment**](./rlhf-alignment) | LLM fine-tuning | Reward modeling, preference learning, DPO | 75 min |
| [**Advanced Topics**](./advanced-topics) | Cutting edge | Multi-agent RL, offline RL, model-based RL | 60 min |

**Total estimated time: 9-10 hours** (Standard Path)

---

## Prerequisites

Before diving into RL, ensure you are comfortable with:

### Required Knowledge

| Topic | Why It Matters | Where to Review |
|-------|----------------|-----------------|
| **Probability & Statistics** | MDPs are probabilistic; value functions are expectations | Statistics basics |
| **Neural Networks** | Deep RL uses NNs as function approximators | [Neural Networks](../neural-networks/) |
| **Gradient Descent** | Policy gradients optimize via gradients | [Gradient Descent](../interview-faq/optimization/gradient-descent) |
| **Basic Calculus** | Understanding policy gradient derivations | Calculus review |

### Helpful (Not Required)

- Dynamic programming concepts (value iteration builds on DP)
- Game theory basics (useful for multi-agent RL)
- Information theory (entropy regularization in PPO)

---

## Quick Reference Card

### Core RL Formalism

```
Agent-Environment Loop:
  State (s) --> Agent --> Action (a) --> Environment --> Reward (r), Next State (s')

MDP Components:
  - S: State space
  - A: Action space
  - P(s'|s,a): Transition probability
  - R(s,a): Reward function
  - gamma: Discount factor (0 <= gamma <= 1)
```

### Key Equations

| Concept | Equation | Meaning |
|---------|----------|---------|
| **Value Function** | V(s) = E[sum of discounted rewards from s] | Expected return from state s |
| **Q-Function** | Q(s,a) = E[sum of discounted rewards after taking a in s] | Expected return from state-action pair |
| **Bellman (V)** | V(s) = max over a of [ R(s,a) + gamma * sum over s' of P(s'\|s,a) * V(s') ] | Recursive value definition |
| **Q-Learning Update** | Q(s,a) <- Q(s,a) + alpha * (r + gamma * max Q(s',a') - Q(s,a)) | TD update with max |
| **Policy Gradient** | grad J = E[grad log pi(a\|s) * Q(s,a)] | Gradient of expected return |

### Algorithm Selection Guide

```
Problem Type Decision Tree:

Discrete actions + small state space?
  --> Q-Learning or SARSA

Continuous actions?
  --> Policy Gradients (PPO, SAC)

Sample efficiency critical?
  --> Model-based RL or offline RL

Training LLMs with human feedback?
  --> PPO (RLHF) or DPO

Multi-objective optimization?
  --> Constrained RL or Multi-agent approaches
```

### Common Interview Questions

**Fundamentals:**
- "Explain the difference between on-policy and off-policy learning"
- "Why do we discount future rewards?"
- "What is the credit assignment problem?"

**Algorithms:**
- "Walk through a Q-learning update step"
- "Why does DQN use experience replay and target networks?"
- "How does PPO's clipped objective prevent large policy changes?"

**RLHF:**
- "How is a reward model trained?"
- "Why use RL instead of just supervised fine-tuning on preferred outputs?"
- "What are the failure modes of RLHF?"

---

## Study Strategy

### For Technical Interviews

1. **Master the vocabulary** — Know MDP, policy, value function, Q-function cold
2. **Understand one algorithm deeply** — Q-learning or PPO are good choices
3. **Know the RLHF pipeline** — This is the most practical application today
4. **Practice explaining tradeoffs** — On-policy vs off-policy, model-free vs model-based

### Common Mistakes to Avoid

- Confusing value functions V(s) with Q-functions Q(s,a)
- Forgetting the discount factor's role in convergence
- Not understanding why policy gradients have high variance
- Oversimplifying RLHF as "just fine-tuning with RL"

### What Distinguishes Strong Candidates

1. Can derive policy gradient theorem from first principles
2. Understands why PPO uses clipping (not just that it does)
3. Knows practical challenges: reward hacking, distribution shift, sample efficiency
4. Can discuss RLHF alternatives (DPO, RLAIF) and their tradeoffs

---

## Resources for Deeper Learning

### Essential Reading

- **Sutton & Barto** — "Reinforcement Learning: An Introduction" (the bible of RL)
- **Schulman et al. (2017)** — "Proximal Policy Optimization Algorithms" (PPO paper)
- **Christiano et al. (2017)** — "Deep Reinforcement Learning from Human Preferences" (RLHF foundations)
- **Ouyang et al. (2022)** — "Training language models to follow instructions" (InstructGPT/RLHF for LLMs)

### Online Courses

- David Silver's RL Course (DeepMind/UCL)
- Sergey Levine's Deep RL Course (UC Berkeley)
- Spinning Up in Deep RL (OpenAI)

### Implementation References

- Stable Baselines3 — Production-quality RL implementations
- CleanRL — Single-file implementations for learning
- TRL (Transformers Reinforcement Learning) — RLHF for LLMs

---

## Next Steps

1. **New to RL?** Start with [MDP Foundations](./mdp-foundations) to build intuition
2. **Short on time?** Follow the Quick Path focusing on TD Learning and RLHF
3. **Preparing for senior roles?** Complete the Standard Path with coding implementations
4. **Interview this week?** Jump to [RLHF & Alignment](./rlhf-alignment) for LLM-specific questions

---

*Last updated: January 2026*

*Reinforcement Learning bridges the gap between theoretical decision-making and practical AI systems. From game-playing agents to aligned language models, RL provides the framework for learning optimal behavior through interaction. Master these fundamentals, and you will be prepared for the increasingly common RL questions in ML interviews.*
