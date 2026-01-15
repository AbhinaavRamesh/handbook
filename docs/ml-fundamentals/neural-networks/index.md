# Neural Networks

> **The foundation of deep learning** — from single neurons to modern architectures

---

## Overview

Neural networks are the backbone of modern machine learning, powering everything from image recognition to language models. Understanding neural networks deeply is essential for any ML interview, as they form the foundation for more advanced architectures like transformers and diffusion models.

This comprehensive guide covers neural networks from first principles through advanced topics, with a focus on the concepts and intuitions that interviewers expect candidates to understand.

---

## Why Neural Networks Matter for Interviews

### Interview Question Types

| Type | Example Questions |
|------|-------------------|
| **Foundations** | "Why do we need non-linear activation functions?", "Explain backpropagation" |
| **Architecture** | "How do CNNs achieve translation equivariance?", "Why do LSTMs solve vanishing gradients?" |
| **Training** | "Compare Adam vs SGD", "When would you use BatchNorm vs LayerNorm?" |
| **Debugging** | "Your model isn't learning, what do you check?", "How do you diagnose gradient issues?" |

---

## Learning Paths

### Quick Path (4-5 hours)

For time-constrained preparation focusing on interview essentials:

1. **Perceptrons & MLPs** — Core building blocks (45 min)
2. **Activation Functions** — Most asked topic (45 min)
3. **Backpropagation** — Must-know algorithm (60 min)
4. **Optimizers** — SGD, Adam, and when to use each (45 min)
5. **Gradient Problems** — Vanishing/exploding gradients (45 min)

*Outcome: Solid foundation for most NN interview questions*

### Standard Path (8-10 hours)

Complete coverage for thorough preparation:

1. Perceptrons & MLPs
2. Activation Functions
3. Backpropagation
4. Weight Initialization
5. Normalization
6. CNNs
7. RNNs, LSTM, GRU
8. Autoencoders & VAE
9. GANs
10. Optimizers
11. Regularization
12. Gradient Problems

*Outcome: Comprehensive understanding for senior-level interviews*

---

## Module Overview

| Module | Focus | Key Topics | Time |
|--------|-------|------------|------|
| [**Perceptrons & MLPs**](./01-perceptrons-mlps) | Building blocks | Single neuron, XOR problem, universal approximation | 45 min |
| [**Activation Functions**](./02-activation-functions) | Non-linearity | ReLU, GELU, dying ReLU, choosing activations | 45 min |
| [**Backpropagation**](./03-backpropagation) | Core algorithm | Chain rule, computation graphs, implementation | 60 min |
| [**Weight Initialization**](./04-weight-initialization) | Starting right | Xavier, He, variance preservation | 40 min |
| [**Normalization**](./05-normalization) | Stabilizing training | BatchNorm, LayerNorm, GroupNorm | 45 min |
| [**CNNs**](./06-cnns) | Vision architectures | Convolutions, pooling, ResNet, receptive field | 50 min |
| [**RNNs, LSTM, GRU**](./07-rnns-lstm-gru) | Sequence modeling | Recurrence, gates, gradient flow | 50 min |
| [**Autoencoders & VAE**](./08-autoencoders-vae) | Generative models | Latent space, reparameterization trick, ELBO | 45 min |
| [**GANs**](./09-gans) | Adversarial training | Generator/Discriminator, mode collapse, WGAN | 45 min |
| [**Optimizers**](./10-optimizers) | Training dynamics | SGD, momentum, Adam, learning rate schedules | 50 min |
| [**Regularization**](./11-regularization) | Preventing overfitting | Dropout, L1/L2, early stopping | 40 min |
| [**Gradient Problems**](./12-gradient-problems) | Training challenges | Vanishing/exploding gradients, solutions | 45 min |

**Total estimated time: 9-10 hours** (Standard Path)

---

## Prerequisites

Before diving into neural networks, ensure comfort with:

### Required Knowledge

| Topic | Why It Matters | Where to Review |
|-------|----------------|-----------------|
| **Linear Algebra** | Matrix operations, vector spaces | Linear algebra basics |
| **Calculus** | Derivatives, chain rule | Calculus review |
| **Probability** | Distributions, expectations | [Statistics basics](../interview-faq/statistics/hypothesis-testing) |
| **Python/NumPy** | Implementation exercises | Python review |

---

## Quick Reference Card

### Neural Network Components

```
NEURON
────────────────────────────────────────────────────────
Input:    x = [x_1, x_2, ..., x_n]
Weights:  w = [w_1, w_2, ..., w_n]
Bias:     b
Output:   y = activation(w^T x + b)

LAYER TYPES
────────────────────────────────────────────────────────
Dense/FC:     y = activation(Wx + b)
Conv2D:       y = activation(W * x + b)  (* = convolution)
RNN:          h_t = activation(W_h h_{t-1} + W_x x_t + b)

COMMON ACTIVATIONS
────────────────────────────────────────────────────────
ReLU:         f(x) = max(0, x)
Sigmoid:      f(x) = 1 / (1 + e^{-x})
Tanh:         f(x) = (e^x - e^{-x}) / (e^x + e^{-x})
GELU:         f(x) = x * Phi(x)  (Gaussian CDF)
Softmax:      f(x_i) = e^{x_i} / sum(e^{x_j})
```

### Key Equations

| Concept | Equation | Use Case |
|---------|----------|----------|
| **Forward Pass** | $a^{[l]} = g(W^{[l]} a^{[l-1]} + b^{[l]})$ | Computing predictions |
| **Cross-Entropy Loss** | $L = -\sum y_i \log(\hat{y}_i)$ | Classification |
| **MSE Loss** | $L = \frac{1}{n}\sum(y - \hat{y})^2$ | Regression |
| **Gradient Update** | $\theta = \theta - \alpha \nabla_\theta L$ | Training |
| **Adam Update** | $\theta = \theta - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$ | Adaptive optimization |

### Architecture Selection Guide

```
Problem Type Decision Tree:

Image data?
  --> CNNs (ResNet, EfficientNet)

Sequential data?
  --> RNNs/LSTM for short sequences
  --> Transformers for long sequences

Tabular data?
  --> MLPs or gradient boosting

Generation task?
  --> VAE for structured latent space
  --> GAN for high-quality samples
  --> Diffusion for state-of-the-art

Small dataset?
  --> Transfer learning + fine-tuning
  --> Strong regularization
```

---

## Common Interview Questions

**Foundations:**
- "Why do neural networks need non-linear activation functions?"
- "Walk through backpropagation for a 2-layer network"
- "What happens if you initialize all weights to zero?"

**Architecture:**
- "Why do CNNs use convolutions instead of fully connected layers?"
- "How do skip connections help train deep networks?"
- "What's the difference between BatchNorm and LayerNorm?"

**Training:**
- "Compare Adam vs SGD. When would you use each?"
- "How does dropout prevent overfitting?"
- "What causes vanishing gradients and how do you fix it?"

**Debugging:**
- "Your model's training loss is stuck. What do you check?"
- "Loss suddenly becomes NaN. What happened?"
- "Model overfits quickly. What regularization would you try?"

---

## Study Strategy

### For Technical Interviews

1. **Master the fundamentals** — Know forward pass, backprop, gradient descent cold
2. **Understand one architecture deeply** — CNNs or transformers are good choices
3. **Know the tradeoffs** — Adam vs SGD, BatchNorm vs LayerNorm, ReLU vs GELU
4. **Practice debugging scenarios** — Common failure modes and solutions

### What Distinguishes Strong Candidates

1. Can derive backpropagation from first principles
2. Understands *why* techniques work, not just *that* they work
3. Knows practical challenges: initialization, normalization, regularization
4. Can discuss modern developments: GELU, AdamW, layer normalization

---

## Next Steps

1. **New to neural networks?** Start with [Perceptrons & MLPs](./01-perceptrons-mlps)
2. **Reviewing for interviews?** Focus on [Backpropagation](./03-backpropagation) and [Optimizers](./10-optimizers)
3. **Architecture-focused role?** Deep dive into [CNNs](./06-cnns) or [RNNs](./07-rnns-lstm-gru)
4. **Debugging skills?** Study [Gradient Problems](./12-gradient-problems) and [Weight Initialization](./04-weight-initialization)

---

*Neural networks transform simple linear operations into powerful function approximators through clever use of non-linearity, depth, and gradient-based optimization. Master these fundamentals, and you'll be prepared for the neural network questions that appear in virtually every ML interview.*
