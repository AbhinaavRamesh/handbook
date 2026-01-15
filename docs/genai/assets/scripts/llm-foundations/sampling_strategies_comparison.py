#!/usr/bin/env python3
"""
Sampling Strategies Comparison Visualization

This script creates visualizations comparing different LLM sampling strategies:
- Temperature scaling
- Top-K sampling
- Top-P (nucleus) sampling

Output: ../images/llm-foundations/sampling_*.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Ensure output directory exists
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "images", "llm-foundations"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature-scaled softmax."""
    if temperature == 0:
        result = np.zeros_like(logits)
        result[np.argmax(logits)] = 1.0
        return result

    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))  # Numerical stability
    return exp_scaled / exp_scaled.sum()


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Apply top-k filtering."""
    result = np.zeros_like(probs)
    top_k_indices = np.argsort(probs)[-k:]
    result[top_k_indices] = probs[top_k_indices]
    return result / result.sum()


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Apply top-p (nucleus) filtering."""
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumsum = np.cumsum(sorted_probs)

    # Find cutoff
    cutoff_idx = np.searchsorted(cumsum, p) + 1

    result = np.zeros_like(probs)
    result[sorted_indices[:cutoff_idx]] = probs[sorted_indices[:cutoff_idx]]
    return result / result.sum()


def create_temperature_comparison():
    """Visualize effect of temperature on probability distribution."""
    # Create sample logits (some tokens more likely than others)
    np.random.seed(42)
    vocab_size = 20
    logits = np.random.randn(vocab_size) * 2
    logits[0] = 4.0  # Make one token clearly most likely
    logits[1] = 3.0
    logits[2] = 2.5

    temperatures = [0.3, 0.7, 1.0, 1.5, 2.0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Temperature Effect on Probability Distribution", fontsize=16, fontweight='bold')

    # First subplot: original logits
    ax = axes[0, 0]
    bars = ax.bar(range(vocab_size), logits, color='steelblue', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title("Raw Logits (before softmax)", fontsize=12)
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Logit Value")
    ax.set_xlim(-0.5, vocab_size - 0.5)

    # Temperature variations
    for idx, temp in enumerate(temperatures):
        ax = axes.flat[idx + 1]
        probs = softmax(logits, temp)

        colors = ['crimson' if p == max(probs) else 'steelblue' for p in probs]
        ax.bar(range(vocab_size), probs, color=colors, alpha=0.7)
        ax.set_title(f"Temperature = {temp}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Token Index")
        ax.set_ylabel("Probability")
        ax.set_xlim(-0.5, vocab_size - 0.5)
        ax.set_ylim(0, max(probs) * 1.1)

        # Add entropy annotation
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        ax.annotate(f"Entropy: {entropy:.2f}", xy=(0.95, 0.95),
                    xycoords='axes fraction', ha='right', va='top',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "sampling_temperature.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved temperature comparison to: {output_path}")


def create_topk_comparison():
    """Visualize effect of top-k filtering."""
    np.random.seed(42)
    vocab_size = 30
    logits = np.random.randn(vocab_size) * 1.5
    logits[:5] = [4.0, 3.5, 3.0, 2.5, 2.0]  # Make some clearly higher

    probs = softmax(logits, temperature=1.0)
    k_values = [3, 5, 10, 20]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Top-K Sampling: Filtering to K Most Likely Tokens", fontsize=16, fontweight='bold')

    # Original distribution
    ax = axes[0, 0]
    sorted_indices = np.argsort(probs)[::-1]
    ax.bar(range(vocab_size), probs[sorted_indices], color='steelblue', alpha=0.7)
    ax.set_title("Original Distribution (sorted)", fontsize=12)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Probability")

    # Cumulative distribution
    ax = axes[0, 1]
    cumsum = np.cumsum(probs[sorted_indices])
    ax.plot(range(vocab_size), cumsum, 'b-', linewidth=2)
    ax.fill_between(range(vocab_size), cumsum, alpha=0.3)
    for k in k_values:
        ax.axvline(x=k - 1, color='red', linestyle='--', alpha=0.5)
        ax.annotate(f'K={k}', xy=(k, cumsum[k - 1]), fontsize=9)
    ax.set_title("Cumulative Probability", fontsize=12)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1.05)

    # K variations
    for idx, k in enumerate(k_values):
        ax = axes.flat[idx + 2]
        filtered = top_k_filter(probs, k)

        colors = ['crimson' if p > 0 else 'lightgray' for p in filtered]
        ax.bar(range(vocab_size), probs, color='lightgray', alpha=0.5)
        ax.bar(range(vocab_size), filtered, color='steelblue', alpha=0.8)

        ax.set_title(f"Top-K = {k}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Token Index")
        ax.set_ylabel("Probability")

        # Annotate with probability mass captured
        mass = np.sum(probs[np.argsort(probs)[-k:]])
        ax.annotate(f"Mass: {mass:.1%}", xy=(0.95, 0.95),
                    xycoords='axes fraction', ha='right', va='top',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "sampling_topk.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved top-k comparison to: {output_path}")


def create_topp_comparison():
    """Visualize effect of top-p (nucleus) filtering."""
    np.random.seed(42)
    vocab_size = 30

    # Create two different distributions to show adaptive nature
    # Sharp distribution (confident)
    logits_sharp = np.random.randn(vocab_size) * 0.5
    logits_sharp[0] = 5.0
    probs_sharp = softmax(logits_sharp)

    # Flat distribution (uncertain)
    logits_flat = np.random.randn(vocab_size) * 1.0
    logits_flat[:10] += 1.5
    probs_flat = softmax(logits_flat)

    p_values = [0.7, 0.9, 0.95]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig)

    fig.suptitle("Top-P (Nucleus) Sampling: Adaptive Token Selection", fontsize=16, fontweight='bold')

    # Sharp distribution examples
    ax = fig.add_subplot(gs[0, 0])
    sorted_idx = np.argsort(probs_sharp)[::-1]
    ax.bar(range(vocab_size), probs_sharp[sorted_idx], color='steelblue', alpha=0.7)
    ax.set_title("Sharp Distribution\n(High Confidence)", fontsize=11)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Probability")

    for idx, p in enumerate(p_values):
        ax = fig.add_subplot(gs[0, idx + 1])
        filtered = top_p_filter(probs_sharp, p)

        # Count tokens included
        n_tokens = np.sum(filtered > 0)

        ax.bar(range(vocab_size), probs_sharp, color='lightgray', alpha=0.5)
        ax.bar(range(vocab_size), filtered, color='crimson', alpha=0.8)

        ax.set_title(f"P = {p}\n({n_tokens} tokens)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Token Index")

    # Flat distribution examples
    ax = fig.add_subplot(gs[1, 0])
    sorted_idx = np.argsort(probs_flat)[::-1]
    ax.bar(range(vocab_size), probs_flat[sorted_idx], color='steelblue', alpha=0.7)
    ax.set_title("Flat Distribution\n(Low Confidence)", fontsize=11)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Probability")

    for idx, p in enumerate(p_values):
        ax = fig.add_subplot(gs[1, idx + 1])
        filtered = top_p_filter(probs_flat, p)

        n_tokens = np.sum(filtered > 0)

        ax.bar(range(vocab_size), probs_flat, color='lightgray', alpha=0.5)
        ax.bar(range(vocab_size), filtered, color='forestgreen', alpha=0.8)

        ax.set_title(f"P = {p}\n({n_tokens} tokens)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Token Index")

    # Add explanation text
    fig.text(0.5, 0.02,
             "Top-P is adaptive: Sharp distributions select fewer tokens, flat distributions select more",
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, "sampling_topp.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved top-p comparison to: {output_path}")


def create_combined_comparison():
    """Create a combined visualization showing all strategies together."""
    np.random.seed(42)
    vocab_size = 20
    logits = np.random.randn(vocab_size) * 1.5
    logits[0] = 4.0
    logits[1] = 3.2
    logits[2] = 2.8

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sampling Strategies Comparison", fontsize=16, fontweight='bold')

    # 1. Greedy (temperature -> 0)
    ax = axes[0, 0]
    probs = softmax(logits, temperature=0.1)
    colors = ['crimson' if i == 0 else 'lightgray' for i in range(vocab_size)]
    ax.bar(range(vocab_size), probs, color=colors, alpha=0.8)
    ax.set_title("Greedy (T→0)\nAlways picks highest", fontsize=12, fontweight='bold')
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Probability")
    ax.annotate("Deterministic, may repeat", xy=(0.5, 0.9),
                xycoords='axes fraction', ha='center', fontsize=9,
                color='darkred')

    # 2. Temperature sampling
    ax = axes[0, 1]
    probs = softmax(logits, temperature=0.8)
    ax.bar(range(vocab_size), probs, color='steelblue', alpha=0.8)
    ax.set_title("Temperature (T=0.8)\nScales all logits", fontsize=12, fontweight='bold')
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Probability")
    ax.annotate("Smooth, but may sample unlikely tokens", xy=(0.5, 0.9),
                xycoords='axes fraction', ha='center', fontsize=9,
                color='darkblue')

    # 3. Top-K
    ax = axes[1, 0]
    probs = softmax(logits, temperature=1.0)
    filtered = top_k_filter(probs, k=5)
    ax.bar(range(vocab_size), probs, color='lightgray', alpha=0.5, label='Original')
    ax.bar(range(vocab_size), filtered, color='forestgreen', alpha=0.8, label='After Top-K')
    ax.set_title("Top-K (K=5)\nFixed number of candidates", fontsize=12, fontweight='bold')
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Probability")
    ax.legend(loc='upper right', fontsize=8)
    ax.annotate("Simple, but not adaptive", xy=(0.5, 0.9),
                xycoords='axes fraction', ha='center', fontsize=9,
                color='darkgreen')

    # 4. Top-P
    ax = axes[1, 1]
    probs = softmax(logits, temperature=1.0)
    filtered = top_p_filter(probs, p=0.9)
    n_tokens = np.sum(filtered > 0)
    ax.bar(range(vocab_size), probs, color='lightgray', alpha=0.5, label='Original')
    ax.bar(range(vocab_size), filtered, color='darkorange', alpha=0.8, label='After Top-P')
    ax.set_title(f"Top-P (P=0.9)\nAdaptive: selected {n_tokens} tokens", fontsize=12, fontweight='bold')
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Probability")
    ax.legend(loc='upper right', fontsize=8)
    ax.annotate("Adaptive to distribution shape", xy=(0.5, 0.9),
                xycoords='axes fraction', ha='center', fontsize=9,
                color='darkorange')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "sampling_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved combined comparison to: {output_path}")


if __name__ == "__main__":
    create_temperature_comparison()
    create_topk_comparison()
    create_topp_comparison()
    create_combined_comparison()
    print("All sampling visualizations created successfully!")
