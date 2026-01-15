"""
Variance Scaling Demonstration (sqrt(d_k) Impact)

This script demonstrates why scaling by sqrt(d_k) is critical in attention:
1. Shows distribution of QK^T dot products for different d_k values
2. Compares variance before and after scaling
3. Shows how softmax behaves differently with/without scaling
4. Demonstrates that scaling normalizes variance to ~1

Key insight: Without scaling, large d_k leads to large variance in dot products,
causing softmax to produce near-one-hot distributions (gradient vanishing).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def softmax(x, axis=-1):
    """Compute softmax values."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def create_variance_scaling_demo():
    """Create a 2x2 figure demonstrating the importance of sqrt(d_k) scaling."""

    np.random.seed(42)
    n_samples = 10000
    seq_len = 6

    # Two different d_k values
    d_k_small = 64
    d_k_large = 512

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Why Scale by $\\sqrt{d_k}$? Variance Normalization in Attention",
                 fontsize=14, fontweight='bold', y=0.98)

    # Generate Q and K with unit variance
    Q_small = np.random.randn(n_samples, d_k_small)
    K_small = np.random.randn(n_samples, d_k_small)
    Q_large = np.random.randn(n_samples, d_k_large)
    K_large = np.random.randn(n_samples, d_k_large)

    # Compute dot products
    dot_products_small = np.sum(Q_small * K_small, axis=1)
    dot_products_large = np.sum(Q_large * K_large, axis=1)

    # Scaled dot products
    scaled_small = dot_products_small / np.sqrt(d_k_small)
    scaled_large = dot_products_large / np.sqrt(d_k_large)

    # Color scheme
    color_unscaled = '#1f77b4'
    color_scaled = '#2ca02c'

    # ============== Top Left: d_k=64 distributions ==============
    ax1 = axes[0, 0]
    ax1.hist(dot_products_small, bins=50, alpha=0.6, density=True,
             label=f'Unscaled (var={np.var(dot_products_small):.1f})', color=color_unscaled)
    ax1.hist(scaled_small, bins=50, alpha=0.6, density=True,
             label=f'Scaled by $\\sqrt{{{d_k_small}}}$ (var={np.var(scaled_small):.2f})', color=color_scaled)
    ax1.set_title(f'$d_k = {d_k_small}$: Distribution of QK$^T$ Dot Products', fontsize=11)
    ax1.set_xlabel('Dot Product Value', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Mean=0')

    # ============== Top Right: d_k=512 distributions ==============
    ax2 = axes[0, 1]
    ax2.hist(dot_products_large, bins=50, alpha=0.6, density=True,
             label=f'Unscaled (var={np.var(dot_products_large):.1f})', color=color_unscaled)
    ax2.hist(scaled_large, bins=50, alpha=0.6, density=True,
             label=f'Scaled by $\\sqrt{{{d_k_large}}}$ (var={np.var(scaled_large):.2f})', color=color_scaled)
    ax2.set_title(f'$d_k = {d_k_large}$: Distribution of QK$^T$ Dot Products', fontsize=11)
    ax2.set_xlabel('Dot Product Value', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # ============== Bottom Left: Softmax outputs comparison ==============
    ax3 = axes[1, 0]

    # Generate attention scores for a single query attending to seq_len keys
    np.random.seed(123)
    n_examples = 5

    bar_width = 0.35
    x = np.arange(seq_len)

    for i in range(n_examples):
        q = np.random.randn(d_k_large)
        K = np.random.randn(seq_len, d_k_large)
        scores_unscaled = K @ q
        scores_scaled = scores_unscaled / np.sqrt(d_k_large)

        attn_unscaled = softmax(scores_unscaled)
        attn_scaled = softmax(scores_scaled)

        if i == 0:  # Only show one example in detail
            ax3.bar(x - bar_width/2, attn_unscaled, bar_width, label='Unscaled', color=color_unscaled, alpha=0.8)
            ax3.bar(x + bar_width/2, attn_scaled, bar_width, label='Scaled', color=color_scaled, alpha=0.8)

    ax3.set_title(f'Softmax Output Comparison ($d_k={d_k_large}$)\nUnscaled: Near One-Hot vs Scaled: Distributed', fontsize=11)
    ax3.set_xlabel('Key Position', fontsize=10)
    ax3.set_ylabel('Attention Weight', fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'K{i}' for i in range(seq_len)])
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 1.1)

    # Add horizontal line at 1/seq_len (uniform attention)
    ax3.axhline(y=1/seq_len, color='gray', linestyle=':', alpha=0.7, label='Uniform')
    ax3.text(seq_len-0.5, 1/seq_len + 0.02, 'Uniform', fontsize=8, color='gray')

    # ============== Bottom Right: Entropy/Peakiness across many samples ==============
    ax4 = axes[1, 1]

    def compute_entropy(probs):
        """Compute entropy of probability distribution."""
        probs = np.clip(probs, 1e-10, 1)
        return -np.sum(probs * np.log(probs), axis=-1)

    # Generate many attention distributions
    n_trials = 1000
    entropies_unscaled = []
    entropies_scaled = []
    max_weights_unscaled = []
    max_weights_scaled = []

    for _ in range(n_trials):
        q = np.random.randn(d_k_large)
        K = np.random.randn(seq_len, d_k_large)
        scores_unscaled = K @ q
        scores_scaled = scores_unscaled / np.sqrt(d_k_large)

        attn_unscaled = softmax(scores_unscaled)
        attn_scaled = softmax(scores_scaled)

        entropies_unscaled.append(compute_entropy(attn_unscaled))
        entropies_scaled.append(compute_entropy(attn_scaled))
        max_weights_unscaled.append(np.max(attn_unscaled))
        max_weights_scaled.append(np.max(attn_scaled))

    # Plot max attention weight distributions
    ax4.hist(max_weights_unscaled, bins=30, alpha=0.6, density=True,
             label=f'Unscaled (mean max={np.mean(max_weights_unscaled):.2f})', color=color_unscaled)
    ax4.hist(max_weights_scaled, bins=30, alpha=0.6, density=True,
             label=f'Scaled (mean max={np.mean(max_weights_scaled):.2f})', color=color_scaled)
    ax4.set_title(f'Distribution of Maximum Attention Weight ($d_k={d_k_large}$)\nUnscaled: Peaked, Scaled: Distributed', fontsize=11)
    ax4.set_xlabel('Maximum Attention Weight', fontsize=10)
    ax4.set_ylabel('Density', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.axvline(x=1/seq_len, color='gray', linestyle=':', alpha=0.7)
    ax4.text(1/seq_len + 0.02, ax4.get_ylim()[1] * 0.9, 'Uniform\nmax', fontsize=8, color='gray')

    plt.tight_layout()

    # Add annotation box
    textstr = (f'Key Insight:\n'
               f'Without scaling: Var(QK$^T$) = $d_k$\n'
               f'With scaling: Var(QK$^T$/$\\sqrt{{d_k}}$) = 1\n\n'
               f'Large variance -> softmax saturation\n'
               f'-> vanishing gradients')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    fig.text(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom',
             bbox=props, family='monospace')

    # Save the figure
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/variance_scaling_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path

if __name__ == "__main__":
    create_variance_scaling_demo()
