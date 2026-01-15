"""
Attention QKV Heatmap Visualization

This script visualizes the three stages of attention score computation:
1. Raw attention scores (QK^T) - the dot product similarities
2. Scaled scores (QK^T / sqrt(d_k)) - normalized for stable softmax
3. Final attention weights (after softmax) - probability distribution

Uses the example sentence: "The cat sat on the mat"
to demonstrate how attention patterns form between tokens.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Blues")

def softmax(x, axis=-1):
    """Compute softmax values for each set of scores."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def create_attention_heatmap():
    """Create a 3-panel heatmap showing attention score progression."""

    # Example sentence tokens
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    n_tokens = len(tokens)
    d_k = 64  # Key dimension

    # Simulate Q and K matrices with realistic patterns
    # Creating patterns where semantically related words have higher similarity
    np.random.seed(42)

    # Base random Q and K
    Q = np.random.randn(n_tokens, d_k) * 0.5
    K = np.random.randn(n_tokens, d_k) * 0.5

    # Add semantic biases to create interesting patterns
    # "cat" should attend to "sat" (subject-verb relationship)
    # "sat" should attend to "cat" and "mat" (action context)
    # "the" tokens should have lower attention overall

    # Compute raw attention scores
    raw_scores = Q @ K.T

    # Add meaningful structure to the scores
    structure = np.array([
        [0.5, 0.3, 0.2, 0.1, 0.1, 0.2],   # The -> mostly self
        [0.2, 0.8, 0.6, 0.1, 0.1, 0.3],   # cat -> cat, sat
        [0.1, 0.7, 0.9, 0.3, 0.1, 0.5],   # sat -> cat, sat, mat
        [0.2, 0.1, 0.4, 0.5, 0.2, 0.3],   # on -> sat, mat
        [0.4, 0.1, 0.1, 0.2, 0.5, 0.3],   # the -> first the, mat
        [0.1, 0.3, 0.6, 0.3, 0.2, 0.8],   # mat -> cat, sat, mat
    ])

    raw_scores = raw_scores + structure * 2

    # Scaled scores
    scaling_factor = np.sqrt(d_k)
    scaled_scores = raw_scores / scaling_factor

    # Attention weights after softmax
    attention_weights = softmax(scaled_scores, axis=-1)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Attention Score Computation Pipeline\n\"The cat sat on the mat\"",
                 fontsize=14, fontweight='bold', y=1.02)

    # Common heatmap parameters
    cmap = 'Blues'

    # 1. Raw scores (before scaling)
    ax1 = axes[0]
    im1 = ax1.imshow(raw_scores, cmap=cmap, aspect='auto')
    ax1.set_title(f"Raw Scores (QK$^T$)\nVariance ~ d_k = {d_k}", fontsize=11)
    ax1.set_xlabel("Key Positions (K)", fontsize=10)
    ax1.set_ylabel("Query Positions (Q)", fontsize=10)
    ax1.set_xticks(range(n_tokens))
    ax1.set_yticks(range(n_tokens))
    ax1.set_xticklabels(tokens, fontsize=9)
    ax1.set_yticklabels(tokens, fontsize=9)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Score', fontsize=9)

    # Add values to cells
    for i in range(n_tokens):
        for j in range(n_tokens):
            text_color = 'white' if raw_scores[i, j] > np.mean(raw_scores) else 'black'
            ax1.text(j, i, f'{raw_scores[i, j]:.1f}', ha='center', va='center',
                    fontsize=8, color=text_color)

    # 2. Scaled scores
    ax2 = axes[1]
    im2 = ax2.imshow(scaled_scores, cmap=cmap, aspect='auto')
    ax2.set_title(f"Scaled Scores (QK$^T$ / $\\sqrt{{d_k}}$)\nVariance ~ 1", fontsize=11)
    ax2.set_xlabel("Key Positions (K)", fontsize=10)
    ax2.set_ylabel("Query Positions (Q)", fontsize=10)
    ax2.set_xticks(range(n_tokens))
    ax2.set_yticks(range(n_tokens))
    ax2.set_xticklabels(tokens, fontsize=9)
    ax2.set_yticklabels(tokens, fontsize=9)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Score', fontsize=9)

    for i in range(n_tokens):
        for j in range(n_tokens):
            text_color = 'white' if scaled_scores[i, j] > np.mean(scaled_scores) else 'black'
            ax2.text(j, i, f'{scaled_scores[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color=text_color)

    # 3. Attention weights (after softmax)
    ax3 = axes[2]
    im3 = ax3.imshow(attention_weights, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    ax3.set_title("Attention Weights (softmax)\nRows sum to 1.0", fontsize=11)
    ax3.set_xlabel("Key Positions (K)", fontsize=10)
    ax3.set_ylabel("Query Positions (Q)", fontsize=10)
    ax3.set_xticks(range(n_tokens))
    ax3.set_yticks(range(n_tokens))
    ax3.set_xticklabels(tokens, fontsize=9)
    ax3.set_yticklabels(tokens, fontsize=9)
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label('Weight', fontsize=9)

    for i in range(n_tokens):
        for j in range(n_tokens):
            text_color = 'white' if attention_weights[i, j] > 0.25 else 'black'
            ax3.text(j, i, f'{attention_weights[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color=text_color)

    plt.tight_layout()

    # Save the figure
    output_path = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/attention_qkv_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path

if __name__ == "__main__":
    create_attention_heatmap()
