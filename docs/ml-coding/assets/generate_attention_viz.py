#!/usr/bin/env python3
"""
Attention Mechanism Visualization Generator

This script generates visualizations for understanding attention mechanisms
in transformer models, including:
1. Self-attention heatmap (tokens attending to tokens)
2. Multi-head attention visualization (multiple heads side by side)
3. Query-Key-Value flow diagram
4. Positional encoding visualization (sinusoidal pattern)
5. Attention weights across layers

Usage:
    python generate_attention_viz.py

Outputs:
    - self_attention_heatmap.png
    - multi_head_attention.png
    - positional_encoding.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from pathlib import Path

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = Path(__file__).parent


def softmax(x, axis=-1):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def compute_attention_weights(query, key, d_k):
    """
    Compute scaled dot-product attention weights.

    attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Returns the attention weights (before multiplying by V).
    """
    scores = np.dot(query, key.T) / np.sqrt(d_k)
    return softmax(scores)


def generate_self_attention_heatmap():
    """
    Generate a self-attention heatmap showing how tokens attend to each other.
    Uses the example sentence: "The cat sat on the mat"
    """
    # Example sentence tokens
    tokens = ["<s>", "The", "cat", "sat", "on", "the", "mat", "</s>"]
    n_tokens = len(tokens)

    # Embedding dimension
    d_model = 64
    d_k = d_model  # Key dimension

    # Generate random embeddings (in practice, these come from the model)
    np.random.seed(42)
    embeddings = np.random.randn(n_tokens, d_model)

    # Simulate learned Q, K projections
    W_q = np.random.randn(d_model, d_k) * 0.1
    W_k = np.random.randn(d_model, d_k) * 0.1

    Q = np.dot(embeddings, W_q)
    K = np.dot(embeddings, W_k)

    # Compute attention weights
    attention_weights = compute_attention_weights(Q, K, d_k)

    # Manually adjust some weights to show realistic patterns
    # (e.g., "cat" attends strongly to "The", "sat" attends to "cat")
    attention_weights[2, 1] += 0.3  # cat -> The
    attention_weights[3, 2] += 0.4  # sat -> cat
    attention_weights[6, 2] += 0.2  # mat -> cat
    attention_weights[5, 1] += 0.25  # the -> The (same word attention)
    attention_weights[4, 3] += 0.2  # on -> sat

    # Re-normalize rows
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'},
        vmin=0,
        vmax=1,
        linewidths=0.5,
        square=True
    )

    ax.set_xlabel('Key (attending to)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query (from)', fontsize=12, fontweight='bold')
    ax.set_title('Self-Attention Heatmap\n"The cat sat on the mat"',
                 fontsize=14, fontweight='bold', pad=20)

    # Add explanation text
    fig.text(0.5, 0.02,
             'Each row shows how much a token (Query) attends to all other tokens (Keys).\n'
             'Row values sum to 1.0 due to softmax normalization.',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save
    output_path = OUTPUT_DIR / 'self_attention_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")

    return attention_weights


def generate_multi_head_attention():
    """
    Generate visualization of multi-head attention with multiple attention heads
    shown side by side, demonstrating how different heads learn different patterns.
    """
    tokens = ["The", "cat", "sat", "on", "mat"]
    n_tokens = len(tokens)
    n_heads = 4
    d_model = 64
    d_k = d_model // n_heads

    np.random.seed(42)

    # Generate different attention patterns for each head
    # Each head learns to focus on different aspects
    head_patterns = []
    head_descriptions = [
        "Head 1: Syntactic\n(adjacent words)",
        "Head 2: Subject-Verb\n(semantic relations)",
        "Head 3: Positional\n(nearby context)",
        "Head 4: Global\n(long-range)"
    ]

    for h in range(n_heads):
        embeddings = np.random.randn(n_tokens, d_model)
        W_q = np.random.randn(d_model, d_k) * 0.1
        W_k = np.random.randn(d_model, d_k) * 0.1

        Q = np.dot(embeddings, W_q)
        K = np.dot(embeddings, W_k)

        weights = compute_attention_weights(Q, K, d_k)

        # Add specific patterns for each head to make them interpretable
        if h == 0:  # Syntactic - adjacent words
            for i in range(n_tokens):
                if i > 0:
                    weights[i, i-1] += 0.5
                if i < n_tokens - 1:
                    weights[i, i+1] += 0.3
        elif h == 1:  # Subject-Verb relations
            weights[2, 0] += 0.5  # sat -> The
            weights[2, 1] += 0.6  # sat -> cat
            weights[4, 1] += 0.4  # mat -> cat
        elif h == 2:  # Positional (nearby)
            for i in range(n_tokens):
                for j in range(n_tokens):
                    distance = abs(i - j)
                    weights[i, j] += max(0, 0.5 - 0.15 * distance)
        elif h == 3:  # Global patterns
            weights[:, 0] += 0.3  # Everyone attends to "The"
            weights[:, -1] += 0.2  # Everyone attends to "mat"

        # Normalize
        weights = weights / weights.sum(axis=1, keepdims=True)
        head_patterns.append(weights)

    # Create figure with subplots
    fig, axes = plt.subplots(1, n_heads, figsize=(16, 5))

    for h, (ax, weights, desc) in enumerate(zip(axes, head_patterns, head_descriptions)):
        sns.heatmap(
            weights,
            xticklabels=tokens,
            yticklabels=tokens if h == 0 else False,
            annot=True,
            fmt='.2f',
            cmap='Purples',
            ax=ax,
            cbar=False,
            vmin=0,
            vmax=0.8,
            linewidths=0.5,
            square=True
        )
        ax.set_title(desc, fontsize=10, fontweight='bold')
        ax.set_xlabel('Key', fontsize=9)
        if h == 0:
            ax.set_ylabel('Query', fontsize=9)

    # Add main title
    fig.suptitle('Multi-Head Attention: Different Heads Learn Different Patterns\n'
                 '"The cat sat on mat"',
                 fontsize=14, fontweight='bold', y=1.02)

    # Add colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='Purples', norm=plt.Normalize(vmin=0, vmax=0.8))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Attention Weight', fontsize=10)

    # Add explanation
    fig.text(0.5, -0.05,
             'Multi-head attention allows the model to jointly attend to information from different\n'
             'representation subspaces at different positions. Each head learns different patterns.',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.05, 0.9, 1])

    # Save
    output_path = OUTPUT_DIR / 'multi_head_attention.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def generate_qkv_flow_diagram():
    """
    Generate a Query-Key-Value flow diagram showing the attention mechanism.
    This is included in the main visualization but can be called separately.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    colors = {
        'input': '#E8F5E9',
        'query': '#FFCDD2',
        'key': '#BBDEFB',
        'value': '#FFF9C4',
        'attention': '#E1BEE7',
        'output': '#B2DFDB'
    }

    # Title
    ax.text(7, 9.5, 'Query-Key-Value Attention Mechanism',
            fontsize=16, fontweight='bold', ha='center')

    # Input embeddings
    input_box = FancyBboxPatch((0.5, 7), 3, 1.5, boxstyle="round,pad=0.05",
                                facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 7.75, 'Input Embeddings\n(n x d_model)', ha='center', va='center', fontsize=10)

    # Linear projections
    proj_y = 5
    proj_height = 1.2
    proj_width = 2.5

    # Q projection
    q_box = FancyBboxPatch((0.5, proj_y), proj_width, proj_height, boxstyle="round,pad=0.05",
                            facecolor=colors['query'], edgecolor='black', linewidth=2)
    ax.add_patch(q_box)
    ax.text(1.75, proj_y + 0.6, 'W_Q\nQuery', ha='center', va='center', fontsize=9, fontweight='bold')

    # K projection
    k_box = FancyBboxPatch((3.5, proj_y), proj_width, proj_height, boxstyle="round,pad=0.05",
                            facecolor=colors['key'], edgecolor='black', linewidth=2)
    ax.add_patch(k_box)
    ax.text(4.75, proj_y + 0.6, 'W_K\nKey', ha='center', va='center', fontsize=9, fontweight='bold')

    # V projection
    v_box = FancyBboxPatch((6.5, proj_y), proj_width, proj_height, boxstyle="round,pad=0.05",
                            facecolor=colors['value'], edgecolor='black', linewidth=2)
    ax.add_patch(v_box)
    ax.text(7.75, proj_y + 0.6, 'W_V\nValue', ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrows from input to projections
    arrow_style = dict(arrowstyle='->', color='black', lw=1.5,
                       connectionstyle="arc3,rad=0")

    ax.annotate('', xy=(1.75, 6.2), xytext=(2, 7),
                arrowprops=arrow_style)
    ax.annotate('', xy=(4.75, 6.2), xytext=(2, 7),
                arrowprops=arrow_style)
    ax.annotate('', xy=(7.75, 6.2), xytext=(3, 7),
                arrowprops=arrow_style)

    # Attention computation box
    attn_box = FancyBboxPatch((10, 4.5), 3.5, 2),
    attn_box = FancyBboxPatch((10, 4.5), 3.5, 2, boxstyle="round,pad=0.05",
                               facecolor=colors['attention'], edgecolor='black', linewidth=2)
    ax.add_patch(attn_box)
    ax.text(11.75, 5.5, 'Scaled Dot-Product\nAttention', ha='center', va='center',
            fontsize=10, fontweight='bold')

    # Formula below attention box
    ax.text(11.75, 3.8, r'$\frac{QK^T}{\sqrt{d_k}}$', ha='center', va='center', fontsize=14)
    ax.text(11.75, 3.2, r'$\downarrow$ softmax', ha='center', va='center', fontsize=10)
    ax.text(11.75, 2.7, r'Attention Weights', ha='center', va='center', fontsize=9)

    # Arrows from Q, K to attention
    ax.annotate('', xy=(10, 5.5), xytext=(3, 5.6),
                arrowprops=dict(arrowstyle='->', color='#E53935', lw=2))
    ax.annotate('', xy=(10, 5.3), xytext=(6, 5.6),
                arrowprops=dict(arrowstyle='->', color='#1E88E5', lw=2))

    # Arrow from V to multiplication
    ax.annotate('', xy=(10, 2), xytext=(7.75, 5),
                arrowprops=dict(arrowstyle='->', color='#FDD835', lw=2,
                               connectionstyle="arc3,rad=-0.3"))

    # Multiplication symbol
    ax.text(10.5, 2, r'$\times$', ha='center', va='center', fontsize=20)

    # Output box
    output_box = FancyBboxPatch((10, 0.5), 3.5, 1.2, boxstyle="round,pad=0.05",
                                 facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(11.75, 1.1, 'Output\n(Weighted Values)', ha='center', va='center', fontsize=10)

    # Arrow to output
    ax.annotate('', xy=(11.75, 1.7), xytext=(11.75, 2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Legend
    legend_y = 1
    legend_items = [
        ('Query (Q)', colors['query']),
        ('Key (K)', colors['key']),
        ('Value (V)', colors['value']),
    ]
    for i, (label, color) in enumerate(legend_items):
        ax.add_patch(plt.Rectangle((0.5 + i*2.5, legend_y), 0.4, 0.4,
                                    facecolor=color, edgecolor='black'))
        ax.text(1 + i*2.5, legend_y + 0.2, label, fontsize=9, va='center')

    plt.tight_layout()
    return fig


def generate_positional_encoding():
    """
    Generate visualization of sinusoidal positional encoding.
    Shows the pattern of sine/cosine waves at different frequencies.
    """
    # Parameters
    max_seq_len = 50
    d_model = 64

    # Create positional encoding
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = np.zeros((max_seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # Even indices
    pe[:, 1::2] = np.cos(position * div_term)  # Odd indices

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(14, 12))

    # 1. Full positional encoding heatmap
    ax1 = plt.subplot(2, 2, 1)
    sns.heatmap(pe[:30, :32], cmap='RdBu_r', center=0, ax=ax1,
                cbar_kws={'label': 'Encoding Value'})
    ax1.set_xlabel('Embedding Dimension', fontsize=10)
    ax1.set_ylabel('Position in Sequence', fontsize=10)
    ax1.set_title('Positional Encoding Matrix (first 30 positions, 32 dims)',
                  fontsize=11, fontweight='bold')

    # 2. Sinusoidal waves at different dimensions
    ax2 = plt.subplot(2, 2, 2)
    positions = np.arange(max_seq_len)
    dims_to_show = [0, 4, 8, 16, 32, 48]
    colors_wave = plt.cm.viridis(np.linspace(0, 1, len(dims_to_show)))

    for dim, color in zip(dims_to_show, colors_wave):
        if dim < d_model:
            ax2.plot(positions, pe[:, dim], label=f'dim {dim}', color=color, linewidth=2)

    ax2.set_xlabel('Position', fontsize=10)
    ax2.set_ylabel('Encoding Value', fontsize=10)
    ax2.set_title('Positional Encoding Waves by Dimension', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Position similarity matrix
    ax3 = plt.subplot(2, 2, 3)

    # Compute cosine similarity between position encodings
    pe_normalized = pe[:20] / np.linalg.norm(pe[:20], axis=1, keepdims=True)
    similarity = np.dot(pe_normalized, pe_normalized.T)

    sns.heatmap(similarity, cmap='YlOrRd', ax=ax3, square=True,
                cbar_kws={'label': 'Cosine Similarity'})
    ax3.set_xlabel('Position', fontsize=10)
    ax3.set_ylabel('Position', fontsize=10)
    ax3.set_title('Position Similarity (Cosine)\nNearby positions are more similar',
                  fontsize=11, fontweight='bold')

    # 4. Example with actual tokens
    ax4 = plt.subplot(2, 2, 4)

    tokens = ["<s>", "The", "cat", "sat", "on", "the", "mat", "</s>"]
    n_tokens = len(tokens)

    # Show positional encoding for these tokens (first 16 dimensions)
    pe_tokens = pe[:n_tokens, :16]

    sns.heatmap(pe_tokens, cmap='RdBu_r', center=0, ax=ax4,
                yticklabels=tokens, annot=False,
                cbar_kws={'label': 'Encoding Value'})
    ax4.set_xlabel('Dimension', fontsize=10)
    ax4.set_ylabel('Token', fontsize=10)
    ax4.set_title('Positional Encoding for "The cat sat on the mat"',
                  fontsize=11, fontweight='bold')

    # Main title
    fig.suptitle('Sinusoidal Positional Encoding Visualization\n'
                 'PE(pos, 2i) = sin(pos / 10000^(2i/d_model))  |  '
                 'PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))',
                 fontsize=13, fontweight='bold', y=1.02)

    # Add explanation
    fig.text(0.5, -0.02,
             'Positional encodings allow the model to understand token order.\n'
             'Different dimensions use different frequencies, enabling the model to '
             'learn relative positions.',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    # Save
    output_path = OUTPUT_DIR / 'positional_encoding.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def generate_layer_attention_weights():
    """
    Generate visualization showing how attention patterns change across layers.
    This demonstrates the hierarchical learning in transformer models.
    """
    tokens = ["The", "cat", "sat", "on", "mat"]
    n_tokens = len(tokens)
    n_layers = 4
    d_k = 16

    np.random.seed(42)

    fig, axes = plt.subplots(1, n_layers, figsize=(16, 4))

    layer_descriptions = [
        "Layer 1: Local Patterns\n(nearby tokens)",
        "Layer 2: Phrase Structure\n(noun-verb groups)",
        "Layer 3: Sentence Relations\n(subject-object)",
        "Layer 4: Global Context\n(full sentence)"
    ]

    for layer in range(n_layers):
        ax = axes[layer]

        # Generate base attention
        embeddings = np.random.randn(n_tokens, d_k * 4)
        W_q = np.random.randn(d_k * 4, d_k) * 0.1
        W_k = np.random.randn(d_k * 4, d_k) * 0.1

        Q = np.dot(embeddings, W_q)
        K = np.dot(embeddings, W_k)
        weights = compute_attention_weights(Q, K, d_k)

        # Add layer-specific patterns
        if layer == 0:  # Local - strong diagonal and adjacent
            for i in range(n_tokens):
                weights[i, i] += 0.3
                if i > 0:
                    weights[i, i-1] += 0.4
                if i < n_tokens - 1:
                    weights[i, i+1] += 0.3
        elif layer == 1:  # Phrase structure
            weights[1, 0] += 0.5  # cat -> The
            weights[2, 1] += 0.4  # sat -> cat
            weights[4, 3] += 0.3  # mat -> on
        elif layer == 2:  # Sentence relations
            weights[2, 1] += 0.6  # sat -> cat (subject-verb)
            weights[4, 1] += 0.4  # mat -> cat (object relation)
            weights[2, 4] += 0.3  # sat -> mat
        elif layer == 3:  # Global
            weights[:, 0] += 0.2  # Everyone attends to "The"
            weights[:, 1] += 0.3  # Everyone attends to "cat"
            weights[:, 2] += 0.2  # Everyone attends to "sat"

        # Normalize
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Plot
        sns.heatmap(
            weights,
            xticklabels=tokens,
            yticklabels=tokens if layer == 0 else False,
            annot=True,
            fmt='.2f',
            cmap='Greens',
            ax=ax,
            cbar=False,
            vmin=0,
            vmax=0.7,
            linewidths=0.5,
            square=True
        )
        ax.set_title(layer_descriptions[layer], fontsize=10, fontweight='bold')
        ax.set_xlabel('Key', fontsize=9)
        if layer == 0:
            ax.set_ylabel('Query', fontsize=9)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=0, vmax=0.7))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Attention Weight', fontsize=10)

    fig.suptitle('Attention Patterns Across Transformer Layers\n'
                 'Early layers capture local patterns, later layers capture global context',
                 fontsize=13, fontweight='bold', y=1.05)

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # This is part of the multi-head visualization
    # Save separately if needed
    return fig


def main():
    """Generate all attention mechanism visualizations."""
    print("Generating Attention Mechanism Visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    # 1. Self-attention heatmap
    print("\n1. Generating self-attention heatmap...")
    generate_self_attention_heatmap()

    # 2. Multi-head attention
    print("\n2. Generating multi-head attention visualization...")
    generate_multi_head_attention()

    # 3. Positional encoding
    print("\n3. Generating positional encoding visualization...")
    generate_positional_encoding()

    print("\n" + "-" * 50)
    print("All visualizations generated successfully!")
    print("\nGenerated files:")
    print(f"  - {OUTPUT_DIR / 'self_attention_heatmap.png'}")
    print(f"  - {OUTPUT_DIR / 'multi_head_attention.png'}")
    print(f"  - {OUTPUT_DIR / 'positional_encoding.png'}")


if __name__ == "__main__":
    main()
