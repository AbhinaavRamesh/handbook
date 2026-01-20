#!/usr/bin/env python3
"""
Attention Mechanism Visualization Generator

This script generates visualizations for understanding attention mechanisms
in transformer models, including:
1. Self-attention heatmap (tokens attending to tokens)
2. Multi-head attention visualization (multiple heads side by side)
3. Query-Key-Value flow diagram
4. Positional encoding visualization (sinusoidal pattern)
5. Attention score evolution during training (GIF)

Usage:
    python generate_attention_viz.py

Outputs:
    - self_attention_heatmap.png
    - multi_head_attention.png
    - qkv_flow_diagram.png
    - positional_encoding.png
    - attention_training_evolution.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import matplotlib.animation as animation
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
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

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
    fig, axes = plt.subplots(1, n_heads, figsize=(16, 5), dpi=150)

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
    Enhanced version with clear visual flow.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_aspect('equal')

    # Colors
    colors = {
        'input': '#E3F2FD',
        'query': '#FFCDD2',
        'key': '#BBDEFB',
        'value': '#FFF9C4',
        'scores': '#E1BEE7',
        'softmax': '#C8E6C9',
        'output': '#B2DFDB'
    }

    # Title
    ax.text(5, 5.7, 'Scaled Dot-Product Attention', fontsize=14, fontweight='bold',
            ha='center', va='center')
    ax.text(5, 5.3, r'Attention(Q, K, V) = softmax($\frac{QK^T}{\sqrt{d_k}}$) V',
            fontsize=11, ha='center', va='center', style='italic')

    # Input box
    input_box = FancyBboxPatch((0.3, 3.8), 1.4, 0.8, boxstyle="round,pad=0.05",
                                facecolor=colors['input'], edgecolor='#1976D2', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1, 4.2, 'Input X', ha='center', va='center', fontsize=10, fontweight='bold')

    # Q, K, V boxes
    box_width, box_height = 1.2, 0.7

    # Query
    q_box = FancyBboxPatch((0.3, 2.5), box_width, box_height, boxstyle="round,pad=0.05",
                            facecolor=colors['query'], edgecolor='#D32F2F', linewidth=2)
    ax.add_patch(q_box)
    ax.text(0.9, 2.85, 'Query (Q)', ha='center', va='center', fontsize=9, fontweight='bold')

    # Key
    k_box = FancyBboxPatch((2.0, 2.5), box_width, box_height, boxstyle="round,pad=0.05",
                            facecolor=colors['key'], edgecolor='#1976D2', linewidth=2)
    ax.add_patch(k_box)
    ax.text(2.6, 2.85, 'Key (K)', ha='center', va='center', fontsize=9, fontweight='bold')

    # Value
    v_box = FancyBboxPatch((3.7, 2.5), box_width, box_height, boxstyle="round,pad=0.05",
                            facecolor=colors['value'], edgecolor='#FFA000', linewidth=2)
    ax.add_patch(v_box)
    ax.text(4.3, 2.85, 'Value (V)', ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrows from input to Q, K, V
    arrow_props = dict(arrowstyle='->', color='#424242', lw=1.5,
                       connectionstyle="arc3,rad=0")

    ax.annotate('', xy=(0.9, 3.2), xytext=(0.9, 3.8), arrowprops=arrow_props)
    ax.annotate('', xy=(2.6, 3.2), xytext=(1.4, 3.8), arrowprops=dict(arrowstyle='->', color='#424242', lw=1.5, connectionstyle="arc3,rad=0.1"))
    ax.annotate('', xy=(4.3, 3.2), xytext=(1.7, 3.8), arrowprops=dict(arrowstyle='->', color='#424242', lw=1.5, connectionstyle="arc3,rad=0.2"))

    # W_q, W_k, W_v labels
    ax.text(0.5, 3.5, r'$W_Q$', fontsize=9, ha='center')
    ax.text(1.7, 3.6, r'$W_K$', fontsize=9, ha='center')
    ax.text(2.8, 3.7, r'$W_V$', fontsize=9, ha='center')

    # QK^T computation box
    scores_box = FancyBboxPatch((5.5, 2.5), 1.8, 0.7, boxstyle="round,pad=0.05",
                                 facecolor=colors['scores'], edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(scores_box)
    ax.text(6.4, 2.85, r'$\frac{QK^T}{\sqrt{d_k}}$', ha='center', va='center', fontsize=11)

    # Arrows from Q, K to scores
    ax.annotate('', xy=(5.5, 2.85), xytext=(1.5, 2.85),
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2))
    ax.annotate('', xy=(5.5, 2.7), xytext=(3.2, 2.7),
                arrowprops=dict(arrowstyle='->', color='#1976D2', lw=2))

    # Softmax box
    softmax_box = FancyBboxPatch((5.5, 1.3), 1.8, 0.7, boxstyle="round,pad=0.05",
                                  facecolor=colors['softmax'], edgecolor='#388E3C', linewidth=2)
    ax.add_patch(softmax_box)
    ax.text(6.4, 1.65, 'softmax', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow from scores to softmax
    ax.annotate('', xy=(6.4, 2.0), xytext=(6.4, 2.5), arrowprops=arrow_props)

    # Multiplication with V
    mult_box = FancyBboxPatch((7.8, 1.3), 0.6, 0.7, boxstyle="round,pad=0.05",
                               facecolor='white', edgecolor='#424242', linewidth=2)
    ax.add_patch(mult_box)
    ax.text(8.1, 1.65, r'$\times$', ha='center', va='center', fontsize=14)

    # Arrow from softmax to mult
    ax.annotate('', xy=(7.8, 1.65), xytext=(7.3, 1.65), arrowprops=arrow_props)

    # Arrow from V to mult
    ax.annotate('', xy=(8.1, 2.0), xytext=(4.3, 2.5),
                arrowprops=dict(arrowstyle='->', color='#FFA000', lw=2,
                               connectionstyle="arc3,rad=-0.3"))

    # Output box
    output_box = FancyBboxPatch((8.0, 0.3), 1.6, 0.7, boxstyle="round,pad=0.05",
                                 facecolor=colors['output'], edgecolor='#00796B', linewidth=2)
    ax.add_patch(output_box)
    ax.text(8.8, 0.65, 'Output', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow from mult to output
    ax.annotate('', xy=(8.8, 1.0), xytext=(8.1, 1.3), arrowprops=arrow_props)

    # Legend
    legend_y = 0.3
    legend_items = [
        ('Query', colors['query'], '#D32F2F'),
        ('Key', colors['key'], '#1976D2'),
        ('Value', colors['value'], '#FFA000'),
    ]
    for i, (label, facecolor, edgecolor) in enumerate(legend_items):
        ax.add_patch(plt.Rectangle((0.3 + i*1.5, legend_y), 0.3, 0.3,
                                    facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5))
        ax.text(0.7 + i*1.5, legend_y + 0.15, label, fontsize=9, va='center')

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'qkv_flow_diagram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


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
    fig = plt.figure(figsize=(14, 12), dpi=150)

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


def generate_attention_training_evolution():
    """
    Generate a GIF showing how attention patterns evolve during training.
    Shows progression from random -> uniform -> learned patterns.
    """
    tokens = ["The", "cat", "sat", "on", "mat"]
    n_tokens = len(tokens)
    n_frames = 30
    d_model = 64
    d_k = 16

    np.random.seed(42)

    # Create figure for animation
    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)

    # Generate evolution of attention weights
    # Stage 1 (frames 0-9): Random/noisy attention
    # Stage 2 (frames 10-19): Gradually focusing
    # Stage 3 (frames 20-29): Learned meaningful patterns

    # Target pattern (what we want to learn)
    target_weights = np.ones((n_tokens, n_tokens)) * 0.1
    target_weights[2, 1] = 0.6  # sat -> cat (subject-verb)
    target_weights[4, 1] = 0.5  # mat -> cat (object relation)
    target_weights[2, 0] = 0.3  # sat -> The
    target_weights[3, 2] = 0.4  # on -> sat
    # Normalize
    target_weights = target_weights / target_weights.sum(axis=1, keepdims=True)

    def compute_frame_weights(frame):
        """Compute attention weights for a given frame."""
        progress = frame / (n_frames - 1)

        # Start with random weights
        np.random.seed(frame * 7 + 42)
        random_weights = np.random.rand(n_tokens, n_tokens)
        random_weights = random_weights / random_weights.sum(axis=1, keepdims=True)

        # Uniform weights
        uniform_weights = np.ones((n_tokens, n_tokens)) / n_tokens

        if progress < 0.3:
            # Early training: mostly random noise
            t = progress / 0.3
            weights = (1 - t) * random_weights + t * uniform_weights
            # Add some noise
            noise = np.random.randn(n_tokens, n_tokens) * 0.1 * (1 - t)
            weights = weights + noise
        elif progress < 0.7:
            # Middle training: transitioning from uniform to pattern
            t = (progress - 0.3) / 0.4
            weights = (1 - t) * uniform_weights + t * target_weights
            # Add decreasing noise
            noise = np.random.randn(n_tokens, n_tokens) * 0.05 * (1 - t)
            weights = weights + noise
        else:
            # Late training: refining the pattern
            t = (progress - 0.7) / 0.3
            weights = target_weights.copy()
            # Very small noise that decreases
            noise = np.random.randn(n_tokens, n_tokens) * 0.02 * (1 - t)
            weights = weights + noise

        # Ensure valid probability distribution
        weights = np.clip(weights, 0.01, None)
        weights = weights / weights.sum(axis=1, keepdims=True)

        return weights

    # Store frames
    frames = []

    def update(frame):
        ax.clear()

        weights = compute_frame_weights(frame)
        epoch = int(frame * 100 / n_frames)  # Simulate epoch number

        sns.heatmap(
            weights,
            xticklabels=tokens,
            yticklabels=tokens,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            ax=ax,
            cbar=False,
            vmin=0,
            vmax=0.7,
            linewidths=0.5,
            square=True
        )

        ax.set_xlabel('Key (attending to)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Query (from)', fontsize=11, fontweight='bold')

        # Stage indicator
        progress = frame / (n_frames - 1)
        if progress < 0.3:
            stage = "Early Training: Random Initialization"
            color = '#F44336'
        elif progress < 0.7:
            stage = "Mid Training: Learning Patterns"
            color = '#FF9800'
        else:
            stage = "Late Training: Refined Attention"
            color = '#4CAF50'

        ax.set_title(f'Attention Score Evolution During Training\n'
                     f'Epoch: {epoch} | {stage}',
                     fontsize=12, fontweight='bold', pad=15, color=color)

        return [ax]

    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                    interval=200, blit=False)

    # Save as GIF
    output_path = OUTPUT_DIR / 'attention_training_evolution.gif'
    anim.save(output_path, writer='pillow', fps=5)
    plt.close()
    print(f"Saved: {output_path}")

    # Also save key frames as static images
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    frames_to_save = [0, 15, 29]  # Early, mid, late
    titles = ['Epoch 0: Random', 'Epoch 50: Learning', 'Epoch 100: Converged']

    for idx, (ax, frame, title) in enumerate(zip(axes, frames_to_save, titles)):
        weights = compute_frame_weights(frame)
        sns.heatmap(
            weights,
            xticklabels=tokens,
            yticklabels=tokens if idx == 0 else False,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            ax=ax,
            cbar=False,
            vmin=0,
            vmax=0.7,
            linewidths=0.5,
            square=True
        )
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Key', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Query', fontsize=10)

    fig2.suptitle('Attention Weights Evolution: Before, During, and After Training',
                  fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    output_path2 = OUTPUT_DIR / 'attention_training_stages.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path2}")


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

    # 3. QKV flow diagram
    print("\n3. Generating QKV flow diagram...")
    generate_qkv_flow_diagram()

    # 4. Positional encoding
    print("\n4. Generating positional encoding visualization...")
    generate_positional_encoding()

    # 5. Attention training evolution
    print("\n5. Generating attention training evolution GIF...")
    generate_attention_training_evolution()

    print("\n" + "-" * 50)
    print("All visualizations generated successfully!")
    print("\nGenerated files:")
    print(f"  - {OUTPUT_DIR / 'self_attention_heatmap.png'}")
    print(f"  - {OUTPUT_DIR / 'multi_head_attention.png'}")
    print(f"  - {OUTPUT_DIR / 'qkv_flow_diagram.png'}")
    print(f"  - {OUTPUT_DIR / 'positional_encoding.png'}")
    print(f"  - {OUTPUT_DIR / 'attention_training_evolution.gif'}")
    print(f"  - {OUTPUT_DIR / 'attention_training_stages.png'}")


if __name__ == "__main__":
    main()
