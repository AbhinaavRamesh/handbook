#!/usr/bin/env python3
"""
Single-Head vs Multi-Head Attention Comparison

Demonstrates why multiple attention heads are more expressive than a single head.
- Single head: One attention pattern must capture all relationships
- Multi-head: Different heads can specialize in different patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

np.random.seed(42)

# Define sequence
tokens = ["The", "cat", "sat", "on", "the", "mat", "and", "purred"]
seq_len = len(tokens)

# Custom colormaps
purple_cmap = LinearSegmentedColormap.from_list(
    'purple', ['#ffffff', '#e8daef', '#9b59b6', '#4a235a']
)
gray_cmap = LinearSegmentedColormap.from_list(
    'gray', ['#ffffff', '#d5d5d5', '#888888', '#333333']
)

def create_single_head_attention(seq_len):
    """
    Single head must compromise between different pattern types.
    Results in a muddled attention pattern.
    """
    attention = np.zeros((seq_len, seq_len))

    # Try to capture local patterns
    for i in range(seq_len):
        for j in range(seq_len):
            dist = abs(i - j)
            attention[i, j] += 0.3 * np.exp(-dist * 0.5)

    # Try to capture syntactic patterns (subject-verb)
    attention[2, 1] += 0.15  # sat -> cat
    attention[7, 1] += 0.1   # purred -> cat

    # Try to capture long-range (conjunction)
    attention[7, 2] += 0.1   # purred -> sat

    # Everything competes - result is compromised
    attention = attention / attention.sum(axis=1, keepdims=True)
    return attention

def create_multi_head_patterns(seq_len):
    """Create 4 specialized head patterns"""
    patterns = []

    # Head 1: Strong local attention
    h1 = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            dist = abs(i - j)
            if dist <= 1:
                h1[i, j] = np.exp(-dist * 0.3)
    h1 = h1 / h1.sum(axis=1, keepdims=True)
    patterns.append((h1, "Local Context"))

    # Head 2: Subject-verb
    h2 = np.random.uniform(0.02, 0.05, (seq_len, seq_len))
    h2[2, 1] = 0.8  # sat -> cat
    h2[7, 1] = 0.7  # purred -> cat
    h2 = h2 / h2.sum(axis=1, keepdims=True)
    patterns.append((h2, "Subject-Verb"))

    # Head 3: Verb connections
    h3 = np.random.uniform(0.02, 0.05, (seq_len, seq_len))
    h3[7, 2] = 0.75  # purred -> sat
    h3[2, 7] = 0.6   # sat -> purred
    h3[6, 2] = 0.5   # and -> sat
    h3 = h3 / h3.sum(axis=1, keepdims=True)
    patterns.append((h3, "Long-Range"))

    # Head 4: Position-based
    h4 = np.random.uniform(0.03, 0.06, (seq_len, seq_len))
    h4[:, 0] += 0.4  # attend to first
    for i in range(1, seq_len):
        h4[i, i-1] += 0.3
    h4 = h4 / h4.sum(axis=1, keepdims=True)
    patterns.append((h4, "Positional"))

    return patterns

def plot_softmax_distribution(ax, attention_row, title, color, tokens):
    """Plot softmax distribution as bar chart"""
    x = np.arange(len(tokens))
    bars = ax.bar(x, attention_row, color=color, alpha=0.8, edgecolor='#333333', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Attention Weight', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=1/len(tokens), color='#999999', linestyle='--', alpha=0.5, label='Uniform')
    return bars

# Create figure
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 4, height_ratios=[1.2, 0.8, 0.8], hspace=0.4, wspace=0.3)

# Get attention patterns
single_head = create_single_head_attention(seq_len)
multi_head_patterns = create_multi_head_patterns(seq_len)

# ============ Row 1: Heatmaps ============
# Single head heatmap (spans 2 columns)
ax_single = fig.add_subplot(gs[0, 0:2])
im1 = ax_single.imshow(single_head, cmap=gray_cmap, aspect='auto', vmin=0, vmax=0.6)
ax_single.set_title('Single-Head Attention\n(One Pattern Must Capture Everything)',
                   fontsize=12, fontweight='bold', color='#333333')
ax_single.set_xticks(range(seq_len))
ax_single.set_yticks(range(seq_len))
ax_single.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
ax_single.set_yticklabels(tokens, fontsize=9)
ax_single.set_xlabel('Key (attending to)', fontsize=10)
ax_single.set_ylabel('Query (from)', fontsize=10)
plt.colorbar(im1, ax=ax_single, shrink=0.8)

# Multi-head combined view (spans 2 columns) - show average
ax_multi = fig.add_subplot(gs[0, 2:4])
combined = np.mean([p[0] for p in multi_head_patterns], axis=0)
im2 = ax_multi.imshow(combined, cmap=purple_cmap, aspect='auto', vmin=0, vmax=0.6)
ax_multi.set_title('Multi-Head Attention (Combined)\n(Specialized Heads Averaged)',
                  fontsize=12, fontweight='bold', color='#4a235a')
ax_multi.set_xticks(range(seq_len))
ax_multi.set_yticks(range(seq_len))
ax_multi.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
ax_multi.set_yticklabels(tokens, fontsize=9)
ax_multi.set_xlabel('Key (attending to)', fontsize=10)
ax_multi.set_ylabel('Query (from)', fontsize=10)
plt.colorbar(im2, ax=ax_multi, shrink=0.8)

# ============ Row 2: Individual head heatmaps ============
for i, (pattern, name) in enumerate(multi_head_patterns):
    ax = fig.add_subplot(gs[1, i])
    im = ax.imshow(pattern, cmap=purple_cmap, aspect='auto', vmin=0, vmax=0.8)
    ax.set_title(f'Head {i+1}: {name}', fontsize=10, fontweight='bold', color='#7d3c98')
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(tokens, fontsize=7)

# ============ Row 3: Softmax distributions for "purred" token ============
query_idx = 7  # "purred"

# Single head distribution
ax_dist1 = fig.add_subplot(gs[2, 0:2])
plot_softmax_distribution(ax_dist1, single_head[query_idx],
                         f'Single-Head: Where does "{tokens[query_idx]}" attend?',
                         '#888888', tokens)
ax_dist1.annotate('Compromised pattern:\nmust balance all relationships',
                 xy=(0.7, 0.7), xycoords='axes fraction',
                 fontsize=9, style='italic', color='#666666')

# Multi-head distributions (stacked for comparison)
ax_dist2 = fig.add_subplot(gs[2, 2:4])
x = np.arange(len(tokens))
width = 0.2
colors = ['#d5a6e6', '#bb8fce', '#9b59b6', '#7d3c98']

for i, (pattern, name) in enumerate(multi_head_patterns):
    ax_dist2.bar(x + i*width - 0.3, pattern[query_idx], width,
                label=f'Head {i+1}: {name}', color=colors[i], alpha=0.9)

ax_dist2.set_xticks(x)
ax_dist2.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
ax_dist2.set_ylabel('Attention Weight', fontsize=9)
ax_dist2.set_title(f'Multi-Head: Where does "{tokens[query_idx]}" attend? (4 heads)',
                  fontsize=10, fontweight='bold')
ax_dist2.legend(loc='upper right', fontsize=8)
ax_dist2.set_ylim(0, 1.0)

# Add main title and explanation
fig.suptitle('Why Multiple Heads Beat a Single Large Head', fontsize=14, fontweight='bold', y=0.98)

fig.text(0.5, 0.01,
         'Single-head attention creates a bottleneck: one distribution must capture all relationship types.\n'
         'Multi-head attention allows specialized patterns, then combines them for richer representations.',
         ha='center', fontsize=10, style='italic', color='#666666')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/single_vs_multi_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Generated: single_vs_multi_comparison.png")
