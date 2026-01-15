#!/usr/bin/env python3
"""
Transformer vs RNN Comparison
Multi-panel figure showing why transformers scale better than RNNs
Panel 1: Gradient path length
Panel 2: Parallelization comparison
Panel 3: Long-range dependencies
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, FancyBboxPatch
from matplotlib.lines import Line2D
import os

# Create figure with 3 panels
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ==============================================================================
# Panel 1: Gradient Path Length
# ==============================================================================
ax1 = axes[0]
ax1.set_xlim(-0.5, 10.5)
ax1.set_ylim(-1, 4)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Gradient Path Length', fontsize=14, fontweight='bold', pad=20)

# RNN gradient path (top)
rnn_y = 2.5
for i in range(8):
    # Hidden state boxes
    rect = FancyBboxPatch((i + 0.1, rnn_y - 0.3), 0.8, 0.6,
                           boxstyle="round,pad=0.05", facecolor='#FFCDD2',
                           edgecolor='#C62828', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(i + 0.5, rnn_y, f'h{i}', ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrows between hidden states
    if i < 7:
        ax1.annotate('', xy=(i + 1.1, rnn_y), xytext=(i + 0.9, rnn_y),
                    arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))

# RNN gradient path annotation
ax1.annotate('', xy=(0.5, rnn_y + 0.5), xytext=(7.5, rnn_y + 0.5),
            arrowprops=dict(arrowstyle='<-', color='red', lw=2,
                           connectionstyle='arc3,rad=0.1'))
ax1.text(4, rnn_y + 0.9, 'Gradient: O(T) steps', ha='center', fontsize=10,
         color='red', fontweight='bold')
ax1.text(-0.3, rnn_y, 'RNN:', ha='right', va='center', fontsize=11, fontweight='bold')

# Transformer gradient path (bottom)
trans_y = 0.5
for i in range(8):
    # Token position boxes
    rect = FancyBboxPatch((i + 0.1, trans_y - 0.3), 0.8, 0.6,
                           boxstyle="round,pad=0.05", facecolor='#C8E6C9',
                           edgecolor='#388E3C', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(i + 0.5, trans_y, f'x{i}', ha='center', va='center', fontsize=9, fontweight='bold')

# Direct attention connections (only show a few)
for i in [0, 3, 7]:
    ax1.annotate('', xy=(i + 0.5, trans_y + 0.3), xytext=(4 + 0.5, trans_y + 0.7),
                arrowprops=dict(arrowstyle='->', color='#388E3C', lw=1.5, alpha=0.7,
                               connectionstyle='arc3,rad=0.2'))

ax1.text(4.5, trans_y + 1.1, 'Gradient: O(1) direct path', ha='center', fontsize=10,
         color='#388E3C', fontweight='bold')
ax1.text(-0.3, trans_y, 'Transformer:', ha='right', va='center', fontsize=11, fontweight='bold')

# ==============================================================================
# Panel 2: Parallelization
# ==============================================================================
ax2 = axes[1]
ax2.set_xlim(-0.5, 10.5)
ax2.set_ylim(-0.5, 5.5)
ax2.axis('off')
ax2.set_title('Computation Parallelization', fontsize=14, fontweight='bold', pad=20)

# RNN: Sequential (top)
rnn_y_base = 4
colors_rnn = plt.cm.Reds(np.linspace(0.3, 0.8, 8))
for i in range(8):
    rect = FancyBboxPatch((i + 0.1, rnn_y_base - 0.3), 0.8, 0.6,
                           boxstyle="round,pad=0.05", facecolor=colors_rnn[i],
                           edgecolor='#C62828', linewidth=1.5)
    ax1.add_patch(rect)

# Time steps for RNN
for i in range(8):
    rect = Rectangle((i + 0.15, rnn_y_base - 0.25), 0.7, 0.5,
                      facecolor=colors_rnn[i], edgecolor='#8B0000', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(i + 0.5, rnn_y_base, f't{i}', ha='center', va='center',
             fontsize=9, fontweight='bold', color='white')

ax2.text(-0.3, rnn_y_base, 'RNN:', ha='right', va='center', fontsize=11, fontweight='bold')
ax2.annotate('', xy=(8.3, rnn_y_base), xytext=(0.1, rnn_y_base),
            arrowprops=dict(arrowstyle='->', color='#8B0000', lw=2))
ax2.text(4, rnn_y_base - 0.7, 'Sequential: O(T) time', ha='center', fontsize=10,
         color='#8B0000', fontweight='bold')

# Transformer: Parallel (bottom)
trans_y_base = 1.5
for i in range(8):
    rect = Rectangle((i + 0.15, trans_y_base - 0.25), 0.7, 0.5,
                      facecolor='#4CAF50', edgecolor='#1B5E20', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(i + 0.5, trans_y_base, f'x{i}', ha='center', va='center',
             fontsize=9, fontweight='bold', color='white')

ax2.text(-0.3, trans_y_base, 'Transformer:', ha='right', va='center', fontsize=11, fontweight='bold')

# Show parallel processing with vertical lines
for i in range(8):
    ax2.plot([i + 0.5, i + 0.5], [trans_y_base + 0.3, trans_y_base + 0.8],
             color='#1B5E20', lw=2)
    ax2.plot([i + 0.5], [trans_y_base + 0.8], 'v', color='#1B5E20', markersize=6)

ax2.plot([0.5, 7.5], [trans_y_base + 1.0, trans_y_base + 1.0], '--',
         color='#1B5E20', lw=2)
ax2.text(4, trans_y_base + 1.3, 'Parallel: O(1) time on GPU', ha='center', fontsize=10,
         color='#1B5E20', fontweight='bold')

# ==============================================================================
# Panel 3: Long-Range Dependencies
# ==============================================================================
ax3 = axes[2]
ax3.set_xlim(-0.5, 10.5)
ax3.set_ylim(-0.5, 5)
ax3.axis('off')
ax3.set_title('Long-Range Dependencies', fontsize=14, fontweight='bold', pad=20)

# RNN: Information bottleneck (top)
rnn_y = 3.5
tokens = ['The', 'cat', 'that', 'sat', 'on', 'the', 'mat', 'was']

for i, tok in enumerate(tokens):
    alpha_val = max(0.2, 1 - i * 0.1)  # Fading effect
    rect = Rectangle((i + 0.05, rnn_y - 0.25), 0.9, 0.5,
                      facecolor=plt.cm.Reds(0.3 + 0.5 * alpha_val),
                      edgecolor='#C62828', linewidth=1, alpha=alpha_val)
    ax3.add_patch(rect)
    ax3.text(i + 0.5, rnn_y, tok, ha='center', va='center', fontsize=8,
             color='white' if alpha_val > 0.5 else 'black', fontweight='bold')

# Show information decay
ax3.annotate('Information\ndecays', xy=(7.5, rnn_y), xytext=(9, rnn_y + 0.5),
            fontsize=9, ha='center', color='#8B0000',
            arrowprops=dict(arrowstyle='->', color='#8B0000', lw=1.5))
ax3.text(-0.3, rnn_y, 'RNN:', ha='right', va='center', fontsize=11, fontweight='bold')
ax3.text(4, rnn_y - 0.7, 'Hidden state bottleneck', ha='center', fontsize=10,
         color='#8B0000', style='italic')

# Transformer: Direct attention (bottom)
trans_y = 1
for i, tok in enumerate(tokens):
    rect = Rectangle((i + 0.05, trans_y - 0.25), 0.9, 0.5,
                      facecolor='#4CAF50', edgecolor='#1B5E20', linewidth=1)
    ax3.add_patch(rect)
    ax3.text(i + 0.5, trans_y, tok, ha='center', va='center', fontsize=8,
             color='white', fontweight='bold')

# Show attention connections from last token to earlier ones
attention_weights = [0.3, 0.5, 0.1, 0.05, 0.02, 0.01, 0.01, 0.0]
for i, weight in enumerate(attention_weights[:-1]):
    if weight > 0.05:
        ax3.annotate('', xy=(i + 0.5, trans_y + 0.3), xytext=(7.5, trans_y + 0.6),
                    arrowprops=dict(arrowstyle='->', color='#1B5E20',
                                   lw=1 + weight * 3, alpha=0.5 + weight,
                                   connectionstyle='arc3,rad=0.3'))

ax3.text(-0.3, trans_y, 'Transformer:', ha='right', va='center', fontsize=11, fontweight='bold')
ax3.text(4, trans_y - 0.7, 'Direct attention to any position', ha='center', fontsize=10,
         color='#1B5E20', style='italic')

# Add legend at bottom
legend_elements = [
    Line2D([0], [0], color='#C62828', lw=3, label='RNN: Sequential, O(T) gradient path'),
    Line2D([0], [0], color='#388E3C', lw=3, label='Transformer: Parallel, O(1) gradient path')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11,
           frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.02))

plt.suptitle('Why Transformers Scale Better Than RNNs', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, '..', 'images', 'transformer_vs_rnn_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")
plt.close()
