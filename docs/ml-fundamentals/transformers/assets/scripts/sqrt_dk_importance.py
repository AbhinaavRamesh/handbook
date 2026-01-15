#!/usr/bin/env python3
"""
Why sqrt(d_k) Scaling Matters
Before/after comparison showing attention weight distributions
and gradient behavior with and without proper scaling
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

np.random.seed(42)

# Create figure
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

# Parameters
seq_len = 8
d_k_values = [64, 256, 512]  # Different dimensions to show effect

# ==============================================================================
# Panel 1: Attention Weights WITHOUT Scaling (Near One-Hot)
# ==============================================================================
ax1 = fig.add_subplot(gs[0, 0])

# Generate random Q and K for demonstration
d_k = 64
Q = np.random.randn(1, d_k) * np.sqrt(d_k)  # Not scaled
K = np.random.randn(seq_len, d_k)

# Compute attention WITHOUT scaling
scores_unscaled = Q @ K.T  # Shape: (1, seq_len)
attention_unscaled = np.exp(scores_unscaled) / np.exp(scores_unscaled).sum(axis=1, keepdims=True)

ax1.bar(range(seq_len), attention_unscaled[0], color='#E57373', edgecolor='#B71C1C', linewidth=2)
ax1.set_xlabel('Key Position', fontsize=11)
ax1.set_ylabel('Attention Weight', fontsize=11)
ax1.set_title('WITHOUT $\\sqrt{d_k}$ Scaling\n(Near One-Hot Distribution)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 1.05)
ax1.set_xticks(range(seq_len))
ax1.axhline(y=1/seq_len, color='gray', linestyle='--', alpha=0.7, label='Uniform')

# Add annotation
max_idx = np.argmax(attention_unscaled[0])
ax1.annotate(f'Max: {attention_unscaled[0, max_idx]:.3f}',
             xy=(max_idx, attention_unscaled[0, max_idx]),
             xytext=(max_idx + 1, 0.8),
             fontsize=10,
             arrowprops=dict(arrowstyle='->', color='#B71C1C'))

ax1.text(0.5, 0.95, 'Problem: Attention collapses\nto single position',
         transform=ax1.transAxes, fontsize=10, color='#B71C1C',
         ha='center', va='top', style='italic',
         bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))

# ==============================================================================
# Panel 2: Attention Weights WITH Scaling (Distributed)
# ==============================================================================
ax2 = fig.add_subplot(gs[0, 1])

# Compute attention WITH scaling
scores_scaled = (Q @ K.T) / np.sqrt(d_k)  # Properly scaled
attention_scaled = np.exp(scores_scaled) / np.exp(scores_scaled).sum(axis=1, keepdims=True)

ax2.bar(range(seq_len), attention_scaled[0], color='#81C784', edgecolor='#1B5E20', linewidth=2)
ax2.set_xlabel('Key Position', fontsize=11)
ax2.set_ylabel('Attention Weight', fontsize=11)
ax2.set_title('WITH $\\sqrt{d_k}$ Scaling\n(Well-Distributed Attention)', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1.05)
ax2.set_xticks(range(seq_len))
ax2.axhline(y=1/seq_len, color='gray', linestyle='--', alpha=0.7, label='Uniform')

# Calculate entropy as measure of distribution
entropy_unscaled = -np.sum(attention_unscaled * np.log(attention_unscaled + 1e-10))
entropy_scaled = -np.sum(attention_scaled * np.log(attention_scaled + 1e-10))

ax2.text(0.5, 0.95, 'Good: Attention spreads\nacross relevant positions',
         transform=ax2.transAxes, fontsize=10, color='#1B5E20',
         ha='center', va='top', style='italic',
         bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))

# ==============================================================================
# Panel 3: Effect of d_k on Score Variance
# ==============================================================================
ax3 = fig.add_subplot(gs[1, 0])

d_k_range = np.arange(16, 513, 16)
score_variance_unscaled = d_k_range  # Variance grows linearly with d_k
score_variance_scaled = np.ones_like(d_k_range)  # Variance stays at 1

ax3.plot(d_k_range, score_variance_unscaled, 'r-', linewidth=2.5, label='Without scaling')
ax3.plot(d_k_range, score_variance_scaled, 'g-', linewidth=2.5, label='With $\\sqrt{d_k}$ scaling')
ax3.fill_between(d_k_range, score_variance_unscaled, alpha=0.3, color='red')
ax3.fill_between(d_k_range, 0, score_variance_scaled, alpha=0.3, color='green')

ax3.set_xlabel('Dimension $d_k$', fontsize=11)
ax3.set_ylabel('Variance of QK^T scores', fontsize=11)
ax3.set_title('Score Variance vs Dimension', fontsize=13, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.set_xlim(0, 520)
ax3.set_ylim(0, 550)

# Add annotation
ax3.annotate('Variance explodes\nwithout scaling!',
             xy=(400, 400), xytext=(300, 300),
             fontsize=10, color='#B71C1C',
             arrowprops=dict(arrowstyle='->', color='#B71C1C'))
ax3.annotate('Stable variance\nwith scaling',
             xy=(400, 1), xytext=(350, 100),
             fontsize=10, color='#1B5E20',
             arrowprops=dict(arrowstyle='->', color='#1B5E20'))

# ==============================================================================
# Panel 4: Gradient Magnitude Comparison
# ==============================================================================
ax4 = fig.add_subplot(gs[1, 1])

# Simulate gradient behavior for different d_k values
d_k_values = [32, 64, 128, 256, 512]
x = np.linspace(0.01, 0.99, 100)

# Without scaling: gradients vanish as softmax becomes sharper
ax4.set_xlabel('Attention Weight (softmax output)', fontsize=11)
ax4.set_ylabel('Gradient Magnitude', fontsize=11)
ax4.set_title('Softmax Gradient Behavior', fontsize=13, fontweight='bold')

# Softmax gradient: p * (1 - p) for diagonal terms
# This peaks at p=0.5 and vanishes at p=0 or p=1
gradient = x * (1 - x)
ax4.plot(x, gradient, 'k-', linewidth=2.5, label='Softmax gradient: p(1-p)')

# Show regions
ax4.axvspan(0, 0.1, alpha=0.3, color='red', label='Vanishing (unscaled)')
ax4.axvspan(0.9, 1.0, alpha=0.3, color='red')
ax4.axvspan(0.1, 0.9, alpha=0.2, color='green', label='Healthy (scaled)')

ax4.axvline(x=0.125, color='#1B5E20', linestyle='--', linewidth=1.5)
ax4.text(0.125, 0.3, f'1/{seq_len}', fontsize=9, color='#1B5E20', ha='center')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 0.3)
ax4.legend(loc='upper right', fontsize=9)

# Add annotations
ax4.annotate('Without scaling:\nweights near 0 or 1\n-> zero gradients',
             xy=(0.95, 0.02), xytext=(0.7, 0.15),
             fontsize=9, color='#B71C1C',
             arrowprops=dict(arrowstyle='->', color='#B71C1C'))

ax4.annotate('With scaling:\nweights spread\n-> healthy gradients',
             xy=(0.4, 0.24), xytext=(0.2, 0.22),
             fontsize=9, color='#1B5E20',
             arrowprops=dict(arrowstyle='->', color='#1B5E20'))

# Main title
fig.suptitle('Why Dividing by $\\sqrt{d_k}$ is Critical for Training',
             fontsize=16, fontweight='bold', y=0.98)

# Add summary text
fig.text(0.5, 0.01,
         'Key Insight: Without scaling, dot products grow with d_k, causing softmax to saturate (near 0 or 1),\n'
         'which kills gradients. Dividing by sqrt(d_k) keeps variance at 1, maintaining healthy gradient flow.',
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout(rect=[0, 0.05, 1, 0.96])

# Save
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, '..', 'images', 'sqrt_dk_importance.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")
plt.close()
