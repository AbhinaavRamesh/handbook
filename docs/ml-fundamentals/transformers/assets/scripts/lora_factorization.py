#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) Concept Visualization

Shows the W = W_0 + BA^T factorization and visualizes dimension reduction:
Full weight matrix (d x d) vs Low-rank matrices (d x r + r x d)
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'images')
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Dimensions
d = 768  # Embedding dimension (like BERT)
r = 8    # Low rank

# Create sample matrices for visualization
np.random.seed(42)
W0 = np.random.randn(32, 32)  # Pretrained weights (scaled down for viz)
A = np.random.randn(32, 4)    # Low-rank matrix A
B = np.random.randn(32, 4)    # Low-rank matrix B
delta_W = B @ A.T             # Low-rank update
W_new = W0 + delta_W          # Updated weights

fig = plt.figure(figsize=(16, 12))

# Create grid spec for custom layout
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8], hspace=0.4, wspace=0.3)

# Title
fig.suptitle('LoRA: Low-Rank Adaptation of Large Language Models\n$W_{new} = W_0 + \\Delta W = W_0 + BA^T$',
             fontsize=16, fontweight='bold', y=0.98)

# Row 1: Original weights (frozen) and concept
ax1 = fig.add_subplot(gs[0, :2])
im1 = ax1.imshow(W0, cmap='RdBu', aspect='equal', vmin=-3, vmax=3)
ax1.set_title('$W_0$ (Pretrained Weights)\nFrozen during LoRA fine-tuning', fontsize=12, fontweight='bold')
ax1.set_xlabel(f'd = {d}')
ax1.set_ylabel(f'd = {d}')
ax1.set_xticks([])
ax1.set_yticks([])
plt.colorbar(im1, ax=ax1, shrink=0.8)

# Add "FROZEN" overlay
ax1.text(0.5, 0.5, 'FROZEN', transform=ax1.transAxes, fontsize=24,
         color='gray', alpha=0.3, ha='center', va='center', fontweight='bold', rotation=45)

# Concept diagram on the right
ax_concept = fig.add_subplot(gs[0, 2:])
ax_concept.axis('off')
ax_concept.set_xlim(0, 10)
ax_concept.set_ylim(0, 6)

# Draw the equation components
ax_concept.text(5, 5.5, 'LoRA Factorization', fontsize=14, fontweight='bold', ha='center')
ax_concept.text(5, 4.5, '$W_{new} = W_0 + BA^T$', fontsize=16, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

# Parameter comparison
ax_concept.text(1, 3, 'Full Fine-Tuning:', fontsize=11, fontweight='bold')
ax_concept.text(1, 2.3, f'Parameters: d x d = {d} x {d}', fontsize=10)
ax_concept.text(1, 1.7, f'= {d*d:,} params', fontsize=10, color='red')

ax_concept.text(6, 3, 'LoRA (rank r={}):', fontsize=11, fontweight='bold'.format(r))
ax_concept.text(6, 2.3, f'Parameters: d x r + r x d', fontsize=10)
ax_concept.text(6, 1.7, f'= {d}x{r} + {r}x{d} = {2*d*r:,} params', fontsize=10, color='green')

# Reduction ratio
reduction = (d * d) / (2 * d * r)
ax_concept.text(5, 0.5, f'Parameter Reduction: {reduction:.0f}x', fontsize=13,
                fontweight='bold', ha='center', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green'))

# Row 2: Low-rank matrices
ax_A = fig.add_subplot(gs[1, 0])
im_A = ax_A.imshow(A, cmap='Greens', aspect='auto')
ax_A.set_title('Matrix A\n(d x r)', fontsize=11, fontweight='bold')
ax_A.set_xlabel(f'r = {r}')
ax_A.set_ylabel(f'd = {d}')
ax_A.set_xticks([])
ax_A.set_yticks([])
ax_A.text(0.5, -0.15, 'Trainable', transform=ax_A.transAxes, ha='center',
          fontsize=10, color='green', fontweight='bold')

ax_B = fig.add_subplot(gs[1, 1])
im_B = ax_B.imshow(B, cmap='Blues', aspect='auto')
ax_B.set_title('Matrix B\n(d x r)', fontsize=11, fontweight='bold')
ax_B.set_xlabel(f'r = {r}')
ax_B.set_ylabel(f'd = {d}')
ax_B.set_xticks([])
ax_B.set_yticks([])
ax_B.text(0.5, -0.15, 'Trainable', transform=ax_B.transAxes, ha='center',
          fontsize=10, color='blue', fontweight='bold')

# Delta W (the low-rank update)
ax_delta = fig.add_subplot(gs[1, 2])
im_delta = ax_delta.imshow(delta_W, cmap='RdBu', aspect='equal', vmin=-3, vmax=3)
ax_delta.set_title('$\\Delta W = BA^T$\n(Low-Rank Update)', fontsize=11, fontweight='bold')
ax_delta.set_xlabel(f'd = {d}')
ax_delta.set_ylabel(f'd = {d}')
ax_delta.set_xticks([])
ax_delta.set_yticks([])
plt.colorbar(im_delta, ax=ax_delta, shrink=0.8)

# Final W_new
ax_new = fig.add_subplot(gs[1, 3])
im_new = ax_new.imshow(W_new, cmap='RdBu', aspect='equal', vmin=-3, vmax=3)
ax_new.set_title('$W_{new} = W_0 + \\Delta W$\n(Adapted Weights)', fontsize=11, fontweight='bold')
ax_new.set_xlabel(f'd = {d}')
ax_new.set_ylabel(f'd = {d}')
ax_new.set_xticks([])
ax_new.set_yticks([])
plt.colorbar(im_new, ax=ax_new, shrink=0.8)

# Row 3: Summary bar chart
ax_bar = fig.add_subplot(gs[2, :])

# Data for bar chart
methods = ['Full\nFine-Tuning', 'LoRA r=4', 'LoRA r=8', 'LoRA r=16', 'LoRA r=64']
params = [d*d, 2*d*4, 2*d*8, 2*d*16, 2*d*64]
params_percent = [100, 100*2*4/d, 100*2*8/d, 100*2*16/d, 100*2*64/d]

colors_bar = ['#e57373', '#81c784', '#81c784', '#81c784', '#81c784']
bars = ax_bar.bar(methods, params_percent, color=colors_bar, edgecolor='black', linewidth=1.2)

ax_bar.set_ylabel('Parameters (%)', fontweight='bold')
ax_bar.set_title('Trainable Parameters Comparison (BERT-base, d=768)', fontweight='bold')
ax_bar.set_ylim(0, 110)

# Add value labels
for bar, pct, param in zip(bars, params_percent, params):
    height = bar.get_height()
    ax_bar.annotate(f'{pct:.2f}%\n({param:,})',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add horizontal line at different thresholds
ax_bar.axhline(y=1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
ax_bar.text(4.5, 2, '1% threshold', fontsize=9, color='orange')

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save figure
output_path = os.path.join(output_dir, 'lora_factorization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"Figure saved to: {output_path}")
