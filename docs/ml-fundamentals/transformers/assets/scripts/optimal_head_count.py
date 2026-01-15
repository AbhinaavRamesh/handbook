#!/usr/bin/env python3
"""
Optimal Head Count Guidance

Shows the relationship between d_model and recommended number of heads.
Rule of thumb: h = d_model / 64 (typical for NLP)

Marks sweet spots and known model configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# d_model range
d_model_range = np.arange(128, 2049, 1)

# Optimal head count (h = d_model / 64)
h_optimal = d_model_range / 64

# Plot optimal line
ax.plot(d_model_range, h_optimal, 'b-', linewidth=3, color='#9b59b6',
        label='Optimal: h = d_model / 64')

# Shade "too few" and "too many" regions
# Too few: below h_optimal * 0.5
# Too many: above h_optimal * 2

d_dense = np.linspace(128, 2048, 500)
h_opt_dense = d_dense / 64

# Too few heads region (less than half optimal)
ax.fill_between(d_dense, 0, h_opt_dense * 0.5,
                alpha=0.2, color='#e74c3c', label='Too few heads (bottleneck)')

# Too many heads region (more than 2x optimal)
ax.fill_between(d_dense, h_opt_dense * 2, 40,
                alpha=0.2, color='#f39c12', label='Too many heads (redundancy)')

# Sweet spot region (0.7x to 1.3x optimal)
ax.fill_between(d_dense, h_opt_dense * 0.7, h_opt_dense * 1.3,
                alpha=0.3, color='#27ae60', label='Sweet spot')

# Mark known models
models = [
    # (name, d_model, heads, is_standard)
    ('Small Transformer', 256, 4, True),
    ('Base Transformer', 512, 8, True),
    ('BERT-base', 768, 12, True),
    ('GPT-2 Small', 768, 12, True),
    ('BERT-large', 1024, 16, True),
    ('GPT-2 Medium', 1024, 16, True),
    ('GPT-2 Large', 1280, 20, True),
    ('T5-base', 768, 12, True),
    ('RoBERTa', 768, 12, True),
    # Some non-standard for comparison
    ('Custom (too few)', 512, 2, False),
    ('Custom (too many)', 512, 32, False),
]

# Plot model points
for name, d_model, heads, is_standard in models:
    if is_standard:
        ax.scatter([d_model], [heads], s=200, c='#3498db', marker='o',
                   edgecolors='white', linewidth=2, zorder=10)
    else:
        ax.scatter([d_model], [heads], s=150, c='#e74c3c', marker='x',
                   linewidth=3, zorder=10)

# Add annotations for standard models (avoid overlap)
annotations = [
    ('Small', 256, 4, (256, 6)),
    ('Base (Original)', 512, 8, (520, 10.5)),
    ('BERT-base\nGPT-2 Small', 768, 12, (680, 14.5)),
    ('BERT-large\nGPT-2 Med', 1024, 16, (950, 19)),
    ('GPT-2 Large', 1280, 20, (1350, 22)),
]

for name, d_model, heads, (tx, ty) in annotations:
    ax.annotate(name, xy=(d_model, heads), xytext=(tx, ty),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='#666666', alpha=0.7),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#3498db', alpha=0.9))

# Non-standard annotations
ax.annotate('Too few\nheads', xy=(512, 2), xytext=(400, 1),
            fontsize=9, ha='center', color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b'))
ax.annotate('Too many\nheads', xy=(512, 32), xytext=(600, 35),
            fontsize=9, ha='center', color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b'))

# Labels and title
ax.set_xlabel('Model Dimension (d_model)', fontsize=12)
ax.set_ylabel('Number of Heads (h)', fontsize=12)
ax.set_title('Optimal Head Count: h = d_model / 64\nKnown Model Configurations',
             fontsize=14, fontweight='bold')

# Set axis limits
ax.set_xlim(100, 2100)
ax.set_ylim(0, 40)

# Add grid
ax.grid(True, alpha=0.3)

# Create custom legend
legend_elements = [
    plt.Line2D([0], [0], color='#9b59b6', linewidth=3, label='Optimal: h = d_model/64'),
    mpatches.Patch(facecolor='#27ae60', alpha=0.3, label='Sweet spot (0.7x - 1.3x)'),
    mpatches.Patch(facecolor='#e74c3c', alpha=0.2, label='Too few heads (<0.5x)'),
    mpatches.Patch(facecolor='#f39c12', alpha=0.2, label='Too many heads (>2x)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=10, label='Standard models'),
    plt.Line2D([0], [0], marker='x', color='#e74c3c', markersize=10,
               linewidth=3, label='Non-standard (warning)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# Add guidance text box
guidance_text = '\n'.join([
    'Guidance:',
    '=' * 35,
    'd_model=256  --> h=4  heads',
    'd_model=512  --> h=8  heads',
    'd_model=768  --> h=12 heads (BERT)',
    'd_model=1024 --> h=16 heads',
    'd_model=2048 --> h=32 heads',
    '',
    'Why d_k=64?',
    '- Enough capacity per head',
    '- Not too redundant',
    '- Empirically validated',
])
props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#9b59b6', alpha=0.95)
ax.text(0.98, 0.02, guidance_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/optimal_head_count.png',
            dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Generated: optimal_head_count.png")
