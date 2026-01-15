#!/usr/bin/env python3
"""
Parameter Count Comparison: Single-Head vs Multi-Head Attention

Key insight: Multi-head attention has the SAME number of parameters as single-head,
but achieves much higher expressiveness through specialized projections.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration
d_model = 512

# Single-head parameters
single_head = {
    'W_Q': d_model * d_model,      # 512 x 512
    'W_K': d_model * d_model,      # 512 x 512
    'W_V': d_model * d_model,      # 512 x 512
    'W_O': d_model * d_model,      # 512 x 512
}

# Multi-head parameters (8 heads, d_k = 64)
h = 8
d_k = d_model // h  # 64

multi_head = {
    'W_Q (8 heads)': h * d_model * d_k,   # 8 x 512 x 64 = 262,144
    'W_K (8 heads)': h * d_model * d_k,   # 8 x 512 x 64 = 262,144
    'W_V (8 heads)': h * d_model * d_k,   # 8 x 512 x 64 = 262,144
    'W_O': d_model * d_model,              # 512 x 512 = 262,144
}

# Calculate totals
single_total = sum(single_head.values())
multi_total = sum(multi_head.values())

# Colors (purple theme)
colors_single = ['#a0a0a0', '#888888', '#707070', '#585858']
colors_multi = ['#d5a6e6', '#bb8fce', '#9b59b6', '#7d3c98']

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 8))

# ============ Left: Single-Head Stacked Bar ============
ax1 = axes[0]
bottom = 0
bars1 = []
for i, (name, value) in enumerate(single_head.items()):
    bar = ax1.bar('Single-Head\n(d_model=512)', value/1e6, bottom=bottom/1e6,
                  color=colors_single[i], edgecolor='#333333', linewidth=1)
    bars1.append(bar)
    # Add label
    ax1.text(0, (bottom + value/2)/1e6, f'{name}\n{value:,}',
             ha='center', va='center', fontsize=9, fontweight='bold')
    bottom += value

ax1.set_ylabel('Parameters (Millions)', fontsize=12)
ax1.set_title('Single-Head Attention', fontsize=13, fontweight='bold', color='#333333')
ax1.set_ylim(0, 1.3)
ax1.axhline(y=single_total/1e6, color='#333333', linestyle='--', alpha=0.7)
ax1.text(0.5, single_total/1e6 + 0.05, f'Total: {single_total:,}',
         ha='center', fontsize=11, fontweight='bold')

# ============ Middle: Multi-Head Stacked Bar ============
ax2 = axes[1]
bottom = 0
bars2 = []
for i, (name, value) in enumerate(multi_head.items()):
    bar = ax2.bar('Multi-Head\n(8 heads, d_k=64)', value/1e6, bottom=bottom/1e6,
                  color=colors_multi[i], edgecolor='#333333', linewidth=1)
    bars2.append(bar)
    # Add label
    ax2.text(0, (bottom + value/2)/1e6, f'{name}\n{value:,}',
             ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    bottom += value

ax2.set_ylabel('Parameters (Millions)', fontsize=12)
ax2.set_title('Multi-Head Attention', fontsize=13, fontweight='bold', color='#4a235a')
ax2.set_ylim(0, 1.3)
ax2.axhline(y=multi_total/1e6, color='#4a235a', linestyle='--', alpha=0.7)
ax2.text(0.5, multi_total/1e6 + 0.05, f'Total: {multi_total:,}',
         ha='center', fontsize=11, fontweight='bold', color='#4a235a')

# ============ Right: Breakdown Comparison ============
ax3 = axes[2]

# Bar positions
categories = ['Query\nProjections', 'Key\nProjections', 'Value\nProjections', 'Output\nProjection']
x = np.arange(len(categories))
width = 0.35

single_values = [single_head['W_Q'], single_head['W_K'], single_head['W_V'], single_head['W_O']]
multi_values = [multi_head['W_Q (8 heads)'], multi_head['W_K (8 heads)'],
                multi_head['W_V (8 heads)'], multi_head['W_O']]

bars_s = ax3.bar(x - width/2, [v/1e6 for v in single_values], width,
                 label='Single-Head', color='#888888', edgecolor='#333333')
bars_m = ax3.bar(x + width/2, [v/1e6 for v in multi_values], width,
                 label='Multi-Head', color='#9b59b6', edgecolor='#333333')

# Add value labels on bars
for bar, val in zip(bars_s, single_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val//1000}K', ha='center', va='bottom', fontsize=8)

for bar, val in zip(bars_m, multi_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val//1000}K', ha='center', va='bottom', fontsize=8, color='#4a235a')

ax3.set_ylabel('Parameters (Millions)', fontsize=12)
ax3.set_title('Parameter Distribution by Component', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(categories, fontsize=10)
ax3.legend(loc='upper right', fontsize=10)
ax3.set_ylim(0, 0.35)

# Add annotation box
textstr = '\n'.join([
    'Key Insight:',
    '------------------------',
    f'Single-Head:  {single_total:,} params',
    f'Multi-Head:   {multi_total:,} params',
    '',
    'SAME PARAMETERS!',
    '',
    'But multi-head is more expressive:',
    '- 8 independent projections',
    '- Diverse attention patterns',
    '- Better gradient flow',
])

props = dict(boxstyle='round,pad=0.5', facecolor='#f5eef8', edgecolor='#9b59b6', alpha=0.9)
fig.text(0.5, -0.02, textstr, fontsize=10, ha='center', va='top', bbox=props,
         family='monospace')

# Main title
fig.suptitle('Multi-Head Attention: Same Parameters, Higher Expressiveness',
             fontsize=15, fontweight='bold', y=1.02)

# Formula annotations
fig.text(0.17, 0.92, 'Single: 4 x d_model^2\n= 4 x 512^2',
         ha='center', fontsize=10, family='monospace', color='#666666')
fig.text(0.5, 0.92, 'Multi: h x 3 x d_model x d_k + d_model^2\n= 8 x 3 x 512 x 64 + 512^2',
         ha='center', fontsize=10, family='monospace', color='#666666')

plt.tight_layout(rect=[0, 0.15, 1, 0.9])
plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/parameter_count_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Generated: parameter_count_comparison.png")
