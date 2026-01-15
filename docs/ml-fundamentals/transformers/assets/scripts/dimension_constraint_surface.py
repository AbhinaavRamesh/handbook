#!/usr/bin/env python3
"""
Dimension Constraint Surface: d_model = h x d_k

3D visualization showing the relationship between:
- Number of heads (h)
- Head dimension (d_k)
- Resulting model dimension (d_model)

Marks real configurations from known models (BERT, GPT-2, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# Create figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Define ranges
h_range = np.arange(1, 17, 1)       # 1 to 16 heads
d_k_range = np.arange(32, 257, 8)   # 32 to 256 head dimension

# Create meshgrid
H, D_K = np.meshgrid(h_range, d_k_range)
D_MODEL = H * D_K  # Constraint: d_model = h * d_k

# Custom purple colormap
purple_cmap = LinearSegmentedColormap.from_list(
    'purple_gradient',
    ['#e8daef', '#d5a6e6', '#bb8fce', '#9b59b6', '#7d3c98', '#4a235a']
)

# Plot surface
surf = ax.plot_surface(H, D_K, D_MODEL, cmap=purple_cmap, alpha=0.7,
                       linewidth=0.5, antialiased=True, edgecolor='#7d3c98')

# Real model configurations
models = [
    # (name, h, d_k, d_model, color)
    ('BERT-base', 12, 64, 768, '#e74c3c'),
    ('BERT-large', 16, 64, 1024, '#c0392b'),
    ('GPT-2 (small)', 12, 64, 768, '#3498db'),
    ('GPT-2 (medium)', 16, 64, 1024, '#2980b9'),
    ('GPT-2 (large)', 20, 64, 1280, '#1f618d'),
    ('Transformer (original)', 8, 64, 512, '#27ae60'),
    ('DistilBERT', 12, 64, 768, '#f39c12'),
    ('RoBERTa', 12, 64, 768, '#9b59b6'),
]

# Plot model points
for name, h, d_k, d_model, color in models:
    ax.scatter([h], [d_k], [d_model], c=color, s=200, marker='o',
               edgecolors='white', linewidth=2, zorder=10)
    # Add vertical line to surface
    ax.plot([h, h], [d_k, d_k], [0, d_model], color=color,
            linestyle='--', alpha=0.5, linewidth=1)

# Labels and title
ax.set_xlabel('Number of Heads (h)', fontsize=12, labelpad=10)
ax.set_ylabel('Head Dimension (d_k)', fontsize=12, labelpad=10)
ax.set_zlabel('Model Dimension (d_model)', fontsize=12, labelpad=10)
ax.set_title('Dimension Constraint Surface: d_model = h x d_k\nValid Configurations for Multi-Head Attention',
             fontsize=14, fontweight='bold', pad=20)

# Set axis limits
ax.set_xlim(1, 16)
ax.set_ylim(32, 256)
ax.set_zlim(0, 4100)

# Add colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('d_model', fontsize=11)

# Create legend for models
legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=10, label=name)
                   for name, h, d_k, d_model, color in models]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
          title='Real Models', title_fontsize=10)

# Add annotation box
textstr = '\n'.join([
    'Constraint: d_model = h x d_k',
    '',
    'Common pattern: d_k = 64',
    '- 8 heads: d_model = 512',
    '- 12 heads: d_model = 768',
    '- 16 heads: d_model = 1024',
])
props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#9b59b6', alpha=0.9)
fig.text(0.02, 0.02, textstr, fontsize=10, va='bottom', bbox=props, family='monospace')

# Adjust viewing angle
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-fundamentals/transformers/assets/images/dimension_constraint_surface.png',
            dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Generated: dimension_constraint_surface.png")
