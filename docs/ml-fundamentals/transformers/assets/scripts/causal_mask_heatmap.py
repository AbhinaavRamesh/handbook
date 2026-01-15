#!/usr/bin/env python3
"""
Causal Mask Heatmap Visualization

Creates a 6x6 lower triangular attention mask heatmap showing
which positions can attend to which other positions in a decoder.
- White = allowed (can attend to past + current)
- Black = masked (cannot attend to future)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Set up the output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'images')
os.makedirs(output_dir, exist_ok=True)

# Create causal mask (lower triangular)
seq_len = 6
mask = np.tril(np.ones((seq_len, seq_len)))

# Create figure with decoder-themed colors
fig, ax = plt.subplots(figsize=(8, 7))

# Create custom colormap: white for allowed, dark orange/red for masked
from matplotlib.colors import LinearSegmentedColormap
colors = ['#1a1a1a', '#ffffff']  # Black for masked (0), White for allowed (1)
cmap = LinearSegmentedColormap.from_list('causal', colors)

# Plot heatmap
im = ax.imshow(mask, cmap=cmap, aspect='equal', vmin=0, vmax=1)

# Add grid lines
for i in range(seq_len + 1):
    ax.axhline(i - 0.5, color='#ff6b35', linewidth=1.5)
    ax.axvline(i - 0.5, color='#ff6b35', linewidth=1.5)

# Add checkmarks and X marks
for i in range(seq_len):
    for j in range(seq_len):
        if mask[i, j] == 1:
            ax.text(j, i, '\u2713', ha='center', va='center',
                   fontsize=16, color='#2e7d32', fontweight='bold')
        else:
            ax.text(j, i, '\u2717', ha='center', va='center',
                   fontsize=16, color='#c62828', fontweight='bold')

# Labels
position_labels = [f'Position {i+1}' for i in range(seq_len)]
ax.set_xticks(range(seq_len))
ax.set_yticks(range(seq_len))
ax.set_xticklabels(position_labels, fontsize=10, rotation=45, ha='right')
ax.set_yticklabels(position_labels, fontsize=10)

# Axis labels
ax.set_xlabel('Key Position (attending to)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Query Position (from)', fontsize=12, fontweight='bold', labelpad=10)

# Title
ax.set_title('Causal (Lower Triangular) Attention Mask\nDecoder Self-Attention',
             fontsize=14, fontweight='bold', color='#d94a1e', pad=15)

# Legend
allowed_patch = mpatches.Patch(facecolor='white', edgecolor='#333',
                                label='Allowed (can attend)')
masked_patch = mpatches.Patch(facecolor='#1a1a1a', edgecolor='#333',
                               label='Masked (cannot attend)')
ax.legend(handles=[allowed_patch, masked_patch], loc='upper left',
          bbox_to_anchor=(1.02, 1), fontsize=10)

# Add annotation explaining the pattern
annotation_text = "Each position can only\nattend to itself and\nprevious positions"
ax.annotate(annotation_text, xy=(0.5, -0.18), xycoords='axes fraction',
            ha='center', va='top', fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3e6', edgecolor='#ff6b35'))

plt.tight_layout()

# Save figure
output_path = os.path.join(output_dir, 'causal_mask_heatmap.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Saved causal mask heatmap to: {output_path}")
