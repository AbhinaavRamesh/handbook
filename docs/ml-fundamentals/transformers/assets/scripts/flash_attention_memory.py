#!/usr/bin/env python3
"""
Flash Attention Memory and Speed Comparison
Shows memory reduction and speedup for different sequence lengths
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Create figure with 2 panels
fig = plt.figure(figsize=(14, 6))
gs = GridSpec(1, 2, figure=fig, wspace=0.3)

# Sequence lengths to compare
seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
seq_labels = ['512', '1K', '2K', '4K', '8K', '16K']

# ==============================================================================
# Panel 1: Memory Usage Comparison
# ==============================================================================
ax1 = fig.add_subplot(gs[0, 0])

# Memory calculations (approximate)
# Standard attention: O(n^2) for attention matrix + O(n*d) for Q, K, V
# Flash attention: O(n*d) for Q, K, V (attention computed in blocks, not materialized)
d_model = 768  # typical hidden dimension

# Memory in GB (approximate)
# Standard: n^2 * 4 bytes (float32) for attention matrix
# Flash: Only needs block-sized memory
standard_memory = [(n ** 2 * 4) / 1e9 for n in seq_lengths]  # GB
flash_memory = [(n * d_model * 4 * 3) / 1e9 for n in seq_lengths]  # Only Q, K, V

x = np.arange(len(seq_lengths))
width = 0.35

bars1 = ax1.bar(x - width/2, standard_memory, width, label='Standard Attention',
                color='#E57373', edgecolor='#B71C1C', linewidth=2)
bars2 = ax1.bar(x + width/2, flash_memory, width, label='Flash Attention',
                color='#81C784', edgecolor='#1B5E20', linewidth=2)

ax1.set_xlabel('Sequence Length', fontsize=12)
ax1.set_ylabel('Memory Usage (GB)', fontsize=12)
ax1.set_title('Memory Usage: Standard vs Flash Attention', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(seq_labels)
ax1.legend(loc='upper left', fontsize=10)
ax1.set_yscale('log')
ax1.set_ylim(0.001, 10)
ax1.grid(True, alpha=0.3, axis='y')

# Add reduction percentages
for i, (std, flash) in enumerate(zip(standard_memory, flash_memory)):
    reduction = (1 - flash / std) * 100
    if std > 0.01:  # Only annotate visible bars
        ax1.annotate(f'{reduction:.0f}%\nreduction',
                     xy=(x[i], std), xytext=(x[i], std * 1.3),
                     ha='center', fontsize=8, color='#1B5E20', fontweight='bold')

# Add OOM line
ax1.axhline(y=8, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.text(5.5, 8.5, 'Typical GPU Memory Limit (8GB)', fontsize=9, color='red', ha='right')

# ==============================================================================
# Panel 2: Speedup Comparison
# ==============================================================================
ax2 = fig.add_subplot(gs[0, 1])

# Speedup factors (approximate based on Flash Attention paper)
# Speedup varies by sequence length - more benefit for longer sequences
speedup_factors = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # Approximate

# Time (normalized) for standard vs flash
standard_time = [1.0] * len(seq_lengths)
flash_time = [1.0 / s for s in speedup_factors]

bars3 = ax2.bar(x - width/2, standard_time, width, label='Standard Attention',
                color='#E57373', edgecolor='#B71C1C', linewidth=2)
bars4 = ax2.bar(x + width/2, flash_time, width, label='Flash Attention',
                color='#81C784', edgecolor='#1B5E20', linewidth=2)

ax2.set_xlabel('Sequence Length', fontsize=12)
ax2.set_ylabel('Relative Computation Time', fontsize=12)
ax2.set_title('Speed: Standard vs Flash Attention', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(seq_labels)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_ylim(0, 1.3)
ax2.grid(True, alpha=0.3, axis='y')

# Add speedup annotations
for i, speedup in enumerate(speedup_factors):
    ax2.annotate(f'{speedup}x\nfaster',
                 xy=(x[i] + width/2, flash_time[i]),
                 xytext=(x[i] + width/2, flash_time[i] + 0.15),
                 ha='center', fontsize=9, color='#1B5E20', fontweight='bold')

# Add explanatory text box
ax2.text(0.98, 0.98,
         'Flash Attention Key Insight:\n'
         '- Computes attention in small blocks\n'
         '- Never materializes full n x n matrix\n'
         '- Uses GPU SRAM instead of HBM\n'
         '- Same mathematical result, faster',
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Main title
fig.suptitle('Flash Attention: Memory-Efficient & Faster Attention',
             fontsize=16, fontweight='bold', y=1.02)

# Add bottom explanation
fig.text(0.5, -0.02,
         'Flash Attention (Dao et al., 2022) enables training with longer sequences on the same hardware\n'
         'by computing attention in blocks and avoiding the O(n^2) memory bottleneck.',
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

plt.tight_layout()

# Save
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, '..', 'images', 'flash_attention_memory.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")
plt.close()
