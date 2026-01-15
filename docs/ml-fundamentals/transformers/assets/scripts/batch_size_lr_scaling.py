#!/usr/bin/env python3
"""
Batch Size vs Learning Rate Scaling
Shows the relationship: lr_scaled = lr_base * sqrt(batch_new / batch_base)
Marks common configurations and includes efficiency vs stability annotations
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
BASE_BATCH = 32
BASE_LR = 1e-4
BATCH_SIZES = np.array([32, 64, 128, 256, 512, 1024, 2048, 4096])

# Calculate scaled learning rates using square root scaling
def sqrt_scaling(batch_new, batch_base, lr_base):
    return lr_base * np.sqrt(batch_new / batch_base)

def linear_scaling(batch_new, batch_base, lr_base):
    return lr_base * (batch_new / batch_base)

lr_sqrt = sqrt_scaling(BATCH_SIZES, BASE_BATCH, BASE_LR)
lr_linear = linear_scaling(BATCH_SIZES, BASE_BATCH, BASE_LR)

# Dense points for smooth curves
batch_dense = np.linspace(32, 4096, 200)
lr_sqrt_dense = sqrt_scaling(batch_dense, BASE_BATCH, BASE_LR)
lr_linear_dense = linear_scaling(batch_dense, BASE_BATCH, BASE_LR)

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Color scheme
colors = {
    'sqrt': '#1f77b4',      # Blue
    'linear': '#ff7f0e',    # Orange
    'points': '#2ca02c',    # Green
    'stability': '#90EE90',
    'efficiency': '#FFB6C1',
}

# Plot curves
ax.plot(batch_dense, lr_sqrt_dense * 1e4, color=colors['sqrt'], linewidth=2.5,
        label=r'$\sqrt{\text{scaling}}$: $lr = lr_{base} \times \sqrt{\frac{batch_{new}}{batch_{base}}}$')
ax.plot(batch_dense, lr_linear_dense * 1e4, color=colors['linear'], linewidth=2.5, linestyle='--',
        label=r'Linear scaling: $lr = lr_{base} \times \frac{batch_{new}}{batch_{base}}$')

# Mark common configurations
common_configs = {
    32: 'Single GPU\n(small model)',
    64: 'Single GPU\n(standard)',
    256: 'Multi-GPU\n(common)',
    1024: 'Large scale\n(8+ GPUs)',
}

for batch, label in common_configs.items():
    lr = sqrt_scaling(batch, BASE_BATCH, BASE_LR) * 1e4
    ax.scatter([batch], [lr], color=colors['points'], s=150, zorder=5, edgecolors='white', linewidths=2)

    # Offset for label positioning
    y_offset = 0.3 if batch < 500 else 0.5
    ax.annotate(f'{label}\nbatch={batch}\nlr={lr:.2f}e-4',
                xy=(batch, lr),
                xytext=(batch, lr + y_offset),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

# Shade regions for efficiency vs stability trade-off
# Stability region (smaller batches)
ax.axvspan(32, 256, alpha=0.15, color=colors['stability'], zorder=0)
# Efficiency region (larger batches)
ax.axvspan(256, 4096, alpha=0.15, color=colors['efficiency'], zorder=0)

# Region labels
ax.text(120, 0.2, 'STABILITY\nZONE', fontsize=11, ha='center', va='bottom',
        color='darkgreen', fontweight='bold', alpha=0.8)
ax.text(1000, 0.2, 'EFFICIENCY\nZONE', fontsize=11, ha='center', va='bottom',
        color='darkred', fontweight='bold', alpha=0.8)

# Add trade-off annotations
tradeoffs = """
Trade-offs:
- Small batches: More stable training, noisier gradients, slower throughput
- Large batches: Higher throughput, requires careful LR tuning, may generalize worse
"""

ax.text(0.98, 0.02, tradeoffs.strip(), transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', ha='right', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#fffff0', alpha=0.95))

# Why sqrt scaling annotation
why_sqrt = """
Why sqrt scaling?
- Gradient noise decreases as 1/sqrt(batch_size)
- LR should compensate proportionally
- Linear scaling often unstable for large batches
"""

ax.text(0.02, 0.98, why_sqrt.strip(), transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#e6f2ff', alpha=0.95))

# Formatting
ax.set_xlabel('Batch Size', fontsize=12)
ax.set_ylabel('Learning Rate (x10$^{-4}$)', fontsize=12)
ax.set_title('Batch Size vs Learning Rate Scaling\n(Base: batch=32, lr=1e-4)',
             fontsize=14, fontweight='bold')

ax.set_xscale('log', base=2)
ax.set_xticks(BATCH_SIZES)
ax.set_xticklabels([str(b) for b in BATCH_SIZES])

ax.set_xlim(25, 5000)
ax.set_ylim(0, 4.5)

ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3, which='both')

# Add formula prominently
formula = r'$lr_{scaled} = lr_{base} \times \sqrt{\frac{batch_{new}}{batch_{base}}}$'
ax.text(0.5, 0.93, formula, transform=ax.transAxes, fontsize=14,
        ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='blue', linewidth=2))

plt.tight_layout()

# Save
output_path = Path(__file__).parent.parent / 'images' / 'batch_size_lr_scaling.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {output_path}")
