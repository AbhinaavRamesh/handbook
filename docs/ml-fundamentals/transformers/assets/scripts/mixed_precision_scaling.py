#!/usr/bin/env python3
"""
Mixed Precision Scaling Visualization
Histogram of gradient magnitudes showing:
- Float16 underflow zone (gray)
- Valid zone (green)
- Overflow zone (red)
Compares: without scaling vs with scaling
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

# Configuration
N_GRADIENTS = 50_000

# Float16 bounds (approximate)
FP16_MIN = 6e-8      # Smallest positive subnormal
FP16_MAX = 6.5e4     # Largest finite value
LOSS_SCALE = 2**14   # Common loss scale factor (16384)

# Generate synthetic gradient magnitudes (log-normal distribution typical for deep networks)
def generate_gradient_magnitudes(n, mean_log=-6, std_log=2):
    """Generate log-normally distributed gradient magnitudes"""
    log_grads = np.random.normal(mean_log, std_log, n)
    return np.power(10.0, log_grads)

# Without loss scaling: many gradients underflow
grads_no_scale = generate_gradient_magnitudes(N_GRADIENTS, mean_log=-8, std_log=2.5)

# With loss scaling: shifted into valid range
grads_scaled = grads_no_scale * LOSS_SCALE

# Create figure with two panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bin edges in log space
log_bins = np.logspace(-12, 6, 100)

# Colors for zones
zone_colors = {
    'underflow': '#d0d0d0',   # Gray
    'valid': '#90EE90',       # Light green
    'overflow': '#ffcccc',    # Light red
}

def plot_gradient_histogram(ax, grads, title, show_legend=True):
    """Plot histogram with colored zones"""

    # Zone boundaries for shading
    ax.axvspan(1e-12, FP16_MIN, alpha=0.4, color=zone_colors['underflow'], label='Underflow zone')
    ax.axvspan(FP16_MIN, FP16_MAX, alpha=0.3, color=zone_colors['valid'], label='Valid FP16 range')
    ax.axvspan(FP16_MAX, 1e6, alpha=0.4, color=zone_colors['overflow'], label='Overflow zone')

    # Vertical lines at boundaries
    ax.axvline(x=FP16_MIN, color='gray', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(x=FP16_MAX, color='red', linestyle='--', linewidth=2, alpha=0.8)

    # Histogram
    ax.hist(grads, bins=log_bins, color='#1f77b4', alpha=0.7, edgecolor='white', linewidth=0.3)

    # Calculate statistics
    underflow_pct = (grads < FP16_MIN).mean() * 100
    valid_pct = ((grads >= FP16_MIN) & (grads <= FP16_MAX)).mean() * 100
    overflow_pct = (grads > FP16_MAX).mean() * 100

    # Statistics box
    stats_text = f'Underflow: {underflow_pct:.1f}%\nValid: {valid_pct:.1f}%\nOverflow: {overflow_pct:.1f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))

    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Gradient Magnitude', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(1e-12, 1e6)
    ax.grid(True, alpha=0.3, which='both')

    if show_legend:
        ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

    return underflow_pct, valid_pct, overflow_pct

# LEFT PANEL: Without loss scaling
stats_no_scale = plot_gradient_histogram(
    ax1, grads_no_scale,
    'WITHOUT Loss Scaling\n(Most gradients underflow to zero)',
    show_legend=True
)

# Add problem annotation
ax1.annotate('Lost gradients!\n(become zero)',
            xy=(1e-10, ax1.get_ylim()[1] * 0.5),
            fontsize=11, ha='center', fontweight='bold',
            color='gray',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax1.text(FP16_MIN * 0.3, ax1.get_ylim()[1] * 0.15, 'FP16 MIN\n(6e-8)',
         fontsize=9, ha='center', rotation=90, color='gray')

# RIGHT PANEL: With loss scaling
stats_scaled = plot_gradient_histogram(
    ax2, grads_scaled,
    f'WITH Loss Scaling (scale = {LOSS_SCALE:,})\n(Most gradients in valid range)',
    show_legend=True
)

# Add success annotation
ax2.annotate('Gradients preserved!',
            xy=(1e-2, ax2.get_ylim()[1] * 0.5),
            fontsize=11, ha='center', fontweight='bold',
            color='darkgreen',
            bbox=dict(boxstyle='round', facecolor='#e6ffe6', alpha=0.9))

ax2.text(FP16_MIN * 0.3, ax2.get_ylim()[1] * 0.15, 'FP16 MIN\n(6e-8)',
         fontsize=9, ha='center', rotation=90, color='gray')

ax2.text(FP16_MAX * 3, ax2.get_ylim()[1] * 0.15, 'FP16 MAX\n(6.5e4)',
         fontsize=9, ha='center', rotation=90, color='red')

# Formula annotation at bottom
formula_text = (
    r'Loss Scaling: $\nabla_{scaled} = \nabla \mathcal{L} \times scale$, '
    r'then unscale before weight update: $\nabla_{actual} = \nabla_{scaled} / scale$'
)
fig.text(0.5, 0.02, formula_text, ha='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

# Memory/speed benefit annotation
benefit_text = 'FP16 Benefits: 2x memory savings, 2-8x faster compute on modern GPUs (Tensor Cores)'
fig.text(0.5, -0.04, benefit_text, ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='#e6f2ff', alpha=0.9, edgecolor='blue'))

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

# Save
output_path = Path(__file__).parent.parent / 'images' / 'mixed_precision_scaling.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {output_path}")
