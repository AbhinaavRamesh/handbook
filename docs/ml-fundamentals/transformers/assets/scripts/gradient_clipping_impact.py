#!/usr/bin/env python3
"""
Gradient Clipping Impact
Time series plot showing gradient norm over training:
- Unclipped gradients (with spikes)
- Clipped gradients (capped at 1.0)
- Color zones: Green (normal), Yellow (warning), Red (danger)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

# Configuration
TOTAL_STEPS = 10_000
MAX_CLIP_NORM = 1.0

# Generate synthetic gradient norm data
steps = np.arange(TOTAL_STEPS)

def generate_unclipped_gradients(steps):
    """Generate realistic unclipped gradient norms with spikes"""
    n = len(steps)

    # Base gradient norm: starts high, decays with noise
    base = 2.0 * np.exp(-steps / 3000) + 0.3 + np.random.exponential(0.2, n)

    # Add periodic instability in early training
    early_noise = np.where(steps < 3000, np.random.exponential(0.5, n), 0)

    # Add random spikes (gradient explosions)
    spike_mask = np.random.random(n) < 0.005  # 0.5% chance of spike
    spikes = np.zeros(n)
    spikes[spike_mask] = np.random.uniform(20, 150, spike_mask.sum())

    # More spikes in early training
    early_spike_mask = (steps < 2000) & (np.random.random(n) < 0.02)
    spikes[early_spike_mask] = np.random.uniform(50, 200, early_spike_mask.sum())

    return base + early_noise + spikes

# Generate unclipped gradients
grad_unclipped = generate_unclipped_gradients(steps)

# Clipped gradients
grad_clipped = np.minimum(grad_unclipped, MAX_CLIP_NORM)

# Create figure
fig, ax = plt.subplots(figsize=(14, 7))

# Color zones
zone_colors = {
    'green': '#e6ffe6',   # Normal: 0-1
    'yellow': '#fffae6',  # Warning: 1-10
    'red': '#ffe6e6'      # Danger: 10+
}

# Fill zones
ax.axhspan(0, 1, alpha=0.4, color=zone_colors['green'], label='Normal (0-1)', zorder=0)
ax.axhspan(1, 10, alpha=0.4, color=zone_colors['yellow'], label='Warning (1-10)', zorder=0)
ax.axhspan(10, 200, alpha=0.4, color=zone_colors['red'], label='Danger (10+)', zorder=0)

# Horizontal threshold lines
ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=10.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

# Plot gradient norms
ax.plot(steps, grad_unclipped, color='#ff7f0e', linewidth=0.8, alpha=0.8,
        label=f'Unclipped (max: {grad_unclipped.max():.1f})')
ax.plot(steps, grad_clipped, color='#1f77b4', linewidth=1.5, alpha=0.9,
        label=f'Clipped (max_norm={MAX_CLIP_NORM})')

# Annotations
# Find a major spike to annotate
major_spike_idx = np.argmax(grad_unclipped)
ax.annotate(f'Gradient explosion!\n({grad_unclipped[major_spike_idx]:.0f})',
            xy=(major_spike_idx, min(grad_unclipped[major_spike_idx], 180)),
            xytext=(major_spike_idx + 500, 120),
            fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            color='red', fontweight='bold')

# Clipping threshold annotation
ax.annotate(f'Clipping threshold\n(max_norm = {MAX_CLIP_NORM})',
            xy=(7000, MAX_CLIP_NORM),
            xytext=(7500, 15),
            fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            color='green')

# Add text for zone labels on right side
ax.text(TOTAL_STEPS + 100, 0.5, 'NORMAL', fontsize=9, va='center', color='darkgreen', fontweight='bold')
ax.text(TOTAL_STEPS + 100, 5, 'WARNING', fontsize=9, va='center', color='darkorange', fontweight='bold')
ax.text(TOTAL_STEPS + 100, 50, 'DANGER', fontsize=9, va='center', color='darkred', fontweight='bold')

# Statistics box
unclipped_stats = f"Unclipped Stats:\n  Mean: {grad_unclipped.mean():.2f}\n  Max: {grad_unclipped.max():.1f}\n  Spikes (>10): {(grad_unclipped > 10).sum()}"
clipped_stats = f"Clipped Stats:\n  Mean: {grad_clipped.mean():.2f}\n  Max: {grad_clipped.max():.1f}\n  All within normal range"

ax.text(0.02, 0.98, unclipped_stats, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#fff0e6', alpha=0.9))

ax.text(0.02, 0.72, clipped_stats, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#e6f0ff', alpha=0.9))

# Formatting
ax.set_xlabel('Training Steps', fontsize=12)
ax.set_ylabel('Gradient L2 Norm', fontsize=12)
ax.set_title('Gradient Clipping Impact on Training Stability\n(Typical Transformer Training)',
             fontsize=14, fontweight='bold')

ax.set_xlim(0, TOTAL_STEPS)
ax.set_ylim(0, 180)
ax.set_yscale('symlog', linthresh=1)  # Logarithmic scale for better visibility

ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()

# Save
output_path = Path(__file__).parent.parent / 'images' / 'gradient_clipping_impact.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {output_path}")
