#!/usr/bin/env python3
"""
Gradient Flow Through Transformer Layers

Visualizes how gradient magnitude changes through layers:
- With residual connections: gradient remains relatively flat
- Without residuals: exponential decay (vanishing gradients)

This demonstrates why residual connections are critical for deep transformers.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set up the figure with green color theme
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

# Layer configuration (BERT-base has 12 layers)
layers = np.arange(1, 13)

# Simulate gradient magnitudes
# With residuals: gradient path includes identity, so gradient stays ~1
# y = f(x) + x => dy/dx = df/dx + I
# The identity component preserves gradient magnitude
np.random.seed(42)
with_residuals = 1.0 * np.exp(-0.02 * (layers - 1)) + np.random.normal(0, 0.05, len(layers))
with_residuals = np.clip(with_residuals, 0.7, 1.3)  # Small variation around 1.0

# Without residuals: gradient decays exponentially
# Each layer multiplies gradient by factor < 1 (typically 0.5-0.8)
decay_factor = 0.65  # Typical Jacobian spectral norm
without_residuals = np.power(decay_factor, layers - 1)

# Green color palette
GREEN_DARK = '#1B5E20'
GREEN_MEDIUM = '#43A047'
GREEN_LIGHT = '#81C784'
GREEN_PALE = '#C8E6C9'

# Plot with residuals (flat line)
ax.semilogy(layers, with_residuals,
            marker='o', markersize=10, linewidth=3,
            color=GREEN_DARK, markerfacecolor=GREEN_MEDIUM,
            markeredgecolor=GREEN_DARK, markeredgewidth=2,
            label='With Residual Connections (stable)')

# Plot without residuals (exponential decay)
ax.semilogy(layers, without_residuals,
            marker='s', markersize=10, linewidth=3,
            color='#D32F2F', markerfacecolor='#EF5350',
            markeredgecolor='#D32F2F', markeredgewidth=2,
            label='Without Residual Connections (vanishing)', linestyle='--')

# Add annotations
ax.annotate('Gradient magnitude\nstays ~1.0',
            xy=(8, with_residuals[7]), xytext=(9.5, 0.5),
            fontsize=11, color=GREEN_DARK,
            arrowprops=dict(arrowstyle='->', color=GREEN_DARK, lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor=GREEN_PALE, edgecolor=GREEN_DARK))

ax.annotate(f'Gradient decays to\n{without_residuals[-1]:.2e} at layer 12',
            xy=(12, without_residuals[11]), xytext=(9, 1e-4),
            fontsize=11, color='#D32F2F',
            arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#D32F2F'))

# Add formula annotations
formula_text = (
    "Residual connection: y = f(x) + x\n"
    "Gradient: dy/dx = df/dx + I\n"
    "Identity I ensures gradient >= 1"
)
ax.text(0.02, 0.02, formula_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=GREEN_PALE, edgecolor=GREEN_DARK, alpha=0.9))

# Styling
ax.set_xlabel('Layer Number (from input to output)', fontsize=14, fontweight='bold')
ax.set_ylabel('Gradient Magnitude (log scale)', fontsize=14, fontweight='bold')
ax.set_title('Gradient Flow Through Transformer Layers\n(BERT-base: 12 layers)',
             fontsize=16, fontweight='bold', color=GREEN_DARK)

ax.set_xlim(0.5, 12.5)
ax.set_ylim(1e-5, 5)
ax.set_xticks(layers)
ax.legend(loc='upper right', fontsize=12, framealpha=0.95)

# Add horizontal reference line at 1.0
ax.axhline(y=1.0, color=GREEN_LIGHT, linestyle=':', linewidth=2, alpha=0.7)
ax.text(1, 1.2, 'Initial gradient magnitude', fontsize=10, color=GREEN_MEDIUM)

# Add shaded region for "trainable" zone
ax.axhspan(0.1, 10, alpha=0.1, color=GREEN_MEDIUM)
ax.text(6.5, 2, 'Trainable gradient range', fontsize=11, color=GREEN_DARK,
        ha='center', style='italic')

# Add danger zone annotation
ax.axhspan(1e-5, 1e-3, alpha=0.1, color='#D32F2F')
ax.text(6.5, 3e-4, 'Vanishing gradient zone', fontsize=11, color='#D32F2F',
        ha='center', style='italic')

plt.tight_layout()

# Save the figure
output_dir = Path(__file__).parent.parent / 'images'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'gradient_flow_depth.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved: {output_path}")

plt.close()
