#!/usr/bin/env python3
"""
GELU vs ReLU Activation Function Comparison

Side-by-side comparison of:
1. GELU (Gaussian Error Linear Unit) - used in modern transformers
2. ReLU (Rectified Linear Unit) - classical activation

Shows both the activation functions and their gradients.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Green color palette
GREEN_DARK = '#1B5E20'
GREEN_MEDIUM = '#43A047'
GREEN_LIGHT = '#81C784'
GREEN_PALE = '#C8E6C9'
GREEN_VERY_PALE = '#E8F5E9'
RED = '#D32F2F'
RED_LIGHT = '#EF5350'

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Generate x values
x = np.linspace(-4, 4, 1000)

# ===== Calculate activation functions =====
# ReLU: max(0, x)
relu = np.maximum(0, x)

# GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# ===== Calculate gradients =====
# ReLU gradient: 0 if x < 0, 1 if x > 0 (undefined at 0)
relu_grad = np.where(x > 0, 1.0, 0.0)

# GELU gradient (numerical approximation)
dx = x[1] - x[0]
gelu_grad = np.gradient(gelu, dx)

# ===== Subplot 1: ReLU Activation =====
ax1 = axes[0, 0]
ax1.plot(x, relu, color=RED, linewidth=3, label='ReLU(x) = max(0, x)')
ax1.axhline(0, color='gray', linewidth=0.5)
ax1.axvline(0, color='gray', linewidth=0.5)

# Highlight the hard cutoff
ax1.fill_between(x, relu, where=(x < 0), alpha=0.3, color=RED_LIGHT,
                  label='Zero region (info lost)')
ax1.scatter([0], [0], color=RED, s=100, zorder=5, marker='o')
ax1.annotate('Hard cutoff\nat x=0', xy=(0, 0), xytext=(1, -0.8),
             fontsize=11, color=RED,
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE', edgecolor=RED))

ax1.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Output', fontsize=12, fontweight='bold')
ax1.set_title('ReLU Activation\n(Classical, Hard Threshold)', fontsize=14, fontweight='bold', color=RED)
ax1.legend(loc='upper left', fontsize=10)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-1.5, 4)
ax1.grid(True, alpha=0.3)

# Add formula
ax1.text(0.98, 0.02, 'ReLU(x) = max(0, x)', transform=ax1.transAxes, fontsize=11,
         ha='right', va='bottom', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor=RED))

# ===== Subplot 2: GELU Activation =====
ax2 = axes[0, 1]
ax2.plot(x, gelu, color=GREEN_DARK, linewidth=3, label='GELU(x)')
ax2.axhline(0, color='gray', linewidth=0.5)
ax2.axvline(0, color='gray', linewidth=0.5)

# Highlight the smooth transition
ax2.fill_between(x, gelu, where=(x < 0), alpha=0.3, color=GREEN_LIGHT,
                  label='Smooth scaling (info preserved)')
ax2.scatter([0], [0], color=GREEN_DARK, s=100, zorder=5, marker='o')
ax2.annotate('Smooth transition\nat x=0 (value=0)', xy=(0, 0), xytext=(1.5, -0.5),
             fontsize=11, color=GREEN_DARK,
             arrowprops=dict(arrowstyle='->', color=GREEN_DARK, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.2', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK))

ax2.annotate('Negative values\nscaled (not zeroed)', xy=(-2, gelu[250]), xytext=(-3.5, 1),
             fontsize=11, color=GREEN_DARK,
             arrowprops=dict(arrowstyle='->', color=GREEN_DARK, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.2', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK))

ax2.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Output', fontsize=12, fontweight='bold')
ax2.set_title('GELU Activation\n(Modern, Smooth Threshold)', fontsize=14, fontweight='bold', color=GREEN_DARK)
ax2.legend(loc='upper left', fontsize=10)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-1.5, 4)
ax2.grid(True, alpha=0.3)

# Add formula
ax2.text(0.98, 0.02, 'GELU(x) = x * Phi(x)', transform=ax2.transAxes, fontsize=11,
         ha='right', va='bottom', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=GREEN_PALE, edgecolor=GREEN_DARK))

# ===== Subplot 3: ReLU Gradient =====
ax3 = axes[1, 0]
ax3.plot(x, relu_grad, color=RED, linewidth=3, label="ReLU'(x)")
ax3.axhline(0, color='gray', linewidth=0.5)
ax3.axvline(0, color='gray', linewidth=0.5)

# Highlight the discontinuity
ax3.scatter([0], [0], color=RED, s=100, marker='o', facecolor='white', edgecolor=RED, linewidth=2, zorder=5)
ax3.scatter([0], [1], color=RED, s=100, marker='o', facecolor='white', edgecolor=RED, linewidth=2, zorder=5)
ax3.annotate('Discontinuous!\nGradient undefined at 0', xy=(0, 0.5), xytext=(1.5, 0.3),
             fontsize=11, color=RED,
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE', edgecolor=RED))

ax3.annotate('Zero gradient\n(dead neurons)', xy=(-2, 0), xytext=(-3.5, 0.5),
             fontsize=11, color=RED,
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE', edgecolor=RED))

ax3.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Gradient', fontsize=12, fontweight='bold')
ax3.set_title('ReLU Gradient\n(Step Function)', fontsize=14, fontweight='bold', color=RED)
ax3.legend(loc='upper left', fontsize=10)
ax3.set_xlim(-4, 4)
ax3.set_ylim(-0.5, 1.5)
ax3.grid(True, alpha=0.3)

# ===== Subplot 4: GELU Gradient =====
ax4 = axes[1, 1]
ax4.plot(x, gelu_grad, color=GREEN_DARK, linewidth=3, label="GELU'(x)")
ax4.axhline(0, color='gray', linewidth=0.5)
ax4.axvline(0, color='gray', linewidth=0.5)

# Highlight the smooth gradient
ax4.scatter([0], [0.5], color=GREEN_DARK, s=100, zorder=5, marker='o')
ax4.annotate('Smooth gradient\nat x=0 (value=0.5)', xy=(0, 0.5), xytext=(1.5, 0.3),
             fontsize=11, color=GREEN_DARK,
             arrowprops=dict(arrowstyle='->', color=GREEN_DARK, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.2', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK))

ax4.annotate('Non-zero gradients\nfor negative x', xy=(-2, gelu_grad[250]), xytext=(-3.5, 0.7),
             fontsize=11, color=GREEN_DARK,
             arrowprops=dict(arrowstyle='->', color=GREEN_DARK, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.2', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK))

ax4.set_xlabel('Input (x)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Gradient', fontsize=12, fontweight='bold')
ax4.set_title('GELU Gradient\n(Smooth, Continuous)', fontsize=14, fontweight='bold', color=GREEN_DARK)
ax4.legend(loc='upper left', fontsize=10)
ax4.set_xlim(-4, 4)
ax4.set_ylim(-0.5, 1.5)
ax4.grid(True, alpha=0.3)

# ===== Add comparison table at bottom =====
comparison_text = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GELU vs ReLU Comparison                              │
├───────────────────┬────────────────────────┬────────────────────────────────┤
│ Aspect            │ ReLU                   │ GELU                           │
├───────────────────┼────────────────────────┼────────────────────────────────┤
│ Formula           │ max(0, x)              │ x * Phi(x)                     │
│ Negative values   │ Zero (info lost)       │ Scaled (info preserved)        │
│ Gradient at x=0   │ Undefined (0 or 1)     │ Smooth (0.5)                   │
│ Dead neurons      │ Yes (if x < 0 always)  │ No (gradients always flow)     │
│ Modern usage      │ CNNs, older models     │ BERT, GPT, modern LLMs         │
│ NLP performance   │ Baseline               │ +1-2% improvement              │
└───────────────────┴────────────────────────┴────────────────────────────────┘
"""

plt.suptitle('GELU vs ReLU: Activation Function Comparison', fontsize=16,
             fontweight='bold', color=GREEN_DARK, y=1.02)

plt.tight_layout()

# Save the figure
output_dir = Path(__file__).parent.parent / 'images'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'gelu_vs_relu.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved: {output_path}")

plt.close()
