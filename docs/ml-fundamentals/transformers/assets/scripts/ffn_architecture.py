#!/usr/bin/env python3
"""
Feed-Forward Network (FFN) Architecture Visualization

Shows the FFN structure in transformer encoders:
- Expansion from d_model (768) to d_ff (3072) - 4x expansion
- GELU activation
- Contraction back to d_model (768)

BERT-base dimensions used as reference.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
from pathlib import Path

# Green color palette
GREEN_DARK = '#1B5E20'
GREEN_MEDIUM = '#43A047'
GREEN_LIGHT = '#81C784'
GREEN_PALE = '#C8E6C9'
GREEN_VERY_PALE = '#E8F5E9'

# Create figure with two subplots
fig = plt.figure(figsize=(16, 10))

# ===== Subplot 1: Block Diagram =====
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_xlim(0, 16)
ax1.set_ylim(0, 8)
ax1.axis('off')

# Dimensions (BERT-base)
d_model = 768
d_ff = 3072  # 4x expansion

# Draw the blocks with height proportional to dimension
max_height = 6
input_height = max_height * (d_model / d_ff)
expanded_height = max_height
output_height = input_height

# Block positions
blocks = [
    (1.5, 'Input\n(d_model)', d_model, input_height, GREEN_LIGHT),
    (5, 'Linear W₁\n768→3072', None, None, None),  # Arrow label
    (7, 'Expanded\n(d_ff)', d_ff, expanded_height, GREEN_MEDIUM),
    (10, 'GELU\nActivation', None, None, None),  # Arrow label
    (12, 'Activated\n(d_ff)', d_ff, expanded_height, GREEN_DARK),
    (15, 'Linear W₂\n3072→768', None, None, None),  # Arrow label
]

# Draw blocks
y_center = 4
block_width = 1.5

# Input block
ax1.add_patch(FancyBboxPatch((0.5, y_center - input_height/2), block_width, input_height,
                              boxstyle="round,pad=0.05", facecolor=GREEN_LIGHT,
                              edgecolor=GREEN_DARK, linewidth=2))
ax1.text(1.25, y_center, f'Input\n({d_model})', ha='center', va='center',
         fontsize=11, fontweight='bold', color=GREEN_DARK)

# Expanded block
ax1.add_patch(FancyBboxPatch((5.5, y_center - expanded_height/2), block_width * 1.5, expanded_height,
                              boxstyle="round,pad=0.05", facecolor=GREEN_MEDIUM,
                              edgecolor=GREEN_DARK, linewidth=2))
ax1.text(6.625, y_center, f'Expanded\n({d_ff})', ha='center', va='center',
         fontsize=11, fontweight='bold', color='white')

# Activated block
ax1.add_patch(FancyBboxPatch((10.5, y_center - expanded_height/2), block_width * 1.5, expanded_height,
                              boxstyle="round,pad=0.05", facecolor=GREEN_DARK,
                              edgecolor='#0D3311', linewidth=2))
ax1.text(11.625, y_center, f'After GELU\n({d_ff})', ha='center', va='center',
         fontsize=11, fontweight='bold', color='white')

# Output block
ax1.add_patch(FancyBboxPatch((14, y_center - output_height/2), block_width, output_height,
                              boxstyle="round,pad=0.05", facecolor=GREEN_LIGHT,
                              edgecolor=GREEN_DARK, linewidth=2))
ax1.text(14.75, y_center, f'Output\n({d_model})', ha='center', va='center',
         fontsize=11, fontweight='bold', color=GREEN_DARK)

# Draw arrows with labels
arrow_style = dict(arrowstyle='->', color=GREEN_DARK, lw=2.5,
                   connectionstyle='arc3,rad=0')

# Arrow 1: Input to Expanded
ax1.annotate('', xy=(5.5, y_center), xytext=(2, y_center),
             arrowprops=arrow_style)
ax1.text(3.75, y_center + 0.8, 'Linear W₁\n(768 → 3072)', ha='center', va='bottom',
         fontsize=10, fontweight='bold', color=GREEN_DARK,
         bbox=dict(boxstyle='round,pad=0.2', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_MEDIUM))

# Arrow 2: Expanded to Activated (GELU)
ax1.annotate('', xy=(10.5, y_center), xytext=(7.25, y_center),
             arrowprops=arrow_style)
ax1.text(8.875, y_center + 0.8, 'GELU\nActivation', ha='center', va='bottom',
         fontsize=10, fontweight='bold', color=GREEN_DARK,
         bbox=dict(boxstyle='round,pad=0.2', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_MEDIUM))

# Arrow 3: Activated to Output
ax1.annotate('', xy=(14, y_center), xytext=(12.25, y_center),
             arrowprops=arrow_style)
ax1.text(13.125, y_center + 0.8, 'Linear W₂\n(3072 → 768)', ha='center', va='bottom',
         fontsize=10, fontweight='bold', color=GREEN_DARK,
         bbox=dict(boxstyle='round,pad=0.2', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_MEDIUM))

# Add title and formula
ax1.text(8, 7.5, 'FFN Architecture: Expansion → Activation → Contraction',
         ha='center', fontsize=14, fontweight='bold', color=GREEN_DARK)
ax1.text(8, 0.5, r'FFN(x) = GELU(xW₁ + b₁)W₂ + b₂    where W₁ ∈ ℝ^{768×3072}, W₂ ∈ ℝ^{3072×768}',
         ha='center', fontsize=11, fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=GREEN_PALE, edgecolor=GREEN_DARK))

# Add 4x expansion annotation
ax1.annotate('4× expansion', xy=(6, y_center - expanded_height/2 - 0.3),
             xytext=(6, 0.3), fontsize=10, fontweight='bold', color=GREEN_DARK,
             arrowprops=dict(arrowstyle='->', color=GREEN_DARK, lw=1.5),
             ha='center')

# ===== Subplot 2: GELU Activation Shape =====
ax2 = fig.add_subplot(2, 1, 2)

# Generate GELU curve
x = np.linspace(-4, 4, 1000)

# GELU approximation
gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# Also show ReLU for comparison
relu = np.maximum(0, x)

# Plot
ax2.plot(x, gelu, color=GREEN_DARK, linewidth=3, label='GELU (used in BERT/GPT)')
ax2.plot(x, relu, color='#D32F2F', linewidth=2, linestyle='--', alpha=0.7, label='ReLU (for comparison)')
ax2.axhline(0, color='gray', linewidth=0.5)
ax2.axvline(0, color='gray', linewidth=0.5)

# Fill the area where GELU differs from zero
ax2.fill_between(x, gelu, where=(x < 0), alpha=0.3, color=GREEN_MEDIUM,
                  label='Smooth transition (vs hard cutoff)')

ax2.set_xlabel('Input Value (x)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Output Value', fontsize=12, fontweight='bold')
ax2.set_title('GELU Activation Function', fontsize=14, fontweight='bold', color=GREEN_DARK)
ax2.legend(loc='upper left', fontsize=10)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-1, 4)
ax2.grid(True, alpha=0.3)

# Add annotations
ax2.annotate('Smooth curve\nat x=0', xy=(0, 0), xytext=(1.5, -0.5),
             fontsize=10, color=GREEN_DARK,
             arrowprops=dict(arrowstyle='->', color=GREEN_DARK),
             bbox=dict(boxstyle='round,pad=0.2', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK))

ax2.annotate('Preserves negative info\n(scaled, not zeroed)', xy=(-2, gelu[250]), xytext=(-3.5, 1),
             fontsize=10, color=GREEN_DARK,
             arrowprops=dict(arrowstyle='->', color=GREEN_DARK),
             bbox=dict(boxstyle='round,pad=0.2', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK))

# Add formula
formula = r'GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))'
ax2.text(0.98, 0.02, formula, transform=ax2.transAxes, fontsize=10,
         ha='right', va='bottom', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.3', facecolor=GREEN_PALE, edgecolor=GREEN_DARK))

plt.tight_layout()

# Save the figure
output_dir = Path(__file__).parent.parent / 'images'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'ffn_architecture.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved: {output_path}")

plt.close()
