#!/usr/bin/env python3
"""
Layer Normalization Visualization

Shows the LayerNorm process:
1. Before: Raw activations with varying mean and variance
2. After normalization: Zero mean, unit variance
3. After learnable affine transform (gamma, beta): Scaled and shifted

Formula: y = gamma * (x - mu) / sqrt(sigma^2 + epsilon) + beta
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set up the figure
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Green color palette
GREEN_DARK = '#1B5E20'
GREEN_MEDIUM = '#43A047'
GREEN_LIGHT = '#81C784'
GREEN_PALE = '#C8E6C9'
GREEN_VERY_PALE = '#E8F5E9'

# Simulate activations for one token (embedding dimension = 768)
np.random.seed(42)
d_model = 768

# Original activations (not normalized - varying mean and std)
original_mean = 2.5
original_std = 1.8
x_original = np.random.normal(original_mean, original_std, d_model)

# After normalization (zero mean, unit variance)
mu = np.mean(x_original)
sigma = np.std(x_original)
epsilon = 1e-6
x_normalized = (x_original - mu) / np.sqrt(sigma**2 + epsilon)

# Learnable parameters
gamma = 1.2  # Scale factor
beta = 0.5   # Shift factor
x_final = gamma * x_normalized + beta

# ===== Plot 1: Original Activations Distribution =====
ax1 = axes[0, 0]
n, bins, patches = ax1.hist(x_original, bins=50, density=True,
                             color=GREEN_LIGHT, edgecolor=GREEN_DARK, alpha=0.8)
ax1.axvline(mu, color='#D32F2F', linewidth=2.5, linestyle='--', label=f'Mean = {mu:.2f}')
ax1.axvline(mu + sigma, color='#FF9800', linewidth=2, linestyle=':', label=f'Std = {sigma:.2f}')
ax1.axvline(mu - sigma, color='#FF9800', linewidth=2, linestyle=':')

ax1.set_xlabel('Activation Value', fontsize=12, fontweight='bold')
ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
ax1.set_title('Step 1: Original Activations\n(Before LayerNorm)', fontsize=14, fontweight='bold', color=GREEN_DARK)
ax1.legend(loc='upper right', fontsize=10)
ax1.text(0.02, 0.98, f'Shape: ({d_model},)\n(embedding dim)',
         transform=ax1.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK))

# ===== Plot 2: Normalized Activations =====
ax2 = axes[0, 1]
ax2.hist(x_normalized, bins=50, density=True,
         color=GREEN_MEDIUM, edgecolor=GREEN_DARK, alpha=0.8)
ax2.axvline(0, color='#D32F2F', linewidth=2.5, linestyle='--', label='Mean = 0')
ax2.axvline(1, color='#FF9800', linewidth=2, linestyle=':', label='Std = 1')
ax2.axvline(-1, color='#FF9800', linewidth=2, linestyle=':')

ax2.set_xlabel('Activation Value', fontsize=12, fontweight='bold')
ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
ax2.set_title('Step 2: After Normalization\n(Zero Mean, Unit Variance)', fontsize=14, fontweight='bold', color=GREEN_DARK)
ax2.legend(loc='upper right', fontsize=10)

# Add formula
formula = r'$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$'
ax2.text(0.5, 0.98, formula, transform=ax2.transAxes, fontsize=14, va='top', ha='center',
         bbox=dict(boxstyle='round', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK))

# ===== Plot 3: After Learnable Affine Transform =====
ax3 = axes[1, 0]
ax3.hist(x_final, bins=50, density=True,
         color=GREEN_DARK, edgecolor='white', alpha=0.9)
final_mean = np.mean(x_final)
final_std = np.std(x_final)
ax3.axvline(final_mean, color='#D32F2F', linewidth=2.5, linestyle='--', label=f'Mean = {final_mean:.2f}')
ax3.axvline(final_mean + final_std, color='#FF9800', linewidth=2, linestyle=':', label=f'Std = {final_std:.2f}')
ax3.axvline(final_mean - final_std, color='#FF9800', linewidth=2, linestyle=':')

ax3.set_xlabel('Activation Value', fontsize=12, fontweight='bold')
ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
ax3.set_title(f'Step 3: After Affine Transform\n(gamma={gamma}, beta={beta})', fontsize=14, fontweight='bold', color=GREEN_DARK)
ax3.legend(loc='upper right', fontsize=10)

formula2 = r'$y = \gamma \cdot \hat{x} + \beta$'
ax3.text(0.5, 0.98, formula2, transform=ax3.transAxes, fontsize=14, va='top', ha='center',
         bbox=dict(boxstyle='round', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK))

# ===== Plot 4: Process Overview Diagram =====
ax4 = axes[1, 1]
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')

# Draw the process flow
boxes = [
    (1, 7, 'Input x\n(raw activations)', GREEN_PALE),
    (4, 7, 'Compute\nμ, σ²', GREEN_LIGHT),
    (7, 7, 'Normalize\n(x-μ)/√(σ²+ε)', GREEN_MEDIUM),
    (4, 3, 'Scale by γ\n(learnable)', GREEN_MEDIUM),
    (7, 3, 'Shift by β\n(learnable)', GREEN_DARK),
]

for x, y, text, color in boxes:
    text_color = 'white' if color in [GREEN_DARK, GREEN_MEDIUM] else GREEN_DARK
    ax4.add_patch(plt.Rectangle((x-0.8, y-0.6), 1.6, 1.2,
                                 facecolor=color, edgecolor=GREEN_DARK, linewidth=2))
    ax4.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', color=text_color)

# Draw arrows
arrow_props = dict(arrowstyle='->', color=GREEN_DARK, lw=2)
ax4.annotate('', xy=(3.2, 7), xytext=(1.8, 7), arrowprops=arrow_props)
ax4.annotate('', xy=(6.2, 7), xytext=(4.8, 7), arrowprops=arrow_props)
ax4.annotate('', xy=(7, 6.4), xytext=(7, 5), arrowprops=arrow_props)
ax4.annotate('', xy=(7, 3.6), xytext=(7, 4.4), arrowprops=arrow_props)
ax4.annotate('', xy=(5.8, 3), xytext=(4.8, 3), arrowprops=arrow_props)

# Add title and key insight
ax4.text(5, 9.5, 'LayerNorm Process Flow', fontsize=14, fontweight='bold',
         ha='center', color=GREEN_DARK)

ax4.text(5, 1,
         'Key: LayerNorm normalizes across the embedding dimension (d_model=768),\n'
         'making each token independent of batch size. γ and β are learnable\n'
         'parameters that allow the model to learn optimal scale and shift.',
         fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor=GREEN_VERY_PALE, edgecolor=GREEN_DARK))

# Add output label
ax4.text(9, 3, 'Output y', fontsize=11, fontweight='bold', color=GREEN_DARK,
         bbox=dict(boxstyle='round', facecolor=GREEN_PALE, edgecolor=GREEN_DARK))
ax4.annotate('', xy=(8.3, 3), xytext=(7.8, 3), arrowprops=arrow_props)

plt.suptitle('Layer Normalization in Transformer Encoders', fontsize=16, fontweight='bold',
             color=GREEN_DARK, y=1.02)
plt.tight_layout()

# Save the figure
output_dir = Path(__file__).parent.parent / 'images'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'layernorm_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved: {output_path}")

plt.close()
