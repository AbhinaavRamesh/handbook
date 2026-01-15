#!/usr/bin/env python3
"""
Sinusoidal Positional Encoding Waves Visualization

Demonstrates how different dimensions have different frequencies in sinusoidal PE.
Shows waves for dimensions 0, 1, 64, 128, 255 (for d_model=512).

Formula:
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
D_MODEL = 512
POSITIONS = np.arange(0, 100, 0.1)  # 0 to 100 with fine resolution
DIMENSIONS = [0, 1, 64, 128, 255]  # Dimensions to visualize

# Color palette for consistent styling
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']


def compute_pe(pos, dim, d_model=512):
    """
    Compute positional encoding for given position and dimension.

    Args:
        pos: Position index (can be array)
        dim: Dimension index
        d_model: Model embedding dimension

    Returns:
        PE value(s) for the given position(s) and dimension
    """
    # Get the pair index (each sin/cos pair shares the same frequency)
    i = dim // 2

    # Compute the divisor: 10000^(2i/d_model)
    divisor = np.power(10000, (2 * i) / d_model)

    # Compute angle
    angle = pos / divisor

    # Apply sin for even dimensions, cos for odd
    if dim % 2 == 0:
        return np.sin(angle)
    else:
        return np.cos(angle)


def main():
    # Create figure with two subplots: sin waves and cos waves
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=150)

    # Plot settings
    fig.suptitle('Sinusoidal Positional Encoding: Multiple Frequency Bands',
                 fontsize=16, fontweight='bold', y=0.98)

    # ===== Top subplot: Sin waves (even dimensions) =====
    ax1 = axes[0]
    ax1.set_title('Sine Components (Even Dimensions)', fontsize=14, pad=10)

    for idx, dim in enumerate(DIMENSIONS):
        # Use even dimension for sin
        even_dim = (dim // 2) * 2
        pe_values = compute_pe(POSITIONS, even_dim, D_MODEL)

        # Compute wavelength for annotation
        i = even_dim // 2
        wavelength = 2 * np.pi * np.power(10000, (2 * i) / D_MODEL)

        label = f'dim={even_dim} (wavelength={wavelength:.1f})'
        ax1.plot(POSITIONS, pe_values, color=COLORS[idx], linewidth=2,
                 label=label, alpha=0.8)

    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('PE Value', fontsize=12)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-1.2, 1.2)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Add formula annotation
    ax1.annotate(r'$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$',
                 xy=(0.02, 0.95), xycoords='axes fraction',
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ===== Bottom subplot: Cos waves (odd dimensions) =====
    ax2 = axes[1]
    ax2.set_title('Cosine Components (Odd Dimensions)', fontsize=14, pad=10)

    for idx, dim in enumerate(DIMENSIONS):
        # Use odd dimension for cos
        odd_dim = (dim // 2) * 2 + 1
        pe_values = compute_pe(POSITIONS, odd_dim, D_MODEL)

        # Compute wavelength for annotation
        i = odd_dim // 2
        wavelength = 2 * np.pi * np.power(10000, (2 * i) / D_MODEL)

        label = f'dim={odd_dim} (wavelength={wavelength:.1f})'
        ax2.plot(POSITIONS, pe_values, color=COLORS[idx], linewidth=2,
                 label=label, alpha=0.8, linestyle='--')

    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('PE Value', fontsize=12)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-1.2, 1.2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add formula annotation
    ax2.annotate(r'$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$',
                 xy=(0.02, 0.95), xycoords='axes fraction',
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add explanatory text
    fig.text(0.5, 0.02,
             'Key Insight: Low dimensions have long wavelengths (capture global position), '
             'high dimensions have short wavelengths (capture fine details)',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    # Save the figure
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'sinusoidal_pe_waves.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')

    plt.close()


if __name__ == '__main__':
    main()
