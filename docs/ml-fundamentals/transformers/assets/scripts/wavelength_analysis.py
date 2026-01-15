#!/usr/bin/env python3
"""
Wavelength vs Dimension Analysis for Positional Encoding

Shows how wavelength grows exponentially with dimension index.
Uses log-scale to demonstrate the exponential relationship.

Key insight: The 10000 base factor creates wavelengths ranging from
2*pi to 2*pi*10000 across the embedding dimensions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
D_MODEL = 512


def compute_wavelength(dim_pair_index, d_model):
    """
    Compute the wavelength for a given dimension pair index.

    Wavelength = 2 * pi * 10000^(2i/d_model)

    Args:
        dim_pair_index: The pair index i (0, 1, 2, ...)
        d_model: Model embedding dimension

    Returns:
        Wavelength value
    """
    return 2 * np.pi * np.power(10000, (2 * dim_pair_index) / d_model)


def main():
    # Compute wavelengths for all dimension pairs
    num_pairs = D_MODEL // 2
    pair_indices = np.arange(num_pairs)
    wavelengths = compute_wavelength(pair_indices, D_MODEL)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

    # ===== Left plot: Log scale =====
    ax1 = axes[0]

    ax1.semilogy(pair_indices * 2, wavelengths, 'b-', linewidth=2.5,
                 label='Wavelength', marker='o', markersize=3, markevery=10)

    ax1.set_title('Wavelength vs Dimension Index (Log Scale)',
                  fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Dimension Index (2i)', fontsize=12)
    ax1.set_ylabel('Wavelength (log scale)', fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')

    # Mark key points
    key_dims = [0, 128, 256, 384, 510]
    for dim in key_dims:
        i = dim // 2
        wl = compute_wavelength(i, D_MODEL)
        ax1.scatter([dim], [wl], s=100, c='red', zorder=5)
        if dim == 0:
            ax1.annotate(f'dim={dim}\n\u03bb={wl:.1f}\n(2\u03c0)',
                         xy=(dim, wl), xytext=(dim + 30, wl * 1.5),
                         fontsize=9, ha='left',
                         arrowprops=dict(arrowstyle='->', color='gray'))
        elif dim == 510:
            ax1.annotate(f'dim={dim}\n\u03bb={wl:.1f}\n(2\u03c0\u00d710000)',
                         xy=(dim, wl), xytext=(dim - 100, wl * 0.1),
                         fontsize=9, ha='left',
                         arrowprops=dict(arrowstyle='->', color='gray'))
        else:
            ax1.annotate(f'dim={dim}\n\u03bb={wl:.1f}',
                         xy=(dim, wl), xytext=(dim + 20, wl * 2),
                         fontsize=9, ha='left',
                         arrowprops=dict(arrowstyle='->', color='gray'))

    # Add the formula
    ax1.annotate(r'$\lambda(i) = 2\pi \cdot 10000^{2i/d_{model}}$',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=13, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # Mark the 10000 base factor
    ax1.axhline(y=2 * np.pi * 10000, color='green', linestyle='--',
                alpha=0.7, linewidth=2, label='Max wavelength (2\u03c0\u00d710000)')
    ax1.axhline(y=2 * np.pi, color='orange', linestyle='--',
                alpha=0.7, linewidth=2, label='Min wavelength (2\u03c0)')
    ax1.legend(loc='lower right', fontsize=10)

    # ===== Right plot: Frequency interpretation =====
    ax2 = axes[1]

    # Compute frequencies (inverse of wavelength)
    frequencies = 1 / wavelengths

    ax2.semilogy(pair_indices * 2, frequencies, 'purple', linewidth=2.5,
                 marker='s', markersize=3, markevery=10)

    ax2.set_title('Frequency vs Dimension Index (Log Scale)',
                  fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Dimension Index (2i)', fontsize=12)
    ax2.set_ylabel('Frequency = 1/Wavelength (log scale)', fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')

    # Add interpretation boxes
    ax2.annotate('Low dimensions:\nLow frequency\nCaptures GLOBAL position',
                 xy=(50, frequencies[25]), xycoords='data',
                 fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    ax2.annotate('High dimensions:\nHigh frequency\nCaptures FINE details',
                 xy=(450, frequencies[225]), xycoords='data',
                 fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

    # Add explanation
    ax2.annotate(r'$f(i) = \frac{1}{2\pi \cdot 10000^{2i/d_{model}}}$',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=13, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # Add overall title and explanation
    fig.suptitle('Understanding the 10000 Base Factor in Positional Encoding',
                 fontsize=16, fontweight='bold', y=1.02)

    fig.text(0.5, -0.02,
             'The base factor 10000 creates a geometric progression of wavelengths, '
             'allowing the model to capture position information at multiple scales simultaneously.\n'
             'This is analogous to Fourier decomposition: different frequencies encode different spatial patterns.',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    # Save the figure
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'wavelength_analysis.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')

    plt.close()


if __name__ == '__main__':
    main()
