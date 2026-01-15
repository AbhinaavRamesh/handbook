#!/usr/bin/env python3
"""
Positional Encoding Matrix Heatmap Visualization

Creates a heatmap showing PE values for positions 0-100, dimensions 0-512.
The sinusoidal wave patterns are clearly visible in the heatmap.

Uses a diverging colormap (blue-white-red) centered at 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# Configuration
D_MODEL = 512
MAX_POSITIONS = 100


def compute_pe_matrix(max_pos, d_model):
    """
    Compute the full positional encoding matrix.

    Args:
        max_pos: Maximum position index
        d_model: Model embedding dimension

    Returns:
        PE matrix of shape (max_pos, d_model)
    """
    pe_matrix = np.zeros((max_pos, d_model))

    positions = np.arange(max_pos)[:, np.newaxis]  # Shape: (max_pos, 1)
    dims = np.arange(d_model)[np.newaxis, :]       # Shape: (1, d_model)

    # Compute the pair indices
    i_values = dims // 2

    # Compute divisors for each dimension
    divisors = np.power(10000, (2 * i_values) / d_model)

    # Compute angles
    angles = positions / divisors

    # Apply sin to even indices, cos to odd indices
    pe_matrix[:, 0::2] = np.sin(angles[:, 0::2])
    pe_matrix[:, 1::2] = np.cos(angles[:, 1::2])

    return pe_matrix


def main():
    # Compute the full PE matrix
    pe_matrix = compute_pe_matrix(MAX_POSITIONS, D_MODEL)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

    # Create diverging colormap centered at 0
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    # Create heatmap
    im = ax.imshow(pe_matrix.T, aspect='auto', cmap='RdBu_r', norm=norm,
                   origin='lower', interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('PE Value', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Labels and title
    ax.set_title('Positional Encoding Matrix Heatmap\n'
                 r'$d_{model}=512$, showing sinusoidal wave patterns',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Position (sequence index)', fontsize=12)
    ax.set_ylabel('Embedding Dimension', fontsize=12)

    # Set tick labels
    ax.set_xticks(np.arange(0, MAX_POSITIONS + 1, 10))
    ax.set_yticks(np.arange(0, D_MODEL + 1, 64))

    # Add annotation boxes explaining the pattern
    # Low dimension annotation (long wavelength)
    ax.annotate('Low dimensions:\nLong wavelength\n(slow variation)',
                xy=(80, 30), xycoords='data',
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

    # High dimension annotation (short wavelength)
    ax.annotate('High dimensions:\nShort wavelength\n(rapid variation)',
                xy=(80, 480), xycoords='data',
                fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

    # Add formula annotation
    ax.text(0.02, 0.98,
            r'$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$' + '\n' +
            r'$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Add wavelength scale reference
    fig.text(0.5, 0.02,
             'Wavelength ranges from 2\u03c0 (dim 510-511) to ~62,832 (dim 0-1) | '
             'Each column is a unique position fingerprint',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Save the figure
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'pe_matrix_heatmap.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')

    plt.close()


if __name__ == '__main__':
    main()
