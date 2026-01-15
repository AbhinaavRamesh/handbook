#!/usr/bin/env python3
"""
Relative Position Dot Product Visualization

Creates a heatmap showing dot products between positional encoding vectors
for all pairs of positions: PE(i) . PE(j).

Key insight: The dot product shows a diagonal band structure where
nearby positions have higher similarity than distant positions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# Configuration
D_MODEL = 512
MAX_POSITIONS = 50  # Smaller for clearer visualization


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

    positions = np.arange(max_pos)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

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
    # Compute PE matrix
    pe_matrix = compute_pe_matrix(MAX_POSITIONS, D_MODEL)

    # Compute pairwise dot products: (max_pos, d_model) @ (d_model, max_pos)
    dot_products = pe_matrix @ pe_matrix.T

    # Create figure with main heatmap and side plots
    fig = plt.figure(figsize=(14, 12), dpi=150)

    # Main heatmap
    ax_main = fig.add_axes([0.1, 0.3, 0.65, 0.6])

    # Use diverging colormap
    vmax = np.max(np.abs(dot_products))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax_main.imshow(dot_products, cmap='RdBu_r', norm=norm,
                        origin='lower', interpolation='nearest')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_main, shrink=0.8, pad=0.02)
    cbar.set_label(r'$PE(i) \cdot PE(j)$ (dot product)', fontsize=12)

    ax_main.set_title('Positional Encoding Dot Product Matrix\n'
                      r'$PE(i) \cdot PE(j)$ for all position pairs',
                      fontsize=14, fontweight='bold', pad=10)
    ax_main.set_xlabel('Position j', fontsize=12)
    ax_main.set_ylabel('Position i', fontsize=12)

    # Add diagonal line
    ax_main.plot([0, MAX_POSITIONS - 1], [0, MAX_POSITIONS - 1],
                 'k--', alpha=0.5, linewidth=1, label='i = j (diagonal)')

    # Annotations
    ax_main.annotate('Diagonal: i=j\n(self-similarity)',
                     xy=(25, 25), xycoords='data',
                     xytext=(35, 10), textcoords='data',
                     fontsize=10, ha='center',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                     arrowprops=dict(arrowstyle='->', color='black'))

    ax_main.annotate('High similarity\n(nearby positions)',
                     xy=(20, 25), xycoords='data',
                     xytext=(5, 40), textcoords='data',
                     fontsize=10, ha='center',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
                     arrowprops=dict(arrowstyle='->', color='blue'))

    ax_main.annotate('Lower similarity\n(distant positions)',
                     xy=(5, 45), xycoords='data',
                     xytext=(15, 48), textcoords='data',
                     fontsize=10, ha='left',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                     arrowprops=dict(arrowstyle='->', color='orange'))

    # ===== Bottom subplot: Cross-section at fixed i =====
    ax_cross = fig.add_axes([0.1, 0.08, 0.65, 0.15])

    # Show cross-sections for positions 0, 10, 25
    positions_to_show = [0, 10, 25]
    colors = ['#2196F3', '#4CAF50', '#E91E63']

    for pos, color in zip(positions_to_show, colors):
        ax_cross.plot(np.arange(MAX_POSITIONS), dot_products[pos, :],
                      color=color, linewidth=2, label=f'PE({pos}) . PE(j)')

    ax_cross.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_cross.set_xlabel('Position j', fontsize=11)
    ax_cross.set_ylabel('Dot Product', fontsize=11)
    ax_cross.set_title('Cross-sections: Dot product from fixed positions',
                       fontsize=12, pad=5)
    ax_cross.legend(loc='upper right', fontsize=9)
    ax_cross.grid(True, alpha=0.3)
    ax_cross.set_xlim(0, MAX_POSITIONS - 1)

    # ===== Right subplot: Diagonal profile =====
    ax_diag = fig.add_axes([0.8, 0.3, 0.15, 0.6])

    # Compute diagonal values (should be constant for normalized PE)
    diagonal = np.diag(dot_products)

    ax_diag.barh(np.arange(MAX_POSITIONS), diagonal, color='purple', alpha=0.7)
    ax_diag.axvline(x=D_MODEL / 2, color='red', linestyle='--',
                    label=f'd_model/2 = {D_MODEL // 2}')
    ax_diag.set_xlabel('Self dot product\nPE(i) . PE(i)', fontsize=10)
    ax_diag.set_ylabel('Position i', fontsize=10)
    ax_diag.set_title('Diagonal\n(self-similarity)', fontsize=11, pad=5)
    ax_diag.legend(fontsize=8)
    ax_diag.set_ylim(0, MAX_POSITIONS - 1)

    # Add explanatory text
    fig.text(0.5, 0.01,
             'Key Insight: The diagonal band structure shows that nearby positions have '
             'similar embeddings (higher dot product),\n'
             'while distant positions have lower similarity. '
             'This enables the model to learn relative position relationships.',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.9))

    # Save the figure
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'relative_position_dotproduct.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')

    plt.close()


if __name__ == '__main__':
    main()
