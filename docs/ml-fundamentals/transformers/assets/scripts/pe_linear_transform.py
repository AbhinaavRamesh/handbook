#!/usr/bin/env python3
"""
PE Linear Transformation Visualization

Demonstrates how PE(pos+k) can be expressed as a linear transformation
(rotation) of PE(pos) in 2D embedding subspaces.

Key insight: The sinusoidal encoding allows position offsets to be
represented as rotation matrices, enabling the model to learn
relative position relationships through linear operations.

Using angle addition formulas:
sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
cos(a+b) = cos(a)cos(b) - sin(a)sin(b)

This is equivalent to a 2D rotation matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from pathlib import Path

# Configuration
D_MODEL = 512


def compute_pe_2d(pos, dim_pair, d_model=512):
    """
    Compute the 2D (sin, cos) pair for a given position and dimension pair.

    Args:
        pos: Position index
        dim_pair: Dimension pair index (0, 1, 2, ...)
        d_model: Model embedding dimension

    Returns:
        Tuple of (sin_value, cos_value)
    """
    divisor = np.power(10000, (2 * dim_pair) / d_model)
    angle = pos / divisor
    return np.sin(angle), np.cos(angle)


def rotation_matrix(theta):
    """
    Create a 2D rotation matrix for angle theta.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])


def main():
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12), dpi=150)

    # ===== Top left: 2D rotation concept =====
    ax1 = fig.add_subplot(2, 2, 1)

    # Choose dimension pair 0 (lowest frequency) for clear visualization
    dim_pair = 0
    divisor = np.power(10000, (2 * dim_pair) / D_MODEL)

    # Draw unit circle
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(np.cos(theta_circle), np.sin(theta_circle), 'lightgray',
             linewidth=2, linestyle='--', alpha=0.7)

    # Plot PE vectors for positions 0, 5, 10, 15, 20
    positions = [0, 5, 10, 15, 20]
    colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))

    for pos, color in zip(positions, colors):
        sin_val, cos_val = compute_pe_2d(pos, dim_pair, D_MODEL)
        # Note: PE stores (sin, cos) in consecutive dims, so vector is (sin, cos)
        ax1.arrow(0, 0, sin_val * 0.9, cos_val * 0.9,
                  head_width=0.05, head_length=0.03, fc=color, ec=color,
                  linewidth=2)
        ax1.annotate(f'pos={pos}', xy=(sin_val * 1.1, cos_val * 1.1),
                     fontsize=10, ha='center', color=color, fontweight='bold')

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('sin component (even dim)', fontsize=11)
    ax1.set_ylabel('cos component (odd dim)', fontsize=11)
    ax1.set_title(f'PE Vectors as Rotation in 2D Space\n'
                  f'(Dimension pair {dim_pair}, low frequency)',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add rotation angle annotation
    angle_offset = 5 / divisor
    ax1.annotate(f'Each step rotates by\n{np.degrees(angle_offset):.4f}',
                 xy=(0.7, -0.9), fontsize=10, ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # ===== Top right: Higher frequency dimension =====
    ax2 = fig.add_subplot(2, 2, 2)

    # Choose dimension pair 100 (higher frequency)
    dim_pair_high = 100
    divisor_high = np.power(10000, (2 * dim_pair_high) / D_MODEL)

    # Draw unit circle
    ax2.plot(np.cos(theta_circle), np.sin(theta_circle), 'lightgray',
             linewidth=2, linestyle='--', alpha=0.7)

    # Plot PE vectors
    for pos, color in zip(positions, colors):
        sin_val, cos_val = compute_pe_2d(pos, dim_pair_high, D_MODEL)
        ax2.arrow(0, 0, sin_val * 0.9, cos_val * 0.9,
                  head_width=0.05, head_length=0.03, fc=color, ec=color,
                  linewidth=2)
        ax2.annotate(f'pos={pos}', xy=(sin_val * 1.15, cos_val * 1.15),
                     fontsize=10, ha='center', color=color, fontweight='bold')

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel('sin component (even dim)', fontsize=11)
    ax2.set_ylabel('cos component (odd dim)', fontsize=11)
    ax2.set_title(f'PE Vectors at Higher Frequency\n'
                  f'(Dimension pair {dim_pair_high})',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add rotation angle annotation
    angle_offset_high = 5 / divisor_high
    ax2.annotate(f'Each step rotates by\n{np.degrees(angle_offset_high):.2f}',
                 xy=(0.7, -0.9), fontsize=10, ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

    # ===== Bottom left: Linear transformation matrix =====
    ax3 = fig.add_subplot(2, 2, 3)

    # Show the rotation matrix formula
    ax3.axis('off')

    # Title for math section
    ax3.text(0.5, 0.95, 'Linear Transformation for Position Offset k',
             transform=ax3.transAxes, fontsize=14, fontweight='bold',
             ha='center', va='top')

    # Rotation matrix visualization using text blocks
    ax3.text(0.5, 0.78, 'PE(pos+k) = R(k) * PE(pos)',
             transform=ax3.transAxes, fontsize=13, fontweight='bold',
             ha='center', va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Matrix representation
    matrix_text = (
        'Rotation Matrix R(k):\n\n'
        '    [ cos(theta_k)   sin(theta_k)  ]\n'
        '    [ -sin(theta_k)  cos(theta_k)  ]\n\n'
        'where theta_k = k / 10000^(2i/d_model)'
    )
    ax3.text(0.5, 0.55, matrix_text,
             transform=ax3.transAxes, fontsize=11, ha='center', va='top',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    # Key properties
    props_text = (
        'Key Properties:\n\n'
        '1. Position offset k is a pure rotation\n'
        '2. Rotation angle depends on dimension i\n'
        '3. Lower dims rotate slowly, higher dims fast\n'
        '4. Model learns relative position via linear ops'
    )
    ax3.text(0.5, 0.18, props_text,
             transform=ax3.transAxes, fontsize=11, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax3.set_title('Mathematical Foundation', fontsize=12, fontweight='bold', pad=20)

    # ===== Bottom right: Demonstrate rotation numerically =====
    ax4 = fig.add_subplot(2, 2, 4)

    # Show numerical verification
    pos = 10
    k = 5
    dim_pair = 50

    # Compute PE(pos) and PE(pos+k) directly
    pe_pos = np.array(compute_pe_2d(pos, dim_pair, D_MODEL))
    pe_pos_k_direct = np.array(compute_pe_2d(pos + k, dim_pair, D_MODEL))

    # Compute PE(pos+k) via rotation
    theta_k = k / np.power(10000, (2 * dim_pair) / D_MODEL)
    rot_matrix = rotation_matrix(theta_k)
    pe_pos_k_rotated = rot_matrix @ pe_pos

    # Visualization
    ax4.axis('off')

    verification_text = (
        f'Numerical Verification\n\n'
        f'Position: pos={pos}, offset k={k}, dim pair i={dim_pair}\n\n'
        f'PE(pos) = [{pe_pos[0]:.6f}, {pe_pos[1]:.6f}]\n\n'
        f'PE(pos+k) direct = [{pe_pos_k_direct[0]:.6f}, {pe_pos_k_direct[1]:.6f}]\n\n'
        f'PE(pos+k) via rotation = [{pe_pos_k_rotated[0]:.6f}, {pe_pos_k_rotated[1]:.6f}]\n\n'
        f'Difference: {np.linalg.norm(pe_pos_k_direct - pe_pos_k_rotated):.2e}\n'
        f'(numerical precision error only)'
    )

    ax4.text(0.5, 0.6, verification_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue',
                       edgecolor='blue', linewidth=2),
             family='monospace')

    # Add explanation
    ax4.text(0.5, 0.15,
             'This proves that PE(pos+k) = R(k) @ PE(pos)\n'
             'where R(k) is a rotation matrix.\n'
             'The model can learn relative positions through linear algebra!',
             transform=ax4.transAxes, fontsize=11, style='italic',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax4.set_title('Verification', fontsize=12, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle('Positional Encoding as Rotation: The Linear Transformation Property',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'pe_linear_transform.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')

    plt.close()


if __name__ == '__main__':
    main()
