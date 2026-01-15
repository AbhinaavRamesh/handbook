#!/usr/bin/env python3
"""
Generate visualizations for Autoencoders & VAE module.

Outputs:
- ae_architecture.svg - Autoencoder architecture diagram
- ae_vae_latent.svg - AE vs VAE latent space comparison
- vae_architecture.svg - VAE architecture with sampling
- vae_reparam.svg - Reparameterization trick visualization
- ae_vae_interpolation.svg - Latent space interpolation comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Style settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

COLORS = {
    'encoder': '#4ECDC4',
    'decoder': '#FF6B6B',
    'latent': '#45B7D1',
    'input': '#95E1D3',
    'output': '#F38181',
    'mu': '#A8E6CF',
    'sigma': '#FFD93D',
}


def create_ae_architecture():
    """Create autoencoder architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Autoencoder Architecture', fontsize=14, fontweight='bold')

    # Input layer
    input_heights = [6, 5, 4, 3, 2]
    for i, y in enumerate(input_heights):
        rect = FancyBboxPatch((0.5, y - 0.3), 0.8, 0.6,
                              boxstyle="round,pad=0.02",
                              facecolor=COLORS['input'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
    ax.text(0.9, 7, 'Input\n(n dims)', ha='center', fontsize=10, fontweight='bold')

    # Encoder layers (shrinking)
    encoder_widths = [(2.5, 4), (4, 3), (5.5, 2)]
    for x, count in encoder_widths:
        for i in range(count):
            y = 4 + (count - 1) / 2 - i
            rect = FancyBboxPatch((x, y - 0.25), 0.6, 0.5,
                                  boxstyle="round,pad=0.02",
                                  facecolor=COLORS['encoder'], edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)

    # Encoder label
    ax.text(4, 1.2, 'Encoder', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['encoder'], linewidth=2))

    # Latent space (bottleneck)
    latent_x = 7
    circle = Circle((latent_x, 4), 0.5, facecolor=COLORS['latent'], edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(latent_x, 4, 'z', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(latent_x, 2.8, 'Latent\nSpace', ha='center', fontsize=10, fontweight='bold')

    # Decoder layers (expanding)
    decoder_widths = [(8.5, 2), (10, 3), (11.5, 4)]
    for x, count in decoder_widths:
        for i in range(count):
            y = 4 + (count - 1) / 2 - i
            rect = FancyBboxPatch((x, y - 0.25), 0.6, 0.5,
                                  boxstyle="round,pad=0.02",
                                  facecolor=COLORS['decoder'], edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)

    # Decoder label
    ax.text(10, 1.2, 'Decoder', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['decoder'], linewidth=2))

    # Output layer
    for i, y in enumerate(input_heights):
        rect = FancyBboxPatch((12.7, y - 0.3), 0.8, 0.6,
                              boxstyle="round,pad=0.02",
                              facecolor=COLORS['output'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
    ax.text(13.1, 7, 'Output\n(n dims)', ha='center', fontsize=10, fontweight='bold')

    # Arrows
    ax.annotate('', xy=(2.3, 4), xytext=(1.5, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(6.4, 4), xytext=(6.2, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(8.3, 4), xytext=(7.6, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(12.5, 4), xytext=(12.3, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Loss function annotation
    ax.annotate('', xy=(13.1, 1.5), xytext=(0.9, 1.5),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2, ls='--'))
    ax.text(7, 1.0, 'Reconstruction Loss: ||x - x\'||²', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lavender', edgecolor='purple'))

    plt.savefig(os.path.join(OUTPUT_DIR, 'ae_architecture.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: ae_architecture.svg")


def create_vae_architecture():
    """Create VAE architecture diagram with sampling."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Variational Autoencoder (VAE) Architecture', fontsize=14, fontweight='bold')

    # Input
    for i in range(4):
        y = 5.5 - i * 1.0
        rect = FancyBboxPatch((0.5, y - 0.25), 0.6, 0.5,
                              boxstyle="round,pad=0.02",
                              facecolor=COLORS['input'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
    ax.text(0.8, 6.5, 'Input x', ha='center', fontsize=10, fontweight='bold')

    # Encoder
    rect = FancyBboxPatch((2, 2.5), 1.5, 3,
                          boxstyle="round,pad=0.05",
                          facecolor=COLORS['encoder'], edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(rect)
    ax.text(2.75, 4, 'Encoder\nq(z|x)', ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrow to encoder
    ax.annotate('', xy=(1.9, 4), xytext=(1.2, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Mu and Sigma outputs
    rect_mu = FancyBboxPatch((4.5, 5), 1.2, 1,
                             boxstyle="round,pad=0.02",
                             facecolor=COLORS['mu'], edgecolor='black', linewidth=1.5)
    ax.add_patch(rect_mu)
    ax.text(5.1, 5.5, 'μ', ha='center', va='center', fontsize=14, fontweight='bold')

    rect_sigma = FancyBboxPatch((4.5, 2), 1.2, 1,
                                boxstyle="round,pad=0.02",
                                facecolor=COLORS['sigma'], edgecolor='black', linewidth=1.5)
    ax.add_patch(rect_sigma)
    ax.text(5.1, 2.5, 'σ', ha='center', va='center', fontsize=14, fontweight='bold')

    # Arrows from encoder to mu/sigma
    ax.annotate('', xy=(4.4, 5.5), xytext=(3.6, 4.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(4.4, 2.5), xytext=(3.6, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Sampling box
    rect = FancyBboxPatch((6.5, 3.2), 1.5, 1.6,
                          boxstyle="round,pad=0.05",
                          facecolor='white', edgecolor='purple', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(7.25, 4, 'z = μ + σε\nε ~ N(0,1)', ha='center', va='center', fontsize=10)
    ax.text(7.25, 5.2, 'Reparameterization', ha='center', fontsize=9, color='purple', fontweight='bold')

    # Arrows to sampling
    ax.annotate('', xy=(6.4, 4.5), xytext=(5.8, 5.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(6.4, 3.5), xytext=(5.8, 2.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Latent z
    circle = Circle((9, 4), 0.4, facecolor=COLORS['latent'], edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(9, 4, 'z', ha='center', va='center', fontsize=12, fontweight='bold')

    ax.annotate('', xy=(8.5, 4), xytext=(8.1, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Decoder
    rect = FancyBboxPatch((10, 2.5), 1.5, 3,
                          boxstyle="round,pad=0.05",
                          facecolor=COLORS['decoder'], edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(rect)
    ax.text(10.75, 4, 'Decoder\np(x|z)', ha='center', va='center', fontsize=11, fontweight='bold')

    ax.annotate('', xy=(9.9, 4), xytext=(9.5, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Output
    for i in range(4):
        y = 5.5 - i * 1.0
        rect = FancyBboxPatch((12.5, y - 0.25), 0.6, 0.5,
                              boxstyle="round,pad=0.02",
                              facecolor=COLORS['output'], edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
    ax.text(12.8, 6.5, "Output x'", ha='center', fontsize=10, fontweight='bold')

    ax.annotate('', xy=(12.4, 4), xytext=(11.6, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Loss annotation
    ax.text(7, 0.8, 'Loss = Reconstruction + KL Divergence', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))

    plt.savefig(os.path.join(OUTPUT_DIR, 'vae_architecture.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: vae_architecture.svg")


def create_vae_reparam():
    """Create reparameterization trick visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Without reparameterization (non-differentiable)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Without Reparameterization\n(Non-differentiable)', fontsize=12, fontweight='bold', color='red')

    # Encoder output
    rect = FancyBboxPatch((1, 3), 2, 2,
                          boxstyle="round,pad=0.05",
                          facecolor=COLORS['encoder'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(2, 4, 'μ, σ', ha='center', va='center', fontsize=12, fontweight='bold')

    # Sampling operation
    circle = Circle((5, 4), 0.6, facecolor='white', edgecolor='red', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.text(5, 4, '~', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(5, 2.8, 'Sample from\nN(μ, σ²)', ha='center', fontsize=9)

    # Z output
    circle = Circle((8, 4), 0.5, facecolor=COLORS['latent'], edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(8, 4, 'z', ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrows
    ax.annotate('', xy=(4.3, 4), xytext=(3.1, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(7.4, 4), xytext=(5.7, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Gradient blocked
    ax.text(5, 6.5, '❌ Gradients cannot flow\nthrough sampling', ha='center', fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='mistyrose', edgecolor='red'))

    # Right: With reparameterization (differentiable)
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('With Reparameterization\n(Differentiable)', fontsize=12, fontweight='bold', color='green')

    # Encoder outputs (μ and σ)
    rect_mu = FancyBboxPatch((0.5, 5), 1.5, 1,
                             boxstyle="round,pad=0.02",
                             facecolor=COLORS['mu'], edgecolor='black', linewidth=1.5)
    ax.add_patch(rect_mu)
    ax.text(1.25, 5.5, 'μ', ha='center', va='center', fontsize=12, fontweight='bold')

    rect_sigma = FancyBboxPatch((0.5, 2), 1.5, 1,
                                boxstyle="round,pad=0.02",
                                facecolor=COLORS['sigma'], edgecolor='black', linewidth=1.5)
    ax.add_patch(rect_sigma)
    ax.text(1.25, 2.5, 'σ', ha='center', va='center', fontsize=12, fontweight='bold')

    # Epsilon (external noise)
    circle = Circle((3.5, 0.8), 0.4, facecolor='lightgray', edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(3.5, 0.8, 'ε', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(3.5, 0.1, '~ N(0,1)', ha='center', fontsize=9)

    # Multiplication (σ * ε)
    circle = Circle((4.5, 2.5), 0.4, facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(4.5, 2.5, '×', ha='center', va='center', fontsize=14, fontweight='bold')

    # Addition (μ + σε)
    circle = Circle((6.5, 4), 0.4, facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(6.5, 4, '+', ha='center', va='center', fontsize=14, fontweight='bold')

    # Z output
    circle = Circle((8.5, 4), 0.5, facecolor=COLORS['latent'], edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(8.5, 4, 'z', ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrows
    ax.annotate('', xy=(4.1, 2.5), xytext=(2.1, 2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(4.2, 1.3), xytext=(3.8, 0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(6.1, 3.7), xytext=(4.9, 2.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(6.1, 4.3), xytext=(2.1, 5.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(8, 4), xytext=(7, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Formula
    ax.text(5.5, 7, 'z = μ + σ ⊙ ε', ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green', linewidth=2))

    # Gradient flows
    ax.text(5, 6, '✓ Gradients flow through\ndeterministic operations', ha='center', fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'vae_reparam.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: vae_reparam.svg")


def create_ae_vae_latent():
    """Create AE vs VAE latent space comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    np.random.seed(42)

    # Left: Standard AE latent space (irregular clusters)
    ax = axes[0]
    ax.set_title('Standard Autoencoder\nLatent Space', fontsize=12, fontweight='bold')

    # Create irregular, scattered clusters
    clusters = [
        (np.random.randn(30, 2) * 0.3 + [2, 2], 'Class 1'),
        (np.random.randn(30, 2) * 0.5 + [-1, 1], 'Class 2'),
        (np.random.randn(30, 2) * 0.2 + [1, -2], 'Class 3'),
        (np.random.randn(30, 2) * 0.4 + [-2, -1], 'Class 4'),
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for (points, label), color in zip(clusters, colors):
        ax.scatter(points[:, 0], points[:, 1], c=color, s=50, alpha=0.7, label=label)

    # Show gaps
    ax.annotate('Gaps\n(unused space)', xy=(0, 0), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='mistyrose', edgecolor='red'))

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: VAE latent space (smooth, continuous)
    ax = axes[1]
    ax.set_title('Variational Autoencoder\nLatent Space', fontsize=12, fontweight='bold')

    # Create smooth, overlapping Gaussian clusters
    theta = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    radius = 1.5
    clusters_vae = []
    for i, t in enumerate(theta):
        center = [radius * np.cos(t), radius * np.sin(t)]
        points = np.random.randn(30, 2) * 0.6 + center
        clusters_vae.append(points)

    for points, color in zip(clusters_vae, colors):
        ax.scatter(points[:, 0], points[:, 1], c=color, s=50, alpha=0.7)

    # Draw unit Gaussian prior
    circle = plt.Circle((0, 0), 2, fill=False, color='purple', linestyle='--', linewidth=2, label='N(0,I) prior')
    ax.add_patch(circle)

    ax.annotate('Smooth\ninterpolation', xy=(0, 0), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green'))

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('Latent dim 1')
    ax.set_ylabel('Latent dim 2')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ae_vae_latent.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: ae_vae_latent.svg")


def create_ae_vae_interpolation():
    """Create latent space interpolation comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top: AE interpolation (discontinuous)
    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('Standard AE: Interpolation in Latent Space', fontsize=12, fontweight='bold')

    # Draw interpolation path
    x_positions = np.linspace(1, 11, 6)

    # Point A
    circle = Circle((x_positions[0], 2), 0.4, facecolor=COLORS['encoder'], edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x_positions[0], 2, 'A', ha='center', va='center', fontsize=12, fontweight='bold')

    # Point B
    circle = Circle((x_positions[-1], 2), 0.4, facecolor=COLORS['decoder'], edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x_positions[-1], 2, 'B', ha='center', va='center', fontsize=12, fontweight='bold')

    # Intermediate points (showing discontinuity)
    intermediate_colors = ['#FFD93D', 'gray', 'gray', '#FFD93D']
    for i, x in enumerate(x_positions[1:-1]):
        circle = Circle((x, 2), 0.3, facecolor=intermediate_colors[i], edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        if i == 1 or i == 2:
            ax.text(x, 2, '?', ha='center', va='center', fontsize=10, fontweight='bold')

    # Dashed line showing path
    ax.plot([x_positions[0] + 0.5, x_positions[-1] - 0.5], [2, 2], 'k--', lw=1.5, alpha=0.5)

    # Warning
    ax.text(6, 0.5, '⚠ May pass through "empty" regions → meaningless outputs',
            ha='center', fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='mistyrose', edgecolor='red'))

    # Bottom: VAE interpolation (smooth)
    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('VAE: Interpolation in Latent Space', fontsize=12, fontweight='bold')

    # Point A
    circle = Circle((x_positions[0], 2), 0.4, facecolor=COLORS['encoder'], edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x_positions[0], 2, 'A', ha='center', va='center', fontsize=12, fontweight='bold')

    # Point B
    circle = Circle((x_positions[-1], 2), 0.4, facecolor=COLORS['decoder'], edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x_positions[-1], 2, 'B', ha='center', va='center', fontsize=12, fontweight='bold')

    # Intermediate points (smooth gradient)
    gradient_colors = ['#4ECDC4', '#6DD5C7', '#95E1D3', '#F38181']
    for i, x in enumerate(x_positions[1:-1]):
        circle = Circle((x, 2), 0.3, facecolor=gradient_colors[i], edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, 2, '✓', ha='center', va='center', fontsize=10, fontweight='bold', color='green')

    # Solid line showing smooth path
    ax.plot([x_positions[0] + 0.5, x_positions[-1] - 0.5], [2, 2], 'g-', lw=2, alpha=0.7)

    # Success message
    ax.text(6, 0.5, '✓ Smooth transitions → meaningful intermediate outputs',
            ha='center', fontsize=10, color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ae_vae_interpolation.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: ae_vae_interpolation.svg")


if __name__ == '__main__':
    print("Generating Autoencoder & VAE visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    create_ae_architecture()
    create_vae_architecture()
    create_vae_reparam()
    create_ae_vae_latent()
    create_ae_vae_interpolation()

    print("\nAll Autoencoder & VAE visualizations generated!")
