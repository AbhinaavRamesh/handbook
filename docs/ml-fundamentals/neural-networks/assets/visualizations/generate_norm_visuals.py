#!/usr/bin/env python3
"""
Generate visualizations for Normalization module.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

COLORS = {
    'batch': '#E74C3C',
    'layer': '#3498DB',
    'instance': '#27AE60',
    'group': '#9B59B6',
}


def create_norm_dimensions():
    """Show which dimensions each normalization type normalizes over."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    norm_types = [
        ('Batch Normalization', 'batch', [(0, 2, 3)], 'Normalize over (N, H, W)\nStats per Channel'),
        ('Layer Normalization', 'layer', [(1, 2, 3)], 'Normalize over (C, H, W)\nStats per Sample'),
        ('Instance Normalization', 'instance', [(2, 3)], 'Normalize over (H, W)\nStats per Sample, Channel'),
        ('Group Normalization', 'group', [(1, 2, 3)], 'Normalize over (C/G, H, W)\nStats per Sample, Group'),
    ]

    for ax, (name, ntype, dims, desc) in zip(axes.flatten(), norm_types):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(name, fontsize=13, fontweight='bold')

        # Draw 3D tensor representation
        # N x C x H x W visualized as stacked grids
        n_samples = 2
        n_channels = 3

        for n in range(n_samples):
            for c in range(n_channels):
                x = 1 + c * 2.5 + n * 0.3
                y = 7 - n * 3

                # Color based on what's normalized together
                if ntype == 'batch':
                    color = plt.cm.Set3(c / n_channels)
                elif ntype == 'layer':
                    color = plt.cm.Set3(n / n_samples)
                elif ntype == 'instance':
                    color = plt.cm.Set3((n * n_channels + c) / (n_samples * n_channels))
                else:  # group
                    group = c // 2
                    color = plt.cm.Set3((n * 2 + group) / (n_samples * 2))

                rect = Rectangle((x, y), 1.5, 2, facecolor=color,
                                edgecolor='black', linewidth=1.5, alpha=0.8)
                ax.add_patch(rect)

                # Label
                ax.text(x + 0.75, y + 1, f'N{n}C{c}', ha='center', va='center', fontsize=8)

        # Description
        ax.text(5, 1.5, desc, ha='center', va='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

        # Dimension labels
        ax.text(0.5, 5, 'N\n(batch)', ha='center', va='center', fontsize=9)
        ax.text(5, 9.5, 'C (channels)', ha='center', va='center', fontsize=9)

    fig.suptitle('Normalization Types: Which Dimensions Are Normalized', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, 'norm_dimensions.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: norm_dimensions.svg")


def create_norm_batchnorm():
    """Show BatchNorm algorithm steps."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    np.random.seed(42)
    data = np.random.randn(5, 8) * 2 + 3  # Mean ~3, std ~2

    titles = [
        'Input Activations',
        '1. Compute μ, σ²',
        '2. Normalize: (x-μ)/σ',
        '3. Scale & Shift: γx+β'
    ]

    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=11, fontweight='bold')

        if i == 0:
            im = ax.imshow(data, cmap='RdYlBu', aspect='auto', vmin=-2, vmax=8)
            ax.set_xlabel('Features')
            ax.set_ylabel('Batch samples')
            plt.colorbar(im, ax=ax)
            ax.text(4, -1, f'μ={np.mean(data):.2f}, σ={np.std(data):.2f}',
                   ha='center', fontsize=9)

        elif i == 1:
            # Show mean and variance per feature
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0)
            x = np.arange(len(means))
            ax.bar(x - 0.2, means, 0.4, label='μ', color='blue', alpha=0.7)
            ax.bar(x + 0.2, stds, 0.4, label='σ', color='orange', alpha=0.7)
            ax.set_xlabel('Feature')
            ax.set_ylabel('Value')
            ax.legend()
            ax.axhline(y=0, color='gray', linestyle='--')

        elif i == 2:
            normalized = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
            im = ax.imshow(normalized, cmap='RdYlBu', aspect='auto', vmin=-2, vmax=2)
            ax.set_xlabel('Features')
            ax.set_ylabel('Batch samples')
            plt.colorbar(im, ax=ax)
            ax.text(4, -1, f'μ≈0, σ≈1', ha='center', fontsize=9)

        else:
            gamma = 1.5
            beta = 0.5
            normalized = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
            output = gamma * normalized + beta
            im = ax.imshow(output, cmap='RdYlBu', aspect='auto', vmin=-2, vmax=3)
            ax.set_xlabel('Features')
            ax.set_ylabel('Batch samples')
            plt.colorbar(im, ax=ax)
            ax.text(4, -1, f'γ={gamma}, β={beta} (learned)', ha='center', fontsize=9)

    fig.suptitle('Batch Normalization: Step by Step', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, 'norm_batchnorm.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: norm_batchnorm.svg")


def create_norm_loss_landscape():
    """Show smoother loss landscape with BatchNorm."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Create loss surfaces
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # Without BatchNorm: more irregular
    Z_without = np.sin(3*X) * np.cos(3*Y) + 0.5 * (X**2 + Y**2) + np.sin(5*X*Y) * 0.3

    # With BatchNorm: smoother
    Z_with = 0.5 * (X**2 + Y**2) + 0.1 * np.sin(2*X) * np.cos(2*Y)

    for ax, Z, title in [(axes[0], Z_without, 'Without BatchNorm'),
                         (axes[1], Z_with, 'With BatchNorm')]:
        contour = ax.contour(X, Y, Z, levels=15, cmap='viridis')
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_xlabel('Weight 1')
        ax.set_ylabel('Weight 2')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')

        # Add optimization path
        if 'Without' in title:
            path_x = [-1.5, -1.0, -0.3, 0.2, -0.1, 0.3, 0.1, 0.0]
            path_y = [1.5, 0.8, 0.5, 0.8, 0.3, 0.1, 0.2, 0.0]
            ax.plot(path_x, path_y, 'r.-', linewidth=2, markersize=8, label='Optimization path')
            ax.text(-1, -1.5, 'Zigzag path\n(slow convergence)', fontsize=9, color='red',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            path_x = [-1.5, -1.0, -0.5, -0.2, 0.0]
            path_y = [1.5, 1.0, 0.5, 0.2, 0.0]
            ax.plot(path_x, path_y, 'g.-', linewidth=2, markersize=8, label='Optimization path')
            ax.text(-1, -1.5, 'Direct path\n(fast convergence)', fontsize=9, color='green',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.legend(loc='upper right')

    fig.suptitle('BatchNorm Smooths the Loss Landscape', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, 'norm_loss_landscape.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: norm_loss_landscape.svg")


if __name__ == '__main__':
    print("Generating Normalization visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    create_norm_dimensions()
    create_norm_batchnorm()
    create_norm_loss_landscape()

    print("\nAll normalization visualizations generated!")
