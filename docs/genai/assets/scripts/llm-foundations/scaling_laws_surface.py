#!/usr/bin/env python3
"""
Scaling Laws 3D Surface Visualization

This script creates 3D visualizations of LLM scaling laws,
showing the relationship between parameters, data, and loss.

Output: ../images/llm-foundations/scaling_*.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

# Ensure output directory exists
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "images", "llm-foundations"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def chinchilla_loss(N: float, D: float) -> float:
    """
    Compute loss using Chinchilla scaling law.

    L(N, D) = E + A/N^alpha + B/D^beta

    Where:
    - E ≈ 1.69 (irreducible loss)
    - A ≈ 406.4, alpha ≈ 0.34
    - B ≈ 410.7, beta ≈ 0.28
    """
    E = 1.69
    A = 406.4
    B = 410.7
    alpha = 0.34
    beta = 0.28

    return E + A / (N ** alpha) + B / (D ** beta)


def compute_optimal_allocation(compute: float) -> tuple:
    """
    For a given compute budget, find optimal N and D.

    Compute ≈ 6 * N * D
    Optimal ratio: D/N ≈ 20
    """
    # D = 20 * N, so C = 6 * N * 20 * N = 120 * N^2
    N_opt = np.sqrt(compute / 120)
    D_opt = 20 * N_opt
    return N_opt, D_opt


def create_scaling_surface():
    """Create 3D surface plot of scaling law."""
    # Parameter ranges (in billions for N, tokens for D)
    N_range = np.logspace(8, 12, 50)  # 100M to 1T parameters
    D_range = np.logspace(9, 14, 50)  # 1B to 100T tokens

    N_grid, D_grid = np.meshgrid(N_range, D_range)
    loss_grid = chinchilla_loss(N_grid, D_grid)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(
        np.log10(N_grid), np.log10(D_grid), loss_grid,
        cmap=cm.viridis, alpha=0.8,
        linewidth=0, antialiased=True
    )

    # Add color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Loss', fontsize=12)

    # Plot compute-optimal frontier
    compute_levels = np.logspace(20, 25, 20)
    N_opt = []
    D_opt = []
    L_opt = []

    for C in compute_levels:
        N, D = compute_optimal_allocation(C)
        if 1e8 <= N <= 1e12 and 1e9 <= D <= 1e14:
            N_opt.append(np.log10(N))
            D_opt.append(np.log10(D))
            L_opt.append(chinchilla_loss(N, D))

    ax.plot(N_opt, D_opt, L_opt, 'r-', linewidth=3, label='Compute-Optimal Frontier')

    # Mark some famous models
    models = {
        'GPT-3': (175e9, 300e9),
        'Chinchilla': (70e9, 1.4e12),
        'LLaMA 2 70B': (70e9, 2e12),
        'LLaMA 3 405B': (405e9, 15e12),
    }

    for name, (N, D) in models.items():
        loss = chinchilla_loss(N, D)
        ax.scatter([np.log10(N)], [np.log10(D)], [loss],
                   s=100, c='red', marker='*', zorder=5)
        ax.text(np.log10(N), np.log10(D), loss + 0.05, name, fontsize=9)

    ax.set_xlabel('log₁₀(Parameters)', fontsize=11)
    ax.set_ylabel('log₁₀(Tokens)', fontsize=11)
    ax.set_zlabel('Loss', fontsize=11)
    ax.set_title('Chinchilla Scaling Law Surface', fontsize=14, fontweight='bold')

    # Adjust view angle
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "scaling_surface_3d.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved 3D scaling surface to: {output_path}")


def create_isocompute_curves():
    """Create 2D plot showing iso-compute curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Iso-compute curves
    ax = axes[0]

    compute_levels = [1e21, 1e22, 1e23, 1e24, 1e25]
    colors = cm.plasma(np.linspace(0.2, 0.8, len(compute_levels)))

    N_range = np.logspace(8, 12, 100)

    for C, color in zip(compute_levels, colors):
        D_range = C / (6 * N_range)  # C = 6 * N * D
        valid = (D_range >= 1e9) & (D_range <= 1e14)

        losses = [chinchilla_loss(N, D) for N, D in zip(N_range[valid], D_range[valid])]

        ax.plot(N_range[valid] / 1e9, losses, color=color, linewidth=2,
                label=f'C = {C:.0e} FLOPs')

        # Mark optimal point
        N_opt, D_opt = compute_optimal_allocation(C)
        if 1e8 <= N_opt <= 1e12:
            L_opt = chinchilla_loss(N_opt, D_opt)
            ax.scatter([N_opt / 1e9], [L_opt], color=color, s=100, zorder=5, edgecolor='black')

    ax.set_xscale('log')
    ax.set_xlabel('Parameters (Billions)', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Iso-Compute Curves (Loss vs Parameters)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add annotation for optimal line
    ax.annotate('Optimal points\n(D/N ≈ 20)', xy=(70, 2.1), fontsize=10,
                ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Right: Optimal scaling
    ax = axes[1]

    compute_range = np.logspace(20, 26, 100)
    N_optimal = []
    D_optimal = []
    L_optimal = []

    for C in compute_range:
        N, D = compute_optimal_allocation(C)
        N_optimal.append(N / 1e9)
        D_optimal.append(D / 1e12)
        L_optimal.append(chinchilla_loss(N, D))

    ax.plot(compute_range, N_optimal, 'b-', linewidth=2, label='Optimal N (B)')
    ax.plot(compute_range, D_optimal, 'g-', linewidth=2, label='Optimal D (T)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute Budget (FLOPs)', fontsize=11)
    ax.set_ylabel('Scale', fontsize=11)
    ax.set_title('Optimal Scaling: Parameters and Data', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add famous model markers
    models = {
        'GPT-3': 3.64e23,
        'Chinchilla': 5.76e23,
        'LLaMA 3 405B': 3.9e25,
    }

    for name, C in models.items():
        N, D = compute_optimal_allocation(C)
        ax.axvline(x=C, color='red', linestyle='--', alpha=0.3)
        ax.annotate(name, xy=(C, N / 1e9), fontsize=8, rotation=90)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "scaling_isocompute.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved iso-compute curves to: {output_path}")


def create_chinchilla_comparison():
    """Compare GPT-3 vs Chinchilla allocation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Same compute budget
    compute = 5.76e23  # Chinchilla's compute

    # GPT-3 style (over-parameterized)
    gpt3_N = 175e9
    gpt3_D = compute / (6 * gpt3_N)

    # Chinchilla style (balanced)
    chinchilla_N, chinchilla_D = compute_optimal_allocation(compute)

    # Left: Bar comparison
    ax = axes[0]

    x = np.arange(2)
    width = 0.35

    params = [gpt3_N / 1e9, chinchilla_N / 1e9]
    tokens = [gpt3_D / 1e12, chinchilla_D / 1e12]

    bars1 = ax.bar(x - width / 2, params, width, label='Parameters (B)', color='steelblue')
    bars2 = ax.bar(x + width / 2, tokens, width, label='Tokens (T)', color='coral')

    ax.set_ylabel('Scale (log)', fontsize=11)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(['GPT-3 Style', 'Chinchilla Style'])
    ax.legend(fontsize=10)
    ax.set_title('Same Compute, Different Allocation', fontsize=12, fontweight='bold')

    # Add ratio annotations
    for i, (n, d) in enumerate(zip(params, tokens)):
        ratio = d * 1000 / n  # tokens(B) / params(B)
        ax.annotate(f'D/N = {ratio:.0f}', xy=(i, max(n, d) * 1.5),
                    ha='center', fontsize=10, fontweight='bold')

    # Right: Loss comparison
    ax = axes[1]

    # Compute losses
    gpt3_loss = chinchilla_loss(gpt3_N, gpt3_D)
    chinchilla_loss_val = chinchilla_loss(chinchilla_N, chinchilla_D)

    losses = [gpt3_loss, chinchilla_loss_val]
    colors = ['steelblue', 'forestgreen']

    bars = ax.bar(['GPT-3 Style', 'Chinchilla Style'], losses, color=colors, alpha=0.8)

    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Resulting Loss (Lower is Better)', fontsize=12, fontweight='bold')

    # Annotate improvement
    improvement = (gpt3_loss - chinchilla_loss_val) / gpt3_loss * 100
    ax.annotate(f'{improvement:.1f}% better', xy=(1, chinchilla_loss_val),
                xytext=(1.3, chinchilla_loss_val + 0.05),
                fontsize=11, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

    # Add loss values on bars
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{loss:.3f}', ha='center', fontsize=11)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "scaling_chinchilla_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved Chinchilla comparison to: {output_path}")


def create_compute_estimation():
    """Visualize compute requirements for various model sizes."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Model sizes to show
    param_sizes = [1e9, 7e9, 13e9, 30e9, 70e9, 175e9, 405e9, 1e12]
    param_labels = ['1B', '7B', '13B', '30B', '70B', '175B', '405B', '1T']

    # Compute for various data sizes
    token_multipliers = [5, 10, 20, 50, 100]  # tokens per parameter
    colors = cm.viridis(np.linspace(0.2, 0.9, len(token_multipliers)))

    for mult, color in zip(token_multipliers, colors):
        computes = [6 * N * (mult * N) for N in param_sizes]
        ax.plot(param_sizes, computes, 'o-', color=color, linewidth=2,
                label=f'{mult} tokens/param', markersize=8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Model Parameters', fontsize=12)
    ax.set_ylabel('Training Compute (FLOPs)', fontsize=12)
    ax.set_title('Training Compute Requirements', fontsize=14, fontweight='bold')

    # Custom x-ticks
    ax.set_xticks(param_sizes)
    ax.set_xticklabels(param_labels)

    # Add reference lines for GPU capabilities
    a100_year = 312e12 * 0.5 * 3600 * 24 * 365  # A100 FLOPs in a year at 50% util
    h100_year = 989e12 * 0.5 * 3600 * 24 * 365

    ax.axhline(y=a100_year, color='gray', linestyle='--', alpha=0.5)
    ax.text(1.5e9, a100_year * 1.2, '1 A100-year', fontsize=9, color='gray')

    ax.axhline(y=1000 * a100_year, color='gray', linestyle='--', alpha=0.5)
    ax.text(1.5e9, 1000 * a100_year * 1.2, '1000 A100-years', fontsize=9, color='gray')

    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add Chinchilla line
    ax.annotate('Chinchilla-optimal\n(20 tokens/param)',
                xy=(70e9, 6 * 70e9 * 20 * 70e9),
                xytext=(10e9, 1e25),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "scaling_compute_requirements.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved compute requirements to: {output_path}")


if __name__ == "__main__":
    create_scaling_surface()
    create_isocompute_curves()
    create_chinchilla_comparison()
    create_compute_estimation()
    print("All scaling law visualizations created successfully!")
