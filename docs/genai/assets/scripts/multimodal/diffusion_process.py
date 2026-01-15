#!/usr/bin/env python3
"""
Diffusion Process Visualization

Generates visualizations showing:
1. The forward diffusion process (adding noise)
2. The reverse diffusion process (denoising)
3. The noise schedule

Output: ../images/multimodal/diffusion_process.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'multimodal')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_diffusion_visualization():
    """Create a comprehensive diffusion process visualization."""
    fig = plt.figure(figsize=(16, 12))

    # Create grid for subplots
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.3)

    # Color scheme
    colors = {
        'clean': '#E8F5E9',
        'noise': '#FFEBEE',
        'process': '#E3F2FD',
        'arrow': '#37474F',
        'highlight': '#FFF9C4'
    }

    # === Forward Process ===
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 16)
    ax1.set_ylim(0, 3)
    ax1.axis('off')
    ax1.set_title('Forward Diffusion Process (Fixed): Gradually Adding Noise', fontsize=14, fontweight='bold', pad=10)

    # Draw timesteps
    timesteps = ['t=0', 't=250', 't=500', 't=750', 't=1000']
    x_positions = [1, 4, 7, 10, 13]

    for i, (t, x) in enumerate(zip(timesteps, x_positions)):
        # Calculate noise level for visual effect
        noise_level = i / (len(timesteps) - 1)

        # Draw image box with increasing noise
        box_color = tuple(
            c1 * (1 - noise_level) + c2 * noise_level
            for c1, c2 in zip(
                plt.cm.colors.hex2color(colors['clean']),
                plt.cm.colors.hex2color(colors['noise'])
            )
        )

        rect = FancyBboxPatch((x - 0.8, 0.8), 1.6, 1.6, boxstyle="round,pad=0.05",
                               facecolor=box_color, edgecolor=colors['arrow'], linewidth=2)
        ax1.add_patch(rect)

        # Add noise dots
        np.random.seed(42 + i)
        n_dots = int(noise_level * 30) + 5
        for _ in range(n_dots):
            dot_x = x + np.random.uniform(-0.6, 0.6)
            dot_y = 1.6 + np.random.uniform(-0.6, 0.6)
            dot_size = np.random.uniform(5, 15) * noise_level
            ax1.scatter(dot_x, dot_y, s=dot_size, c='gray', alpha=0.5)

        # Label
        ax1.text(x, 0.4, t, ha='center', va='center', fontsize=10, fontweight='bold')

        # Arrow to next
        if i < len(timesteps) - 1:
            ax1.annotate('', xy=(x_positions[i+1] - 1, 1.6),
                        xytext=(x + 1, 1.6),
                        arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))
            ax1.text((x + x_positions[i+1]) / 2, 2.2, '+noise', ha='center', fontsize=9)

    # Add labels
    ax1.text(1, 2.7, 'Clean Image\nx₀', ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor=colors['clean'], edgecolor='none'))
    ax1.text(13, 2.7, 'Pure Noise\nxₜ', ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor=colors['noise'], edgecolor='none'))

    # Formula
    ax1.text(7, 0.05, r'$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon$',
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor=colors['highlight'], edgecolor='none'))

    # === Reverse Process ===
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 16)
    ax2.set_ylim(0, 3)
    ax2.axis('off')
    ax2.set_title('Reverse Diffusion Process (Learned): Iterative Denoising', fontsize=14, fontweight='bold', pad=10)

    timesteps_rev = ['t=1000', 't=750', 't=500', 't=250', 't=0']

    for i, (t, x) in enumerate(zip(timesteps_rev, x_positions)):
        noise_level = 1 - i / (len(timesteps_rev) - 1)

        box_color = tuple(
            c1 * (1 - noise_level) + c2 * noise_level
            for c1, c2 in zip(
                plt.cm.colors.hex2color(colors['clean']),
                plt.cm.colors.hex2color(colors['noise'])
            )
        )

        rect = FancyBboxPatch((x - 0.8, 0.8), 1.6, 1.6, boxstyle="round,pad=0.05",
                               facecolor=box_color, edgecolor=colors['arrow'], linewidth=2)
        ax2.add_patch(rect)

        # Add noise dots
        np.random.seed(42 + i)
        n_dots = int(noise_level * 30) + 5
        for _ in range(n_dots):
            dot_x = x + np.random.uniform(-0.6, 0.6)
            dot_y = 1.6 + np.random.uniform(-0.6, 0.6)
            dot_size = np.random.uniform(5, 15) * noise_level
            ax2.scatter(dot_x, dot_y, s=dot_size, c='gray', alpha=0.5)

        ax2.text(x, 0.4, t, ha='center', va='center', fontsize=10, fontweight='bold')

        if i < len(timesteps_rev) - 1:
            ax2.annotate('', xy=(x_positions[i+1] - 1, 1.6),
                        xytext=(x + 1, 1.6),
                        arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))
            ax2.text((x + x_positions[i+1]) / 2, 2.2, 'denoise', ha='center', fontsize=9, color='#4CAF50')

    # U-Net annotation
    unet_box = FancyBboxPatch((6.5, 2.5), 3, 0.4, boxstyle="round,pad=0.05",
                               facecolor=colors['process'], edgecolor=colors['arrow'], linewidth=1)
    ax2.add_patch(unet_box)
    ax2.text(8, 2.7, 'U-Net predicts noise', ha='center', va='center', fontsize=9)

    # Labels
    ax2.text(1, 2.7, 'Pure Noise\nxₜ', ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor=colors['noise'], edgecolor='none'))
    ax2.text(13, 2.7, 'Generated\nImage', ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor=colors['clean'], edgecolor='none'))

    ax2.text(7, 0.05, r'$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)) + \sigma_t z$',
             ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor=colors['highlight'], edgecolor='none'))

    # === Noise Schedule ===
    ax3 = fig.add_subplot(gs[2])

    # Create noise schedule data
    t = np.linspace(0, 1000, 1001)

    # Linear schedule
    beta_start, beta_end = 0.0001, 0.02
    betas_linear = np.linspace(beta_start, beta_end, len(t))
    alphas_linear = 1 - betas_linear
    alpha_bar_linear = np.cumprod(alphas_linear)

    # Cosine schedule
    s = 0.008
    f_t = np.cos((t / 1000 + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar_cosine = f_t / f_t[0]

    # Plot
    ax3.plot(t, alpha_bar_linear, 'b-', linewidth=2, label='Linear Schedule')
    ax3.plot(t, alpha_bar_cosine, 'r-', linewidth=2, label='Cosine Schedule')

    ax3.set_xlabel('Timestep t', fontsize=11)
    ax3.set_ylabel(r'$\bar{\alpha}_t$ (Signal Remaining)', fontsize=11)
    ax3.set_title('Noise Schedules: Linear vs Cosine', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1000)
    ax3.set_ylim(0, 1.05)

    # Add annotations
    ax3.annotate('More signal\npreserved early', xy=(200, 0.85), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    ax3.annotate('Cosine preserves\nmore signal overall', xy=(700, 0.35), fontsize=9, ha='center',
                 bbox=dict(boxstyle='round', facecolor='#FFCDD2', edgecolor='gray'))

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'diffusion_process.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved diffusion process diagram to: {output_path}")
    return output_path


def create_diffusion_animation_frames():
    """Create individual frames showing the diffusion process step by step."""
    num_frames = 10

    for frame_idx in range(num_frames):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.axis('off')

        # Noise level increases with frame
        noise_level = frame_idx / (num_frames - 1)

        # Create a simple "image" that gets progressively noisier
        np.random.seed(42)
        base_x = np.random.randn(50) * 0.3
        base_y = np.random.randn(50) * 0.3

        # Add circular pattern
        theta = np.linspace(0, 2*np.pi, 50)
        circle_x = 0.8 * np.cos(theta)
        circle_y = 0.8 * np.sin(theta)

        # Mix with noise
        noise_x = np.random.randn(50) * noise_level * 1.5
        noise_y = np.random.randn(50) * noise_level * 1.5

        final_x = circle_x * (1 - noise_level) + noise_x
        final_y = circle_y * (1 - noise_level) + noise_y

        # Plot
        color = plt.cm.RdYlGn(1 - noise_level)
        ax.scatter(final_x, final_y, s=100, c=[color], alpha=0.8)

        # Title
        t = int(noise_level * 1000)
        ax.set_title(f'Forward Diffusion: t = {t}', fontsize=14, fontweight='bold')

        # Add noise level indicator
        ax.text(0, -1.7, f'Signal: {(1-noise_level)*100:.0f}%  |  Noise: {noise_level*100:.0f}%',
                ha='center', fontsize=11)

        # Save frame
        frame_path = os.path.join(OUTPUT_DIR, f'diffusion_frame_{frame_idx:02d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()

    print(f"Saved {num_frames} diffusion animation frames")


if __name__ == "__main__":
    create_diffusion_visualization()
    # Optionally create animation frames
    # create_diffusion_animation_frames()
