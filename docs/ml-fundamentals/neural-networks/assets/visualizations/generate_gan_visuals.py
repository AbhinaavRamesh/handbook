#!/usr/bin/env python3
"""
Generate visualizations for GANs module.

Outputs:
- gan_architecture.svg - GAN architecture diagram
- gan_training.gif - GAN training progression
- gan_mode_collapse.svg - Mode collapse visualization
- gan_wgan_comparison.svg - WGAN vs GAN loss comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Style settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

COLORS = {
    'generator': '#4ECDC4',
    'discriminator': '#FF6B6B',
    'noise': '#95E1D3',
    'real': '#45B7D1',
    'fake': '#F38181',
}


def create_gan_architecture():
    """Create GAN architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Generative Adversarial Network (GAN) Architecture', fontsize=14, fontweight='bold')

    # Noise input (z)
    circle = Circle((1, 4), 0.6, facecolor=COLORS['noise'], edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(1, 4, 'z', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(1, 2.8, 'Random\nNoise', ha='center', fontsize=9)
    ax.text(1, 5.2, '~ N(0, I)', ha='center', fontsize=9)

    # Generator
    rect = FancyBboxPatch((3, 2.5), 2.5, 3,
                          boxstyle="round,pad=0.05",
                          facecolor=COLORS['generator'], edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(4.25, 4, 'Generator\nG(z)', ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrow from noise to generator
    ax.annotate('', xy=(2.9, 4), xytext=(1.7, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Generated (fake) samples
    rect = FancyBboxPatch((6.5, 3.2), 1.2, 1.6,
                          boxstyle="round,pad=0.02",
                          facecolor=COLORS['fake'], edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(7.1, 4, 'Fake\nG(z)', ha='center', va='center', fontsize=10, fontweight='bold')

    ax.annotate('', xy=(6.4, 4), xytext=(5.6, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Real samples
    rect = FancyBboxPatch((6.5, 5.5), 1.2, 1.6,
                          boxstyle="round,pad=0.02",
                          facecolor=COLORS['real'], edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(7.1, 6.3, 'Real\nx', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7.1, 7.5, 'Training Data', ha='center', fontsize=9)

    # Discriminator
    rect = FancyBboxPatch((9, 2.5), 2.5, 5,
                          boxstyle="round,pad=0.05",
                          facecolor=COLORS['discriminator'], edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(10.25, 5, 'Discriminator\nD(x)', ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrows to discriminator
    ax.annotate('', xy=(8.9, 4), xytext=(7.8, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(8.9, 6), xytext=(7.8, 6.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Output
    circle = Circle((13, 5), 0.5, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(13, 5, 'P', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(13, 4, 'Real or\nFake?', ha='center', fontsize=9)

    ax.annotate('', xy=(12.4, 5), xytext=(11.6, 5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Feedback loops
    ax.annotate('', xy=(10.25, 2.3), xytext=(10.25, 1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='--',
                               connectionstyle='arc3,rad=0'))
    ax.annotate('', xy=(4.25, 2.3), xytext=(4.25, 1),
                arrowprops=dict(arrowstyle='<-', color='green', lw=2, ls='--'))
    ax.plot([4.25, 10.25], [1, 1], 'k--', lw=1.5)

    # Loss labels
    ax.text(4.25, 0.5, 'G tries to\nfool D', ha='center', fontsize=9, color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green'))
    ax.text(10.25, 0.5, 'D tries to\ndetect fakes', ha='center', fontsize=9, color='red',
            bbox=dict(boxstyle='round', facecolor='mistyrose', edgecolor='red'))

    plt.savefig(os.path.join(OUTPUT_DIR, 'gan_architecture.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: gan_architecture.svg")


def create_gan_training():
    """Create GAN training progression animation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    np.random.seed(42)

    # Target distribution (mixture of Gaussians)
    def target_dist():
        n = 500
        mix = np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3])
        centers = [(-2, 0), (2, 2), (2, -2)]
        points = np.zeros((n, 2))
        for i, m in enumerate(mix):
            points[i] = np.random.randn(2) * 0.5 + centers[m]
        return points

    real_data = target_dist()

    def init():
        for ax in axes:
            ax.clear()
        return []

    def animate(frame):
        for ax in axes:
            ax.clear()

        # Epoch progression
        epochs = [1, 50, 200]
        titles = ['Epoch 1\n(Random noise)', 'Epoch 50\n(Learning structure)', 'Epoch 200\n(Converged)']

        for idx, (ax, epoch, title) in enumerate(zip(axes, epochs, titles)):
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_title(title, fontsize=12, fontweight='bold')

            # Real data (blue)
            ax.scatter(real_data[:, 0], real_data[:, 1], c='blue', s=20, alpha=0.5, label='Real')

            # Generated data (red) - evolves over time
            progress = min(1.0, (frame + idx * 10) / 30)

            if idx == 0:
                # Random noise
                fake = np.random.randn(500, 2) * 2
            elif idx == 1:
                # Partially learned
                fake = real_data * progress + np.random.randn(500, 2) * (1 - progress) * 2
            else:
                # Well learned
                fake = real_data + np.random.randn(500, 2) * 0.3 * (1 - progress * 0.8)

            ax.scatter(fake[:, 0], fake[:, 1], c='red', s=20, alpha=0.5, label='Generated')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        return []

    anim = FuncAnimation(fig, animate, init_func=init, frames=30, interval=200, blit=False)
    anim.save(os.path.join(OUTPUT_DIR, 'gan_training.gif'), writer=PillowWriter(fps=5))
    plt.close()
    print("Generated: gan_training.gif")


def create_gan_mode_collapse():
    """Create mode collapse visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    np.random.seed(42)

    # Left: Healthy GAN (covers all modes)
    ax = axes[0]
    ax.set_title('Healthy GAN\n(All modes covered)', fontsize=12, fontweight='bold', color='green')

    # Target modes (3 Gaussian clusters)
    centers = [(-2, 2), (2, 2), (0, -2)]
    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']

    for center, color in zip(centers, colors):
        # Real data
        real = np.random.randn(100, 2) * 0.5 + center
        ax.scatter(real[:, 0], real[:, 1], c=color, s=30, alpha=0.3, marker='o')
        # Generated data (covers the same area)
        fake = np.random.randn(100, 2) * 0.6 + center
        ax.scatter(fake[:, 0], fake[:, 1], c=color, s=30, alpha=0.7, marker='x')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    # Legend
    ax.scatter([], [], c='gray', marker='o', label='Real data', alpha=0.3)
    ax.scatter([], [], c='gray', marker='x', label='Generated', alpha=0.7)
    ax.legend(loc='upper right')

    # Right: Mode collapse
    ax = axes[1]
    ax.set_title('Mode Collapse\n(Only one mode captured)', fontsize=12, fontweight='bold', color='red')

    # Show all real modes
    for center, color in zip(centers, colors):
        real = np.random.randn(100, 2) * 0.5 + center
        ax.scatter(real[:, 0], real[:, 1], c=color, s=30, alpha=0.3, marker='o')

    # But generator only produces one mode
    collapsed_center = centers[1]  # Only covering middle mode
    fake = np.random.randn(300, 2) * 0.4 + collapsed_center
    ax.scatter(fake[:, 0], fake[:, 1], c='red', s=30, alpha=0.7, marker='x')

    # Mark collapsed modes
    for i, center in enumerate(centers):
        if i != 1:
            circle = plt.Circle(center, 1.2, fill=False, color='red', linestyle='--', linewidth=2)
            ax.add_patch(circle)
            ax.text(center[0], center[1] - 1.8, 'Missing!', ha='center', fontsize=9, color='red')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    # Legend
    ax.scatter([], [], c='gray', marker='o', label='Real data', alpha=0.3)
    ax.scatter([], [], c='red', marker='x', label='Generated (collapsed)', alpha=0.7)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gan_mode_collapse.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: gan_mode_collapse.svg")


def create_gan_wgan_comparison():
    """Create WGAN vs GAN loss comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Standard GAN loss issues
    ax = axes[0]
    ax.set_title('Standard GAN Loss\n(JS Divergence)', fontsize=12, fontweight='bold')

    x = np.linspace(-3, 3, 100)

    # Discriminator loss when distributions don't overlap
    ax.axhline(y=np.log(2), color='red', linestyle='--', label='Saturated loss (no overlap)')
    ax.text(0, np.log(2) + 0.1, 'Loss saturates → vanishing gradients', ha='center', fontsize=9)

    # Show the problematic region
    ax.fill_between(x, 0, np.log(2), alpha=0.2, color='red')
    ax.set_xlabel('Training Progress')
    ax.set_ylabel('Discriminator Loss')
    ax.set_ylim(0, 1.5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Problems list
    ax.text(0, 1.3, 'Problems:', fontsize=10, fontweight='bold', ha='center')
    ax.text(0, 1.15, '• Vanishing gradients', fontsize=9, ha='center')
    ax.text(0, 1.0, '• Mode collapse', fontsize=9, ha='center')
    ax.text(0, 0.85, '• Training instability', fontsize=9, ha='center')

    # Right: WGAN loss (Wasserstein distance)
    ax = axes[1]
    ax.set_title('WGAN Loss\n(Wasserstein Distance)', fontsize=12, fontweight='bold')

    # Smooth, meaningful loss curve
    x = np.linspace(0, 100, 100)
    loss = 5 * np.exp(-x / 30) + 0.5 + np.random.randn(100) * 0.1

    ax.plot(x, loss, 'g-', linewidth=2, label='Critic loss')
    ax.fill_between(x, loss - 0.2, loss + 0.2, alpha=0.2, color='green')

    ax.set_xlabel('Training Progress')
    ax.set_ylabel('Critic Loss (Wasserstein Distance)')
    ax.set_ylim(0, 6)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Benefits list
    ax.text(50, 5.5, 'Benefits:', fontsize=10, fontweight='bold', ha='center')
    ax.text(50, 5.0, '• Meaningful gradients everywhere', fontsize=9, ha='center')
    ax.text(50, 4.5, '• Correlates with sample quality', fontsize=9, ha='center')
    ax.text(50, 4.0, '• More stable training', fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gan_wgan_comparison.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: gan_wgan_comparison.svg")


if __name__ == '__main__':
    print("Generating GAN visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    create_gan_architecture()
    create_gan_training()
    create_gan_mode_collapse()
    create_gan_wgan_comparison()

    print("\nAll GAN visualizations generated!")
