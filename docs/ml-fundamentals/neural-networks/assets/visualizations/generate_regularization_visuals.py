#!/usr/bin/env python3
"""
Generate visualizations for Regularization module.

Outputs:
- reg_overfitting.svg - Overfitting training/validation curves
- reg_l1_l2.svg - L1 vs L2 regularization comparison
- reg_dropout.gif - Dropout mechanism animation
- reg_early_stopping.svg - Early stopping visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Style settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

COLORS = {
    'train': '#3498DB',
    'val': '#E74C3C',
    'l1': '#27AE60',
    'l2': '#9B59B6',
    'active': '#45B7D1',
    'dropped': '#BDC3C7',
    'best': '#F39C12',
    'stopped': '#E74C3C',
}


def create_overfitting_curves():
    """Create overfitting training/validation curves visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    epochs = np.arange(1, 101)

    # Underfitting scenario
    ax = axes[0]
    train_loss = 0.8 - 0.2 * (1 - np.exp(-epochs / 50))
    val_loss = 0.85 - 0.15 * (1 - np.exp(-epochs / 50))
    ax.plot(epochs, train_loss, color=COLORS['train'], linewidth=2, label='Training Loss')
    ax.plot(epochs, val_loss, color=COLORS['val'], linewidth=2, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Underfitting', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.text(50, 0.15, 'Both losses high\nModel too simple', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    # Good fit scenario
    ax = axes[1]
    train_loss = 0.8 * np.exp(-epochs / 30) + 0.05
    val_loss = 0.9 * np.exp(-epochs / 35) + 0.08
    ax.plot(epochs, train_loss, color=COLORS['train'], linewidth=2, label='Training Loss')
    ax.plot(epochs, val_loss, color=COLORS['val'], linewidth=2, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Good Fit', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.text(50, 0.6, 'Both losses low\nSmall gap', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green'))

    # Overfitting scenario
    ax = axes[2]
    train_loss = 0.8 * np.exp(-epochs / 20) + 0.01
    val_loss = 0.9 * np.exp(-epochs / 30) + 0.1 + 0.3 * (1 - np.exp(-epochs / 40))
    ax.plot(epochs, train_loss, color=COLORS['train'], linewidth=2, label='Training Loss')
    ax.plot(epochs, val_loss, color=COLORS['val'], linewidth=2, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Overfitting', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Mark the divergence point
    diverge_epoch = 25
    ax.axvline(x=diverge_epoch, color='gray', linestyle='--', alpha=0.5)
    ax.annotate('Overfitting\nbegins', xy=(diverge_epoch, 0.35), xytext=(50, 0.5),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='red'))

    # Add gap annotation
    ax.annotate('', xy=(80, train_loss[79]), xytext=(80, val_loss[79]),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(85, (train_loss[79] + val_loss[79]) / 2, 'Gap', fontsize=9, va='center')

    fig.suptitle('Overfitting: Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, 'reg_overfitting.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: reg_overfitting.svg")


def create_l1_l2_comparison():
    """Create L1 vs L2 regularization comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top row: Weight distribution
    np.random.seed(42)

    # No regularization
    ax = axes[0, 0]
    weights_no_reg = np.random.randn(100) * 2
    ax.bar(range(100), weights_no_reg, color='gray', alpha=0.7)
    ax.set_title('No Regularization', fontsize=11, fontweight='bold')
    ax.set_xlabel('Weight Index')
    ax.set_ylabel('Weight Value')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylim(-5, 5)

    # L2 regularization
    ax = axes[0, 1]
    weights_l2 = weights_no_reg * 0.4  # All shrunk toward zero
    ax.bar(range(100), weights_l2, color=COLORS['l2'], alpha=0.7)
    ax.set_title('L2 Regularization (Weight Decay)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Weight Index')
    ax.set_ylabel('Weight Value')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylim(-5, 5)
    ax.text(50, 4, 'All weights shrunk\nbut none exactly zero', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lavender', edgecolor=COLORS['l2']))

    # L1 regularization
    ax = axes[1, 0]
    weights_l1 = weights_no_reg.copy()
    weights_l1[np.abs(weights_l1) < 1.5] = 0  # Many exactly zero
    weights_l1 = weights_l1 * 0.6
    ax.bar(range(100), weights_l1, color=COLORS['l1'], alpha=0.7)
    ax.set_title('L1 Regularization (Lasso)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Weight Index')
    ax.set_ylabel('Weight Value')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylim(-5, 5)
    n_zero = np.sum(weights_l1 == 0)
    ax.text(50, 4, f'Many weights exactly zero\n({n_zero}% sparse)', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor=COLORS['l1']))

    # Geometric interpretation
    ax = axes[1, 1]
    theta = np.linspace(0, 2*np.pi, 100)

    # L2 constraint (circle)
    r_l2 = 1.5
    x_l2 = r_l2 * np.cos(theta)
    y_l2 = r_l2 * np.sin(theta)
    ax.plot(x_l2, y_l2, color=COLORS['l2'], linewidth=2, label='L2: $||w||_2 = c$ (circle)')
    ax.fill(x_l2, y_l2, color=COLORS['l2'], alpha=0.2)

    # L1 constraint (diamond)
    r_l1 = 1.8
    x_l1 = [r_l1, 0, -r_l1, 0, r_l1]
    y_l1 = [0, r_l1, 0, -r_l1, 0]
    ax.plot(x_l1, y_l1, color=COLORS['l1'], linewidth=2, label='L1: $||w||_1 = c$ (diamond)')
    ax.fill(x_l1, y_l1, color=COLORS['l1'], alpha=0.2)

    # Loss contours
    w1 = np.linspace(-3, 3, 100)
    w2 = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1, w2)
    # Elliptical loss centered at (2, 1)
    L = 0.5 * ((W1 - 2)**2 / 4 + (W2 - 1)**2)
    contour = ax.contour(W1, W2, L, levels=[0.5, 1, 1.5, 2, 3], colors='gray', alpha=0.5)
    ax.clabel(contour, inline=True, fontsize=8)

    # Optimal points
    ax.plot(0, 1.5, 'o', color=COLORS['l2'], markersize=12, markeredgecolor='black',
            markeredgewidth=2, label='L2 solution', zorder=5)
    ax.plot(0, 1.8, 's', color=COLORS['l1'], markersize=12, markeredgecolor='black',
            markeredgewidth=2, label='L1 solution (sparse!)', zorder=5)
    ax.plot(2, 1, '*', color='red', markersize=15, markeredgecolor='black',
            markeredgewidth=1, label='Unconstrained minimum', zorder=5)

    ax.set_xlabel('$w_1$', fontsize=11)
    ax.set_ylabel('$w_2$', fontsize=11)
    ax.set_title('Geometric View: Why L1 Promotes Sparsity', fontsize=11, fontweight='bold')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Annotate corner intersection
    ax.annotate('L1 hits corner\n(sparse solution)', xy=(0, 1.8), xytext=(-2.2, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['l1']),
                fontsize=9, color=COLORS['l1'])

    fig.suptitle('L1 vs L2 Regularization Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, 'reg_l1_l2.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: reg_l1_l2.svg")


def create_dropout_animation():
    """Create dropout mechanism animation showing neurons being dropped."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Network structure
    layer_x = [2, 5, 8, 11]
    layer_sizes = [4, 6, 6, 3]
    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']

    def get_y_positions(n, center=4, spread=1.2):
        return center + np.linspace(-(n-1)*spread/2, (n-1)*spread/2, n)

    np.random.seed(42)

    # Pre-generate dropout masks for animation
    n_frames = 40
    dropout_masks = []
    for _ in range(n_frames):
        masks = []
        for size in layer_sizes:
            if size == layer_sizes[0] or size == layer_sizes[-1]:
                masks.append(np.ones(size, dtype=bool))  # No dropout on input/output
            else:
                masks.append(np.random.random(size) > 0.5)  # 50% dropout
        dropout_masks.append(masks)

    def animate(frame):
        ax.clear()
        ax.set_xlim(0, 13)
        ax.set_ylim(-1, 9)
        ax.axis('off')

        masks = dropout_masks[frame]

        # Draw connections first (behind nodes)
        for l in range(len(layer_x) - 1):
            y1 = get_y_positions(layer_sizes[l])
            y2 = get_y_positions(layer_sizes[l + 1])
            mask1 = masks[l]
            mask2 = masks[l + 1]

            for i, yi in enumerate(y1):
                for j, yj in enumerate(y2):
                    if mask1[i] and mask2[j]:
                        ax.plot([layer_x[l], layer_x[l + 1]], [yi, yj],
                               color='gray', alpha=0.3, linewidth=0.8, zorder=1)

        # Draw nodes
        for l, (x, size, name) in enumerate(zip(layer_x, layer_sizes, layer_names)):
            y_positions = get_y_positions(size)
            mask = masks[l]

            for i, y in enumerate(y_positions):
                if mask[i]:
                    color = COLORS['active']
                    alpha = 1.0
                else:
                    color = COLORS['dropped']
                    alpha = 0.4

                circle = Circle((x, y), 0.3, facecolor=color, edgecolor='black',
                               linewidth=2, alpha=alpha, zorder=2)
                ax.add_patch(circle)

                if not mask[i]:
                    # Draw X for dropped neurons
                    ax.plot([x-0.15, x+0.15], [y-0.15, y+0.15],
                           color='red', linewidth=2, alpha=0.8, zorder=3)
                    ax.plot([x-0.15, x+0.15], [y+0.15, y-0.15],
                           color='red', linewidth=2, alpha=0.8, zorder=3)

            ax.text(x, 8.2, name, ha='center', va='center', fontsize=11, fontweight='bold')

            # Show dropout rate
            if l > 0 and l < len(layer_sizes) - 1:
                dropped = np.sum(~mask)
                ax.text(x, -0.3, f'Dropped: {dropped}/{size}', ha='center', fontsize=9)

        # Title and explanation
        ax.set_title(f'Dropout in Action (Frame {frame+1}/{n_frames})', fontsize=14, fontweight='bold')

        # Legend box
        legend_box = FancyBboxPatch((0.3, 6.5), 3, 1.8, boxstyle="round,pad=0.1",
                                     facecolor='white', edgecolor='black', alpha=0.9)
        ax.add_patch(legend_box)
        ax.add_patch(Circle((0.7, 7.8), 0.2, facecolor=COLORS['active'], edgecolor='black'))
        ax.text(1.0, 7.8, 'Active', va='center', fontsize=9)
        ax.add_patch(Circle((0.7, 7.2), 0.2, facecolor=COLORS['dropped'], edgecolor='black', alpha=0.4))
        ax.text(1.0, 7.2, 'Dropped (p=0.5)', va='center', fontsize=9)

        # Formula
        ax.text(6.5, -0.8, 'Training: $y = \\frac{x \\cdot mask}{1-p}$ | Inference: $y = x$ (no dropout)',
               ha='center', fontsize=10, style='italic')

        return []

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=250, blit=False)
    writer = PillowWriter(fps=4)
    anim.save(os.path.join(OUTPUT_DIR, 'reg_dropout.gif'), writer=writer)
    plt.close()
    print("Generated: reg_dropout.gif")


def create_early_stopping():
    """Create early stopping visualization."""
    fig, ax = plt.subplots(figsize=(12, 7))

    np.random.seed(42)
    epochs = np.arange(1, 101)

    # Training loss: continues to decrease
    train_loss = 0.8 * np.exp(-epochs / 25) + 0.02 + np.random.randn(100) * 0.01

    # Validation loss: decreases then increases (overfitting)
    val_loss = 0.9 * np.exp(-epochs / 30) + 0.1 + 0.25 * (1 - np.exp(-(epochs - 20) / 30))
    val_loss = np.maximum(val_loss, 0.08)
    val_loss += np.random.randn(100) * 0.015

    # Find best validation loss
    best_epoch = np.argmin(val_loss) + 1
    best_val_loss = val_loss[best_epoch - 1]

    # Early stopping epoch (with patience)
    patience = 10
    stop_epoch = best_epoch + patience

    # Plot full curves (faded after early stop)
    ax.plot(epochs[:stop_epoch], train_loss[:stop_epoch], color=COLORS['train'],
            linewidth=2, label='Training Loss')
    ax.plot(epochs[stop_epoch-1:], train_loss[stop_epoch-1:], color=COLORS['train'],
            linewidth=2, linestyle='--', alpha=0.3)

    ax.plot(epochs[:stop_epoch], val_loss[:stop_epoch], color=COLORS['val'],
            linewidth=2, label='Validation Loss')
    ax.plot(epochs[stop_epoch-1:], val_loss[stop_epoch-1:], color=COLORS['val'],
            linewidth=2, linestyle='--', alpha=0.3)

    # Mark best checkpoint
    ax.scatter([best_epoch], [best_val_loss], color=COLORS['best'], s=200,
               marker='*', edgecolors='black', linewidth=2, zorder=5,
               label=f'Best Model (Epoch {best_epoch})')

    # Mark early stopping point
    ax.axvline(x=stop_epoch, color=COLORS['stopped'], linestyle='--', linewidth=2,
               label=f'Early Stop (Epoch {stop_epoch})')

    # Patience region
    ax.axvspan(best_epoch, stop_epoch, alpha=0.2, color='yellow')
    ax.text((best_epoch + stop_epoch) / 2, 0.65, f'Patience\n({patience} epochs)',
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='orange'))

    # Annotations
    ax.annotate('Best checkpoint saved', xy=(best_epoch, best_val_loss),
                xytext=(best_epoch + 15, best_val_loss - 0.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['best']),
                fontsize=10, color=COLORS['best'])

    ax.annotate('Would overfit\nif continued', xy=(80, val_loss[79]),
                xytext=(85, val_loss[79] + 0.15),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray', alpha=0.7)

    # Add savings annotation
    savings_text = f"Training stopped at epoch {stop_epoch}\nSaved {100-stop_epoch} epochs!"
    ax.text(75, 0.08, savings_text, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green'))

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Early Stopping: Prevent Overfitting & Save Compute', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3)

    # Algorithm box
    algo_text = ("Algorithm:\n"
                 "1. Track best validation loss\n"
                 "2. Save model at each new best\n"
                 "3. Count epochs without improvement\n"
                 "4. Stop if count > patience\n"
                 "5. Load best saved model")
    ax.text(5, 0.55, algo_text, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'),
            verticalalignment='top')

    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, 'reg_early_stopping.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: reg_early_stopping.svg")


if __name__ == '__main__':
    print("Generating Regularization visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    create_overfitting_curves()
    create_l1_l2_comparison()
    create_dropout_animation()
    create_early_stopping()

    print("\nAll regularization visualizations generated!")
