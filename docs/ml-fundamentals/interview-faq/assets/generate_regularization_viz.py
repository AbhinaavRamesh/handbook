"""
Generate visualizations for Regularization FAQ
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Common styling
FIGSIZE = (10, 6)
DPI = 150
COLORS = {
    'primary': '#2563eb',
    'secondary': '#dc2626',
    'tertiary': '#16a34a',
    'quaternary': '#9333ea',
    'fill': '#93c5fd',
}


def plot_l1_l2_geometry():
    """Generate L1 vs L2 geometry visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI)

    # L1 constraint (diamond)
    ax = axes[0]

    # Draw loss contours
    w1 = np.linspace(-3, 3, 100)
    w2 = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1, w2)
    # Loss centered at (2, 1)
    Loss = (W1 - 2)**2 + (W2 - 1)**2
    ax.contour(W1, W2, Loss, levels=15, cmap='Blues', alpha=0.7)

    # Draw L1 constraint (diamond)
    diamond = Polygon([(-1.5, 0), (0, 1.5), (1.5, 0), (0, -1.5)],
                       fill=True, facecolor=COLORS['fill'], edgecolor=COLORS['primary'], linewidth=2, alpha=0.5)
    ax.add_patch(diamond)

    # Mark optimal point (on corner/edge)
    ax.scatter([1.5], [0], color=COLORS['secondary'], s=200, zorder=5, marker='*')
    ax.annotate('Optimal\n(w₂ = 0, sparse!)', xy=(1.5, 0), xytext=(2.2, 0.8),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlabel('w₁', fontsize=12)
    ax.set_ylabel('w₂', fontsize=12)
    ax.set_title('L1 Regularization (Lasso)\nDiamond constraint → solutions at corners (sparse)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')

    # L2 constraint (circle)
    ax = axes[1]

    # Draw same loss contours
    ax.contour(W1, W2, Loss, levels=15, cmap='Blues', alpha=0.7)

    # Draw L2 constraint (circle)
    circle = Circle((0, 0), 1.5, fill=True, facecolor=COLORS['fill'],
                     edgecolor=COLORS['primary'], linewidth=2, alpha=0.5)
    ax.add_patch(circle)

    # Mark optimal point (on circle boundary)
    opt_x = 1.5 * 2 / np.sqrt(5)
    opt_y = 1.5 * 1 / np.sqrt(5)
    ax.scatter([opt_x], [opt_y], color=COLORS['secondary'], s=200, zorder=5, marker='*')
    ax.annotate('Optimal\n(both w₁, w₂ ≠ 0)', xy=(opt_x, opt_y), xytext=(1.8, 1.5),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlabel('w₁', fontsize=12)
    ax.set_ylabel('w₂', fontsize=12)
    ax.set_title('L2 Regularization (Ridge)\nCircular constraint → smooth shrinkage (not sparse)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')

    plt.suptitle('Geometric Interpretation: Why L1 Produces Sparse Solutions',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('regularization_l1_l2_geometry.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: regularization_l1_l2_geometry.png")


def plot_penalty_curves():
    """Generate L1 vs L2 penalty curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

    w = np.linspace(-3, 3, 100)

    # Penalty functions
    ax = axes[0]
    l1_penalty = np.abs(w)
    l2_penalty = w**2

    ax.plot(w, l1_penalty, label='L1: |w|', color=COLORS['primary'], linewidth=2.5)
    ax.plot(w, l2_penalty, label='L2: w²', color=COLORS['secondary'], linewidth=2.5)

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlabel('Weight Value (w)', fontsize=11)
    ax.set_ylabel('Penalty', fontsize=11)
    ax.set_title('Penalty Functions', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 5)

    # Gradient of penalty (derivative)
    ax = axes[1]
    l1_grad = np.sign(w)
    l1_grad[w == 0] = 0
    l2_grad = 2 * w

    ax.plot(w, l1_grad, label='L1 gradient: sign(w)', color=COLORS['primary'], linewidth=2.5)
    ax.plot(w, l2_grad, label='L2 gradient: 2w', color=COLORS['secondary'], linewidth=2.5)

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlabel('Weight Value (w)', fontsize=11)
    ax.set_ylabel('Gradient', fontsize=11)
    ax.set_title('Gradient of Penalties\nL1 has constant gradient (pushes to exactly 0)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    # Annotate key insight
    ax.annotate('L1: Constant push\nregardless of w size', xy=(-1.5, -1), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.annotate('L2: Smaller push\nas w → 0', xy=(1, 1), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Why L1 Drives Weights to Exactly Zero', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('regularization_penalty_curves.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: regularization_penalty_curves.png")


def plot_coefficient_paths():
    """Generate coefficient paths as regularization strength varies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

    np.random.seed(42)

    # Simulate coefficient paths
    alphas = np.logspace(-3, 2, 100)
    n_features = 6

    # L1 (Lasso) - coefficients go to exactly zero
    ax = axes[0]
    for i in range(n_features):
        coef_start = np.random.uniform(-2, 2)
        threshold = np.random.uniform(0.1, 10)
        coefs = coef_start * np.maximum(0, 1 - alphas / threshold)
        ax.plot(alphas, coefs, linewidth=2, label=f'Feature {i+1}')

    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('Regularization Strength (λ)', fontsize=11)
    ax.set_ylabel('Coefficient Value', fontsize=11)
    ax.set_title('L1 (Lasso): Coefficients go to exactly 0', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # L2 (Ridge) - coefficients shrink but don't reach zero
    ax = axes[1]
    for i in range(n_features):
        coef_start = np.random.uniform(-2, 2)
        coefs = coef_start / (1 + alphas * np.random.uniform(0.5, 2))
        ax.plot(alphas, coefs, linewidth=2, label=f'Feature {i+1}')

    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('Regularization Strength (λ)', fontsize=11)
    ax.set_ylabel('Coefficient Value', fontsize=11)
    ax.set_title('L2 (Ridge): Coefficients shrink but never reach 0', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    plt.suptitle('Coefficient Paths: How Weights Change with Regularization', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('regularization_coefficient_paths.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: regularization_coefficient_paths.png")


def plot_dropout():
    """Generate dropout mechanism visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI)

    np.random.seed(42)

    def draw_network(ax, title, dropout_mask=None):
        layer_sizes = [4, 6, 6, 3]

        # Node positions
        positions = []
        for l, size in enumerate(layer_sizes):
            x = l * 2
            for i in range(size):
                y = (size - 1) / 2 - i
                positions.append((x, y, l, i))

        # Draw connections
        for l in range(len(layer_sizes) - 1):
            for i in range(layer_sizes[l]):
                for j in range(layer_sizes[l + 1]):
                    x1, y1 = l * 2, (layer_sizes[l] - 1) / 2 - i
                    x2, y2 = (l + 1) * 2, (layer_sizes[l + 1] - 1) / 2 - j

                    # Check if connection is active
                    if dropout_mask is not None:
                        if not dropout_mask.get((l, i), True) or not dropout_mask.get((l+1, j), True):
                            alpha = 0.1
                            color = 'lightgray'
                        else:
                            alpha = 0.5
                            color = 'gray'
                    else:
                        alpha = 0.3
                        color = 'gray'

                    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=0.5)

        # Draw nodes
        for x, y, l, i in positions:
            if dropout_mask is not None and not dropout_mask.get((l, i), True):
                color = 'lightgray'
                edgecolor = 'gray'
            else:
                color = COLORS['primary'] if l == 0 else (COLORS['tertiary'] if l == len(layer_sizes)-1 else COLORS['secondary'])
                edgecolor = 'black'

            circle = Circle((x, y), 0.25, facecolor=color, edgecolor=edgecolor, linewidth=1.5)
            ax.add_patch(circle)

        ax.set_xlim(-1, 7)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold')

    # Without dropout
    draw_network(axes[0], 'Training WITHOUT Dropout\n(All neurons active)')

    # With dropout
    dropout_mask = {}
    layer_sizes = [4, 6, 6, 3]
    for l in range(1, len(layer_sizes) - 1):  # Don't drop input/output
        for i in range(layer_sizes[l]):
            dropout_mask[(l, i)] = np.random.random() > 0.5  # 50% dropout

    draw_network(axes[1], 'Training WITH Dropout (p=0.5)\n(Random neurons deactivated)')

    # Add legend
    axes[1].text(3, -3, '❌ Dropped neurons shown in gray\nForces network to learn redundant representations',
                 ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Dropout: A Form of Ensemble Regularization', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('regularization_dropout.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: regularization_dropout.png")


def plot_early_stopping():
    """Generate early stopping visualization."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    epochs = np.arange(1, 101)

    # Training loss decreases continuously
    train_loss = 2.0 * np.exp(-0.05 * epochs) + 0.1

    # Validation loss decreases then increases (overfitting)
    val_loss = 2.0 * np.exp(-0.05 * epochs) + 0.15 + 0.003 * (epochs - 30)**2 * (epochs > 30)

    ax.plot(epochs, train_loss, label='Training Loss', color=COLORS['primary'], linewidth=2.5)
    ax.plot(epochs, val_loss, label='Validation Loss', color=COLORS['secondary'], linewidth=2.5)

    # Mark early stopping point
    best_epoch = 30
    ax.axvline(x=best_epoch, color=COLORS['tertiary'], linestyle='--', linewidth=2, label='Early Stopping Point')
    ax.scatter([best_epoch], [val_loss[best_epoch-1]], color=COLORS['tertiary'], s=200, zorder=5, marker='*')

    # Shade overfitting region
    ax.fill_between(epochs[best_epoch:], train_loss[best_epoch:], val_loss[best_epoch:],
                    alpha=0.2, color=COLORS['secondary'], label='Overfitting Gap')

    ax.annotate('Stop here!\nBest validation loss', xy=(best_epoch, val_loss[best_epoch-1]),
                xytext=(best_epoch+15, val_loss[best_epoch-1]+0.3),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.annotate('Overfitting begins', xy=(50, val_loss[49]),
                xytext=(60, val_loss[49]+0.3),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Early Stopping: Regularization by Limiting Training Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 2.5)

    plt.tight_layout()
    plt.savefig('regularization_early_stopping.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: regularization_early_stopping.png")


def plot_comparison():
    """Generate regularization methods comparison."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=DPI)

    methods = [
        ('L1 (Lasso)', '|w|', 'Yes', 'Feature selection', 'Linear/Logistic'),
        ('L2 (Ridge)', 'w²', 'No', 'Multicollinearity', 'Linear/Logistic'),
        ('Elastic Net', 'α|w| + (1-α)w²', 'Yes', 'Groups of features', 'Linear'),
        ('Dropout', 'Random neuron drop', 'Implicit', 'Co-adaptation', 'Neural Networks'),
        ('Early Stopping', 'Limit epochs', 'Implicit', 'Overfitting', 'Any iterative'),
        ('Data Augmentation', 'Expand dataset', 'No', 'Limited data', 'Images/NLP'),
    ]

    columns = ['Method', 'Penalty', 'Sparse?', 'Best For', 'Used In']

    ax.axis('off')

    table = ax.table(
        cellText=methods,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=[COLORS['primary']] * 5,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2)

    # Style header
    for i in range(5):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Regularization Methods Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('regularization_comparison.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: regularization_comparison.png")


if __name__ == '__main__':
    print("Generating Regularization visualizations...")
    plot_l1_l2_geometry()
    plot_penalty_curves()
    plot_coefficient_paths()
    plot_dropout()
    plot_early_stopping()
    plot_comparison()
    print("Done!")
