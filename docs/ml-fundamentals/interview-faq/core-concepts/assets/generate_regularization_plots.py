"""
Regularization Visualizations (L1, L2, Dropout)
Generates plots for ML interview preparation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12

output_dir = os.path.dirname(os.path.abspath(__file__))


def plot_l1_l2_geometry():
    """Show geometric interpretation of L1 vs L2 constraints"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create loss function contours
    w1 = np.linspace(-3, 3, 100)
    w2 = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1, w2)

    # Loss function (elliptical contours, minimum away from origin)
    loss_center = (1.5, 1.5)
    Loss = 0.5 * ((W1 - loss_center[0])**2 + 2*(W2 - loss_center[1])**2)

    # L1 constraint (diamond)
    ax1 = axes[0]
    ax1.contour(W1, W2, Loss, levels=15, cmap='Blues', alpha=0.7)

    # Draw diamond (L1 ball)
    radius = 1.0
    diamond = Polygon([
        (radius, 0), (0, radius), (-radius, 0), (0, -radius)
    ], fill=True, facecolor='red', alpha=0.3, edgecolor='red', linewidth=2)
    ax1.add_patch(diamond)

    # Mark intersection point (at corner = sparse solution)
    ax1.scatter([radius], [0], s=200, c='green', marker='*', zorder=10,
                edgecolors='black', linewidth=2, label='Optimal (sparse)')
    ax1.scatter([loss_center[0]], [loss_center[1]], s=100, c='blue', marker='o',
                zorder=10, label='Unconstrained min')

    ax1.set_xlabel('w₁', fontsize=13)
    ax1.set_ylabel('w₂', fontsize=13)
    ax1.set_title('L1 Regularization (Lasso)\n|w₁| + |w₂| ≤ t', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    ax1.annotate('w₂ = 0\n(Feature removed!)', xy=(1.0, 0.1),
                 fontsize=11, color='green', fontweight='bold')

    # L2 constraint (circle)
    ax2 = axes[1]
    ax2.contour(W1, W2, Loss, levels=15, cmap='Blues', alpha=0.7)

    # Draw circle (L2 ball)
    circle = Circle((0, 0), radius, fill=True, facecolor='orange',
                    alpha=0.3, edgecolor='orange', linewidth=2)
    ax2.add_patch(circle)

    # Mark intersection point (not on axis = no sparsity)
    intersect_angle = np.arctan2(loss_center[1], loss_center[0])
    intersect_x = radius * np.cos(intersect_angle)
    intersect_y = radius * np.sin(intersect_angle)
    ax2.scatter([intersect_x], [intersect_y], s=200, c='green', marker='*',
                zorder=10, edgecolors='black', linewidth=2, label='Optimal (not sparse)')
    ax2.scatter([loss_center[0]], [loss_center[1]], s=100, c='blue', marker='o',
                zorder=10, label='Unconstrained min')

    ax2.set_xlabel('w₁', fontsize=13)
    ax2.set_ylabel('w₂', fontsize=13)
    ax2.set_title('L2 Regularization (Ridge)\nw₁² + w₂² ≤ t', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    ax2.annotate('Both weights\nare non-zero', xy=(intersect_x + 0.1, intersect_y + 0.3),
                 fontsize=11, color='orange', fontweight='bold')

    plt.suptitle('Geometric Interpretation: Why L1 Produces Sparsity',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regularization_l1_l2_geometry.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: regularization_l1_l2_geometry.png")


def plot_penalty_curves():
    """Show L1 vs L2 penalty as function of weight"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    w = np.linspace(-3, 3, 100)

    # Penalty values
    l1_penalty = np.abs(w)
    l2_penalty = w ** 2

    # Left: Penalty curves
    ax1 = axes[0]
    ax1.plot(w, l1_penalty, 'b-', linewidth=3, label='L1: |w|')
    ax1.plot(w, l2_penalty, 'r-', linewidth=3, label='L2: w²')
    ax1.set_xlabel('Weight (w)', fontsize=13)
    ax1.set_ylabel('Penalty', fontsize=13)
    ax1.set_title('Penalty Functions: L1 vs L2', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    # Right: Gradient curves (derivative)
    ax2 = axes[1]

    # L1 gradient: sign function (with discontinuity at 0)
    l1_gradient = np.sign(w)
    l1_gradient[w == 0] = 0  # technically undefined

    # L2 gradient: 2w
    l2_gradient = 2 * w

    ax2.plot(w, l1_gradient, 'b-', linewidth=3, label='L1: sign(w)')
    ax2.plot(w, l2_gradient, 'r-', linewidth=3, label='L2: 2w')

    ax2.scatter([0], [0], s=100, c='blue', marker='o', zorder=5)
    ax2.annotate('Constant gradient!\nPushes toward 0', xy=(0.5, 0.8),
                 fontsize=11, color='blue', fontweight='bold')
    ax2.annotate('Gradient → 0\nas w → 0', xy=(0.3, 0.3),
                 fontsize=11, color='red', fontweight='bold')

    ax2.set_xlabel('Weight (w)', fontsize=13)
    ax2.set_ylabel('Gradient of Penalty', fontsize=13)
    ax2.set_title('Gradients: Why L1 Zeros Out Weights', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_ylim(-3, 3)

    plt.suptitle('L1 vs L2: Penalty and Gradient Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regularization_penalty_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: regularization_penalty_curves.png")


def plot_coefficient_paths():
    """Show how coefficients shrink with increasing lambda"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    np.random.seed(42)

    # Lambda values (log scale)
    lambdas = np.logspace(-3, 2, 100)

    # Simulate coefficient paths for 5 features
    n_features = 5
    initial_coefs = np.array([2.5, 1.8, 1.2, 0.5, 0.2])
    colors = plt.cm.viridis(np.linspace(0, 1, n_features))

    # L1 (Lasso) - coefficients go to exactly zero
    ax1 = axes[0]
    for i, (coef, color) in enumerate(zip(initial_coefs, colors)):
        # L1 path: soft thresholding
        lasso_coefs = np.maximum(0, coef - lambdas * 0.3)
        ax1.plot(lambdas, lasso_coefs, linewidth=2.5, color=color, label=f'Feature {i+1}')

    ax1.set_xscale('log')
    ax1.set_xlabel('λ (Regularization Strength)', fontsize=13)
    ax1.set_ylabel('Coefficient Value', fontsize=13)
    ax1.set_title('L1 (Lasso): Coefficients Hit Zero', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # L2 (Ridge) - coefficients approach but never reach zero
    ax2 = axes[1]
    for i, (coef, color) in enumerate(zip(initial_coefs, colors)):
        # L2 path: gradual shrinkage
        ridge_coefs = coef / (1 + lambdas * 0.5)
        ax2.plot(lambdas, ridge_coefs, linewidth=2.5, color=color, label=f'Feature {i+1}')

    ax2.set_xscale('log')
    ax2.set_xlabel('λ (Regularization Strength)', fontsize=13)
    ax2.set_ylabel('Coefficient Value', fontsize=13)
    ax2.set_title('L2 (Ridge): Coefficients Approach Zero', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Coefficient Paths: Feature Selection in L1 vs L2',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regularization_coefficient_paths.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: regularization_coefficient_paths.png")


def plot_dropout_visualization():
    """Visualize dropout in neural networks"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Helper function to draw a neural network layer
    def draw_network(ax, layer_sizes, title, dropout_mask=None):
        ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
        ax.set_ylim(-0.5, max(layer_sizes) - 0.5)

        for i, size in enumerate(layer_sizes):
            y_offset = (max(layer_sizes) - size) / 2
            for j in range(size):
                y = j + y_offset

                # Check if neuron is dropped
                is_dropped = False
                if dropout_mask is not None and i < len(dropout_mask):
                    if dropout_mask[i] is not None and j < len(dropout_mask[i]):
                        is_dropped = dropout_mask[i][j]

                color = 'lightgray' if is_dropped else 'skyblue'
                alpha = 0.3 if is_dropped else 1.0

                circle = plt.Circle((i, y), 0.15, color=color, ec='black',
                                    linewidth=1.5, alpha=alpha)
                ax.add_patch(circle)

                # Draw connections to next layer
                if i < len(layer_sizes) - 1:
                    next_size = layer_sizes[i + 1]
                    next_y_offset = (max(layer_sizes) - next_size) / 2
                    for k in range(next_size):
                        next_y = k + next_y_offset

                        # Check if connection is dropped
                        conn_dropped = is_dropped
                        if dropout_mask is not None and i + 1 < len(dropout_mask):
                            if dropout_mask[i + 1] is not None and k < len(dropout_mask[i + 1]):
                                conn_dropped = conn_dropped or dropout_mask[i + 1][k]

                        line_alpha = 0.1 if conn_dropped else 0.4
                        ax.plot([i + 0.15, i + 0.85], [y, next_y], 'k-',
                               alpha=line_alpha, linewidth=0.5)

        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=13, fontweight='bold')

    # Full network (no dropout)
    layer_sizes = [4, 6, 6, 3]
    draw_network(axes[0], layer_sizes, 'Standard Network (Inference)')

    # Network with dropout
    np.random.seed(42)
    dropout_mask = [
        None,  # Input layer - no dropout
        [np.random.random() < 0.3 for _ in range(6)],  # Hidden 1
        [np.random.random() < 0.3 for _ in range(6)],  # Hidden 2
        None   # Output layer - no dropout
    ]
    draw_network(axes[1], layer_sizes, 'Network with Dropout (Training)\np = 0.3', dropout_mask)

    axes[1].text(0.5, -0.3, 'Grayed neurons are "dropped out"',
                 transform=axes[1].transAxes, fontsize=11,
                 ha='center', style='italic', color='gray')

    plt.suptitle('Dropout Regularization: Training vs Inference',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regularization_dropout.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: regularization_dropout.png")


def plot_early_stopping():
    """Visualize early stopping"""
    fig, ax = plt.subplots(figsize=(10, 7))

    np.random.seed(42)

    # Simulate training and validation loss
    epochs = np.arange(1, 101)

    # Training loss - keeps decreasing
    train_loss = 2 * np.exp(-0.05 * epochs) + 0.1 + np.random.randn(100) * 0.02

    # Validation loss - decreases then increases (overfitting)
    val_loss = 2 * np.exp(-0.04 * epochs) + 0.15 + 0.005 * (epochs - 30) ** 2 * (epochs > 30)
    val_loss += np.random.randn(100) * 0.03

    ax.plot(epochs, train_loss, 'b-', linewidth=2.5, label='Training Loss')
    ax.plot(epochs, val_loss, 'r-', linewidth=2.5, label='Validation Loss')

    # Find optimal stopping point
    optimal_epoch = np.argmin(val_loss[:60]) + 1
    ax.axvline(x=optimal_epoch, color='green', linestyle='--', linewidth=2,
               label=f'Early Stopping (epoch {optimal_epoch})')
    ax.scatter([optimal_epoch], [val_loss[optimal_epoch-1]], s=200, c='green',
               marker='*', zorder=5, edgecolors='black', linewidth=2)

    # Mark overfitting region
    ax.fill_between(epochs[optimal_epoch:], 0, 3, alpha=0.1, color='red')
    ax.annotate('OVERFITTING ZONE', xy=(70, 2.5), fontsize=12,
                color='red', fontweight='bold', ha='center')

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Early Stopping: Prevent Overfitting by Stopping at Optimal Epoch',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regularization_early_stopping.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: regularization_early_stopping.png")


def plot_regularization_comparison():
    """Create a summary comparison of regularization techniques"""
    fig, ax = plt.subplots(figsize=(12, 8))

    techniques = ['No Regularization', 'L1 (Lasso)', 'L2 (Ridge)', 'Elastic Net', 'Dropout', 'Early Stopping']
    categories = ['Reduces\nOverfitting', 'Feature\nSelection', 'Handles\nMulticollinearity', 'Computational\nCost']

    # Scores (0-5 scale) for each technique on each category
    scores = np.array([
        [1, 0, 0, 5],  # No regularization
        [4, 5, 2, 3],  # L1
        [4, 0, 5, 4],  # L2
        [5, 4, 4, 2],  # Elastic Net
        [5, 0, 2, 3],  # Dropout
        [4, 0, 1, 5],  # Early Stopping
    ])

    # Create heatmap
    im = ax.imshow(scores, cmap='RdYlGn', aspect='auto', vmin=0, vmax=5)

    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(techniques)))
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_yticklabels(techniques, fontsize=11)

    # Add text annotations
    for i in range(len(techniques)):
        for j in range(len(categories)):
            text = ax.text(j, i, scores[i, j], ha='center', va='center',
                          fontsize=12, fontweight='bold',
                          color='white' if scores[i, j] < 2.5 else 'black')

    ax.set_title('Regularization Techniques Comparison\n(Score: 0=Poor, 5=Excellent)',
                 fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Effectiveness Score', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regularization_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: regularization_comparison.png")


if __name__ == '__main__':
    print("Generating Regularization visualizations...")
    plot_l1_l2_geometry()
    plot_penalty_curves()
    plot_coefficient_paths()
    plot_dropout_visualization()
    plot_early_stopping()
    plot_regularization_comparison()
    print("\nAll visualizations generated successfully!")
