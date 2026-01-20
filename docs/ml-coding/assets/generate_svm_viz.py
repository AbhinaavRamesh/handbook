#!/usr/bin/env python3
"""
SVM Visualization Generator

This script generates educational visualizations for Support Vector Machines:
1. Maximum margin hyperplane with support vectors highlighted
2. Kernel trick visualization (2D non-separable to 3D separable)
3. Soft margin C parameter effect (multiple panels)
4. Kernel comparison (linear, poly, RBF, sigmoid) 2x2 grid

Outputs:
- svm_maximum_margin.png: Maximum margin hyperplane with support vectors
- svm_kernel_trick.png: Kernel trick transformation visualization
- svm_c_parameter.png: Effect of C parameter on decision boundary
- svm_kernel_comparison.png: Comparison of different kernel functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, art3d
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons, make_blobs
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Set random seed for reproducibility
np.random.seed(42)

# Color palette (colorblind-friendly)
COLORS = {
    'class_pos': '#E63946',      # Red for positive class
    'class_neg': '#457B9D',      # Blue for negative class
    'support_vector': '#2A9D8F', # Teal for support vectors
    'hyperplane': '#1D3557',     # Dark blue for hyperplane
    'margin': '#F4A261',         # Orange for margins
    'background': '#F8F9FA'      # Light gray background
}

# Output directory
OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-coding/assets/'


def generate_linear_separable_data(n_samples=100, margin=1.0, random_state=42):
    """Generate linearly separable 2D data with a clear margin."""
    np.random.seed(random_state)

    n_each = n_samples // 2

    # Class 1 (positive): points above the line y = x + margin
    X_pos = np.random.randn(n_each, 2)
    X_pos[:, 1] += X_pos[:, 0] + margin

    # Class 2 (negative): points below the line y = x - margin
    X_neg = np.random.randn(n_each, 2)
    X_neg[:, 1] += X_neg[:, 0] - margin

    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_each + [-1] * n_each)

    # Shuffle
    perm = np.random.permutation(n_samples)
    return X[perm], y[perm]


def create_maximum_margin_plot(output_path=None):
    """
    Create visualization of maximum margin hyperplane with support vectors.

    Shows:
    - Data points from two classes
    - Decision boundary (hyperplane)
    - Margin boundaries (dashed lines)
    - Support vectors highlighted with circles
    - Margin width annotation
    """
    print("Generating maximum margin hyperplane visualization...")

    # Generate data
    X, y = generate_linear_separable_data(n_samples=80, margin=2.0, random_state=42)

    # Train SVM
    clf = SVC(kernel='linear', C=1000)  # Large C for hard margin behavior
    clf.fit(X, y)

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.set_facecolor(COLORS['background'])

    # Compute bounds
    margin_pad = 1.5
    x_min, x_max = X[:, 0].min() - margin_pad, X[:, 0].max() + margin_pad
    y_min, y_max = X[:, 1].min() - margin_pad, X[:, 1].max() + margin_pad

    # Create mesh for decision function
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision regions with gradient
    levels = np.linspace(-3, 3, 30)
    contourf = ax.contourf(xx, yy, Z, levels=levels, cmap='RdBu', alpha=0.3)

    # Plot decision boundary (hyperplane) and margins
    ax.contour(xx, yy, Z, levels=[-1], colors=[COLORS['margin']],
               linestyles=['--'], linewidths=2, alpha=0.8)
    ax.contour(xx, yy, Z, levels=[0], colors=[COLORS['hyperplane']],
               linestyles=['-'], linewidths=3)
    ax.contour(xx, yy, Z, levels=[1], colors=[COLORS['margin']],
               linestyles=['--'], linewidths=2, alpha=0.8)

    # Plot data points
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=COLORS['class_pos'],
               marker='o', s=80, edgecolors='white', linewidth=1.5,
               label='Class +1', zorder=5)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c=COLORS['class_neg'],
               marker='s', s=80, edgecolors='white', linewidth=1.5,
               label='Class -1', zorder=5)

    # Highlight support vectors
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=250, facecolors='none',
               edgecolors=COLORS['support_vector'], linewidths=3,
               label=f'Support Vectors (n={len(sv)})', zorder=10)

    # Add margin width annotation
    # Find a support vector and draw margin
    w = clf.coef_[0]
    w_norm = np.linalg.norm(w)
    margin_width = 2 / w_norm

    # Draw double-headed arrow for margin
    sv_idx = 0
    sv_point = sv[sv_idx]

    # Calculate perpendicular direction
    perp_dir = w / w_norm

    # Find point on hyperplane closest to this support vector
    # Project support vector onto hyperplane
    dist_to_hyperplane = clf.decision_function(sv_point.reshape(1, -1))[0]
    hyperplane_point = sv_point - dist_to_hyperplane * perp_dir

    # Add annotation for margin
    mid_point = hyperplane_point + 0.5 * perp_dir
    ax.annotate(f'Margin = {margin_width:.2f}',
                xy=mid_point, xytext=(mid_point[0] + 1.5, mid_point[1] + 0.5),
                fontsize=11, fontweight='bold', color=COLORS['hyperplane'],
                arrowprops=dict(arrowstyle='->', color=COLORS['hyperplane'], lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # Labels and title
    ax.set_xlabel('Feature $x_1$', fontsize=12)
    ax.set_ylabel('Feature $x_2$', fontsize=12)
    ax.set_title('SVM Maximum Margin Hyperplane\n'
                 'Decision Boundary: $\\mathbf{w}^T\\mathbf{x} + b = 0$',
                 fontsize=14, fontweight='bold', pad=15)

    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

    # Add text annotations for boundary lines
    ax.text(x_max - 0.5, y_min + 0.5, '$\\mathbf{w}^T\\mathbf{x} + b = -1$',
            fontsize=10, color=COLORS['margin'], fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(x_max - 0.5, y_max - 0.5, '$\\mathbf{w}^T\\mathbf{x} + b = +1$',
            fontsize=10, color=COLORS['margin'], fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['background'])
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_kernel_trick_plot(output_path=None):
    """
    Create visualization showing the kernel trick:
    - Left: 2D non-linearly separable data (circles)
    - Right: 3D representation after transformation where data becomes separable

    Shows how phi(x) = [x1, x2, x1^2 + x2^2] transforms circular data to separable.
    """
    print("Generating kernel trick visualization...")

    # Generate circular data (not linearly separable in 2D)
    np.random.seed(42)
    X, y = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=42)

    # Map labels to -1, +1
    y = np.where(y == 0, -1, 1)

    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 6), dpi=150)

    # Left subplot: 2D view
    ax1 = fig.add_subplot(121)
    ax1.set_facecolor(COLORS['background'])

    # Plot 2D data
    ax1.scatter(X[y == 1, 0], X[y == 1, 1], c=COLORS['class_pos'],
                marker='o', s=60, edgecolors='white', linewidth=1,
                label='Class +1 (inner)', alpha=0.8)
    ax1.scatter(X[y == -1, 0], X[y == -1, 1], c=COLORS['class_neg'],
                marker='s', s=60, edgecolors='white', linewidth=1,
                label='Class -1 (outer)', alpha=0.8)

    # Add text showing no linear separator exists
    ax1.text(0, 0, 'No linear\nseparator!', fontsize=12, ha='center', va='center',
             fontweight='bold', color=COLORS['hyperplane'],
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

    ax1.set_xlabel('$x_1$', fontsize=12)
    ax1.set_ylabel('$x_2$', fontsize=12)
    ax1.set_title('Original 2D Space\nNon-linearly Separable',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(-1.8, 1.8)
    ax1.set_ylim(-1.8, 1.8)
    ax1.set_aspect('equal')

    # Right subplot: 3D view after transformation
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor(COLORS['background'])

    # Apply transformation: phi(x) = [x1, x2, x1^2 + x2^2]
    X_transformed = np.column_stack([X[:, 0], X[:, 1], X[:, 0]**2 + X[:, 1]**2])

    # Plot 3D transformed data
    ax2.scatter(X_transformed[y == 1, 0], X_transformed[y == 1, 1],
                X_transformed[y == 1, 2], c=COLORS['class_pos'],
                marker='o', s=40, edgecolors='white', linewidth=0.5,
                label='Class +1', alpha=0.8)
    ax2.scatter(X_transformed[y == -1, 0], X_transformed[y == -1, 1],
                X_transformed[y == -1, 2], c=COLORS['class_neg'],
                marker='s', s=40, edgecolors='white', linewidth=0.5,
                label='Class -1', alpha=0.8)

    # Add separating hyperplane in 3D
    # The hyperplane is z = constant (around 0.3-0.4)
    xx_plane, yy_plane = np.meshgrid(np.linspace(-1.5, 1.5, 20),
                                      np.linspace(-1.5, 1.5, 20))
    zz_plane = np.ones_like(xx_plane) * 0.35

    # Only plot where it's within the circle
    mask = xx_plane**2 + yy_plane**2 <= 2.0
    zz_plane = np.where(mask, zz_plane, np.nan)

    ax2.plot_surface(xx_plane, yy_plane, zz_plane, alpha=0.3,
                     color=COLORS['hyperplane'], edgecolor='none')

    ax2.set_xlabel('$x_1$', fontsize=11)
    ax2.set_ylabel('$x_2$', fontsize=11)
    ax2.set_zlabel('$\\phi_3 = x_1^2 + x_2^2$', fontsize=11)
    ax2.set_title('Feature Space (3D)\n$\\phi(\\mathbf{x}) = [x_1, x_2, x_1^2 + x_2^2]$',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)

    # Set view angle
    ax2.view_init(elev=20, azim=45)

    # Add arrow between plots indicating transformation
    fig.text(0.5, 0.5, r'$\longrightarrow$' + '\n' + r'$\phi(\mathbf{x})$', fontsize=16,
             ha='center', va='center', transform=fig.transFigure,
             fontweight='bold', color=COLORS['hyperplane'])

    # Add main title
    fig.suptitle('The Kernel Trick: Mapping to Higher Dimensions',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['background'])
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_c_parameter_plot(output_path=None):
    """
    Create visualization showing the effect of C parameter on SVM.

    Shows 4 panels with different C values:
    - Very small C (very soft margin, underfitting)
    - Small C (soft margin)
    - Medium C (balanced)
    - Large C (hard margin, overfitting to noise)
    """
    print("Generating C parameter effect visualization...")

    # Generate data with some noise/overlap
    np.random.seed(42)
    n_samples = 150

    # Create overlapping classes
    X1 = np.random.randn(n_samples // 2, 2) + np.array([1, 1])
    X2 = np.random.randn(n_samples // 2, 2) + np.array([-1, -1])

    # Add some outliers
    X_outliers = np.array([[2.5, -1.5], [-2.5, 1.5], [0, 2], [0, -2]])
    y_outliers = np.array([1, -1, 1, -1])

    X = np.vstack([X1, X2, X_outliers])
    y = np.array([1] * (n_samples // 2) + [-1] * (n_samples // 2) + list(y_outliers))

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Different C values
    C_values = [0.01, 0.1, 1.0, 100.0]
    titles = ['C = 0.01 (Very Soft Margin)\nHigh bias, low variance',
              'C = 0.1 (Soft Margin)\nBalanced, some slack allowed',
              'C = 1.0 (Default)\nGood bias-variance tradeoff',
              'C = 100 (Hard Margin)\nLow bias, high variance']

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    axes = axes.flatten()

    # Compute bounds
    margin_pad = 1.5
    x_min, x_max = X[:, 0].min() - margin_pad, X[:, 0].max() + margin_pad
    y_min, y_max = X[:, 1].min() - margin_pad, X[:, 1].max() + margin_pad

    # Create mesh
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))

    for ax, C, title in zip(axes, C_values, titles):
        ax.set_facecolor(COLORS['background'])

        # Train SVM with this C value
        clf = SVC(kernel='rbf', C=C, gamma=0.5)
        clf.fit(X, y)

        # Get decision function
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision regions
        ax.contourf(xx, yy, Z, levels=np.linspace(-2, 2, 20),
                    cmap='RdBu', alpha=0.4)

        # Plot decision boundary and margins
        ax.contour(xx, yy, Z, levels=[-1, 0, 1],
                   colors=[COLORS['margin'], COLORS['hyperplane'], COLORS['margin']],
                   linestyles=['--', '-', '--'], linewidths=[1.5, 2.5, 1.5])

        # Plot data points
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c=COLORS['class_pos'],
                   marker='o', s=50, edgecolors='white', linewidth=1,
                   alpha=0.8)
        ax.scatter(X[y == -1, 0], X[y == -1, 1], c=COLORS['class_neg'],
                   marker='s', s=50, edgecolors='white', linewidth=1,
                   alpha=0.8)

        # Highlight support vectors
        sv = clf.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=150, facecolors='none',
                   edgecolors=COLORS['support_vector'], linewidths=2)

        # Add number of support vectors
        ax.text(0.02, 0.98, f'Support Vectors: {len(sv)}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Add main title
    fig.suptitle('Effect of Regularization Parameter C on SVM Decision Boundary',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['background'])
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_kernel_comparison_plot(output_path=None):
    """
    Create 2x2 grid comparing different kernel functions:
    - Linear kernel
    - Polynomial kernel (degree 3)
    - RBF (Gaussian) kernel
    - Sigmoid kernel

    Uses the same non-linear dataset for fair comparison.
    """
    print("Generating kernel comparison visualization...")

    # Generate non-linear data (moons shape)
    np.random.seed(42)
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
    y = np.where(y == 0, -1, 1)

    # Kernel configurations
    kernels = [
        ('linear', {}, 'Linear Kernel\n$K(\\mathbf{x}, \\mathbf{y}) = \\mathbf{x}^T\\mathbf{y}$'),
        ('poly', {'degree': 3, 'gamma': 'auto', 'coef0': 1},
         'Polynomial Kernel (d=3)\n$K(\\mathbf{x}, \\mathbf{y}) = (\\gamma\\mathbf{x}^T\\mathbf{y} + r)^d$'),
        ('rbf', {'gamma': 2},
         'RBF (Gaussian) Kernel\n$K(\\mathbf{x}, \\mathbf{y}) = \\exp(-\\gamma||\\mathbf{x}-\\mathbf{y}||^2)$'),
        ('sigmoid', {'gamma': 0.5, 'coef0': 0},
         'Sigmoid Kernel\n$K(\\mathbf{x}, \\mathbf{y}) = \\tanh(\\gamma\\mathbf{x}^T\\mathbf{y} + r)$')
    ]

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    axes = axes.flatten()

    # Compute bounds
    margin_pad = 0.5
    x_min, x_max = X[:, 0].min() - margin_pad, X[:, 0].max() + margin_pad
    y_min, y_max = X[:, 1].min() - margin_pad, X[:, 1].max() + margin_pad

    # Create mesh
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    for ax, (kernel, params, title) in zip(axes, kernels):
        ax.set_facecolor(COLORS['background'])

        # Train SVM with this kernel
        clf = SVC(kernel=kernel, C=1.0, **params)
        clf.fit(X, y)

        # Get decision function
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision regions with filled contours
        ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 25),
                    cmap='RdBu', alpha=0.4)

        # Plot decision boundary
        ax.contour(xx, yy, Z, levels=[0], colors=[COLORS['hyperplane']],
                   linewidths=2.5)
        ax.contour(xx, yy, Z, levels=[-1, 1], colors=[COLORS['margin']],
                   linestyles='--', linewidths=1.5, alpha=0.8)

        # Plot data points
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c=COLORS['class_pos'],
                   marker='o', s=50, edgecolors='white', linewidth=1,
                   alpha=0.9, label='Class +1')
        ax.scatter(X[y == -1, 0], X[y == -1, 1], c=COLORS['class_neg'],
                   marker='s', s=50, edgecolors='white', linewidth=1,
                   alpha=0.9, label='Class -1')

        # Highlight support vectors
        sv = clf.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=150, facecolors='none',
                   edgecolors=COLORS['support_vector'], linewidths=2,
                   label=f'SVs (n={len(sv)})')

        # Compute accuracy
        accuracy = clf.score(X, y)

        # Add accuracy text
        ax.text(0.02, 0.98, f'Accuracy: {accuracy:.1%}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    # Add main title
    fig.suptitle('Comparison of SVM Kernel Functions on Non-Linear Data',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=COLORS['background'])
        print(f"  Saved: {output_path}")

    plt.close(fig)


def main():
    """Generate all SVM visualizations."""
    print("=" * 60)
    print("SVM Visualization Generator")
    print("=" * 60)
    print()

    # 1. Maximum margin hyperplane
    create_maximum_margin_plot(output_path=f'{OUTPUT_DIR}svm_maximum_margin.png')

    # 2. Kernel trick visualization
    create_kernel_trick_plot(output_path=f'{OUTPUT_DIR}svm_kernel_trick.png')

    # 3. C parameter effect
    create_c_parameter_plot(output_path=f'{OUTPUT_DIR}svm_c_parameter.png')

    # 4. Kernel comparison
    create_kernel_comparison_plot(output_path=f'{OUTPUT_DIR}svm_kernel_comparison.png')

    print()
    print("=" * 60)
    print("All SVM visualizations generated successfully!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"  1. {OUTPUT_DIR}svm_maximum_margin.png")
    print(f"  2. {OUTPUT_DIR}svm_kernel_trick.png")
    print(f"  3. {OUTPUT_DIR}svm_c_parameter.png")
    print(f"  4. {OUTPUT_DIR}svm_kernel_comparison.png")


if __name__ == '__main__':
    main()
