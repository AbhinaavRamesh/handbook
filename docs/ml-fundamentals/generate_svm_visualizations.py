"""
Generate SVM visualizations for the SVM documentation page.
Creates three SVG visualizations:
1. Linear SVM with maximum margin and support vectors
2. Kernel trick (linear vs RBF decision boundary)
3. Soft margin illustration with slack variables
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, make_circles
import os

# Set style for clean visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# Output directory
OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/images'


def plot_decision_boundary(clf, X, ax, h=0.02, alpha=0.3, cmap='coolwarm'):
    """Helper function to plot decision boundary and margins."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    ax.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#FFCCCC', '#CCCCFF'], alpha=0.5)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['#FF6B6B', '#333333', '#4ECDC4'],
               linestyles=['--', '-', '--'], linewidths=[1.5, 2, 1.5])

    return xx, yy


def generate_margin_visualization():
    """
    Visualization 1: Linear SVM with maximum margin and support vectors highlighted.
    """
    np.random.seed(42)

    # Create linearly separable data
    X, y = make_blobs(n_samples=40, centers=2, cluster_std=1.2, random_state=42)

    # Shift to make it more visually pleasing
    X[:, 0] = X[:, 0] - X[:, 0].mean()
    X[:, 1] = X[:, 1] - X[:, 1].mean()

    # Train SVM
    clf = SVC(kernel='linear', C=1000)  # High C for hard margin
    clf.fit(X, y)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot decision boundary
    plot_decision_boundary(clf, X, ax)

    # Plot all points
    colors = ['#FF6B6B', '#4ECDC4']
    for i, class_val in enumerate([0, 1]):
        mask = y == class_val
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=80,
                   edgecolors='white', linewidths=1.5, label=f'Class {class_val}', zorder=5)

    # Highlight support vectors with larger circle
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=200, facecolors='none', edgecolors='#333333', linewidths=2.5,
               label='Support Vectors', zorder=6)

    # Add annotations
    ax.annotate('Decision Boundary\n$\\mathbf{w} \\cdot \\mathbf{x} + b = 0$',
                xy=(0.5, 0.1), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.annotate('Margin = $\\frac{2}{||\\mathbf{w}||}$',
                xy=(-2.5, 1.5), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='#FFE66D', alpha=0.8))

    # Draw margin width indicator
    w = clf.coef_[0]
    b = clf.intercept_[0]
    margin = 2 / np.linalg.norm(w)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Linear SVM: Maximum Margin Classifier')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(X[:, 0].min() - 1.5, X[:, 0].max() + 1.5)
    ax.set_ylim(X[:, 1].min() - 1.5, X[:, 1].max() + 1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'svm-margin.svg'), format='svg',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: svm-margin.svg")


def generate_kernel_visualization():
    """
    Visualization 2: Kernel trick - linear vs RBF decision boundary.
    """
    np.random.seed(42)

    # Create non-linearly separable data (circles)
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.4, random_state=42)
    X = X * 2  # Scale up for better visualization

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Train Linear SVM
    clf_linear = SVC(kernel='linear', C=1)
    clf_linear.fit(X, y)

    # Train RBF SVM
    clf_rbf = SVC(kernel='rbf', C=1, gamma=0.5)
    clf_rbf.fit(X, y)

    classifiers = [clf_linear, clf_rbf]
    titles = ['Linear Kernel (Fails)', 'RBF Kernel (Succeeds)']

    for ax, clf, title in zip(axes, classifiers, titles):
        # Plot decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
        ax.contour(xx, yy, Z, colors='k', linewidths=1.5)

        # Plot points
        colors = ['#FF6B6B', '#4ECDC4']
        markers = ['o', 's']
        for i, class_val in enumerate([0, 1]):
            mask = y == class_val
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=60,
                       marker=markers[i], edgecolors='white', linewidths=1,
                       label=f'Class {class_val}', zorder=5)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)

    # Add overall title
    fig.suptitle('The Kernel Trick: Non-Linear Decision Boundaries',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'svm-kernel.svg'), format='svg',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: svm-kernel.svg")


def generate_soft_margin_visualization():
    """
    Visualization 3: Soft margin illustration with slack variables.
    """
    np.random.seed(42)

    # Create overlapping data
    n_samples = 30
    X1 = np.random.randn(n_samples, 2) + np.array([2, 2])
    X2 = np.random.randn(n_samples, 2) + np.array([0, 0])

    # Add some overlap points
    X1[0] = [0.5, 0.5]  # Misclassified point
    X1[1] = [1.0, 1.0]  # Point in margin
    X2[0] = [1.5, 1.8]  # Misclassified point
    X2[1] = [1.2, 1.3]  # Point in margin

    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_samples), np.zeros(n_samples)])

    # Create figure with different C values
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    C_values = [0.1, 1, 100]
    titles = ['Small C = 0.1\n(Large margin, more errors)',
              'Medium C = 1\n(Balanced)',
              'Large C = 100\n(Small margin, few errors)']

    for ax, C, title in zip(axes, C_values, titles):
        clf = SVC(kernel='linear', C=C)
        clf.fit(X, y)

        # Plot decision boundary and margins
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Fill regions
        ax.contourf(xx, yy, Z, levels=[-100, -1, 1, 100],
                    colors=['#FFCCCC', '#FFFFCC', '#CCFFCC'], alpha=0.5)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1],
                   colors=['#FF6B6B', '#333333', '#4ECDC4'],
                   linestyles=['--', '-', '--'], linewidths=[1.5, 2, 1.5])

        # Plot points
        colors = ['#4ECDC4', '#FF6B6B']
        for i, class_val in enumerate([0, 1]):
            mask = y == class_val
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], s=60,
                       edgecolors='white', linewidths=1, zorder=5)

        # Highlight support vectors
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                   s=150, facecolors='none', edgecolors='#333333', linewidths=2,
                   zorder=6)

        # Mark slack variable points (violations)
        decision_vals = clf.decision_function(X)
        violations = np.abs(decision_vals) < 1
        if np.any(violations):
            ax.scatter(X[violations, 0], X[violations, 1],
                       s=200, facecolors='none', edgecolors='#FFD93D',
                       linewidths=2.5, linestyle='--', zorder=7)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-2, 5)
        ax.set_ylim(-2, 5)

        # Add support vector count
        n_sv = len(clf.support_vectors_)
        ax.text(0.05, 0.95, f'Support Vectors: {n_sv}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add overall title
    fig.suptitle('Soft Margin SVM: Effect of Regularization Parameter C',
                 fontsize=14, fontweight='bold', y=1.02)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B',
               markersize=10, label='Class 1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4',
               markersize=10, label='Class 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#333333', markersize=10, markeredgewidth=2,
               label='Support Vectors'),
        Line2D([0], [0], linestyle='-', color='#333333', linewidth=2,
               label='Decision Boundary'),
        Line2D([0], [0], linestyle='--', color='#FF6B6B', linewidth=1.5,
               label='Margin Boundaries'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.08), framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'svm-soft-margin.svg'), format='svg',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: svm-soft-margin.svg")


if __name__ == '__main__':
    print("Generating SVM visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    generate_margin_visualization()
    generate_kernel_visualization()
    generate_soft_margin_visualization()

    print("-" * 50)
    print("All visualizations generated successfully!")
