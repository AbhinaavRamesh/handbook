#!/usr/bin/env python3
"""
Generate visualizations for Logistic Regression documentation.
Creates SVG files for sigmoid function, decision boundary, and cross-entropy loss.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set style for clean, professional look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/images/'


def generate_sigmoid_curve():
    """Generate sigmoid function visualization."""
    fig, ax = plt.subplots(figsize=(8, 5))

    z = np.linspace(-8, 8, 500)
    sigmoid = 1 / (1 + np.exp(-z))

    # Plot sigmoid
    ax.plot(z, sigmoid, 'b-', linewidth=2.5, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')

    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)

    # Annotations
    ax.annotate('Threshold (0.5)', xy=(4, 0.5), fontsize=10, color='gray')
    ax.annotate(r'$z \to -\infty$: $\sigma \to 0$', xy=(-7.5, 0.08), fontsize=10)
    ax.annotate(r'$z \to +\infty$: $\sigma \to 1$', xy=(4, 0.92), fontsize=10)

    # Mark key point
    ax.plot(0, 0.5, 'ro', markersize=8, zorder=5)
    ax.annotate(r'$\sigma(0) = 0.5$', xy=(0.3, 0.55), fontsize=10, color='red')

    ax.set_xlabel('z (linear combination: w·x + b)')
    ax.set_ylabel('σ(z) = P(y=1|x)')
    ax.set_title('Sigmoid Function')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc='lower right', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'logistic-sigmoid.svg', format='svg', dpi=150, bbox_inches='tight')
    plt.close()
    print('Generated: logistic-sigmoid.svg')


def generate_decision_boundary():
    """Generate 2D decision boundary visualization."""
    np.random.seed(42)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Generate synthetic 2D data
    n_samples = 100

    # Class 0 (negative)
    X0 = np.random.randn(n_samples, 2) * 0.8 + np.array([-1.5, -1])
    # Class 1 (positive)
    X1 = np.random.randn(n_samples, 2) * 0.8 + np.array([1.5, 1])

    X = np.vstack([X0, X1])
    y = np.array([0] * n_samples + [1] * n_samples)

    # Fit simple logistic regression (gradient descent)
    w = np.array([1.0, 1.0])
    b = 0.0
    lr = 0.1

    for _ in range(1000):
        z = X @ w + b
        y_pred = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        dw = (1/len(y)) * X.T @ (y_pred - y)
        db = (1/len(y)) * np.sum(y_pred - y)
        w -= lr * dw
        b -= lr * db

    # Plot data points
    ax.scatter(X0[:, 0], X0[:, 1], c='#3498db', label='Class 0', alpha=0.7, s=50, edgecolors='white')
    ax.scatter(X1[:, 0], X1[:, 1], c='#e74c3c', label='Class 1', alpha=0.7, s=50, edgecolors='white')

    # Plot decision boundary: w·x + b = 0
    x_boundary = np.linspace(-4, 4, 100)
    y_boundary = -(w[0] * x_boundary + b) / w[1]
    ax.plot(x_boundary, y_boundary, 'k-', linewidth=2, label='Decision Boundary')

    # Plot probability contours
    xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
    Z = 1 / (1 + np.exp(-(w[0]*xx + w[1]*yy + b)))

    contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap='RdBu_r', alpha=0.3)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('P(y=1|x)', fontsize=11)

    # Annotations
    ax.annotate('P(y=1) > 0.5\nPredict Class 1', xy=(2, 2.5), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    ax.annotate('P(y=1) < 0.5\nPredict Class 0', xy=(-2, -2.5), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='#cce5ff', alpha=0.8))

    ax.set_xlabel('Feature $x_1$')
    ax.set_ylabel('Feature $x_2$')
    ax.set_title('Logistic Regression Decision Boundary')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.legend(loc='upper left')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'logistic-decision-boundary.svg', format='svg', dpi=150, bbox_inches='tight')
    plt.close()
    print('Generated: logistic-decision-boundary.svg')


def generate_cross_entropy_loss():
    """Generate cross-entropy loss curve visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    y_pred = np.linspace(0.001, 0.999, 500)

    # Left plot: Loss when y=1
    ax1 = axes[0]
    loss_y1 = -np.log(y_pred)
    ax1.plot(y_pred, loss_y1, 'b-', linewidth=2.5, label=r'$-\log(\hat{y})$')
    ax1.fill_between(y_pred, loss_y1, alpha=0.2, color='blue')
    ax1.set_xlabel(r'Predicted Probability $\hat{y}$')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss when y = 1 (True Positive)')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 5)
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate('Correct prediction\n(low loss)', xy=(0.9, 0.3), fontsize=10,
                 ha='center', color='green')
    ax1.annotate('Wrong prediction\n(high loss)', xy=(0.1, 3), fontsize=10,
                 ha='center', color='red')
    ax1.legend(loc='upper right')

    # Right plot: Loss when y=0
    ax2 = axes[1]
    loss_y0 = -np.log(1 - y_pred)
    ax2.plot(y_pred, loss_y0, 'r-', linewidth=2.5, label=r'$-\log(1-\hat{y})$')
    ax2.fill_between(y_pred, loss_y0, alpha=0.2, color='red')
    ax2.set_xlabel(r'Predicted Probability $\hat{y}$')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss when y = 0 (True Negative)')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 5)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.annotate('Correct prediction\n(low loss)', xy=(0.1, 0.3), fontsize=10,
                 ha='center', color='green')
    ax2.annotate('Wrong prediction\n(high loss)', xy=(0.9, 3), fontsize=10,
                 ha='center', color='red')
    ax2.legend(loc='upper left')

    plt.suptitle('Cross-Entropy Loss: Penalizes Confident Wrong Predictions',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'logistic-cross-entropy.svg', format='svg', dpi=150, bbox_inches='tight')
    plt.close()
    print('Generated: logistic-cross-entropy.svg')


if __name__ == '__main__':
    print('Generating Logistic Regression visualizations...')
    print(f'Output directory: {OUTPUT_DIR}')
    print()

    generate_sigmoid_curve()
    generate_decision_boundary()
    generate_cross_entropy_loss()

    print()
    print('All visualizations generated successfully!')
