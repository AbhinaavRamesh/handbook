#!/usr/bin/env python3
"""
Generate visualizations for the Linear Regression documentation page.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style for clean, professional look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "public" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_linear_regression_fit():
    """
    Plot 1: Simple linear regression fit with data points and best-fit line.
    """
    np.random.seed(42)

    # Generate data
    X = np.linspace(0, 10, 50)
    y_true = 2 * X + 1
    y = y_true + np.random.normal(0, 2, size=X.shape)

    # Fit linear regression (closed form)
    X_design = np.c_[np.ones(len(X)), X]
    w = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    y_pred = X_design @ w

    # Calculate R^2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot data points
    ax.scatter(X, y, c='#4A90D9', s=50, alpha=0.7, label='Data points', edgecolors='white', linewidth=0.5)

    # Plot best-fit line
    ax.plot(X, y_pred, 'r-', linewidth=2.5, label=f'Best fit: $\\hat{{y}} = {w[1]:.2f}x + {w[0]:.2f}$')

    # Plot true line (dashed)
    ax.plot(X, y_true, 'g--', linewidth=1.5, alpha=0.7, label='True: $y = 2x + 1$')

    # Add residual lines for a few points
    indices = [5, 20, 35, 45]
    for i in indices:
        ax.vlines(X[i], min(y[i], y_pred[i]), max(y[i], y_pred[i]),
                  colors='gray', linestyles='dotted', alpha=0.5)

    ax.set_xlabel('Feature $x$')
    ax.set_ylabel('Target $y$')
    ax.set_title(f'Linear Regression Fit ($R^2 = {r2:.3f}$)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(-0.5, 10.5)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'linear-regression-fit.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'linear-regression-fit.svg'}")


def plot_gradient_descent_convergence():
    """
    Plot 2: Gradient descent convergence showing loss vs iterations.
    """
    np.random.seed(42)

    # Generate data
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    true_weights = np.array([3, -2])
    y = X @ true_weights + np.random.randn(n_samples) * 0.5

    # Add bias column
    X_design = np.c_[np.ones(n_samples), X]

    # Gradient descent with different learning rates
    learning_rates = [0.01, 0.1, 0.5]
    colors = ['#E74C3C', '#27AE60', '#3498DB']
    n_iters = 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for lr, color in zip(learning_rates, colors):
        w = np.zeros(3)
        losses = []

        for i in range(n_iters):
            y_pred = X_design @ w
            loss = np.mean((y - y_pred) ** 2)
            losses.append(loss)

            gradient = (2 / n_samples) * X_design.T @ (y_pred - y)
            w = w - lr * gradient

        ax1.plot(losses, color=color, linewidth=2, label=f'$\\alpha = {lr}$')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Gradient Descent Convergence')
    ax1.legend(title='Learning Rate')
    ax1.set_xlim(0, n_iters)
    ax1.set_ylim(bottom=0)

    # Plot 2b: Contour plot of loss surface (2D slice)
    # Use first two weights (excluding bias)
    w0_range = np.linspace(-1, 5, 100)
    w1_range = np.linspace(-4, 2, 100)
    W0, W1 = np.meshgrid(w0_range, w1_range)

    # Calculate loss for each weight combination (with optimal bias)
    Z = np.zeros_like(W0)
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            w_test = np.array([0, W0[i, j], W1[i, j]])
            # Compute optimal bias
            w_test[0] = np.mean(y - X @ w_test[1:])
            y_pred = X_design @ w_test
            Z[i, j] = np.mean((y - y_pred) ** 2)

    contour = ax2.contour(W0, W1, Z, levels=20, cmap='viridis', alpha=0.7)
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

    # Plot gradient descent path
    w = np.zeros(3)
    path_w0, path_w1 = [w[1]], [w[2]]
    lr = 0.1

    for _ in range(50):
        y_pred = X_design @ w
        gradient = (2 / n_samples) * X_design.T @ (y_pred - y)
        w = w - lr * gradient
        path_w0.append(w[1])
        path_w1.append(w[2])

    ax2.plot(path_w0, path_w1, 'r.-', markersize=4, linewidth=1.5, label='GD path')
    ax2.scatter([true_weights[0]], [true_weights[1]], c='gold', s=150,
                marker='*', edgecolors='black', linewidth=1, zorder=5, label='Optimum')
    ax2.scatter([0], [0], c='red', s=80, marker='o', edgecolors='black',
                linewidth=1, zorder=5, label='Start')

    ax2.set_xlabel('Weight $w_1$')
    ax2.set_ylabel('Weight $w_2$')
    ax2.set_title('Loss Contours & GD Path')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'gradient-descent-convergence.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'gradient-descent-convergence.svg'}")


def plot_regularization_effect():
    """
    Plot 3: Regularization effect showing L1 vs L2 weight shrinkage.
    """
    np.random.seed(42)

    # Generate data with some irrelevant features
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)

    # True weights: only first 3 features matter
    true_weights = np.array([3, -2, 1.5, 0, 0, 0, 0, 0, 0, 0])
    y = X @ true_weights + np.random.randn(n_samples) * 0.5

    # Add bias column
    X_design = np.c_[np.ones(n_samples), X]

    # Compute weights for different lambda values
    lambdas = np.logspace(-3, 2, 50)

    # Storage for weight paths
    ridge_weights = []
    lasso_weights = []

    # Ridge regression (closed form)
    for lam in lambdas:
        I = np.eye(X_design.shape[1])
        I[0, 0] = 0  # Don't regularize bias
        w_ridge = np.linalg.inv(X_design.T @ X_design + lam * I) @ X_design.T @ y
        ridge_weights.append(w_ridge[1:])  # Exclude bias

    ridge_weights = np.array(ridge_weights)

    # Lasso regression (coordinate descent approximation)
    for lam in lambdas:
        # Simple soft-thresholding approximation
        w_ols = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
        w_lasso = np.sign(w_ols[1:]) * np.maximum(np.abs(w_ols[1:]) - lam * 0.5, 0)
        lasso_weights.append(w_lasso)

    lasso_weights = np.array(lasso_weights)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Colors for features
    colors = plt.cm.tab10(np.linspace(0, 1, n_features))

    # Plot Ridge weights
    for i in range(n_features):
        linestyle = '-' if i < 3 else '--'
        alpha = 1.0 if i < 3 else 0.5
        ax1.plot(lambdas, ridge_weights[:, i], color=colors[i],
                 linestyle=linestyle, linewidth=2, alpha=alpha, label=f'$w_{i+1}$')

    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xscale('log')
    ax1.set_xlabel('Regularization $\\lambda$')
    ax1.set_ylabel('Weight value')
    ax1.set_title('Ridge (L2): Weights Shrink Smoothly')
    ax1.legend(loc='upper right', ncol=2, fontsize=9)

    # Plot Lasso weights
    for i in range(n_features):
        linestyle = '-' if i < 3 else '--'
        alpha = 1.0 if i < 3 else 0.5
        ax2.plot(lambdas, lasso_weights[:, i], color=colors[i],
                 linestyle=linestyle, linewidth=2, alpha=alpha, label=f'$w_{i+1}$')

    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Regularization $\\lambda$')
    ax2.set_ylabel('Weight value')
    ax2.set_title('Lasso (L1): Weights Go to Exactly Zero')
    ax2.legend(loc='upper right', ncol=2, fontsize=9)

    # Add annotations
    ax1.annotate('Important features\n(solid lines)', xy=(0.02, 2.5), fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.annotate('Sparse solution:\nonly 3 features remain', xy=(0.5, 2.5), fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'regularization-effect.svg', format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'regularization-effect.svg'}")


if __name__ == '__main__':
    print("Generating Linear Regression visualizations...")
    plot_linear_regression_fit()
    plot_gradient_descent_convergence()
    plot_regularization_effect()
    print("Done!")
