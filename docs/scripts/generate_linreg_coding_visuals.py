"""
Generate SVG visualizations for Linear Regression Implementation page.

Visualizations:
1. Ridge vs Lasso coefficient paths
2. Gradient descent convergence comparison
3. Regularization effect on decision boundary
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'public', 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def soft_threshold(z, threshold):
    """Soft thresholding for Lasso."""
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)


def generate_coefficient_paths():
    """
    Generate Ridge vs Lasso coefficient paths as regularization strength varies.
    Shows how Lasso drives coefficients to exactly zero while Ridge shrinks them.
    """
    np.random.seed(42)

    # Generate synthetic data
    n_samples, n_features = 100, 8
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([3.0, -2.5, 2.0, -1.5, 0.0, 0.0, 0.0, 0.0])
    y = X @ true_coef + np.random.randn(n_samples) * 0.5

    # Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std
    y_centered = y - y.mean()

    # Range of alpha values (log scale)
    alphas = np.logspace(-3, 2, 100)

    # Store coefficients
    ridge_coefs = []
    lasso_coefs = []

    for alpha in alphas:
        # Ridge: closed-form solution
        XtX = X_scaled.T @ X_scaled
        Xty = X_scaled.T @ y_centered
        ridge_w = np.linalg.solve(XtX + alpha * np.eye(n_features), Xty)
        ridge_coefs.append(ridge_w)

        # Lasso: coordinate descent
        w = np.zeros(n_features)
        X_norms_sq = np.sum(X_scaled ** 2, axis=0)
        residuals = y_centered.copy()

        for _ in range(100):  # iterations
            for j in range(n_features):
                if X_norms_sq[j] == 0:
                    continue
                residuals += w[j] * X_scaled[:, j]
                rho = X_scaled[:, j] @ residuals
                w[j] = soft_threshold(rho, alpha * n_samples) / X_norms_sq[j]
                residuals -= w[j] * X_scaled[:, j]

        lasso_coefs.append(w.copy())

    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_features))

    # Ridge plot
    ax1 = axes[0]
    for i in range(n_features):
        label = f'$w_{i+1}$' + (' (true=0)' if true_coef[i] == 0 else f' (true={true_coef[i]:.1f})')
        ax1.plot(alphas, ridge_coefs[:, i], color=colors[i], linewidth=2, label=label)

    ax1.set_xscale('log')
    ax1.set_xlabel(r'Regularization Strength $\lambda$')
    ax1.set_ylabel('Coefficient Value')
    ax1.set_title('Ridge Regression (L2) Coefficient Paths')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(alphas[0], alphas[-1])

    # Lasso plot
    ax2 = axes[1]
    for i in range(n_features):
        label = f'$w_{i+1}$' + (' (true=0)' if true_coef[i] == 0 else f' (true={true_coef[i]:.1f})')
        ax2.plot(alphas, lasso_coefs[:, i], color=colors[i], linewidth=2, label=label)

    ax2.set_xscale('log')
    ax2.set_xlabel(r'Regularization Strength $\lambda$')
    ax2.set_ylabel('Coefficient Value')
    ax2.set_title('Lasso Regression (L1) Coefficient Paths')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(alphas[0], alphas[-1])

    # Add annotations
    ax1.annotate('Coefficients shrink\nbut never reach 0',
                xy=(10, 0.5), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.annotate('Coefficients become\nexactly 0 (sparse)',
                xy=(1, 0.5), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'linreg-coefficient-paths.svg'),
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated: linreg-coefficient-paths.svg")


def generate_gd_convergence():
    """
    Compare convergence of different gradient descent variants:
    - Batch GD
    - SGD
    - Mini-batch GD
    """
    np.random.seed(42)

    # Generate data
    n_samples = 500
    X = np.random.randn(n_samples, 5)
    true_w = np.array([1.0, -2.0, 1.5, -0.5, 2.0])
    y = X @ true_w + 3.0 + np.random.randn(n_samples) * 0.5

    # Add intercept column
    X_aug = np.column_stack([np.ones(n_samples), X])
    true_w_aug = np.concatenate([[3.0], true_w])

    def compute_loss(w):
        return np.mean((X_aug @ w - y) ** 2) / 2

    # Batch Gradient Descent
    def batch_gd(lr, n_iters):
        w = np.zeros(6)
        losses = [compute_loss(w)]
        for _ in range(n_iters):
            grad = X_aug.T @ (X_aug @ w - y) / n_samples
            w = w - lr * grad
            losses.append(compute_loss(w))
        return losses

    # SGD
    def sgd(lr, n_epochs):
        w = np.zeros(6)
        losses = [compute_loss(w)]
        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            for i in indices:
                grad = X_aug[i] * (X_aug[i] @ w - y[i])
                w = w - lr * grad
            losses.append(compute_loss(w))
        return losses

    # Mini-batch GD
    def minibatch_gd(lr, n_epochs, batch_size=32):
        w = np.zeros(6)
        losses = [compute_loss(w)]
        n_batches = n_samples // batch_size
        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            for b in range(n_batches):
                batch_idx = indices[b * batch_size:(b + 1) * batch_size]
                X_batch = X_aug[batch_idx]
                y_batch = y[batch_idx]
                grad = X_batch.T @ (X_batch @ w - y_batch) / batch_size
                w = w - lr * grad
            losses.append(compute_loss(w))
        return losses

    # Run optimizations
    n_iters = 100
    batch_losses = batch_gd(0.1, n_iters)
    sgd_losses = sgd(0.01, n_iters)
    minibatch_losses = minibatch_gd(0.05, n_iters, batch_size=32)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss vs iterations (log scale)
    ax1 = axes[0]
    ax1.semilogy(range(len(batch_losses)), batch_losses, 'b-', linewidth=2,
                 label='Batch GD (lr=0.1)')
    ax1.semilogy(range(len(sgd_losses)), sgd_losses, 'r-', linewidth=2,
                 label='SGD (lr=0.01)', alpha=0.8)
    ax1.semilogy(range(len(minibatch_losses)), minibatch_losses, 'g-', linewidth=2,
                 label='Mini-batch GD (lr=0.05, bs=32)')

    ax1.set_xlabel('Epoch / Iteration')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Gradient Descent Convergence Comparison')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Learning rate effect
    ax2 = axes[1]
    for lr, color, style in [(0.01, 'blue', '-'), (0.1, 'green', '-'),
                              (0.5, 'orange', '-'), (1.0, 'red', '--')]:
        try:
            losses = batch_gd(lr, 50)
            ax2.semilogy(range(len(losses)), losses, color=color, linestyle=style,
                        linewidth=2, label=f'lr={lr}')
        except:
            pass  # Skip if diverges

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Effect of Learning Rate on Convergence')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0.01)

    # Add annotations
    ax1.annotate('Batch GD: Smooth but\nmore computation per step',
                xy=(50, batch_losses[50]), xytext=(60, batch_losses[30]),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=9, color='blue')

    ax1.annotate('SGD: Noisy but\nfast per iteration',
                xy=(30, sgd_losses[30]), xytext=(10, sgd_losses[10] * 1.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    ax2.annotate('Too high:\nDiverges!',
                xy=(20, 1e10) if len(batch_gd(1.0, 20)) > 20 else (20, 100),
                fontsize=9, color='red',
                bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'linreg-gd-convergence.svg'),
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated: linreg-gd-convergence.svg")


def generate_regularization_effect():
    """
    Visualize the effect of regularization on model fit:
    - Underfitting (too much regularization)
    - Overfitting (no regularization)
    - Good fit (appropriate regularization)
    """
    np.random.seed(42)

    # Generate polynomial data
    n_samples = 30
    X = np.sort(np.random.uniform(-3, 3, n_samples))
    y_true = np.sin(X) + 0.5 * X
    y = y_true + np.random.randn(n_samples) * 0.3

    # Create polynomial features (degree 10)
    degree = 10

    def poly_features(x, deg):
        return np.column_stack([x ** i for i in range(deg + 1)])

    X_poly = poly_features(X, degree)

    # Fit models with different regularization
    def fit_ridge(X, y, alpha):
        n_features = X.shape[1]
        XtX = X.T @ X
        Xty = X.T @ y
        w = np.linalg.solve(XtX + alpha * np.eye(n_features), Xty)
        return w

    # Different regularization strengths
    alphas = [0, 0.01, 100]  # No reg, good reg, too much reg
    titles = ['No Regularization\n(Overfitting)',
              'Optimal Regularization\n(Good Fit)',
              'Too Much Regularization\n(Underfitting)']

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Fine grid for plotting
    X_plot = np.linspace(-3.5, 3.5, 200)
    X_plot_poly = poly_features(X_plot, degree)

    colors = ['red', 'green', 'blue']

    for i, (alpha, title, color) in enumerate(zip(alphas, titles, colors)):
        ax = axes[i]

        # Fit model
        if alpha == 0:
            # OLS (add small ridge to avoid singular matrix)
            w = fit_ridge(X_poly, y, 1e-10)
        else:
            w = fit_ridge(X_poly, y, alpha)

        # Predict
        y_pred = X_plot_poly @ w

        # Plot
        ax.scatter(X, y, c='black', s=50, alpha=0.7, label='Training data', zorder=5)
        ax.plot(X_plot, np.sin(X_plot) + 0.5 * X_plot, 'k--', linewidth=1.5,
                label='True function', alpha=0.5)
        ax.plot(X_plot, y_pred, color=color, linewidth=2.5, label=f'Fitted (α={alpha})')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title, fontsize=12)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)

        # Add coefficient magnitude info
        coef_norm = np.linalg.norm(w[1:])  # Exclude intercept
        ax.text(0.95, 0.05, f'||w||₂ = {coef_norm:.1f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add overall title
    fig.suptitle('Effect of L2 Regularization on Polynomial Regression', fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'linreg-regularization-effect.svg'),
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated: linreg-regularization-effect.svg")


def main():
    """Generate all visualizations."""
    print("Generating Linear Regression visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    generate_coefficient_paths()
    generate_gd_convergence()
    generate_regularization_effect()

    print("-" * 50)
    print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
