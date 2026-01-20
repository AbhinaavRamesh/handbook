#!/usr/bin/env python3
"""
Linear Regression Visualization Generator

This script generates visualizations demonstrating key concepts in linear regression:

1. Gradient descent convergence animation (GIF) showing optimization path
2. 3D loss surface with trajectory
3. Ridge vs Lasso comparison (L2 circle vs L1 diamond constraints)
4. Residual analysis plot

Author: ML Coding Reference
Date: 2024
License: MIT

Requirements:
    pip install numpy matplotlib pillow

Usage:
    python generate_linear_regression_viz.py

    Or import and use individual functions:
        from generate_linear_regression_viz import generate_all_visualizations
        generate_all_visualizations(output_dir='./assets/')
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Tuple, List, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14


# =============================================================================
# Data Generation
# =============================================================================

def generate_linear_data(
    n_samples: int = 100,
    n_features: int = 1,
    noise_std: float = 0.5,
    true_weights: Optional[np.ndarray] = None,
    true_intercept: float = 2.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate synthetic linear regression data.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise_std: Standard deviation of Gaussian noise
        true_weights: True weight vector (if None, randomly generated)
        true_intercept: True intercept term
        random_state: Random seed

    Returns:
        X, y, true_weights, true_intercept
    """
    np.random.seed(random_state)

    X = np.random.randn(n_samples, n_features) * 2

    if true_weights is None:
        true_weights = np.random.randn(n_features) * 2

    y = X @ true_weights + true_intercept + np.random.randn(n_samples) * noise_std

    return X, y, true_weights, true_intercept


# =============================================================================
# Loss Functions
# =============================================================================

def mse_loss(X: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    """Calculate Mean Squared Error loss."""
    predictions = X.flatten() * w + b
    return np.mean((predictions - y) ** 2) / 2


def mse_gradient(X: np.ndarray, y: np.ndarray, w: float, b: float) -> Tuple[float, float]:
    """Calculate gradient of MSE loss w.r.t. w and b."""
    n = len(y)
    predictions = X.flatten() * w + b
    errors = predictions - y

    grad_w = np.mean(errors * X.flatten())
    grad_b = np.mean(errors)

    return grad_w, grad_b


def ridge_loss(X: np.ndarray, y: np.ndarray, w: float, b: float, alpha: float = 1.0) -> float:
    """Calculate Ridge (L2) loss."""
    mse = mse_loss(X, y, w, b)
    return mse + alpha * w ** 2 / 2


def lasso_loss(X: np.ndarray, y: np.ndarray, w: float, b: float, alpha: float = 1.0) -> float:
    """Calculate Lasso (L1) loss."""
    mse = mse_loss(X, y, w, b)
    return mse + alpha * np.abs(w)


# =============================================================================
# Gradient Descent
# =============================================================================

def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    max_iters: int = 100,
    init_w: float = -3.0,
    init_b: float = -2.0
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Run gradient descent for simple linear regression.

    Returns:
        path: List of (w, b) tuples
        losses: List of loss values
    """
    w, b = init_w, init_b
    path = [(w, b)]
    losses = [mse_loss(X, y, w, b)]

    for _ in range(max_iters):
        grad_w, grad_b = mse_gradient(X, y, w, b)

        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

        path.append((w, b))
        losses.append(mse_loss(X, y, w, b))

    return path, losses


# =============================================================================
# Visualization Functions
# =============================================================================

def create_gradient_descent_animation(
    output_path: str,
    n_samples: int = 50,
    num_iterations: int = 50,
    learning_rate: float = 0.1
):
    """
    Create animated GIF showing gradient descent convergence for linear regression.

    Shows:
    - Left: Data points with evolving regression line
    - Right: Loss curve over iterations
    """
    # Generate data
    X, y, true_w, true_b = generate_linear_data(
        n_samples=n_samples,
        n_features=1,
        noise_std=0.8,
        true_weights=np.array([1.5]),
        true_intercept=2.0
    )

    # Run gradient descent
    path, losses = gradient_descent(
        X, y,
        learning_rate=learning_rate,
        max_iters=num_iterations,
        init_w=-2.0,
        init_b=-1.0
    )

    # Setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # Left plot: Data and regression line
    ax1.scatter(X, y, alpha=0.6, c='steelblue', edgecolors='white', s=50, label='Data')

    # Plot true line
    x_line = np.linspace(X.min() - 0.5, X.max() + 0.5, 100)
    true_line = x_line * true_w[0] + true_b
    ax1.plot(x_line, true_line, 'g--', linewidth=2, alpha=0.7, label='True line')

    # Initialize fitted line
    fitted_line, = ax1.plot([], [], 'r-', linewidth=2, label='Fitted line')

    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title('Linear Regression: Gradient Descent')
    ax1.legend(loc='upper left')
    ax1.set_xlim(X.min() - 0.5, X.max() + 0.5)
    ax1.set_ylim(y.min() - 1, y.max() + 1)

    # Right plot: Loss curve
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_title('Loss Convergence')
    ax2.set_xlim(0, num_iterations)
    ax2.set_ylim(0, max(losses) * 1.1)

    loss_line, = ax2.plot([], [], 'b-', linewidth=2)
    loss_point, = ax2.plot([], [], 'ro', markersize=8)

    # Text annotation for parameters
    param_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                          verticalalignment='top', fontsize=11,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def init():
        fitted_line.set_data([], [])
        loss_line.set_data([], [])
        loss_point.set_data([], [])
        param_text.set_text('')
        return fitted_line, loss_line, loss_point, param_text

    def animate(frame):
        w, b = path[frame]

        # Update fitted line
        y_fit = x_line * w + b
        fitted_line.set_data(x_line, y_fit)

        # Update loss curve
        loss_line.set_data(range(frame + 1), losses[:frame + 1])
        loss_point.set_data([frame], [losses[frame]])

        # Update parameter text
        param_text.set_text(f'Iter: {frame}\nw = {w:.3f}\nb = {b:.3f}\nLoss = {losses[frame]:.4f}')

        return fitted_line, loss_line, loss_point, param_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(path),
        interval=100, blit=True
    )

    plt.tight_layout()
    anim.save(output_path, writer='pillow', fps=10, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def create_3d_loss_surface(
    output_path: str,
    n_samples: int = 50
):
    """
    Create animated 3D loss surface with gradient descent trajectory.
    """
    # Generate data
    X, y, true_w, true_b = generate_linear_data(
        n_samples=n_samples,
        n_features=1,
        noise_std=0.5,
        true_weights=np.array([1.5]),
        true_intercept=2.0
    )

    # Run gradient descent
    path, losses = gradient_descent(
        X, y,
        learning_rate=0.1,
        max_iters=40,
        init_w=-2.0,
        init_b=-1.0
    )

    # Create mesh for loss surface
    w_range = np.linspace(-3, 4, 50)
    b_range = np.linspace(-2, 5, 50)
    W, B = np.meshgrid(w_range, b_range)

    # Calculate loss at each point
    Z = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i, j] = mse_loss(X, y, W[i, j], B[i, j])

    # Setup figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    ax.plot_surface(W, B, Z, cmap='viridis', alpha=0.6, edgecolor='none')

    # Plot contours on the bottom
    ax.contour(W, B, Z, zdir='z', offset=Z.min(), cmap='viridis', alpha=0.5, levels=20)

    # Mark optimal point
    opt_w, opt_b = true_w[0], true_b
    opt_loss = mse_loss(X, y, opt_w, opt_b)
    ax.scatter([opt_w], [opt_b], [opt_loss], c='red', s=100, marker='*', label='Optimal')

    # Initialize path line
    path_w = [p[0] for p in path]
    path_b = [p[1] for p in path]

    line, = ax.plot([], [], [], 'r-', linewidth=2, label='GD Path')
    point, = ax.plot([], [], [], 'ro', markersize=8)

    ax.set_xlabel('Weight (w)')
    ax.set_ylabel('Bias (b)')
    ax.set_zlabel('Loss (MSE)')
    ax.set_title('3D Loss Surface with Gradient Descent Trajectory')
    ax.legend()

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point

    def animate(frame):
        line.set_data(path_w[:frame+1], path_b[:frame+1])
        line.set_3d_properties(losses[:frame+1])
        point.set_data([path_w[frame]], [path_b[frame]])
        point.set_3d_properties([losses[frame]])

        # Rotate view
        ax.view_init(elev=25, azim=frame * 3)

        return line, point

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(path),
        interval=100, blit=False
    )

    anim.save(output_path, writer='pillow', fps=10, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def create_ridge_vs_lasso_comparison(output_path: str):
    """
    Create visualization comparing Ridge (L2) vs Lasso (L1) constraint regions.

    Shows:
    - L2 constraint: Circle (Euclidean ball)
    - L1 constraint: Diamond (L1 ball)
    - Contours of loss function
    - Why L1 promotes sparsity (corner solutions)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # Generate contour data for loss function (elliptical)
    w1_range = np.linspace(-3, 3, 200)
    w2_range = np.linspace(-3, 3, 200)
    W1, W2 = np.meshgrid(w1_range, w2_range)

    # Elliptical loss contours (simulating correlation between features)
    # Center at (2, 1) - the unconstrained optimal
    center = (2.0, 1.0)
    a, b = 1.5, 3.0  # Different scales for ellipse
    angle = np.pi / 6  # Rotation angle

    W1_rot = (W1 - center[0]) * np.cos(angle) + (W2 - center[1]) * np.sin(angle)
    W2_rot = -(W1 - center[0]) * np.sin(angle) + (W2 - center[1]) * np.cos(angle)
    Z = (W1_rot / a) ** 2 + (W2_rot / b) ** 2

    # Plot 1: Ridge (L2) Constraint
    ax1 = axes[0]

    # Loss contours
    contour_levels = np.linspace(0.1, 4, 15)
    ax1.contour(W1, W2, Z, levels=contour_levels, cmap='Blues', alpha=0.7)
    ax1.contourf(W1, W2, Z, levels=contour_levels, cmap='Blues', alpha=0.2)

    # L2 constraint region (circle)
    theta = np.linspace(0, 2 * np.pi, 100)
    radius = 1.2
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    ax1.fill(circle_x, circle_y, alpha=0.3, color='green', label='L2 constraint')
    ax1.plot(circle_x, circle_y, 'g-', linewidth=2)

    # Mark optimal unconstrained point
    ax1.plot(center[0], center[1], 'r*', markersize=15, label='Unconstrained opt.')

    # Mark Ridge solution (on circle boundary, tangent to contour)
    ridge_angle = np.arctan2(center[1], center[0])
    ridge_x = radius * np.cos(ridge_angle)
    ridge_y = radius * np.sin(ridge_angle)
    ax1.plot(ridge_x, ridge_y, 'go', markersize=12, markeredgecolor='black',
             markeredgewidth=2, label='Ridge solution')

    ax1.set_xlabel('$w_1$', fontsize=12)
    ax1.set_ylabel('$w_2$', fontsize=12)
    ax1.set_title('Ridge Regression (L2)\n$||w||_2^2 \\leq t$', fontsize=13)
    ax1.set_xlim(-2.5, 3)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.legend(loc='lower left', fontsize=9)

    # Annotation for Ridge
    ax1.annotate('Non-sparse\nsolution', xy=(ridge_x, ridge_y),
                 xytext=(ridge_x - 1.5, ridge_y + 1.2),
                 fontsize=10, ha='center',
                 arrowprops=dict(arrowstyle='->', color='green'))

    # Plot 2: Lasso (L1) Constraint
    ax2 = axes[1]

    # Loss contours
    ax2.contour(W1, W2, Z, levels=contour_levels, cmap='Blues', alpha=0.7)
    ax2.contourf(W1, W2, Z, levels=contour_levels, cmap='Blues', alpha=0.2)

    # L1 constraint region (diamond)
    t = 1.2
    diamond_x = [t, 0, -t, 0, t]
    diamond_y = [0, t, 0, -t, 0]
    ax2.fill(diamond_x, diamond_y, alpha=0.3, color='orange', label='L1 constraint')
    ax2.plot(diamond_x, diamond_y, 'orange', linewidth=2)

    # Mark optimal unconstrained point
    ax2.plot(center[0], center[1], 'r*', markersize=15, label='Unconstrained opt.')

    # Mark Lasso solution (on diamond corner - sparse!)
    lasso_x, lasso_y = t, 0  # On the corner
    ax2.plot(lasso_x, lasso_y, 'o', color='orange', markersize=12,
             markeredgecolor='black', markeredgewidth=2, label='Lasso solution')

    ax2.set_xlabel('$w_1$', fontsize=12)
    ax2.set_ylabel('$w_2$', fontsize=12)
    ax2.set_title('Lasso Regression (L1)\n$||w||_1 \\leq t$', fontsize=13)
    ax2.set_xlim(-2.5, 3)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.legend(loc='lower left', fontsize=9)

    # Annotation for Lasso
    ax2.annotate('Sparse solution!\n$w_2 = 0$', xy=(lasso_x, lasso_y),
                 xytext=(lasso_x + 0.3, lasso_y + 1.5),
                 fontsize=10, ha='center',
                 arrowprops=dict(arrowstyle='->', color='orange'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_residual_analysis_plot(output_path: str, n_samples: int = 100):
    """
    Create residual analysis plot showing:
    - Residuals vs Fitted values
    - Q-Q plot for normality check
    - Histogram of residuals
    - Residuals vs order (checking independence)
    """
    # Generate data
    X, y, true_w, true_b = generate_linear_data(
        n_samples=n_samples,
        n_features=1,
        noise_std=0.8,
        true_weights=np.array([1.5]),
        true_intercept=2.0
    )

    # Fit model using closed-form solution
    X_aug = np.column_stack([np.ones(len(X)), X])
    w = np.linalg.lstsq(X_aug, y, rcond=None)[0]

    # Predictions and residuals
    y_pred = X_aug @ w
    residuals = y - y_pred

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # Plot 1: Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(y_pred, residuals, alpha=0.6, c='steelblue', edgecolors='white', s=40)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted Values')

    # Add lowess smoothing line approximation
    sorted_idx = np.argsort(y_pred)
    window = n_samples // 5
    smooth_y = np.convolve(residuals[sorted_idx], np.ones(window)/window, mode='valid')
    smooth_x = y_pred[sorted_idx][window//2:-(window//2)+1]
    ax1.plot(smooth_x, smooth_y, 'orange', linewidth=2, label='Trend')
    ax1.legend()

    # Plot 2: Q-Q Plot
    ax2 = axes[0, 1]
    residuals_sorted = np.sort(residuals)
    n = len(residuals)
    theoretical_quantiles = np.array([np.quantile(np.random.randn(10000), (i + 0.5) / n)
                                       for i in range(n)])

    ax2.scatter(theoretical_quantiles, residuals_sorted, alpha=0.6, c='steelblue',
                edgecolors='white', s=40)

    # Reference line
    min_val = min(theoretical_quantiles.min(), residuals_sorted.min())
    max_val = max(theoretical_quantiles.max(), residuals_sorted.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5,
             label='Normal line')

    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Sample Quantiles')
    ax2.set_title('Q-Q Plot (Normality Check)')
    ax2.legend()

    # Plot 3: Histogram of Residuals
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=20, density=True, alpha=0.7, color='steelblue',
             edgecolor='white')

    # Overlay normal distribution
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = (1 / (np.std(residuals) * np.sqrt(2 * np.pi))) * \
             np.exp(-0.5 * ((x_norm - np.mean(residuals)) / np.std(residuals)) ** 2)
    ax3.plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal dist.')

    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Residuals')
    ax3.legend()

    # Plot 4: Residuals vs Order
    ax4 = axes[1, 1]
    ax4.scatter(range(len(residuals)), residuals, alpha=0.6, c='steelblue',
                edgecolors='white', s=40)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Observation Order')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residuals vs Order (Independence)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_all_visualizations(output_dir: str = None):
    """
    Generate all linear regression visualizations.

    Args:
        output_dir: Directory to save outputs (default: script directory)
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating Linear Regression Visualizations")
    print("=" * 60)
    print(f"Output directory: {output_dir}\n")

    # 1. Gradient descent convergence animation
    print("[1/4] Creating gradient descent convergence animation...")
    create_gradient_descent_animation(
        os.path.join(output_dir, 'linreg_gradient_descent.gif'),
        n_samples=50,
        num_iterations=50,
        learning_rate=0.1
    )

    # 2. 3D loss surface
    print("[2/4] Creating 3D loss surface animation...")
    create_3d_loss_surface(
        os.path.join(output_dir, 'linreg_3d_loss_surface.gif'),
        n_samples=50
    )

    # 3. Ridge vs Lasso comparison
    print("[3/4] Creating Ridge vs Lasso comparison plot...")
    create_ridge_vs_lasso_comparison(
        os.path.join(output_dir, 'ridge_vs_lasso_constraints.png')
    )

    # 4. Residual analysis
    print("[4/4] Creating residual analysis plot...")
    create_residual_analysis_plot(
        os.path.join(output_dir, 'linreg_residual_analysis.png'),
        n_samples=100
    )

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - {os.path.join(output_dir, 'linreg_gradient_descent.gif')}")
    print(f"  - {os.path.join(output_dir, 'linreg_3d_loss_surface.gif')}")
    print(f"  - {os.path.join(output_dir, 'ridge_vs_lasso_constraints.png')}")
    print(f"  - {os.path.join(output_dir, 'linreg_residual_analysis.png')}")


if __name__ == '__main__':
    # When run directly, generate all visualizations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generate_all_visualizations(script_dir)
