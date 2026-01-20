#!/usr/bin/env python3
"""
Logistic Regression Visualization Generator

This script generates static visualizations for logistic regression concepts:
1. Sigmoid function with annotations at key points (0, 0.5, 1)
2. 2D decision boundary with probability contours
3. Cross-entropy loss curves for y=0 and y=1
4. Multi-class decision regions (3-class softmax)

Author: ML Coding Reference
Date: 2024
License: MIT

Requirements:
    pip install numpy matplotlib

Usage:
    python generate_logistic_regression_viz.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14


def sigmoid(z):
    """Numerically stable sigmoid function."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


def create_sigmoid_plot(output_path: str):
    """
    Create sigmoid function visualization with annotations at key points.

    Shows:
    - Sigmoid curve from -8 to 8
    - Horizontal lines at y=0, 0.5, 1
    - Vertical line at x=0 (decision boundary)
    - Annotations for key points
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Generate data
    z = np.linspace(-8, 8, 500)
    y = sigmoid(z)

    # Plot sigmoid curve
    ax.plot(z, y, 'b-', linewidth=3, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')

    # Add horizontal reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3, linewidth=1)

    # Add vertical reference line at z=0
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

    # Key points annotations
    key_points = [
        (-8, sigmoid(-8), r'$\sigma(-\infty) \approx 0$', 'right', 'bottom'),
        (0, 0.5, r'$\sigma(0) = 0.5$', 'left', 'bottom'),
        (8, sigmoid(8), r'$\sigma(+\infty) \approx 1$', 'left', 'top'),
    ]

    for x, y_pt, label, ha, va in key_points:
        ax.plot(x, y_pt, 'ro', markersize=10, zorder=5)
        offset_x = 0.3 if ha == 'left' else -0.3
        offset_y = 0.05 if va == 'bottom' else -0.05
        ax.annotate(label, (x + offset_x, y_pt + offset_y), fontsize=12, ha=ha, va=va,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Add decision boundary annotation
    ax.fill_between(z[z < 0], 0, sigmoid(z[z < 0]), alpha=0.2, color='red',
                    label='Predict Class 0 (p < 0.5)')
    ax.fill_between(z[z >= 0], 0, sigmoid(z[z >= 0]), alpha=0.2, color='blue',
                    label='Predict Class 1 (p >= 0.5)')

    # Labels and title
    ax.set_xlabel('z = w^T x + b (Linear Combination)', fontsize=12)
    ax.set_ylabel(r'$\sigma(z)$ (Probability)', fontsize=12)
    ax.set_title('Sigmoid Function: Mapping Linear Output to Probability', fontsize=14, fontweight='bold')

    # Set limits
    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.05, 1.1)

    # Add threshold annotation
    ax.annotate('Decision\nBoundary', xy=(0, 0.5), xytext=(2, 0.3),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               fontsize=11, ha='left',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_decision_boundary_plot(output_path: str):
    """
    Create 2D decision boundary visualization with probability contours.

    Shows:
    - Synthetic 2D data points for two classes
    - Decision boundary (p=0.5 line)
    - Probability contours showing gradient from class 0 to class 1
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100

    # Class 0 (centered at (-1.5, -1))
    X0 = np.random.randn(n_samples, 2) * 0.8 + np.array([-1.5, -1])

    # Class 1 (centered at (1.5, 1))
    X1 = np.random.randn(n_samples, 2) * 0.8 + np.array([1.5, 1])

    # Combine data
    X = np.vstack([X0, X1])
    y = np.array([0] * n_samples + [1] * n_samples)

    # Fit simple logistic regression (analytical solution for visualization)
    # Using approximate weights for a clean boundary
    w = np.array([1.2, 1.0])
    b = 0.0

    # Create mesh grid for contours
    xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
    Z = sigmoid(xx * w[0] + yy * w[1] + b)

    # Plot probability contours
    contour_filled = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11),
                                  cmap='RdYlBu_r', alpha=0.6)
    cbar = plt.colorbar(contour_filled, ax=ax, label='P(y=1|x)')

    # Plot decision boundary (p=0.5)
    decision_boundary = ax.contour(xx, yy, Z, levels=[0.5], colors='black',
                                    linewidths=3, linestyles='-')
    ax.clabel(decision_boundary, fmt='p=0.5', fontsize=11, inline=True)

    # Plot probability contour lines
    contour_lines = ax.contour(xx, yy, Z, levels=[0.1, 0.3, 0.7, 0.9],
                                colors='gray', linewidths=1, linestyles='--', alpha=0.7)
    ax.clabel(contour_lines, fmt='%.1f', fontsize=9, inline=True)

    # Plot data points
    scatter0 = ax.scatter(X0[:, 0], X0[:, 1], c='red', marker='o', s=50,
                          edgecolors='darkred', alpha=0.7, label='Class 0')
    scatter1 = ax.scatter(X1[:, 0], X1[:, 1], c='blue', marker='s', s=50,
                          edgecolors='darkblue', alpha=0.7, label='Class 1')

    # Labels and title
    ax.set_xlabel('Feature $x_1$', fontsize=12)
    ax.set_ylabel('Feature $x_2$', fontsize=12)
    ax.set_title('Logistic Regression: Decision Boundary with Probability Contours',
                 fontsize=14, fontweight='bold')

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_aspect('equal')

    # Add annotation for decision boundary equation
    ax.annotate(r'$w^T x + b = 0$' + '\n' + r'$\sigma(w^T x + b) = 0.5$',
                xy=(0, 0), xytext=(2.5, -2.5),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black'),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_cross_entropy_loss_plot(output_path: str):
    """
    Create cross-entropy loss curves for y=0 and y=1 cases.

    Shows:
    - Loss when true label is y=1: -log(p)
    - Loss when true label is y=0: -log(1-p)
    - Why loss penalizes confident wrong predictions heavily
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Predicted probability range (avoid 0 and 1 for log stability)
    p = np.linspace(0.001, 0.999, 500)

    # Loss for y=1: -log(p)
    loss_y1 = -np.log(p)

    # Loss for y=0: -log(1-p)
    loss_y0 = -np.log(1 - p)

    # Left plot: Loss when y=1
    ax1 = axes[0]
    ax1.plot(p, loss_y1, 'b-', linewidth=3, label=r'$\mathcal{L} = -\log(\hat{y})$')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

    # Annotate key regions
    ax1.fill_between(p[p < 0.5], 0, loss_y1[p < 0.5], alpha=0.2, color='red',
                     label='High loss (wrong prediction)')
    ax1.fill_between(p[p >= 0.5], 0, loss_y1[p >= 0.5], alpha=0.2, color='green',
                     label='Low loss (correct prediction)')

    # Key points
    ax1.plot(0.1, -np.log(0.1), 'ro', markersize=10, zorder=5)
    ax1.annotate(f'p=0.1: Loss={-np.log(0.1):.2f}', (0.1, -np.log(0.1)),
                xytext=(0.25, 3.5), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red'))

    ax1.plot(0.9, -np.log(0.9), 'go', markersize=10, zorder=5)
    ax1.annotate(f'p=0.9: Loss={-np.log(0.9):.2f}', (0.9, -np.log(0.9)),
                xytext=(0.6, 1.5), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green'))

    ax1.set_xlabel(r'Predicted Probability $\hat{y}$', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Loss When True Label y = 1', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 5)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right plot: Loss when y=0
    ax2 = axes[1]
    ax2.plot(p, loss_y0, 'r-', linewidth=3, label=r'$\mathcal{L} = -\log(1-\hat{y})$')
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

    # Annotate key regions
    ax2.fill_between(p[p > 0.5], 0, loss_y0[p > 0.5], alpha=0.2, color='red',
                     label='High loss (wrong prediction)')
    ax2.fill_between(p[p <= 0.5], 0, loss_y0[p <= 0.5], alpha=0.2, color='green',
                     label='Low loss (correct prediction)')

    # Key points
    ax2.plot(0.9, -np.log(0.1), 'ro', markersize=10, zorder=5)
    ax2.annotate(f'p=0.9: Loss={-np.log(0.1):.2f}', (0.9, -np.log(0.1)),
                xytext=(0.6, 3.5), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red'))

    ax2.plot(0.1, -np.log(0.9), 'go', markersize=10, zorder=5)
    ax2.annotate(f'p=0.1: Loss={-np.log(0.9):.2f}', (0.1, -np.log(0.9)),
                xytext=(0.25, 1.5), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green'))

    ax2.set_xlabel(r'Predicted Probability $\hat{y}$', fontsize=12)
    ax2.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax2.set_title('Loss When True Label y = 0', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 5)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def softmax(z):
    """Numerically stable softmax."""
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def create_multiclass_decision_regions(output_path: str):
    """
    Create multi-class decision regions for 3-class softmax classification.

    Shows:
    - Three distinct regions with different colors
    - Decision boundaries between classes
    - Probability shading within regions
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Generate synthetic data for 3 classes
    np.random.seed(42)
    n_samples = 80

    # Class 0 (top)
    X0 = np.random.randn(n_samples, 2) * 0.6 + np.array([0, 2])

    # Class 1 (bottom-left)
    X1 = np.random.randn(n_samples, 2) * 0.6 + np.array([-2, -1])

    # Class 2 (bottom-right)
    X2 = np.random.randn(n_samples, 2) * 0.6 + np.array([2, -1])

    # Combine data
    X = np.vstack([X0, X1, X2])
    y = np.array([0] * n_samples + [1] * n_samples + [2] * n_samples)

    # Approximate weights for visualization (3 classes, 2 features + bias)
    W = np.array([
        [0.0, 1.5],    # Class 0 weights (favors high y)
        [-1.5, -0.8],  # Class 1 weights (favors negative x, low y)
        [1.5, -0.8],   # Class 2 weights (favors positive x, low y)
    ])
    b = np.array([0, 0, 0])

    # Create mesh grid
    xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-3, 4, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Compute softmax probabilities for each point
    logits = grid_points @ W.T + b
    probs = softmax(logits)

    # Get predicted class (argmax)
    predictions = np.argmax(probs, axis=1).reshape(xx.shape)

    # Get max probability for shading
    max_probs = np.max(probs, axis=1).reshape(xx.shape)

    # Custom colormap for 3 classes
    colors_base = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
    cmap = ListedColormap(colors_base)

    # Plot decision regions with probability-based alpha
    for class_idx in range(3):
        mask = predictions == class_idx
        class_probs = probs[:, class_idx].reshape(xx.shape)
        ax.contourf(xx, yy, np.where(mask, class_probs, np.nan),
                   levels=np.linspace(0.33, 1, 10),
                   colors=[colors_base[class_idx]], alpha=0.5)

    # Plot decision boundaries
    for i in range(3):
        for j in range(i+1, 3):
            # Boundary where P(class i) = P(class j)
            prob_diff = probs[:, i].reshape(xx.shape) - probs[:, j].reshape(xx.shape)
            ax.contour(xx, yy, prob_diff, levels=[0], colors='black',
                      linewidths=2, linestyles='-')

    # Plot probability contours for each class (dashed)
    for class_idx, color in enumerate(colors_base):
        class_probs = probs[:, class_idx].reshape(xx.shape)
        cs = ax.contour(xx, yy, class_probs, levels=[0.5, 0.7, 0.9],
                       colors=color, linewidths=1, linestyles='--', alpha=0.8)

    # Plot data points
    markers = ['o', 's', '^']
    labels = ['Class 0', 'Class 1', 'Class 2']
    data_list = [X0, X1, X2]

    for i, (data, marker, label, color) in enumerate(zip(data_list, markers, labels, colors_base)):
        ax.scatter(data[:, 0], data[:, 1], c=color, marker=marker, s=60,
                  edgecolors='black', linewidths=0.5, alpha=0.8, label=label)

    # Labels and title
    ax.set_xlabel('Feature $x_1$', fontsize=12)
    ax.set_ylabel('Feature $x_2$', fontsize=12)
    ax.set_title('Multi-class Softmax: Decision Regions for 3 Classes',
                fontsize=14, fontweight='bold')

    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 4)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')

    # Add annotation explaining softmax
    ax.text(0.02, 0.02,
            r'Softmax: $P(y=k) = \frac{e^{z_k}}{\sum_j e^{z_j}}$' + '\n' +
            'Decision: argmax(P)',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_visualizations(output_dir: str = None):
    """
    Generate all logistic regression visualizations.

    Args:
        output_dir: Directory to save outputs (default: script directory)
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating Logistic Regression Visualizations")
    print("=" * 60)
    print(f"Output directory: {output_dir}\n")

    # 1. Sigmoid function plot
    print("[1/4] Creating sigmoid function plot...")
    create_sigmoid_plot(
        os.path.join(output_dir, 'logreg_sigmoid_function.png')
    )

    # 2. Decision boundary with probability contours
    print("[2/4] Creating decision boundary plot...")
    create_decision_boundary_plot(
        os.path.join(output_dir, 'logreg_decision_boundary.png')
    )

    # 3. Cross-entropy loss curves
    print("[3/4] Creating cross-entropy loss plot...")
    create_cross_entropy_loss_plot(
        os.path.join(output_dir, 'logreg_cross_entropy_loss.png')
    )

    # 4. Multi-class decision regions
    print("[4/4] Creating multi-class decision regions plot...")
    create_multiclass_decision_regions(
        os.path.join(output_dir, 'logreg_multiclass_regions.png')
    )

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - {os.path.join(output_dir, 'logreg_sigmoid_function.png')}")
    print(f"  - {os.path.join(output_dir, 'logreg_decision_boundary.png')}")
    print(f"  - {os.path.join(output_dir, 'logreg_cross_entropy_loss.png')}")
    print(f"  - {os.path.join(output_dir, 'logreg_multiclass_regions.png')}")


if __name__ == '__main__':
    # When run directly, generate all visualizations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generate_all_visualizations(script_dir)
