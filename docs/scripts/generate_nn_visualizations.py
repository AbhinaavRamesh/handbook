#!/usr/bin/env python3
"""
Generate visualizations for the Neural Networks documentation page.
Creates SVG files for:
1. Activation functions comparison
2. Neural network XOR decision boundary
3. Training loss curve example
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# Set style for clean, professional look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# Output directory
OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)


def plot_activation_functions():
    """Plot comparison of common activation functions."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    x = np.linspace(-5, 5, 200)

    # Colors
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    # Sigmoid
    ax = axes[0, 0]
    ax.plot(x, sigmoid(x), color=colors[0], linewidth=2.5, label='Sigmoid')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title('Sigmoid: $\\sigma(x) = \\frac{1}{1 + e^{-x}}$', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('$\\sigma(x)$')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-5, 5)
    ax.fill_between(x, sigmoid(x), alpha=0.2, color=colors[0])
    ax.text(2.5, 0.2, 'Range: (0, 1)', fontsize=9, color='gray')

    # Tanh
    ax = axes[0, 1]
    ax.plot(x, tanh(x), color=colors[1], linewidth=2.5, label='Tanh')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title('Tanh: $\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('$\\tanh(x)$')
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(-5, 5)
    ax.fill_between(x, tanh(x), alpha=0.2, color=colors[1])
    ax.text(2.5, -0.8, 'Range: (-1, 1)', fontsize=9, color='gray')

    # ReLU
    ax = axes[1, 0]
    ax.plot(x, relu(x), color=colors[2], linewidth=2.5, label='ReLU')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title('ReLU: $f(x) = \\max(0, x)$', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('$f(x)$')
    ax.set_ylim(-1, 5.5)
    ax.set_xlim(-5, 5)
    ax.fill_between(x, relu(x), alpha=0.2, color=colors[2])
    ax.text(2.5, 1, 'Range: [0, $\\infty$)', fontsize=9, color='gray')

    # Leaky ReLU
    ax = axes[1, 1]
    ax.plot(x, leaky_relu(x), color=colors[3], linewidth=2.5, label='Leaky ReLU')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_title('Leaky ReLU: $f(x) = \\max(\\alpha x, x)$', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('$f(x)$')
    ax.set_ylim(-1, 5.5)
    ax.set_xlim(-5, 5)
    ax.fill_between(x, leaky_relu(x), alpha=0.2, color=colors[3])
    ax.text(2.5, 1, 'Range: ($-\\infty$, $\\infty$)', fontsize=9, color='gray')
    ax.text(-4.5, -0.3, '$\\alpha = 0.1$', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'nn-activations.svg'), format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'nn-activations.svg')}")


def plot_xor_decision_boundary():
    """Plot XOR problem and neural network decision boundary."""
    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Simple 2-layer neural network trained on XOR
    # Pre-computed weights that solve XOR
    np.random.seed(42)

    # Network: 2 -> 4 -> 1
    W1 = np.array([[ 5.0,  5.0, -5.0, -5.0],
                   [ 5.0, -5.0,  5.0, -5.0]])
    b1 = np.array([-2.5, -2.5, -2.5,  7.5])
    W2 = np.array([[ 5.0], [ 5.0], [ 5.0], [-5.0]])
    b2 = np.array([-2.5])

    def predict(X):
        h = np.maximum(0, X @ W1 + b1)  # ReLU
        out = sigmoid(h @ W2 + b2)
        return out.flatten()

    # Create mesh grid for decision boundary
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                         np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(grid).reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Decision boundary with gradient
    contour = ax.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.8)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

    # Plot data points
    colors = ['#E91E63', '#2196F3']
    markers = ['o', 's']
    labels = ['Class 0 (XOR=0)', 'Class 1 (XOR=1)']

    for i, (c, m, l) in enumerate(zip(colors, markers, labels)):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], c=c, marker=m, s=200,
                   edgecolors='white', linewidths=2, label=l, zorder=5)

    # Annotations
    ax.annotate('(0,0)\nXOR=0', (0, 0), textcoords="offset points",
                xytext=(0, -30), ha='center', fontsize=9)
    ax.annotate('(0,1)\nXOR=1', (0, 1), textcoords="offset points",
                xytext=(0, 20), ha='center', fontsize=9)
    ax.annotate('(1,0)\nXOR=1', (1, 0), textcoords="offset points",
                xytext=(0, -30), ha='center', fontsize=9)
    ax.annotate('(1,1)\nXOR=0', (1, 1), textcoords="offset points",
                xytext=(0, 20), ha='center', fontsize=9)

    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Neural Network Decision Boundary (XOR Problem)', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)

    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, label='Network Output')
    cbar.ax.set_ylabel('$P(y=1|x)$', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'nn-xor-boundary.svg'), format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'nn-xor-boundary.svg')}")


def plot_training_loss_curve():
    """Plot example training and validation loss curves."""
    np.random.seed(42)

    epochs = np.arange(1, 101)

    # Simulated training curves
    # Good fit
    train_loss_good = 2.0 * np.exp(-0.05 * epochs) + 0.1 + np.random.normal(0, 0.02, len(epochs))
    val_loss_good = 2.0 * np.exp(-0.04 * epochs) + 0.15 + np.random.normal(0, 0.03, len(epochs))

    # Overfitting
    train_loss_overfit = 2.0 * np.exp(-0.08 * epochs) + 0.05 + np.random.normal(0, 0.01, len(epochs))
    val_loss_overfit = 2.0 * np.exp(-0.03 * epochs) + np.where(epochs > 30, 0.01 * (epochs - 30), 0) + 0.2 + np.random.normal(0, 0.03, len(epochs))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Good fit plot
    ax = axes[0]
    ax.plot(epochs, train_loss_good, color='#2196F3', linewidth=2, label='Training Loss')
    ax.plot(epochs, val_loss_good, color='#FF9800', linewidth=2, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Good Fit: Convergent Training', fontsize=13)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 2.5)
    ax.fill_between(epochs, train_loss_good, val_loss_good, alpha=0.1, color='gray')
    ax.annotate('Gap stays small', xy=(80, 0.15), fontsize=9, color='gray')

    # Overfitting plot
    ax = axes[1]
    ax.plot(epochs, train_loss_overfit, color='#2196F3', linewidth=2, label='Training Loss')
    ax.plot(epochs, val_loss_overfit, color='#FF9800', linewidth=2, label='Validation Loss')
    ax.axvline(x=30, color='#E91E63', linestyle='--', linewidth=1.5, alpha=0.7, label='Best Checkpoint')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Overfitting: Validation Loss Increases', fontsize=13)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 2.5)

    # Highlight overfitting region
    ax.axvspan(30, 100, alpha=0.1, color='red')
    ax.annotate('Overfitting\nRegion', xy=(65, 1.8), fontsize=10, color='#E91E63',
                ha='center', fontweight='bold')
    ax.annotate('Early stopping\npoint', xy=(30, 0.4), xytext=(45, 0.8),
                arrowprops=dict(arrowstyle='->', color='#E91E63'),
                fontsize=9, color='#E91E63')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'nn-training-curves.svg'), format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'nn-training-curves.svg')}")


if __name__ == '__main__':
    print("Generating Neural Network visualizations...")
    plot_activation_functions()
    plot_xor_decision_boundary()
    plot_training_loss_curve()
    print("\nAll visualizations generated successfully!")
