#!/usr/bin/env python3
"""
Generate visualizations for Perceptrons & MLPs module.

Outputs:
- perceptron_single.svg - Single perceptron diagram
- xor_problem.svg - XOR data with failed linear boundary
- xor_solution.gif - MLP solving XOR
- mlp_architecture.svg - Full MLP diagram
- decision_boundary.gif - Decision boundary evolution
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Style settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

COLORS = {
    'input': '#4ECDC4',
    'hidden': '#45B7D1',
    'output': '#FF6B6B',
    'weight': '#2C3E50',
    'positive': '#27AE60',
    'negative': '#E74C3C',
    'boundary': '#9B59B6',
}


def create_perceptron_single():
    """Create single perceptron diagram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 7)
    ax.axis('off')
    ax.set_title('Single Perceptron', fontsize=14, fontweight='bold')

    # Input nodes
    input_y = [5, 3, 1]
    input_labels = ['$x_1$', '$x_2$', '$x_n$']
    for i, (y, label) in enumerate(zip(input_y, input_labels)):
        circle = Circle((1, y), 0.4, facecolor=COLORS['input'], edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(1, y, label, ha='center', va='center', fontsize=12, fontweight='bold')
        if i < 2:
            # Draw arrow to summation
            ax.annotate('', xy=(4.5, 3), xytext=(1.5, y),
                       arrowprops=dict(arrowstyle='->', color=COLORS['weight'], lw=1.5))
            ax.text(2.5, y - 0.3 + (1-i)*0.5, f'$w_{i+1}$', fontsize=10)

    # Dots for "..."
    ax.text(1, 2, '...', ha='center', va='center', fontsize=16)

    # Summation node
    circle = Circle((5, 3), 0.5, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(5, 3, '$\\Sigma$', ha='center', va='center', fontsize=14, fontweight='bold')

    # Bias input
    ax.annotate('', xy=(5, 2.5), xytext=(5, 0.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['weight'], lw=1.5))
    ax.text(5.3, 1.5, '$b$', fontsize=10)
    ax.text(5, 0.3, 'bias', ha='center', fontsize=9)

    # Arrow to activation
    ax.annotate('', xy=(7.5, 3), xytext=(5.5, 3),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(6.5, 3.4, '$z = \\sum w_i x_i + b$', ha='center', fontsize=10)

    # Activation function
    circle = Circle((8, 3), 0.5, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(8, 3, '$\\sigma$', ha='center', va='center', fontsize=14, fontweight='bold')

    # Arrow to output
    ax.annotate('', xy=(10, 3), xytext=(8.5, 3),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Output node
    circle = Circle((10.5, 3), 0.4, facecolor=COLORS['output'], edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(10.5, 3, '$y$', ha='center', va='center', fontsize=12, fontweight='bold')

    # Labels
    ax.text(1, 6.2, 'Inputs', ha='center', fontsize=11, fontweight='bold')
    ax.text(6.5, 6.2, 'Activation', ha='center', fontsize=11, fontweight='bold')
    ax.text(10.5, 6.2, 'Output', ha='center', fontsize=11, fontweight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, 'perceptron_single.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: perceptron_single.svg")


def create_xor_problem():
    """Create XOR data visualization showing linear inseparability."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Plot points
    for i, (point, label) in enumerate(zip(X, y)):
        color = COLORS['positive'] if label == 1 else COLORS['negative']
        marker = 'o' if label == 1 else 's'
        ax.scatter(point[0], point[1], c=color, s=300, marker=marker, edgecolors='black', linewidth=2, zorder=3)
        ax.annotate(f'({point[0]}, {point[1]}) → {label}',
                   xy=(point[0], point[1]), xytext=(10, 10),
                   textcoords='offset points', fontsize=10)

    # Show failed linear boundaries
    x_line = np.linspace(-0.5, 1.5, 100)
    for slope, intercept, style in [(1, 0.5, '--'), (-1, 1.5, ':'), (0.5, 0.75, '-.')]:
        y_line = slope * x_line + (intercept - slope * 0.5)
        ax.plot(x_line, y_line, color=COLORS['boundary'], linestyle=style, alpha=0.5, linewidth=2)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('XOR Problem: No Linear Boundary Can Separate Classes', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Legend
    ax.scatter([], [], c=COLORS['positive'], s=100, marker='o', label='Class 1 (XOR=1)')
    ax.scatter([], [], c=COLORS['negative'], s=100, marker='s', label='Class 0 (XOR=0)')
    ax.legend(loc='upper right')

    plt.savefig(os.path.join(OUTPUT_DIR, 'xor_problem.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: xor_problem.svg")


def create_xor_solution():
    """Create animation showing MLP solving XOR through space transformation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Pre-compute hidden layer transformations
    W1 = np.array([[1, 1], [1, 1]])
    b1 = np.array([0, -1])

    def sigmoid(x):
        return 1 / (1 + np.exp(-5*x))

    def transform(X, alpha):
        """Interpolate from original to transformed space."""
        z = X @ W1.T + b1
        h = sigmoid(z)
        return (1 - alpha) * X + alpha * h

    frames_data = []
    for alpha in np.linspace(0, 1, 30):
        frames_data.append(transform(X, alpha))

    def animate(frame):
        for ax in axes:
            ax.clear()

        alpha = frame / 29

        # Original space
        axes[0].set_title('Original Space ($x_1, x_2$)', fontsize=11, fontweight='bold')
        for i, (point, label) in enumerate(zip(X, y)):
            color = COLORS['positive'] if label == 1 else COLORS['negative']
            axes[0].scatter(point[0], point[1], c=color, s=200, edgecolors='black', linewidth=2)
        axes[0].set_xlim(-0.3, 1.3)
        axes[0].set_ylim(-0.3, 1.3)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal')

        # Transforming space
        axes[1].set_title(f'Transformation (α={alpha:.2f})', fontsize=11, fontweight='bold')
        transformed = frames_data[frame]
        for i, (point, label) in enumerate(zip(transformed, y)):
            color = COLORS['positive'] if label == 1 else COLORS['negative']
            axes[1].scatter(point[0], point[1], c=color, s=200, edgecolors='black', linewidth=2)
        axes[1].set_xlim(-0.3, 1.3)
        axes[1].set_ylim(-0.3, 1.3)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal')

        # Hidden space (final)
        axes[2].set_title('Hidden Space ($h_1, h_2$)', fontsize=11, fontweight='bold')
        final_h = frames_data[-1]
        for i, (point, label) in enumerate(zip(final_h, y)):
            color = COLORS['positive'] if label == 1 else COLORS['negative']
            axes[2].scatter(point[0], point[1], c=color, s=200, edgecolors='black', linewidth=2)
        # Draw separating line
        if alpha > 0.5:
            x_line = np.linspace(0, 1, 100)
            y_line = 1.5 - x_line
            axes[2].plot(x_line, y_line, color=COLORS['boundary'], linewidth=2, linestyle='--')
            axes[2].text(0.5, 0.8, 'Linearly\nSeparable!', ha='center', fontsize=10, fontweight='bold')
        axes[2].set_xlim(-0.3, 1.3)
        axes[2].set_ylim(-0.3, 1.3)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_aspect('equal')

        fig.suptitle('MLP Solving XOR: Hidden Layer Transforms Space', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return []

    anim = FuncAnimation(fig, animate, frames=30, interval=200, blit=False)
    writer = PillowWriter(fps=5)
    anim.save(os.path.join(OUTPUT_DIR, 'xor_solution.gif'), writer=writer)
    plt.close()
    print("Generated: xor_solution.gif")


def create_mlp_architecture():
    """Create MLP architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 9)
    ax.axis('off')

    layer_x = [1, 4, 7, 10]
    layer_sizes = [4, 5, 5, 3]
    layer_colors = [COLORS['input'], COLORS['hidden'], COLORS['hidden'], COLORS['output']]
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Output\nLayer']

    def get_y_positions(n, total_height=7):
        return np.linspace(total_height, 1, n)

    # Draw connections first (behind nodes)
    for l in range(len(layer_x) - 1):
        y1 = get_y_positions(layer_sizes[l])
        y2 = get_y_positions(layer_sizes[l + 1])
        for yi in y1:
            for yj in y2:
                ax.plot([layer_x[l], layer_x[l + 1]], [yi, yj],
                       color='gray', alpha=0.3, linewidth=0.5, zorder=1)

    # Draw nodes
    for l, (x, size, color, name) in enumerate(zip(layer_x, layer_sizes, layer_colors, layer_names)):
        y_positions = get_y_positions(size)
        for i, y in enumerate(y_positions):
            circle = Circle((x, y), 0.3, facecolor=color, edgecolor='black', linewidth=2, zorder=2)
            ax.add_patch(circle)
        ax.text(x, 8.5, name, ha='center', va='center', fontsize=10, fontweight='bold')

    # Add annotations
    ax.annotate('', xy=(2.5, 4), xytext=(1.5, 4),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(2, 4.5, '$W_1, b_1$', ha='center', fontsize=9)

    ax.annotate('', xy=(5.5, 4), xytext=(4.5, 4),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(5, 4.5, '$W_2, b_2$', ha='center', fontsize=9)

    ax.annotate('', xy=(8.5, 4), xytext=(7.5, 4),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(8, 4.5, '$W_3, b_3$', ha='center', fontsize=9)

    ax.set_title('Multi-Layer Perceptron (MLP) Architecture', fontsize=14, fontweight='bold')

    # Formula
    ax.text(6, -0.5, '$\\hat{y} = \\sigma(W_3 \\cdot \\sigma(W_2 \\cdot \\sigma(W_1 x + b_1) + b_2) + b_3)$',
           ha='center', fontsize=11, style='italic')

    plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_architecture.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: mlp_architecture.svg")


def create_decision_boundary():
    """Create animation showing decision boundary evolution during training."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Generate spiral data
    np.random.seed(42)
    n_points = 100
    theta = np.linspace(0, 4*np.pi, n_points)
    r = np.linspace(0.5, 2, n_points)

    X1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    X2 = np.column_stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])
    X1 += np.random.randn(n_points, 2) * 0.1
    X2 += np.random.randn(n_points, 2) * 0.1

    X = np.vstack([X1, X2])
    y = np.array([0]*n_points + [1]*n_points)

    # Simulate boundaries at different epochs
    def get_boundary_params(epoch):
        """Simulate decision boundary evolution."""
        complexity = min(1, epoch / 50)
        return complexity

    frames = 60

    def animate(frame):
        ax.clear()

        # Plot data
        ax.scatter(X[:n_points, 0], X[:n_points, 1], c=COLORS['negative'], s=30, alpha=0.7, label='Class 0')
        ax.scatter(X[n_points:, 0], X[n_points:, 1], c=COLORS['positive'], s=30, alpha=0.7, label='Class 1')

        # Simulated decision boundary
        complexity = get_boundary_params(frame)

        xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Simple boundary that evolves
        if frame < 10:
            # Linear boundary
            z = grid[:, 0] * 0.5
        elif frame < 30:
            # Curved boundary
            z = grid[:, 0]**2 + grid[:, 1]**2 - 1.5
        else:
            # Complex spiral-like boundary
            angle = np.arctan2(grid[:, 1], grid[:, 0])
            radius = np.sqrt(grid[:, 0]**2 + grid[:, 1]**2)
            z = np.sin(angle * 2 + radius * 2) * min(1, (frame - 30) / 30)

        Z = z.reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0], colors=[COLORS['boundary']], linewidths=2)
        ax.contourf(xx, yy, Z, levels=[-10, 0, 10], colors=[COLORS['negative'], COLORS['positive']], alpha=0.2)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(f'Decision Boundary Evolution (Epoch {frame})', fontsize=12, fontweight='bold')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.legend(loc='upper right')
        ax.set_aspect('equal')

        return []

    anim = FuncAnimation(fig, animate, frames=frames, interval=100, blit=False)
    writer = PillowWriter(fps=10)
    anim.save(os.path.join(OUTPUT_DIR, 'decision_boundary.gif'), writer=writer)
    plt.close()
    print("Generated: decision_boundary.gif")


if __name__ == '__main__':
    print("Generating Perceptron & MLP visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    create_perceptron_single()
    create_xor_problem()
    create_xor_solution()
    create_mlp_architecture()
    create_decision_boundary()

    print("\nAll perceptron visualizations generated!")
