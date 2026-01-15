#!/usr/bin/env python3
"""
Generate visualizations for Optimizers module.

Outputs:
- opt_momentum.gif - SGD vs Momentum comparison
- opt_adam.svg - Adam components visualization
- opt_lr_schedules.svg - Learning rate schedules comparison
- opt_comparison.gif - Multiple optimizers comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Style settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

COLORS = {
    'sgd': '#FF6B6B',
    'momentum': '#4ECDC4',
    'adam': '#45B7D1',
    'rmsprop': '#96CEB4',
}


def rosenbrock(x, y):
    """Rosenbrock function - a common optimization test function."""
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_grad(x, y):
    """Gradient of Rosenbrock function."""
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])


def create_opt_momentum():
    """Create SGD vs Momentum comparison animation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create contour plot data
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    # SGD path
    sgd_path = [(-1.5, 2.5)]
    lr_sgd = 0.001

    # Momentum path
    mom_path = [(-1.5, 2.5)]
    velocity = np.array([0.0, 0.0])
    lr_mom = 0.001
    beta = 0.9

    # Pre-compute paths
    for _ in range(200):
        # SGD update
        pos = sgd_path[-1]
        grad = rosenbrock_grad(pos[0], pos[1])
        new_pos = (pos[0] - lr_sgd * grad[0], pos[1] - lr_sgd * grad[1])
        sgd_path.append(new_pos)

        # Momentum update
        pos = mom_path[-1]
        grad = rosenbrock_grad(pos[0], pos[1])
        velocity = beta * velocity + grad
        new_pos = (pos[0] - lr_mom * velocity[0], pos[1] - lr_mom * velocity[1])
        mom_path.append(new_pos)

    def animate(frame):
        for ax in axes:
            ax.clear()

        idx = min(frame * 5, len(sgd_path) - 1)

        # Left: SGD
        ax = axes[0]
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
        ax.set_title(f'SGD (Step {idx})', fontsize=12, fontweight='bold')

        path = np.array(sgd_path[:idx + 1])
        ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label='SGD path')
        ax.scatter(path[-1, 0], path[-1, 1], c='red', s=100, zorder=5, marker='o')
        ax.scatter(1, 1, c='green', s=100, marker='*', label='Minimum')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper left')

        # Right: Momentum
        ax = axes[1]
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
        ax.set_title(f'SGD + Momentum (Step {idx})', fontsize=12, fontweight='bold')

        path = np.array(mom_path[:idx + 1])
        ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Momentum path')
        ax.scatter(path[-1, 0], path[-1, 1], c='blue', s=100, zorder=5, marker='o')
        ax.scatter(1, 1, c='green', s=100, marker='*', label='Minimum')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper left')

        return []

    anim = FuncAnimation(fig, animate, frames=40, interval=200, blit=False)
    anim.save(os.path.join(OUTPUT_DIR, 'opt_momentum.gif'), writer=PillowWriter(fps=5))
    plt.close()
    print("Generated: opt_momentum.gif")


def create_opt_adam():
    """Create Adam components visualization."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Adam Optimizer Components', fontsize=14, fontweight='bold')

    # Gradient input
    rect = FancyBboxPatch((0.5, 3.5), 1.5, 1.5,
                          boxstyle="round,pad=0.05",
                          facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.25, 4.25, 'g_t\nGradient', ha='center', va='center', fontsize=10, fontweight='bold')

    # First moment (m) - momentum
    rect = FancyBboxPatch((3.5, 5.5), 2.5, 1.5,
                          boxstyle="round,pad=0.05",
                          facecolor='#4ECDC4', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(4.75, 6.25, 'm_t = β₁m_{t-1} + (1-β₁)g_t', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(4.75, 5.2, 'First Moment\n(Momentum)', ha='center', fontsize=9)

    # Second moment (v) - RMSprop
    rect = FancyBboxPatch((3.5, 1.5), 2.5, 1.5,
                          boxstyle="round,pad=0.05",
                          facecolor='#FF6B6B', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(4.75, 2.25, 'v_t = β₂v_{t-1} + (1-β₂)g_t²', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(4.75, 1.2, 'Second Moment\n(Adaptive LR)', ha='center', fontsize=9)

    # Arrows from gradient
    ax.annotate('', xy=(3.4, 6), xytext=(2.1, 4.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(3.4, 2.5), xytext=(2.1, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Bias correction
    rect = FancyBboxPatch((7, 5.5), 2, 1.5,
                          boxstyle="round,pad=0.05",
                          facecolor='#95E1D3', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(8, 6.25, 'm̂_t = m_t/(1-β₁ᵗ)', ha='center', va='center', fontsize=10)

    rect = FancyBboxPatch((7, 1.5), 2, 1.5,
                          boxstyle="round,pad=0.05",
                          facecolor='#F38181', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(8, 2.25, 'v̂_t = v_t/(1-β₂ᵗ)', ha='center', va='center', fontsize=10)

    ax.annotate('', xy=(6.9, 6.25), xytext=(6.1, 6.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(6.9, 2.25), xytext=(6.1, 2.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.text(6.5, 4.5, 'Bias\nCorrection', ha='center', fontsize=9, fontweight='bold')

    # Final update
    rect = FancyBboxPatch((10.5, 3), 3, 2,
                          boxstyle="round,pad=0.05",
                          facecolor='#45B7D1', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(12, 4, 'θ_t = θ_{t-1} - α·m̂_t/(√v̂_t + ε)', ha='center', va='center', fontsize=11, fontweight='bold')

    ax.annotate('', xy=(10.4, 4.5), xytext=(9.1, 6),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(10.4, 3.5), xytext=(9.1, 2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Hyperparameters
    ax.text(7, 0.5, 'Typical values: β₁=0.9, β₂=0.999, ε=1e-8, α=0.001', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    plt.savefig(os.path.join(OUTPUT_DIR, 'opt_adam.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: opt_adam.svg")


def create_opt_lr_schedules():
    """Create learning rate schedules comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = np.arange(100)
    initial_lr = 0.1

    # Step decay
    ax = axes[0, 0]
    step_lr = initial_lr * (0.1 ** (epochs // 30))
    ax.plot(epochs, step_lr, 'b-', linewidth=2)
    ax.set_title('Step Decay', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_ylim(0, 0.12)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=30, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=60, color='gray', linestyle='--', alpha=0.5)
    ax.text(15, 0.08, 'Drop by 10x\nevery 30 epochs', fontsize=9)

    # Exponential decay
    ax = axes[0, 1]
    exp_lr = initial_lr * np.exp(-0.05 * epochs)
    ax.plot(epochs, exp_lr, 'g-', linewidth=2)
    ax.set_title('Exponential Decay', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_ylim(0, 0.12)
    ax.grid(True, alpha=0.3)
    ax.text(50, 0.06, 'lr = lr₀ × e^(-kt)', fontsize=9)

    # Cosine annealing
    ax = axes[1, 0]
    cosine_lr = initial_lr * 0.5 * (1 + np.cos(np.pi * epochs / 100))
    ax.plot(epochs, cosine_lr, 'r-', linewidth=2)
    ax.set_title('Cosine Annealing', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_ylim(0, 0.12)
    ax.grid(True, alpha=0.3)
    ax.text(50, 0.08, 'Smooth decay\nto minimum', fontsize=9)

    # Warmup + decay
    ax = axes[1, 1]
    warmup_epochs = 10
    warmup_lr = np.zeros_like(epochs, dtype=float)
    warmup_lr[:warmup_epochs] = initial_lr * epochs[:warmup_epochs] / warmup_epochs
    warmup_lr[warmup_epochs:] = initial_lr * np.exp(-0.03 * (epochs[warmup_epochs:] - warmup_epochs))
    ax.plot(epochs, warmup_lr, 'm-', linewidth=2)
    ax.set_title('Warmup + Exponential Decay', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_ylim(0, 0.12)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=warmup_epochs, color='gray', linestyle='--', alpha=0.5)
    ax.text(5, 0.03, 'Warmup', fontsize=9)
    ax.text(50, 0.06, 'Decay', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'opt_lr_schedules.svg'), format='svg', bbox_inches='tight')
    plt.close()
    print("Generated: opt_lr_schedules.svg")


def create_opt_comparison():
    """Create optimizer comparison animation on 2D loss surface."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a challenging loss surface (saddle point + valley)
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X ** 2 - Y ** 2 + 0.5 * X ** 2 * Y ** 2  # Saddle-like surface

    def loss_grad(pos):
        x, y = pos
        dx = 2 * x + x * y ** 2
        dy = -2 * y + x ** 2 * y
        return np.array([dx, dy])

    # Initialize optimizers
    start = np.array([-2.5, 2.5])
    lr = 0.05

    paths = {
        'SGD': [start.copy()],
        'Momentum': [start.copy()],
        'RMSprop': [start.copy()],
        'Adam': [start.copy()],
    }

    # State variables
    v_mom = np.zeros(2)
    v_rms = np.zeros(2)
    m_adam = np.zeros(2)
    v_adam = np.zeros(2)

    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    # Pre-compute paths
    for t in range(1, 100):
        # SGD
        pos = paths['SGD'][-1]
        grad = loss_grad(pos)
        paths['SGD'].append(pos - lr * grad)

        # Momentum
        pos = paths['Momentum'][-1]
        grad = loss_grad(pos)
        v_mom = 0.9 * v_mom + grad
        paths['Momentum'].append(pos - lr * v_mom)

        # RMSprop
        pos = paths['RMSprop'][-1]
        grad = loss_grad(pos)
        v_rms = 0.9 * v_rms + 0.1 * grad ** 2
        paths['RMSprop'].append(pos - lr * grad / (np.sqrt(v_rms) + eps))

        # Adam
        pos = paths['Adam'][-1]
        grad = loss_grad(pos)
        m_adam = beta1 * m_adam + (1 - beta1) * grad
        v_adam = beta2 * v_adam + (1 - beta2) * grad ** 2
        m_hat = m_adam / (1 - beta1 ** t)
        v_hat = v_adam / (1 - beta2 ** t)
        paths['Adam'].append(pos - lr * m_hat / (np.sqrt(v_hat) + eps))

    colors = {'SGD': '#FF6B6B', 'Momentum': '#4ECDC4', 'RMSprop': '#96CEB4', 'Adam': '#45B7D1'}

    def animate(frame):
        ax.clear()
        ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
        ax.set_title(f'Optimizer Comparison (Step {frame * 2})', fontsize=14, fontweight='bold')

        idx = min(frame * 2, 99)

        for name, path in paths.items():
            path_arr = np.array(path[:idx + 1])
            ax.plot(path_arr[:, 0], path_arr[:, 1], '-', color=colors[name], linewidth=2, label=name)
            ax.scatter(path_arr[-1, 0], path_arr[-1, 1], c=colors[name], s=100, zorder=5)

        ax.scatter(0, 0, c='green', s=150, marker='*', label='Minimum', zorder=10)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        return []

    anim = FuncAnimation(fig, animate, frames=50, interval=200, blit=False)
    anim.save(os.path.join(OUTPUT_DIR, 'opt_comparison.gif'), writer=PillowWriter(fps=5))
    plt.close()
    print("Generated: opt_comparison.gif")


if __name__ == '__main__':
    print("Generating Optimizer visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    create_opt_momentum()
    create_opt_adam()
    create_opt_lr_schedules()
    create_opt_comparison()

    print("\nAll Optimizer visualizations generated!")
