"""
Generate visualizations for Gradient Descent FAQ
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Common styling
FIGSIZE = (10, 6)
DPI = 150
COLORS = {
    'primary': '#2563eb',
    'secondary': '#dc2626',
    'tertiary': '#16a34a',
    'quaternary': '#9333ea',
    'fill': '#93c5fd',
}


def plot_learning_rate():
    """Generate learning rate effects visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI)

    # Create loss function
    def loss(x):
        return x**2

    x_range = np.linspace(-5, 5, 100)

    learning_rates = [0.01, 0.1, 0.9]
    titles = ['Too Small (lr=0.01)', 'Just Right (lr=0.1)', 'Too Large (lr=0.9)']

    for ax, lr, title in zip(axes, learning_rates, titles):
        ax.plot(x_range, loss(x_range), 'b-', linewidth=2, label='Loss Function')

        # Simulate gradient descent
        x = 4.0
        history = [x]
        for _ in range(20):
            grad = 2 * x  # derivative of x^2
            x = x - lr * grad
            history.append(x)
            if abs(x) > 10:  # divergence check
                break

        # Plot trajectory
        history = np.array(history)
        ax.plot(history, loss(history), 'ro-', markersize=8, linewidth=1.5, alpha=0.7)
        ax.scatter([history[0]], [loss(history[0])], color='green', s=150, zorder=5, label='Start')
        if len(history) > 1 and abs(history[-1]) < 10:
            ax.scatter([history[-1]], [loss(history[-1])], color='red', s=150, zorder=5, marker='*', label='End')

        ax.set_xlabel('Parameter Value', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-1, 20)
        ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('Effect of Learning Rate on Gradient Descent Convergence', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('gradient_descent_learning_rate.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: gradient_descent_learning_rate.png")


def plot_variants():
    """Generate gradient descent variants comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI)

    np.random.seed(42)

    # Create 2D loss landscape (elliptical)
    def loss_landscape(x, y):
        return 0.5 * x**2 + 2 * y**2

    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss_landscape(X, Y)

    variants = [
        ('Batch GD', False, COLORS['primary']),
        ('Mini-Batch GD', True, COLORS['secondary']),
        ('SGD', True, COLORS['tertiary'])
    ]
    noise_levels = [0, 0.3, 0.8]

    for ax, (name, noisy, color), noise in zip(axes, variants, noise_levels):
        ax.contour(X, Y, Z, levels=20, cmap='Blues', alpha=0.7)

        # Simulate descent
        x, y = 3.5, 3.5
        lr = 0.1
        history_x, history_y = [x], [y]

        for _ in range(50):
            grad_x = x + np.random.normal(0, noise)
            grad_y = 4 * y + np.random.normal(0, noise)
            x = x - lr * grad_x
            y = y - lr * grad_y
            history_x.append(x)
            history_y.append(y)

        ax.plot(history_x, history_y, 'o-', color=color, markersize=4, linewidth=1.5, alpha=0.8)
        ax.scatter([history_x[0]], [history_y[0]], color='green', s=150, zorder=5, marker='o')
        ax.scatter([0], [0], color='red', s=200, zorder=5, marker='*')

        ax.set_xlabel('w₁', fontsize=11)
        ax.set_ylabel('w₂', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

    plt.suptitle('Gradient Descent Variants: Trajectory Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('gradient_descent_variants.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: gradient_descent_variants.png")


def plot_saddle_point():
    """Generate saddle point visualization."""
    fig = plt.figure(figsize=(12, 5), dpi=DPI)

    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')

    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2  # Saddle function

    ax1.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8, edgecolor='none')
    ax1.scatter([0], [0], [0], color='black', s=100, zorder=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('3D View of Saddle Point\nf(x,y) = x² - y²', fontsize=12, fontweight='bold')

    # 2D contour view
    ax2 = fig.add_subplot(122)

    contour = ax2.contour(X, Y, Z, levels=20, cmap='coolwarm')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.scatter([0], [0], color='black', s=150, zorder=5, marker='X')

    # Add gradient arrows
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        r = 1.5
        x_start = r * np.cos(angle)
        y_start = r * np.sin(angle)
        dx = -2 * x_start * 0.3  # gradient direction
        dy = 2 * y_start * 0.3
        ax2.annotate('', xy=(x_start + dx, y_start + dy), xytext=(x_start, y_start),
                     arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title('Contour View: Gradient is Zero at Saddle Point\nbut it\'s not a minimum!', fontsize=12, fontweight='bold')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('gradient_descent_saddle_point.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: gradient_descent_saddle_point.png")


def plot_momentum():
    """Generate momentum comparison visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI)

    # Create narrow valley loss landscape
    def loss_landscape(x, y):
        return 0.1 * x**2 + 5 * y**2

    x_range = np.linspace(-10, 10, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss_landscape(X, Y)

    # Standard GD
    ax = axes[0]
    ax.contour(X, Y, Z, levels=30, cmap='Blues', alpha=0.7)

    x, y = 8, 2.5
    lr = 0.05
    history_x, history_y = [x], [y]

    for _ in range(100):
        grad_x = 0.2 * x
        grad_y = 10 * y
        x = x - lr * grad_x
        y = y - lr * grad_y
        history_x.append(x)
        history_y.append(y)

    ax.plot(history_x, history_y, 'o-', color=COLORS['primary'], markersize=3, linewidth=1, alpha=0.8)
    ax.scatter([history_x[0]], [history_y[0]], color='green', s=150, zorder=5)
    ax.scatter([0], [0], color='red', s=200, zorder=5, marker='*')
    ax.set_title('Standard Gradient Descent\n(Oscillates in narrow valley)', fontsize=12, fontweight='bold')
    ax.set_xlabel('w₁', fontsize=11)
    ax.set_ylabel('w₂', fontsize=11)

    # With Momentum
    ax = axes[1]
    ax.contour(X, Y, Z, levels=30, cmap='Blues', alpha=0.7)

    x, y = 8, 2.5
    vx, vy = 0, 0
    lr = 0.05
    momentum = 0.9
    history_x, history_y = [x], [y]

    for _ in range(100):
        grad_x = 0.2 * x
        grad_y = 10 * y
        vx = momentum * vx - lr * grad_x
        vy = momentum * vy - lr * grad_y
        x = x + vx
        y = y + vy
        history_x.append(x)
        history_y.append(y)

    ax.plot(history_x, history_y, 'o-', color=COLORS['tertiary'], markersize=3, linewidth=1, alpha=0.8)
    ax.scatter([history_x[0]], [history_y[0]], color='green', s=150, zorder=5)
    ax.scatter([0], [0], color='red', s=200, zorder=5, marker='*')
    ax.set_title('Gradient Descent with Momentum\n(Dampens oscillations, faster convergence)', fontsize=12, fontweight='bold')
    ax.set_xlabel('w₁', fontsize=11)
    ax.set_ylabel('w₂', fontsize=11)

    for ax in axes:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-3, 3)

    plt.suptitle('Momentum Accelerates Convergence Through Narrow Valleys', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('gradient_descent_momentum.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: gradient_descent_momentum.png")


def plot_optimizers():
    """Generate optimizer comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=DPI)

    # Create complex loss landscape
    def loss_landscape(x, y):
        return 0.5 * x**2 + 2 * y**2 + 0.5 * np.sin(3*x) + 0.5 * np.sin(3*y)

    x_range = np.linspace(-4, 4, 100)
    y_range = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss_landscape(X, Y)

    optimizers = [
        ('SGD', COLORS['primary']),
        ('SGD + Momentum', COLORS['secondary']),
        ('RMSprop', COLORS['tertiary']),
        ('Adam', COLORS['quaternary'])
    ]

    np.random.seed(42)

    for ax, (name, color) in zip(axes.flat, optimizers):
        ax.contour(X, Y, Z, levels=20, cmap='Blues', alpha=0.7)

        x, y = 3.5, 3.5
        lr = 0.1
        history_x, history_y = [x], [y]

        # Simplified optimizer simulation
        vx, vy = 0, 0
        sx, sy = 0, 0
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        for t in range(1, 51):
            grad_x = x + 1.5 * np.cos(3*x) + np.random.normal(0, 0.1)
            grad_y = 4*y + 1.5 * np.cos(3*y) + np.random.normal(0, 0.1)

            if name == 'SGD':
                x = x - lr * grad_x
                y = y - lr * grad_y
            elif name == 'SGD + Momentum':
                vx = 0.9 * vx - lr * grad_x
                vy = 0.9 * vy - lr * grad_y
                x = x + vx
                y = y + vy
            elif name == 'RMSprop':
                sx = 0.9 * sx + 0.1 * grad_x**2
                sy = 0.9 * sy + 0.1 * grad_y**2
                x = x - lr * grad_x / (np.sqrt(sx) + eps)
                y = y - lr * grad_y / (np.sqrt(sy) + eps)
            else:  # Adam
                vx = beta1 * vx + (1 - beta1) * grad_x
                vy = beta1 * vy + (1 - beta1) * grad_y
                sx = beta2 * sx + (1 - beta2) * grad_x**2
                sy = beta2 * sy + (1 - beta2) * grad_y**2
                vx_hat = vx / (1 - beta1**t)
                vy_hat = vy / (1 - beta1**t)
                sx_hat = sx / (1 - beta2**t)
                sy_hat = sy / (1 - beta2**t)
                x = x - lr * vx_hat / (np.sqrt(sx_hat) + eps)
                y = y - lr * vy_hat / (np.sqrt(sy_hat) + eps)

            history_x.append(x)
            history_y.append(y)

        ax.plot(history_x, history_y, 'o-', color=color, markersize=4, linewidth=1.5, alpha=0.8)
        ax.scatter([history_x[0]], [history_y[0]], color='green', s=150, zorder=5)
        ax.scatter([0], [0], color='red', s=200, zorder=5, marker='*')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('w₁', fontsize=10)
        ax.set_ylabel('w₂', fontsize=10)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

    plt.suptitle('Optimizer Trajectories on the Same Loss Landscape', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('gradient_descent_optimizers.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: gradient_descent_optimizers.png")


if __name__ == '__main__':
    print("Generating Gradient Descent visualizations...")
    plot_learning_rate()
    plot_variants()
    plot_saddle_point()
    plot_momentum()
    plot_optimizers()
    print("Done!")
