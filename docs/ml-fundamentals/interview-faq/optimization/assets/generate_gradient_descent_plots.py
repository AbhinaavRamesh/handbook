"""
Gradient Descent & Optimization Visualizations
Generates plots for ML interview preparation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12

output_dir = os.path.dirname(os.path.abspath(__file__))


def plot_learning_rate_comparison():
    """Show effect of different learning rates"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Simple quadratic loss
    def loss(x):
        return x ** 2

    def gradient(x):
        return 2 * x

    learning_rates = [0.01, 0.1, 0.9]
    titles = ['Too Small (lr=0.01)\nSlow Convergence',
              'Just Right (lr=0.1)\nOptimal Convergence',
              'Too Large (lr=0.9)\nOscillation/Divergence']
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    x_range = np.linspace(-3, 3, 100)

    for ax, lr, title, color in zip(axes, learning_rates, titles, colors):
        # Plot loss surface
        ax.plot(x_range, loss(x_range), 'k-', linewidth=2, alpha=0.5)
        ax.fill_between(x_range, 0, loss(x_range), alpha=0.1, color='gray')

        # Gradient descent path
        x = 2.5
        path_x = [x]
        path_y = [loss(x)]

        for _ in range(20):
            x = x - lr * gradient(x)
            path_x.append(x)
            path_y.append(loss(x))
            if abs(x) > 5 or abs(x) < 0.001:
                break

        ax.plot(path_x, path_y, 'o-', color=color, markersize=8, linewidth=2,
                label=f'GD path (lr={lr})')
        ax.scatter([path_x[0]], [path_y[0]], s=150, c='red', marker='*',
                   zorder=5, label='Start')

        ax.set_xlabel('Parameter θ', fontsize=12)
        ax.set_ylabel('Loss L(θ)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold', color=color)
        ax.legend(loc='upper right')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, 8)

    plt.suptitle('Learning Rate: The Most Important Hyperparameter', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_descent_learning_rate.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: gradient_descent_learning_rate.png")


def plot_gd_variants():
    """Compare Batch GD, SGD, and Mini-batch GD"""
    fig, ax = plt.subplots(figsize=(10, 8))

    np.random.seed(42)

    # Define 2D loss landscape (bowl shape)
    def loss_2d(x, y):
        return 0.5 * x**2 + 2 * y**2

    # Create contour plot
    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss_2d(X, Y)

    contour = ax.contour(X, Y, Z, levels=15, cmap='coolwarm', alpha=0.7)
    ax.contourf(X, Y, Z, levels=15, cmap='coolwarm', alpha=0.3)

    # Batch GD - smooth path
    batch_x, batch_y = [2.5], [2.5]
    lr = 0.15
    for _ in range(20):
        batch_x.append(batch_x[-1] - lr * batch_x[-1])
        batch_y.append(batch_y[-1] - lr * 4 * batch_y[-1])
    ax.plot(batch_x, batch_y, 'b-o', linewidth=2, markersize=6, label='Batch GD', alpha=0.8)

    # SGD - noisy path
    sgd_x, sgd_y = [-2.5], [2.5]
    lr = 0.1
    for _ in range(30):
        noise_x = np.random.randn() * 0.3
        noise_y = np.random.randn() * 0.3
        sgd_x.append(sgd_x[-1] - lr * (sgd_x[-1] + noise_x))
        sgd_y.append(sgd_y[-1] - lr * (4 * sgd_y[-1] + noise_y))
    ax.plot(sgd_x, sgd_y, 'r-o', linewidth=1.5, markersize=4, label='SGD (noisy)', alpha=0.8)

    # Mini-batch - moderate noise
    mini_x, mini_y = [2.5], [-2.5]
    lr = 0.12
    for _ in range(25):
        noise_x = np.random.randn() * 0.1
        noise_y = np.random.randn() * 0.1
        mini_x.append(mini_x[-1] - lr * (mini_x[-1] + noise_x))
        mini_y.append(mini_y[-1] - lr * (4 * mini_y[-1] + noise_y))
    ax.plot(mini_x, mini_y, 'g-o', linewidth=2, markersize=5, label='Mini-batch GD', alpha=0.8)

    ax.scatter([0], [0], s=200, c='yellow', marker='*', edgecolors='black',
               linewidth=2, zorder=10, label='Minimum')

    ax.set_xlabel('θ₁', fontsize=13)
    ax.set_ylabel('θ₂', fontsize=13)
    ax.set_title('Gradient Descent Variants: Path Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_descent_variants.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: gradient_descent_variants.png")


def plot_saddle_point():
    """Visualize saddle points and local minima"""
    fig = plt.figure(figsize=(14, 5))

    # 3D saddle point
    ax1 = fig.add_subplot(121, projection='3d')
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2  # Saddle function

    ax1.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8, edgecolor='none')
    ax1.scatter([0], [0], [0], s=200, c='red', marker='o', label='Saddle Point')
    ax1.set_xlabel('θ₁')
    ax1.set_ylabel('θ₂')
    ax1.set_zlabel('Loss')
    ax1.set_title('Saddle Point\n(Min in one direction, Max in other)', fontsize=12, fontweight='bold')
    ax1.view_init(elev=25, azim=45)

    # 2D contour view
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='coolwarm')
    ax2.contourf(X, Y, Z, levels=20, cmap='coolwarm', alpha=0.3)
    ax2.scatter([0], [0], s=150, c='red', marker='X', zorder=5, label='Saddle Point')

    # Show gradient directions
    ax2.annotate('', xy=(0.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.annotate('', xy=(-0.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.annotate('', xy=(0, 0.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax2.annotate('', xy=(0, -0.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax2.text(0.7, 0, 'Min\ndirection', fontsize=10, color='blue', ha='left')
    ax2.text(0, 0.7, 'Max direction', fontsize=10, color='green', ha='center')

    ax2.set_xlabel('θ₁', fontsize=12)
    ax2.set_ylabel('θ₂', fontsize=12)
    ax2.set_title('Saddle Point: Contour View\n(Gradient = 0, but not a minimum)', fontsize=12, fontweight='bold')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_descent_saddle_point.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: gradient_descent_saddle_point.png")


def plot_momentum_comparison():
    """Compare vanilla GD vs GD with momentum"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    np.random.seed(42)

    # Loss landscape with elongated valley
    def loss(x, y):
        return 0.1 * x**2 + 5 * y**2

    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss(X, Y)

    # Vanilla GD
    ax1 = axes[0]
    ax1.contour(X, Y, Z, levels=20, cmap='coolwarm', alpha=0.7)
    ax1.contourf(X, Y, Z, levels=20, cmap='coolwarm', alpha=0.2)

    # GD path with oscillation
    lr = 0.18
    x, y = 4.0, 1.5
    path_x, path_y = [x], [y]
    for _ in range(30):
        grad_x = 0.2 * x
        grad_y = 10 * y
        x = x - lr * grad_x
        y = y - lr * grad_y
        path_x.append(x)
        path_y.append(y)

    ax1.plot(path_x, path_y, 'r-o', linewidth=2, markersize=5, alpha=0.8)
    ax1.scatter([0], [0], s=200, c='yellow', marker='*', edgecolors='black', linewidth=2, zorder=10)
    ax1.set_title('Vanilla Gradient Descent\n(Oscillation in steep direction)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('θ₁ (flat direction)', fontsize=12)
    ax1.set_ylabel('θ₂ (steep direction)', fontsize=12)

    # GD with Momentum
    ax2 = axes[1]
    ax2.contour(X, Y, Z, levels=20, cmap='coolwarm', alpha=0.7)
    ax2.contourf(X, Y, Z, levels=20, cmap='coolwarm', alpha=0.2)

    # Momentum path - smoother
    lr = 0.15
    momentum = 0.9
    x, y = 4.0, 1.5
    vx, vy = 0, 0
    path_x, path_y = [x], [y]
    for _ in range(30):
        grad_x = 0.2 * x
        grad_y = 10 * y
        vx = momentum * vx + lr * grad_x
        vy = momentum * vy + lr * grad_y
        x = x - vx
        y = y - vy
        path_x.append(x)
        path_y.append(y)

    ax2.plot(path_x, path_y, 'g-o', linewidth=2, markersize=5, alpha=0.8)
    ax2.scatter([0], [0], s=200, c='yellow', marker='*', edgecolors='black', linewidth=2, zorder=10)
    ax2.set_title('GD with Momentum\n(Smoother, faster convergence)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('θ₁ (flat direction)', fontsize=12)
    ax2.set_ylabel('θ₂ (steep direction)', fontsize=12)

    for ax in axes:
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2, 2)

    plt.suptitle('Momentum: Dampening Oscillations in Gradient Descent', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_descent_momentum.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: gradient_descent_momentum.png")


def plot_optimizer_comparison():
    """Compare SGD, Adam, RMSprop"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a more complex loss surface
    def loss(x, y):
        return 0.5 * x**2 + 3 * y**2 + 0.5 * np.sin(3*x) * np.sin(3*y)

    x_range = np.linspace(-3, 3, 100)
    y_range = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss(X, Y)

    ax.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
    ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.2)

    np.random.seed(42)

    # SGD path (slow, zigzag)
    sgd_path = [(2.5, 2.5)]
    lr = 0.08
    x, y = 2.5, 2.5
    for _ in range(50):
        x = x - lr * (x + 1.5 * np.cos(3*x) * np.sin(3*y))
        y = y - lr * (6*y + 1.5 * np.sin(3*x) * np.cos(3*y))
        sgd_path.append((x, y))
    sgd_path = np.array(sgd_path)
    ax.plot(sgd_path[:, 0], sgd_path[:, 1], 'b-o', linewidth=2, markersize=4,
            label='SGD', alpha=0.8)

    # Adam path (fast, smooth)
    adam_path = [(-2.5, 2.5)]
    lr = 0.3
    x, y = -2.5, 2.5
    beta1, beta2 = 0.9, 0.999
    m_x, m_y, v_x, v_y = 0, 0, 0, 0
    eps = 1e-8
    for t in range(1, 30):
        g_x = x + 1.5 * np.cos(3*x) * np.sin(3*y)
        g_y = 6*y + 1.5 * np.sin(3*x) * np.cos(3*y)
        m_x = beta1 * m_x + (1-beta1) * g_x
        m_y = beta1 * m_y + (1-beta1) * g_y
        v_x = beta2 * v_x + (1-beta2) * g_x**2
        v_y = beta2 * v_y + (1-beta2) * g_y**2
        m_x_hat = m_x / (1 - beta1**t)
        m_y_hat = m_y / (1 - beta1**t)
        v_x_hat = v_x / (1 - beta2**t)
        v_y_hat = v_y / (1 - beta2**t)
        x = x - lr * m_x_hat / (np.sqrt(v_x_hat) + eps)
        y = y - lr * m_y_hat / (np.sqrt(v_y_hat) + eps)
        adam_path.append((x, y))
    adam_path = np.array(adam_path)
    ax.plot(adam_path[:, 0], adam_path[:, 1], 'r-s', linewidth=2, markersize=6,
            label='Adam', alpha=0.8)

    ax.scatter([0], [0], s=250, c='yellow', marker='*', edgecolors='black',
               linewidth=2, zorder=10, label='Minimum')

    ax.set_xlabel('θ₁', fontsize=13)
    ax.set_ylabel('θ₂', fontsize=13)
    ax.set_title('Optimizer Comparison: SGD vs Adam', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_descent_optimizers.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: gradient_descent_optimizers.png")


if __name__ == '__main__':
    print("Generating Gradient Descent visualizations...")
    plot_learning_rate_comparison()
    plot_gd_variants()
    plot_saddle_point()
    plot_momentum_comparison()
    plot_optimizer_comparison()
    print("\nAll visualizations generated successfully!")
