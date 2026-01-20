"""
Generate visualizations for Bias-Variance FAQ
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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
    'grid': '#e5e7eb'
}


def plot_dartboard():
    """Generate dartboard analogy for bias-variance."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=DPI)

    scenarios = [
        ('Low Bias, Low Variance', 0.0, 0.1, 'Ideal Model'),
        ('Low Bias, High Variance', 0.0, 0.8, 'Overfitting'),
        ('High Bias, Low Variance', 1.2, 0.1, 'Underfitting'),
        ('High Bias, High Variance', 1.0, 0.6, 'Poor Model')
    ]

    for ax, (title, bias, variance, desc) in zip(axes.flat, scenarios):
        # Draw target
        for r in [2, 1.5, 1, 0.5]:
            circle = Circle((0, 0), r, fill=False, color='gray', linewidth=1)
            ax.add_patch(circle)

        # Draw center (bullseye)
        center = Circle((0, 0), 0.1, color=COLORS['tertiary'], zorder=5)
        ax.add_patch(center)

        # Generate dart positions
        np.random.seed(42)
        n_darts = 20
        x = np.random.normal(bias, variance, n_darts)
        y = np.random.normal(0, variance, n_darts)

        ax.scatter(x, y, c=COLORS['secondary'], s=100, zorder=10, alpha=0.7, edgecolors='black')

        # Mark mean position
        ax.scatter([np.mean(x)], [np.mean(y)], c=COLORS['primary'], s=200,
                   marker='X', zorder=15, edgecolors='black', linewidth=2)

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title(f'{title}\n({desc})', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('The Dartboard Analogy: Bias vs Variance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('bias_variance_dartboard.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: bias_variance_dartboard.png")


def plot_complexity():
    """Generate model complexity vs error curve."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    complexity = np.linspace(1, 10, 100)

    # Bias decreases with complexity
    bias_sq = 5 * np.exp(-0.5 * complexity)

    # Variance increases with complexity
    variance = 0.2 * complexity ** 1.5

    # Total error
    total = bias_sq + variance + 0.5  # irreducible error

    ax.plot(complexity, bias_sq, label='Bias²', color=COLORS['primary'], linewidth=2.5)
    ax.plot(complexity, variance, label='Variance', color=COLORS['secondary'], linewidth=2.5)
    ax.plot(complexity, total, label='Total Error', color=COLORS['tertiary'], linewidth=3, linestyle='--')
    ax.axhline(y=0.5, color='gray', linestyle=':', label='Irreducible Error', linewidth=1.5)

    # Mark optimal complexity
    optimal_idx = np.argmin(total)
    ax.axvline(x=complexity[optimal_idx], color=COLORS['quaternary'], linestyle='--', alpha=0.7)
    ax.scatter([complexity[optimal_idx]], [total[optimal_idx]], color=COLORS['quaternary'],
               s=150, zorder=5, marker='*')
    ax.annotate('Optimal\nComplexity', xy=(complexity[optimal_idx], total[optimal_idx]),
                xytext=(complexity[optimal_idx]+1.5, total[optimal_idx]+1),
                fontsize=10, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))

    # Add regions
    ax.fill_between(complexity[:30], 0, 6, alpha=0.1, color=COLORS['primary'], label='Underfitting Zone')
    ax.fill_between(complexity[70:], 0, 6, alpha=0.1, color=COLORS['secondary'], label='Overfitting Zone')

    ax.set_xlabel('Model Complexity', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Bias-Variance Tradeoff: Model Complexity vs Error', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(1, 10)
    ax.set_ylim(0, 6)

    plt.tight_layout()
    plt.savefig('bias_variance_complexity.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: bias_variance_complexity.png")


def plot_learning_curves():
    """Generate learning curves for diagnosing bias/variance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI)

    train_sizes = np.linspace(10, 1000, 50)

    # High Bias (Underfitting)
    ax = axes[0]
    train_error = 0.3 - 0.05 * np.log(train_sizes / 10)
    val_error = 0.35 - 0.03 * np.log(train_sizes / 10)
    ax.plot(train_sizes, train_error, label='Training Error', color=COLORS['primary'], linewidth=2.5)
    ax.plot(train_sizes, val_error, label='Validation Error', color=COLORS['secondary'], linewidth=2.5)
    ax.fill_between(train_sizes, train_error, val_error, alpha=0.2, color='gray')
    ax.set_title('High Bias (Underfitting)', fontsize=12, fontweight='bold')
    ax.annotate('Both errors\nremain high', xy=(600, 0.25), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Optimal
    ax = axes[1]
    train_error = 0.15 - 0.08 * np.log(train_sizes / 10)
    val_error = 0.25 - 0.1 * np.log(train_sizes / 10)
    ax.plot(train_sizes, train_error, label='Training Error', color=COLORS['primary'], linewidth=2.5)
    ax.plot(train_sizes, val_error, label='Validation Error', color=COLORS['secondary'], linewidth=2.5)
    ax.fill_between(train_sizes, train_error, val_error, alpha=0.2, color='gray')
    ax.set_title('Good Fit (Optimal)', fontsize=12, fontweight='bold')
    ax.annotate('Curves\nconverge', xy=(600, 0.12), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # High Variance (Overfitting)
    ax = axes[2]
    train_error = 0.05 * np.ones_like(train_sizes)
    val_error = 0.35 - 0.15 * np.log(train_sizes / 10)
    ax.plot(train_sizes, train_error, label='Training Error', color=COLORS['primary'], linewidth=2.5)
    ax.plot(train_sizes, val_error, label='Validation Error', color=COLORS['secondary'], linewidth=2.5)
    ax.fill_between(train_sizes, train_error, val_error, alpha=0.2, color='gray')
    ax.set_title('High Variance (Overfitting)', fontsize=12, fontweight='bold')
    ax.annotate('Large gap\nbetween curves', xy=(400, 0.2), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    for ax in axes:
        ax.set_xlabel('Training Set Size', fontsize=11)
        ax.set_ylabel('Error', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(0, 0.5)

    plt.suptitle('Learning Curves for Diagnosing Bias/Variance Issues', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('bias_variance_learning_curves.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: bias_variance_learning_curves.png")


def plot_polynomial():
    """Generate polynomial fitting example."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI)

    np.random.seed(42)
    x = np.linspace(0, 10, 30)
    y_true = np.sin(x) + 0.5 * x
    y = y_true + np.random.normal(0, 0.5, len(x))

    x_smooth = np.linspace(0, 10, 200)
    y_true_smooth = np.sin(x_smooth) + 0.5 * x_smooth

    degrees = [1, 4, 15]
    titles = ['Underfitting (Degree 1)', 'Good Fit (Degree 4)', 'Overfitting (Degree 15)']

    for ax, degree, title in zip(axes, degrees, titles):
        # Fit polynomial
        coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(coeffs, x_smooth)

        ax.scatter(x, y, color=COLORS['primary'], s=60, alpha=0.7, label='Data', zorder=5)
        ax.plot(x_smooth, y_true_smooth, '--', color='gray', label='True Function', linewidth=1.5)
        ax.plot(x_smooth, y_pred, color=COLORS['secondary'], linewidth=2.5, label=f'Polynomial (deg={degree})')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_ylim(-2, 8)

    plt.suptitle('Polynomial Fitting: Underfitting vs Optimal vs Overfitting', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('bias_variance_polynomial.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: bias_variance_polynomial.png")


def plot_decomposition():
    """Generate error decomposition visualization."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    categories = ['Simple\nModel', 'Medium\nModel', 'Complex\nModel']
    x = np.arange(len(categories))
    width = 0.5

    bias_sq = [3.0, 1.0, 0.3]
    variance = [0.3, 0.8, 2.5]
    irreducible = [0.5, 0.5, 0.5]

    ax.bar(x, irreducible, width, label='Irreducible Error', color='gray', alpha=0.7)
    ax.bar(x, bias_sq, width, bottom=irreducible, label='Bias²', color=COLORS['primary'])
    ax.bar(x, variance, width, bottom=[i+b for i, b in zip(irreducible, bias_sq)],
           label='Variance', color=COLORS['secondary'])

    # Add total error line
    total = [i + b + v for i, b, v in zip(irreducible, bias_sq, variance)]
    ax.plot(x, total, 'o--', color=COLORS['tertiary'], linewidth=2, markersize=10, label='Total Error')

    # Mark optimal
    optimal_idx = np.argmin(total)
    ax.annotate('Optimal\n(Lowest Total Error)', xy=(optimal_idx, total[optimal_idx]),
                xytext=(optimal_idx + 0.5, total[optimal_idx] + 0.8),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Error Decomposition: Bias² + Variance + Irreducible Error', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig('bias_variance_decomposition.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: bias_variance_decomposition.png")


if __name__ == '__main__':
    print("Generating Bias-Variance visualizations...")
    plot_dartboard()
    plot_complexity()
    plot_learning_curves()
    plot_polynomial()
    plot_decomposition()
    print("Done!")
