"""
Bias-Variance Tradeoff Visualizations
Generates plots for ML interview preparation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12

output_dir = os.path.dirname(os.path.abspath(__file__))

def plot_dartboard_analogy():
    """Create the famous dartboard analogy visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    titles = [
        'Low Bias, Low Variance\n(Ideal)',
        'High Bias, Low Variance\n(Consistent but Wrong)',
        'Low Bias, High Variance\n(Right on Average)',
        'High Bias, High Variance\n(Worst Case)'
    ]

    # Define dart positions for each scenario
    np.random.seed(42)
    scenarios = [
        # Low bias, low variance - clustered near center
        np.random.randn(10, 2) * 0.15,
        # High bias, low variance - clustered away from center
        np.random.randn(10, 2) * 0.15 + np.array([0.6, 0.4]),
        # Low bias, high variance - scattered around center
        np.random.randn(10, 2) * 0.5,
        # High bias, high variance - scattered away from center
        np.random.randn(10, 2) * 0.5 + np.array([0.4, 0.3]),
    ]

    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    for idx, (ax, darts, title, color) in enumerate(zip(axes.flat, scenarios, titles, colors)):
        # Draw target circles
        for r in [1.0, 0.7, 0.4, 0.15]:
            circle = Circle((0, 0), r, fill=False, color='gray', linestyle='--', alpha=0.5)
            ax.add_patch(circle)

        # Draw bullseye
        bullseye = Circle((0, 0), 0.05, fill=True, color='red', alpha=0.8)
        ax.add_patch(bullseye)

        # Plot darts
        ax.scatter(darts[:, 0], darts[:, 1], s=100, c=color, edgecolors='black',
                   linewidth=1.5, zorder=5, alpha=0.8)

        # Mark mean position
        mean_pos = darts.mean(axis=0)
        ax.scatter(mean_pos[0], mean_pos[1], s=200, c='yellow', marker='*',
                   edgecolors='black', linewidth=2, zorder=6, label='Mean')

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Bias-Variance Tradeoff: Dartboard Analogy', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_variance_dartboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: bias_variance_dartboard.png")


def plot_model_complexity_curve():
    """Create the U-shaped error curve vs model complexity"""
    fig, ax = plt.subplots(figsize=(10, 7))

    complexity = np.linspace(0.1, 10, 100)

    # Bias decreases with complexity
    bias_squared = 2 / (complexity + 0.5) ** 0.8

    # Variance increases with complexity
    variance = 0.1 * complexity ** 1.2

    # Irreducible error is constant
    irreducible = np.ones_like(complexity) * 0.3

    # Total error
    total_error = bias_squared + variance + irreducible

    # Find optimal point
    optimal_idx = np.argmin(total_error)
    optimal_complexity = complexity[optimal_idx]

    ax.fill_between(complexity, 0, irreducible, alpha=0.3, color='gray', label='Irreducible Error')
    ax.plot(complexity, bias_squared, 'b-', linewidth=2.5, label='Bias²')
    ax.plot(complexity, variance, 'r-', linewidth=2.5, label='Variance')
    ax.plot(complexity, total_error, 'g-', linewidth=3, label='Total Error')

    # Mark optimal point
    ax.axvline(x=optimal_complexity, color='green', linestyle='--', alpha=0.7)
    ax.scatter([optimal_complexity], [total_error[optimal_idx]], s=150, c='green',
               zorder=5, edgecolors='black', linewidth=2)

    # Add annotations
    ax.annotate('Optimal\nComplexity', xy=(optimal_complexity, total_error[optimal_idx]),
                xytext=(optimal_complexity + 1.5, total_error[optimal_idx] + 0.5),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='green'))

    # Mark underfitting and overfitting regions
    ax.annotate('UNDERFITTING\n(High Bias)', xy=(1.5, 2.5), fontsize=12,
                color='blue', ha='center', fontweight='bold')
    ax.annotate('OVERFITTING\n(High Variance)', xy=(8, 2.5), fontsize=12,
                color='red', ha='center', fontweight='bold')

    ax.set_xlabel('Model Complexity', fontsize=13)
    ax.set_ylabel('Error', fontsize=13)
    ax.set_title('Bias-Variance Tradeoff: Error vs Model Complexity', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_variance_complexity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: bias_variance_complexity.png")


def plot_learning_curves():
    """Create learning curves showing high bias vs high variance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    train_sizes = np.linspace(10, 100, 50)

    # High Bias (Underfitting)
    ax1 = axes[0]
    train_error_bias = 0.35 - 0.05 * np.log(train_sizes / 10)
    val_error_bias = 0.40 - 0.03 * np.log(train_sizes / 10)

    ax1.plot(train_sizes, train_error_bias, 'b-', linewidth=2.5, label='Training Error')
    ax1.plot(train_sizes, val_error_bias, 'r-', linewidth=2.5, label='Validation Error')
    ax1.fill_between(train_sizes, train_error_bias, val_error_bias, alpha=0.2, color='gray')
    ax1.set_xlabel('Training Set Size', fontsize=12)
    ax1.set_ylabel('Error', fontsize=12)
    ax1.set_title('High Bias (Underfitting)\nBoth errors converge at HIGH level', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 0.6)
    ax1.annotate('Small gap\nHigh error', xy=(70, 0.36), fontsize=11,
                 color='purple', ha='center', fontweight='bold')

    # High Variance (Overfitting)
    ax2 = axes[1]
    train_error_var = 0.05 + 0.02 * np.log(train_sizes / 10)
    val_error_var = 0.45 - 0.15 * np.log(train_sizes / 10)

    ax2.plot(train_sizes, train_error_var, 'b-', linewidth=2.5, label='Training Error')
    ax2.plot(train_sizes, val_error_var, 'r-', linewidth=2.5, label='Validation Error')
    ax2.fill_between(train_sizes, train_error_var, val_error_var, alpha=0.2, color='orange')
    ax2.set_xlabel('Training Set Size', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_title('High Variance (Overfitting)\nLarge gap between train/validation', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 0.6)
    ax2.annotate('Large gap\n(decreases with more data)', xy=(50, 0.25), fontsize=11,
                 color='orange', ha='center', fontweight='bold')

    plt.suptitle('Learning Curves: Diagnosing Bias vs Variance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_variance_learning_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: bias_variance_learning_curves.png")


def plot_polynomial_fitting():
    """Show underfitting, optimal fit, and overfitting with polynomial regression"""
    np.random.seed(42)

    # Generate data
    X = np.linspace(0, 1, 30)
    y_true = np.sin(2 * np.pi * X)
    y = y_true + np.random.randn(len(X)) * 0.2

    X_plot = np.linspace(0, 1, 100)
    y_true_plot = np.sin(2 * np.pi * X_plot)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    degrees = [1, 4, 15]
    titles = ['Degree 1 (Underfitting)', 'Degree 4 (Good Fit)', 'Degree 15 (Overfitting)']
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for ax, degree, title, color in zip(axes, degrees, titles, colors):
        # Fit polynomial
        coeffs = np.polyfit(X, y, degree)
        y_pred = np.polyval(coeffs, X_plot)

        ax.scatter(X, y, c='black', s=50, alpha=0.6, label='Data points', zorder=5)
        ax.plot(X_plot, y_true_plot, 'g--', linewidth=2, alpha=0.5, label='True function')
        ax.plot(X_plot, y_pred, color=color, linewidth=3, label=f'Polynomial fit')

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold', color=color)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(-2, 2)

    plt.suptitle('Polynomial Regression: Bias-Variance in Action', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_variance_polynomial.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: bias_variance_polynomial.png")


def plot_error_decomposition():
    """Create a bar chart showing error decomposition"""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['Very Simple\n(High Bias)', 'Simple', 'Moderate', 'Complex', 'Very Complex\n(High Variance)']
    bias_squared = [2.5, 1.5, 0.5, 0.2, 0.1]
    variance = [0.1, 0.2, 0.4, 1.0, 2.0]
    irreducible = [0.3, 0.3, 0.3, 0.3, 0.3]

    x = np.arange(len(models))
    width = 0.6

    bars1 = ax.bar(x, bias_squared, width, label='Bias²', color='#3498db')
    bars2 = ax.bar(x, variance, width, bottom=bias_squared, label='Variance', color='#e74c3c')
    bars3 = ax.bar(x, irreducible, width, bottom=np.array(bias_squared) + np.array(variance),
                   label='Irreducible Error', color='#95a5a6')

    # Add total error line
    total = np.array(bias_squared) + np.array(variance) + np.array(irreducible)
    ax.plot(x, total, 'go-', markersize=10, linewidth=2, label='Total Error')

    # Mark optimal
    optimal_idx = np.argmin(total)
    ax.annotate('Optimal', xy=(optimal_idx, total[optimal_idx] + 0.1),
                fontsize=12, ha='center', fontweight='bold', color='green')

    ax.set_ylabel('Error', fontsize=13)
    ax.set_xlabel('Model Complexity', fontsize=13)
    ax.set_title('Error Decomposition: Bias² + Variance + Irreducible Error', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bias_variance_decomposition.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: bias_variance_decomposition.png")


if __name__ == '__main__':
    print("Generating Bias-Variance Tradeoff visualizations...")
    plot_dartboard_analogy()
    plot_model_complexity_curve()
    plot_learning_curves()
    plot_polynomial_fitting()
    plot_error_decomposition()
    print("\nAll visualizations generated successfully!")
