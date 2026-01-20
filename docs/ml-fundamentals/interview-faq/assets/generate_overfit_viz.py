#!/usr/bin/env python3
"""
Generate visualizations for Overfitting/Underfitting FAQ.
Creates educational diagrams for ML interview preparation.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set consistent styling
plt.style.use('seaborn-v0_8-whitegrid')

# Output directory (same as script location)
OUTPUT_DIR = Path(__file__).parent


def plot_learning_curves_diagnosis():
    """
    Create 3-panel learning curves showing underfit, good fit, and overfit patterns.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    epochs = np.linspace(1, 100, 100)

    # Panel 1: Underfitting (High Bias)
    ax1 = axes[0]
    train_underfit = 0.35 - 0.05 * (1 - np.exp(-epochs/20))
    val_underfit = 0.38 - 0.05 * (1 - np.exp(-epochs/25))
    ax1.plot(epochs, train_underfit, 'b-', linewidth=2.5, label='Training Error')
    ax1.plot(epochs, val_underfit, 'r-', linewidth=2.5, label='Validation Error')
    ax1.fill_between(epochs, train_underfit, val_underfit, alpha=0.2, color='gray')
    ax1.set_xlabel('Training Epochs', fontsize=12)
    ax1.set_ylabel('Error', fontsize=12)
    ax1.set_title('Underfitting (High Bias)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(0, 0.6)
    ax1.annotate('Small gap\nBoth errors high', xy=(70, 0.36), fontsize=10,
                 ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: Good Fit
    ax2 = axes[1]
    train_good = 0.25 * np.exp(-epochs/30) + 0.05
    val_good = 0.30 * np.exp(-epochs/35) + 0.08
    ax2.plot(epochs, train_good, 'b-', linewidth=2.5, label='Training Error')
    ax2.plot(epochs, val_good, 'r-', linewidth=2.5, label='Validation Error')
    ax2.fill_between(epochs, train_good, val_good, alpha=0.2, color='green')
    ax2.set_xlabel('Training Epochs', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_title('Good Fit', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim(0, 0.6)
    ax2.annotate('Small gap\nBoth errors low', xy=(70, 0.12), fontsize=10,
                 ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Panel 3: Overfitting (High Variance)
    ax3 = axes[2]
    train_overfit = 0.3 * np.exp(-epochs/15) + 0.02
    val_overfit = 0.15 * np.exp(-epochs/30) + 0.12 + 0.15 * (1 - np.exp(-epochs/40))
    ax3.plot(epochs, train_overfit, 'b-', linewidth=2.5, label='Training Error')
    ax3.plot(epochs, val_overfit, 'r-', linewidth=2.5, label='Validation Error')
    ax3.fill_between(epochs, train_overfit, val_overfit, alpha=0.2, color='red')
    ax3.set_xlabel('Training Epochs', fontsize=12)
    ax3.set_ylabel('Error', fontsize=12)
    ax3.set_title('Overfitting (High Variance)', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_ylim(0, 0.6)
    ax3.annotate('Large gap\nTrain low, val high', xy=(70, 0.20), fontsize=10,
                 ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'learning_curves_diagnosis.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: learning_curves_diagnosis.png")


def plot_bias_variance_tradeoff():
    """
    Create the classic U-curve showing bias^2, variance, and total error.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    complexity = np.linspace(0.1, 10, 200)

    # Bias decreases with complexity
    bias_squared = 2.5 * np.exp(-0.4 * complexity) + 0.1

    # Variance increases with complexity
    variance = 0.05 * complexity**1.5

    # Irreducible error (constant)
    irreducible = np.ones_like(complexity) * 0.15

    # Total error
    total_error = bias_squared + variance + irreducible

    # Find optimal point
    optimal_idx = np.argmin(total_error)
    optimal_complexity = complexity[optimal_idx]
    optimal_error = total_error[optimal_idx]

    # Plot curves
    ax.plot(complexity, bias_squared, 'b-', linewidth=2.5, label='Bias$^2$')
    ax.plot(complexity, variance, 'orange', linewidth=2.5, label='Variance')
    ax.plot(complexity, irreducible, 'gray', linewidth=2, linestyle='--', label='Irreducible Error')
    ax.plot(complexity, total_error, 'r-', linewidth=3, label='Total Error')

    # Mark optimal point
    ax.scatter([optimal_complexity], [optimal_error], s=150, c='green', zorder=5,
               edgecolors='black', linewidth=2)
    ax.axvline(x=optimal_complexity, color='green', linestyle=':', alpha=0.7)
    ax.annotate('Optimal\nComplexity', xy=(optimal_complexity, optimal_error),
                xytext=(optimal_complexity + 1.5, optimal_error + 0.4),
                fontsize=11, ha='left',
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Add region labels
    ax.text(1.5, 1.8, 'Underfitting\nRegion', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(8, 1.8, 'Overfitting\nRegion', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    ax.set_xlabel('Model Complexity', fontsize=13)
    ax.set_ylabel('Error', fontsize=13)
    ax.set_title('Bias-Variance Tradeoff', fontsize=15, fontweight='bold')
    ax.legend(loc='upper center', fontsize=11, ncol=4)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bias_variance_tradeoff.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: bias_variance_tradeoff.png")


def plot_model_complexity_vs_error():
    """
    Create model complexity vs training/validation error with optimal point marked.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    complexity = np.linspace(1, 15, 100)

    # Training error decreases with complexity
    train_error = 0.6 * np.exp(-0.25 * complexity) + 0.02

    # Validation error has U-shape
    val_error = 0.4 * np.exp(-0.3 * complexity) + 0.05 * (complexity - 5)**2 / 50 + 0.08

    # Find optimal point (minimum validation error)
    optimal_idx = np.argmin(val_error)
    optimal_complexity = complexity[optimal_idx]
    optimal_val_error = val_error[optimal_idx]

    ax.plot(complexity, train_error, 'b-', linewidth=2.5, label='Training Error')
    ax.plot(complexity, val_error, 'r-', linewidth=2.5, label='Validation Error')

    # Mark optimal complexity
    ax.scatter([optimal_complexity], [optimal_val_error], s=150, c='green', zorder=5,
               edgecolors='black', linewidth=2)
    ax.axvline(x=optimal_complexity, color='green', linestyle=':', alpha=0.7, linewidth=2)

    # Fill regions
    ax.fill_between(complexity[complexity < optimal_complexity], 0, 0.8,
                    alpha=0.1, color='blue', label='_nolegend_')
    ax.fill_between(complexity[complexity >= optimal_complexity], 0, 0.8,
                    alpha=0.1, color='red', label='_nolegend_')

    # Add annotations
    ax.annotate('Optimal\nComplexity', xy=(optimal_complexity, optimal_val_error),
                xytext=(optimal_complexity + 2.5, optimal_val_error + 0.15),
                fontsize=11, ha='left',
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax.text(3, 0.6, 'Underfitting\n(Too Simple)', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(12, 0.6, 'Overfitting\n(Too Complex)', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Generalization gap annotation
    mid_idx = 70
    ax.annotate('', xy=(complexity[mid_idx], train_error[mid_idx]),
                xytext=(complexity[mid_idx], val_error[mid_idx]),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text(complexity[mid_idx] + 0.5, (train_error[mid_idx] + val_error[mid_idx])/2,
            'Generalization\nGap', fontsize=10, va='center',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))

    ax.set_xlabel('Model Complexity', fontsize=13)
    ax.set_ylabel('Error', fontsize=13)
    ax.set_title('Model Complexity vs. Error', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(1, 15)
    ax.set_ylim(0, 0.8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_complexity_vs_error.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: model_complexity_vs_error.png")


def plot_polynomial_degree_comparison():
    """
    Show polynomial fits with degree 1, 3, and 15 to demonstrate underfitting/overfitting.
    """
    np.random.seed(42)

    # Generate true underlying function (quadratic) with noise
    n_points = 25
    x = np.linspace(0, 1, n_points)
    y_true = 0.5 + 2*x - 3*x**2
    noise = np.random.normal(0, 0.15, n_points)
    y = y_true + noise

    # Fine grid for plotting fits
    x_fine = np.linspace(0, 1, 200)
    y_true_fine = 0.5 + 2*x_fine - 3*x_fine**2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    degrees = [1, 3, 15]
    titles = ['Degree 1: Underfitting', 'Degree 3: Good Fit', 'Degree 15: Overfitting']
    colors = ['blue', 'green', 'red']

    for ax, degree, title, color in zip(axes, degrees, titles, colors):
        # Fit polynomial
        coeffs = np.polyfit(x, y, degree)
        y_fit = np.polyval(coeffs, x_fine)

        # Calculate training error
        y_train_pred = np.polyval(coeffs, x)
        train_mse = np.mean((y - y_train_pred)**2)

        # Plot
        ax.scatter(x, y, s=60, c='black', alpha=0.7, label='Data points', zorder=3)
        ax.plot(x_fine, y_true_fine, 'gray', linestyle='--', linewidth=2,
                label='True function', alpha=0.7)
        ax.plot(x_fine, y_fit, color=color, linewidth=2.5, label=f'Degree {degree} fit')

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', color=color)
        ax.legend(loc='lower left', fontsize=9)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-1.5, 1.5)
        ax.text(0.95, 0.95, f'Train MSE: {train_mse:.3f}', transform=ax.transAxes,
                fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'polynomial_degree_comparison.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: polynomial_degree_comparison.png")


def plot_double_descent():
    """
    Create the double descent phenomenon curve.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Create piecewise function for double descent
    complexity = np.linspace(0.1, 10, 500)

    # Interpolation threshold around complexity = 5
    threshold = 5.0

    # First descent (classical regime)
    test_error = np.zeros_like(complexity)

    for i, c in enumerate(complexity):
        if c < threshold - 0.5:
            # Classical U-curve behavior
            test_error[i] = 0.8 * np.exp(-0.5 * c) + 0.05 * c + 0.2
        elif c < threshold + 0.5:
            # Sharp peak at interpolation threshold
            dist_to_threshold = abs(c - threshold)
            peak_height = 1.5
            test_error[i] = peak_height * np.exp(-10 * dist_to_threshold**2) + 0.4
        else:
            # Second descent (overparameterized regime)
            test_error[i] = 0.6 * np.exp(-0.3 * (c - threshold)) + 0.15

    # Smooth the curve a bit
    from scipy.ndimage import gaussian_filter1d
    test_error = gaussian_filter1d(test_error, sigma=3)

    # Also create classical U-curve for comparison
    classical = 0.8 * np.exp(-0.4 * complexity) + 0.03 * complexity**1.2 + 0.1

    ax.plot(complexity, test_error, 'b-', linewidth=3, label='Double Descent (Modern)')
    ax.plot(complexity, classical, 'gray', linewidth=2, linestyle='--',
            label='Classical U-curve', alpha=0.7)

    # Mark regions
    ax.axvline(x=threshold, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax.annotate('Interpolation\nThreshold', xy=(threshold, 1.3), fontsize=11,
                ha='center', color='red',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Region labels
    ax.text(2.5, 0.15, 'Underparameterized\nRegime', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(7.5, 0.15, 'Overparameterized\nRegime', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    ax.set_xlabel('Model Complexity (Parameters)', fontsize=13)
    ax.set_ylabel('Test Error', fontsize=13)
    ax.set_title('Double Descent Phenomenon', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'double_descent.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: double_descent.png")


def plot_train_val_test_split():
    """
    Create a diagram showing train/validation/test split workflow.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Remove axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Draw the main data bar
    bar_y = 4.5
    bar_height = 0.8

    # Training set (60%)
    train_rect = plt.Rectangle((0.5, bar_y), 3.6, bar_height,
                                 facecolor='#4CAF50', edgecolor='black', linewidth=2)
    ax.add_patch(train_rect)
    ax.text(2.3, bar_y + bar_height/2, 'Training Set\n(60%)', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')

    # Validation set (20%)
    val_rect = plt.Rectangle((4.1, bar_y), 1.2, bar_height,
                              facecolor='#2196F3', edgecolor='black', linewidth=2)
    ax.add_patch(val_rect)
    ax.text(4.7, bar_y + bar_height/2, 'Validation\n(20%)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Test set (20%)
    test_rect = plt.Rectangle((5.3, bar_y), 1.2, bar_height,
                               facecolor='#FF9800', edgecolor='black', linewidth=2)
    ax.add_patch(test_rect)
    ax.text(5.9, bar_y + bar_height/2, 'Test\n(20%)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

    # Title for the bar
    ax.text(3.25, bar_y + bar_height + 0.3, 'Total Dataset', ha='center', va='bottom',
            fontsize=14, fontweight='bold')

    # Draw boxes with purposes
    box_y = 1.5
    box_height = 1.8
    box_width = 2.5

    # Training box
    train_box = plt.Rectangle((0.3, box_y), box_width, box_height,
                               facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=2)
    ax.add_patch(train_box)
    ax.text(0.3 + box_width/2, box_y + box_height - 0.3, 'Training', ha='center', va='top',
            fontsize=12, fontweight='bold', color='#2E7D32')
    ax.text(0.3 + box_width/2, box_y + box_height/2 - 0.1, 'Learn model\nparameters',
            ha='center', va='center', fontsize=10)
    ax.text(0.3 + box_width/2, box_y + 0.25, 'Used: Every iteration',
            ha='center', va='bottom', fontsize=9, style='italic')

    # Validation box
    val_box = plt.Rectangle((3.1, box_y), box_width, box_height,
                             facecolor='#E3F2FD', edgecolor='#2196F3', linewidth=2)
    ax.add_patch(val_box)
    ax.text(3.1 + box_width/2, box_y + box_height - 0.3, 'Validation', ha='center', va='top',
            fontsize=12, fontweight='bold', color='#1565C0')
    ax.text(3.1 + box_width/2, box_y + box_height/2 - 0.1, 'Tune hyperparameters\nModel selection',
            ha='center', va='center', fontsize=10)
    ax.text(3.1 + box_width/2, box_y + 0.25, 'Used: Many times',
            ha='center', va='bottom', fontsize=9, style='italic')

    # Test box
    test_box = plt.Rectangle((5.9, box_y), box_width, box_height,
                              facecolor='#FFF3E0', edgecolor='#FF9800', linewidth=2)
    ax.add_patch(test_box)
    ax.text(5.9 + box_width/2, box_y + box_height - 0.3, 'Test', ha='center', va='top',
            fontsize=12, fontweight='bold', color='#E65100')
    ax.text(5.9 + box_width/2, box_y + box_height/2 - 0.1, 'Final evaluation\nUnbiased estimate',
            ha='center', va='center', fontsize=10)
    ax.text(5.9 + box_width/2, box_y + 0.25, 'Used: ONCE only',
            ha='center', va='bottom', fontsize=9, style='italic', fontweight='bold', color='red')

    # Draw arrows from bar to boxes
    arrow_style = dict(arrowstyle='->', color='gray', lw=2)
    ax.annotate('', xy=(0.3 + box_width/2, box_y + box_height),
                xytext=(2.3, bar_y), arrowprops=arrow_style)
    ax.annotate('', xy=(3.1 + box_width/2, box_y + box_height),
                xytext=(4.7, bar_y), arrowprops=arrow_style)
    ax.annotate('', xy=(5.9 + box_width/2, box_y + box_height),
                xytext=(5.9, bar_y), arrowprops=arrow_style)

    # Key insight box
    insight_rect = plt.Rectangle((1.5, 0.2), 6, 0.8,
                                  facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2)
    ax.add_patch(insight_rect)
    ax.text(4.5, 0.6, 'Key: Test set must NEVER influence any decisions during development!',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#C62828')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'train_val_test_split.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: train_val_test_split.png")


def main():
    """Generate all visualizations."""
    print("Generating visualizations for Overfitting/Underfitting FAQ...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    plot_learning_curves_diagnosis()
    plot_bias_variance_tradeoff()
    plot_model_complexity_vs_error()
    plot_polynomial_degree_comparison()
    plot_double_descent()
    plot_train_val_test_split()

    print("-" * 50)
    print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
