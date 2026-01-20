"""
Generate visualizations for Ensemble Methods FAQ
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
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
    'orange': '#f97316',
    'fill': '#93c5fd',
}


def plot_bagging_vs_boosting():
    """Generate bagging vs boosting architecture comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=DPI)

    # Bagging
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Data
    rect = FancyBboxPatch((3.5, 8.5), 3, 1, boxstyle="round,pad=0.05",
                          facecolor=COLORS['primary'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 9, 'Full Dataset', ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    # Bootstrap samples
    for i, x in enumerate([1.5, 4, 6.5]):
        ax.annotate('', xy=(x+0.75, 7.2), xytext=(5, 8.4),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        rect = FancyBboxPatch((x, 6.2), 1.5, 0.8, boxstyle="round,pad=0.05",
                              facecolor=COLORS['fill'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x+0.75, 6.6, f'Sample {i+1}', ha='center', va='center', fontsize=9)

    # Models (parallel)
    for i, x in enumerate([1.5, 4, 6.5]):
        ax.annotate('', xy=(x+0.75, 5), xytext=(x+0.75, 6.1),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
        rect = FancyBboxPatch((x, 4), 1.5, 0.8, boxstyle="round,pad=0.05",
                              facecolor=COLORS['tertiary'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x+0.75, 4.4, f'Model {i+1}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # Aggregate
    for x in [1.5, 4, 6.5]:
        ax.annotate('', xy=(5, 2.8), xytext=(x+0.75, 3.9),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    rect = FancyBboxPatch((3.5, 1.8), 3, 0.8, boxstyle="round,pad=0.05",
                          facecolor=COLORS['orange'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 2.2, 'Average / Vote', ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    ax.text(5, 0.8, 'Models trained INDEPENDENTLY\nin PARALLEL', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_title('Bagging (Bootstrap Aggregating)', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Boosting
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Data
    rect = FancyBboxPatch((3.5, 8.5), 3, 1, boxstyle="round,pad=0.05",
                          facecolor=COLORS['primary'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 9, 'Weighted Data', ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    # Sequential models
    y_positions = [6.5, 4.5, 2.5]
    for i, y in enumerate(y_positions):
        # Model box
        rect = FancyBboxPatch((3.5, y), 3, 0.8, boxstyle="round,pad=0.05",
                              facecolor=COLORS['tertiary'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(5, y+0.4, f'Weak Learner {i+1}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

        # Arrow to next
        if i < len(y_positions) - 1:
            ax.annotate('', xy=(5, y_positions[i+1]+0.9), xytext=(5, y-0.1),
                        arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=2))
            ax.text(6.8, y-0.5, 'Focus on\nerrors', fontsize=8, ha='left',
                    color=COLORS['secondary'], style='italic')

    # Arrow from data
    ax.annotate('', xy=(5, 7.4), xytext=(5, 8.4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Final combination
    rect = FancyBboxPatch((3.5, 0.8), 3, 0.8, boxstyle="round,pad=0.05",
                          facecolor=COLORS['orange'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 1.2, 'Weighted Sum', ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    ax.annotate('', xy=(5, 1.7), xytext=(5, 2.4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.text(1.5, 4.5, 'Models trained\nSEQUENTIALLY', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_title('Boosting (Sequential)', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.suptitle('Bagging vs Boosting: Architectural Differences', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('ensemble_bagging_vs_boosting.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: ensemble_bagging_vs_boosting.png")


def plot_bias_variance():
    """Generate bias-variance in ensembles visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI)

    # Bagging effect
    ax = axes[0]
    n_models = np.arange(1, 51)

    base_variance = 2.0
    variance_bagging = base_variance / n_models
    bias_bagging = np.ones_like(n_models) * 0.5  # Bias unchanged
    total_bagging = variance_bagging + bias_bagging

    ax.plot(n_models, bias_bagging, '--', label='Bias²', color=COLORS['primary'], linewidth=2.5)
    ax.plot(n_models, variance_bagging, '-', label='Variance', color=COLORS['secondary'], linewidth=2.5)
    ax.plot(n_models, total_bagging, '-', label='Total Error', color=COLORS['tertiary'], linewidth=3)

    ax.set_xlabel('Number of Models in Ensemble', fontsize=11)
    ax.set_ylabel('Error', fontsize=11)
    ax.set_title('Bagging: Reduces Variance\n(Bias stays constant)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 3)

    # Boosting effect
    ax = axes[1]
    n_rounds = np.arange(1, 51)

    bias_boosting = 2.0 * np.exp(-0.1 * n_rounds)
    variance_boosting = 0.2 + 0.02 * n_rounds  # Variance increases slightly
    total_boosting = bias_boosting + variance_boosting

    ax.plot(n_rounds, bias_boosting, '--', label='Bias²', color=COLORS['primary'], linewidth=2.5)
    ax.plot(n_rounds, variance_boosting, '-', label='Variance', color=COLORS['secondary'], linewidth=2.5)
    ax.plot(n_rounds, total_boosting, '-', label='Total Error', color=COLORS['tertiary'], linewidth=3)

    # Mark optimal point
    optimal_idx = np.argmin(total_boosting)
    ax.axvline(x=n_rounds[optimal_idx], color='gray', linestyle=':', alpha=0.7)
    ax.scatter([n_rounds[optimal_idx]], [total_boosting[optimal_idx]],
               color=COLORS['quaternary'], s=150, zorder=5, marker='*')
    ax.annotate('Optimal\n(before overfitting)', xy=(n_rounds[optimal_idx], total_boosting[optimal_idx]),
                xytext=(n_rounds[optimal_idx]+10, total_boosting[optimal_idx]+0.5),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel('Number of Boosting Rounds', fontsize=11)
    ax.set_ylabel('Error', fontsize=11)
    ax.set_title('Boosting: Reduces Bias\n(Can increase variance if overfit)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 3)

    plt.suptitle('How Bagging and Boosting Address the Bias-Variance Tradeoff', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('ensemble_bias_variance.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: ensemble_bias_variance.png")


def plot_random_forest_features():
    """Generate random forest feature sampling visualization."""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)

    n_features = 8
    n_trees = 5
    features_per_tree = 3

    np.random.seed(42)

    # Draw feature pool
    ax.text(0.5, 0.95, 'All Features', ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
    for i in range(n_features):
        rect = FancyBboxPatch((0.1 + i*0.1, 0.85), 0.08, 0.06, boxstyle="round,pad=0.01",
                              facecolor=COLORS['primary'], edgecolor='black', linewidth=1, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.14 + i*0.1, 0.88, f'F{i+1}', ha='center', va='center', fontsize=9,
                color='white', fontweight='bold', transform=ax.transAxes)

    # Draw trees with random feature subsets
    for tree_idx in range(n_trees):
        x_pos = 0.1 + tree_idx * 0.18

        # Tree label
        ax.text(x_pos + 0.07, 0.65, f'Tree {tree_idx+1}', ha='center', fontsize=10,
                fontweight='bold', transform=ax.transAxes)

        # Selected features
        selected = np.random.choice(n_features, features_per_tree, replace=False)
        selected = sorted(selected)

        for j, feat_idx in enumerate(selected):
            # Draw arrow from feature to tree
            ax.annotate('', xy=(x_pos + 0.04 + j*0.035, 0.68),
                        xytext=(0.14 + feat_idx*0.1, 0.84),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.5),
                        transform=ax.transAxes)

            # Draw selected feature
            rect = FancyBboxPatch((x_pos + j*0.035, 0.58), 0.03, 0.04, boxstyle="round,pad=0.01",
                                  facecolor=COLORS['tertiary'], edgecolor='black', linewidth=1, transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(x_pos + 0.015 + j*0.035, 0.60, f'F{feat_idx+1}', ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold', transform=ax.transAxes)

        # Draw simple tree
        tree_y = 0.25
        rect = FancyBboxPatch((x_pos, tree_y), 0.14, 0.25, boxstyle="round,pad=0.02",
                              facecolor='lightgreen', edgecolor='black', linewidth=1, transform=ax.transAxes, alpha=0.5)
        ax.add_patch(rect)
        ax.text(x_pos + 0.07, tree_y + 0.12, '🌳', ha='center', va='center', fontsize=20, transform=ax.transAxes)

    # Add note
    ax.text(0.5, 0.08, 'Each tree sees a random subset of features (√p or p/3 features)\n'
            'This creates diversity: trees make different splits → uncorrelated errors',
            ha='center', fontsize=10, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Random Forest: Random Feature Subset at Each Split', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('ensemble_random_forest_features.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: ensemble_random_forest_features.png")


def plot_boosting_weights():
    """Generate boosting weight update visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI)

    np.random.seed(42)
    n_samples = 20

    # Initial uniform weights
    weights = [np.ones(n_samples) / n_samples]

    # Simulate boosting rounds
    for _ in range(2):
        # Some samples are misclassified (random selection)
        errors = np.random.choice(n_samples, size=5, replace=False)
        new_weights = weights[-1].copy()
        new_weights[errors] *= 2  # Increase weight of misclassified
        new_weights /= new_weights.sum()  # Normalize
        weights.append(new_weights)

    titles = ['Round 1: Initial Weights', 'Round 2: After First Learner', 'Round 3: After Second Learner']

    for ax, w, title in zip(axes, weights, titles):
        colors = [COLORS['secondary'] if wi > 1.2/n_samples else COLORS['primary'] for wi in w]
        ax.bar(range(n_samples), w, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=1/n_samples, color='gray', linestyle='--', label='Uniform weight')
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Weight', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(w) * 1.2)

    # Legend
    high_patch = mpatches.Patch(color=COLORS['secondary'], label='Misclassified (higher weight)')
    low_patch = mpatches.Patch(color=COLORS['primary'], label='Correctly classified')
    fig.legend(handles=[high_patch, low_patch], loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Boosting: Sample Weights Evolve to Focus on Difficult Cases', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('ensemble_boosting_weights.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: ensemble_boosting_weights.png")


def plot_algorithm_comparison():
    """Generate ensemble algorithm comparison."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=DPI)

    algorithms = [
        ('Random Forest', 'Bagging + Feature Random.', 'Parallel trees', 'Variance', 'High'),
        ('AdaBoost', 'Weighted samples', 'Sequential weak learners', 'Bias', 'Medium'),
        ('Gradient Boosting', 'Fit residuals', 'Sequential trees', 'Bias', 'Low'),
        ('XGBoost', 'Regularized GB', 'Parallel tree building', 'Both', 'High'),
        ('LightGBM', 'Leaf-wise growth', 'Histogram-based', 'Both', 'Very High'),
    ]

    columns = ['Algorithm', 'Key Idea', 'Training', 'Reduces', 'Speed']

    ax.axis('off')

    table = ax.table(
        cellText=algorithms,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=[COLORS['primary']] * 5,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2)

    # Style header
    for i in range(5):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Popular Ensemble Algorithm Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('ensemble_algorithm_comparison.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: ensemble_algorithm_comparison.png")


if __name__ == '__main__':
    print("Generating Ensemble Methods visualizations...")
    plot_bagging_vs_boosting()
    plot_bias_variance()
    plot_random_forest_features()
    plot_boosting_weights()
    plot_algorithm_comparison()
    print("Done!")
