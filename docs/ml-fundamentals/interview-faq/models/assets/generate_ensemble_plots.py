"""
Ensemble Methods Visualizations (Bagging, Boosting, Random Forest, XGBoost)
Generates plots for ML interview preparation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12

output_dir = os.path.dirname(os.path.abspath(__file__))


def plot_bagging_vs_boosting_architecture():
    """Show architectural difference between bagging and boosting"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Bagging (Parallel)
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    # Data box
    data_box = FancyBboxPatch((4, 8.5), 2, 1, boxstyle="round,pad=0.05",
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax1.add_patch(data_box)
    ax1.text(5, 9, 'Original\nData', ha='center', va='center', fontsize=11, fontweight='bold')

    # Bootstrap samples and models (parallel)
    positions = [(1.5, 6), (5, 6), (8.5, 6)]
    for i, (x, y) in enumerate(positions):
        # Bootstrap sample
        sample_box = FancyBboxPatch((x-0.8, y+0.5), 1.6, 0.8, boxstyle="round,pad=0.03",
                                    facecolor='lightyellow', edgecolor='black')
        ax1.add_patch(sample_box)
        ax1.text(x, y+0.9, f'Sample {i+1}', ha='center', va='center', fontsize=9)

        # Arrow from data to sample
        ax1.annotate('', xy=(x, y+1.3), xytext=(5, 8.5),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

        # Model box
        model_box = FancyBboxPatch((x-0.8, y-1.2), 1.6, 1, boxstyle="round,pad=0.03",
                                   facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax1.add_patch(model_box)
        ax1.text(x, y-0.7, f'Model {i+1}', ha='center', va='center', fontsize=10, fontweight='bold')

        # Arrow from sample to model
        ax1.annotate('', xy=(x, y-0.2), xytext=(x, y+0.5),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

        # Arrow from model to aggregation
        ax1.annotate('', xy=(5, 2.8), xytext=(x, y-1.2),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Aggregation box
    agg_box = FancyBboxPatch((3.5, 1.5), 3, 1.3, boxstyle="round,pad=0.05",
                             facecolor='salmon', edgecolor='black', linewidth=2)
    ax1.add_patch(agg_box)
    ax1.text(5, 2.1, 'Aggregate\n(Vote/Average)', ha='center', va='center', fontsize=11, fontweight='bold')

    # Final prediction
    pred_box = FancyBboxPatch((4, 0), 2, 0.8, boxstyle="round,pad=0.03",
                              facecolor='gold', edgecolor='black', linewidth=2)
    ax1.add_patch(pred_box)
    ax1.text(5, 0.4, 'Prediction', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.annotate('', xy=(5, 0.8), xytext=(5, 1.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax1.axis('off')
    ax1.set_title('BAGGING (Parallel)\nModels trained independently', fontsize=14, fontweight='bold', color='blue')

    # Boosting (Sequential)
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    # Data box
    data_box = FancyBboxPatch((0.5, 4), 1.5, 1.5, boxstyle="round,pad=0.05",
                               facecolor='lightblue', edgecolor='black', linewidth=2)
    ax2.add_patch(data_box)
    ax2.text(1.25, 4.75, 'Weighted\nData', ha='center', va='center', fontsize=10, fontweight='bold')

    # Sequential models
    model_positions = [(3.5, 4.75), (5.5, 4.75), (7.5, 4.75)]
    for i, x in enumerate([3.5, 5.5, 7.5]):
        # Model box
        model_box = FancyBboxPatch((x-0.6, 4.2), 1.2, 1.1, boxstyle="round,pad=0.03",
                                   facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax2.add_patch(model_box)
        ax2.text(x, 4.75, f'Model\n{i+1}', ha='center', va='center', fontsize=9, fontweight='bold')

        if i < 2:
            # Arrow to next model
            ax2.annotate('', xy=(x+0.9, 4.75), xytext=(x+0.6, 4.75),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))

            # Error feedback (curved arrow below)
            ax2.annotate('Update\nweights', xy=(x+0.3, 3.5), fontsize=8, ha='center',
                        color='red', style='italic')

    # Arrow from data to first model
    ax2.annotate('', xy=(2.9, 4.75), xytext=(2, 4.75),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Weighted combination
    comb_box = FancyBboxPatch((3, 1.5), 4.5, 1.3, boxstyle="round,pad=0.05",
                              facecolor='salmon', edgecolor='black', linewidth=2)
    ax2.add_patch(comb_box)
    ax2.text(5.25, 2.1, 'Weighted Combination\nα₁f₁ + α₂f₂ + α₃f₃', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows from models to combination
    for x in [3.5, 5.5, 7.5]:
        ax2.annotate('', xy=(5.25, 2.8), xytext=(x, 4.2),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Final prediction
    pred_box = FancyBboxPatch((4.25, 0), 2, 0.8, boxstyle="round,pad=0.03",
                              facecolor='gold', edgecolor='black', linewidth=2)
    ax2.add_patch(pred_box)
    ax2.text(5.25, 0.4, 'Prediction', ha='center', va='center', fontsize=10, fontweight='bold')
    ax2.annotate('', xy=(5.25, 0.8), xytext=(5.25, 1.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Sequential indicator
    ax2.annotate('Sequential Training', xy=(5.5, 6.2), fontsize=12, ha='center',
                color='red', fontweight='bold')
    ax2.annotate('', xy=(7.8, 6), xytext=(3.2, 6),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax2.axis('off')
    ax2.set_title('BOOSTING (Sequential)\nEach model corrects previous errors', fontsize=14, fontweight='bold', color='red')

    plt.suptitle('Bagging vs Boosting: Architectural Difference', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_bagging_vs_boosting.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: ensemble_bagging_vs_boosting.png")


def plot_bias_variance_ensemble():
    """Show how bagging reduces variance and boosting reduces bias"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = np.sin(x) + 0.5 * x

    # Bagging: Multiple high-variance models
    ax1 = axes[0]
    ax1.plot(x, y_true, 'k-', linewidth=3, label='True Function', zorder=10)

    # Individual trees (high variance)
    for i in range(5):
        noise = np.random.randn(100) * 0.8
        y_pred = y_true + noise + np.random.randn() * 0.5
        ax1.plot(x, y_pred, alpha=0.3, linewidth=1.5, label=f'Tree {i+1}' if i < 3 else None)

    # Averaged prediction (reduced variance)
    y_avg = y_true + np.random.randn(100) * 0.2
    ax1.plot(x, y_avg, 'b-', linewidth=3, label='Bagged Prediction', zorder=9)

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Bagging: Reduces Variance\n(Averaging stabilizes predictions)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_ylim(-2, 8)

    # Boosting: Sequential bias reduction
    ax2 = axes[1]
    ax2.plot(x, y_true, 'k-', linewidth=3, label='True Function', zorder=10)

    # Show progressive improvement
    colors = ['#ff9999', '#ff6666', '#ff3333', '#cc0000', '#990000']
    labels = ['Iteration 1 (High Bias)', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Final (Low Bias)']

    # Start with simple model (high bias)
    for i, (color, label) in enumerate(zip(colors, labels)):
        complexity = (i + 1) / 5
        y_pred = y_true * complexity + (1 - complexity) * np.mean(y_true)
        y_pred += np.random.randn(100) * 0.1
        ax2.plot(x, y_pred, color=color, linewidth=2 if i < 4 else 3,
                alpha=0.5 if i < 4 else 1.0, label=label)

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Boosting: Reduces Bias\n(Sequential models correct errors)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_ylim(-2, 8)

    plt.suptitle('How Ensemble Methods Address Bias-Variance Tradeoff', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_bias_variance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: ensemble_bias_variance.png")


def plot_random_forest_feature_sampling():
    """Visualize random feature sampling in Random Forest"""
    fig, ax = plt.subplots(figsize=(12, 8))

    np.random.seed(42)

    # Features
    n_features = 8
    n_trees = 4
    features = [f'Feature {i+1}' for i in range(n_features)]

    # Create grid
    ax.set_xlim(-0.5, n_trees + 0.5)
    ax.set_ylim(-0.5, n_features + 1)

    # Header
    ax.text(n_trees/2, n_features + 0.5, 'Random Forest: Feature Subsampling at Each Split',
            ha='center', fontsize=14, fontweight='bold')

    # Feature labels
    for i, feat in enumerate(features):
        ax.text(-0.3, i + 0.5, feat, ha='right', va='center', fontsize=11)

    # Tree labels
    for j in range(n_trees):
        ax.text(j + 0.5, n_features + 0.1, f'Tree {j+1}', ha='center', fontsize=12, fontweight='bold')

    # Random feature selection for each tree
    for j in range(n_trees):
        # Each tree gets a random subset of features
        selected = np.random.choice(n_features, size=int(np.sqrt(n_features)) + 2, replace=False)

        for i in range(n_features):
            if i in selected:
                rect = plt.Rectangle((j + 0.1, i + 0.1), 0.8, 0.8,
                                     facecolor='lightgreen', edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(j + 0.5, i + 0.5, '✓', ha='center', va='center', fontsize=14, color='green')
            else:
                rect = plt.Rectangle((j + 0.1, i + 0.1), 0.8, 0.8,
                                     facecolor='lightgray', edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                ax.text(j + 0.5, i + 0.5, '✗', ha='center', va='center', fontsize=14, color='gray')

    # Legend
    legend_y = -0.3
    ax.text(0.5, legend_y, '✓ = Feature used', fontsize=10, color='green')
    ax.text(2, legend_y, '✗ = Feature not used', fontsize=10, color='gray')

    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_random_forest_features.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: ensemble_random_forest_features.png")


def plot_boosting_weight_updates():
    """Visualize how boosting updates instance weights"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    np.random.seed(42)

    # Generate sample data
    n_samples = 20
    x = np.random.randn(n_samples)
    y = (x > 0).astype(int)  # Simple classification

    titles = ['Iteration 1: Equal Weights', 'Iteration 2: Misclassified Upweighted',
              'Iteration 3: Further Adjustment']

    # Simulate weight evolution
    weights_history = [
        np.ones(n_samples) / n_samples,  # Initial equal weights
        None,  # Will be computed
        None   # Will be computed
    ]

    # Simulate misclassifications
    misclassified_1 = np.array([1, 5, 8, 12, 15])  # First iteration misclassified
    weights_history[1] = weights_history[0].copy()
    weights_history[1][misclassified_1] *= 2
    weights_history[1] /= weights_history[1].sum()

    misclassified_2 = np.array([3, 8, 18])  # Second iteration misclassified
    weights_history[2] = weights_history[1].copy()
    weights_history[2][misclassified_2] *= 2
    weights_history[2] /= weights_history[2].sum()

    for ax, weights, title, misclassified in zip(
        axes, weights_history, titles,
        [[], misclassified_1, misclassified_2]
    ):
        # Scatter plot with point sizes proportional to weights
        sizes = weights * 5000

        # Color based on class
        colors = ['blue' if yi == 0 else 'red' for yi in y]

        ax.scatter(x, np.zeros(n_samples) + np.random.randn(n_samples) * 0.1,
                   s=sizes, c=colors, alpha=0.6, edgecolors='black', linewidth=1)

        # Highlight misclassified
        if len(misclassified) > 0:
            ax.scatter(x[misclassified],
                      np.zeros(len(misclassified)) + np.random.randn(len(misclassified)) * 0.1,
                      s=sizes[misclassified], facecolors='none',
                      edgecolors='yellow', linewidth=3, label='Misclassified')
            ax.legend(loc='upper right')

        ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Decision boundary')
        ax.set_xlabel('Feature x', fontsize=12)
        ax.set_ylabel('(Jittered for visibility)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 1)

    # Add note about size
    fig.text(0.5, 0.02, 'Point size ∝ Instance weight (larger = harder to classify = more focus)',
             ha='center', fontsize=11, style='italic')

    plt.suptitle('Boosting: Instance Weight Updates After Each Iteration', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'ensemble_boosting_weights.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: ensemble_boosting_weights.png")


def plot_algorithm_comparison():
    """Create a comparison chart of ensemble algorithms"""
    fig, ax = plt.subplots(figsize=(14, 8))

    algorithms = ['Random\nForest', 'AdaBoost', 'Gradient\nBoosting', 'XGBoost', 'LightGBM', 'CatBoost']
    metrics = ['Speed', 'Accuracy', 'Handles\nCategorical', 'Memory\nEfficiency', 'Ease of\nTuning', 'Overfitting\nResistance']

    # Scores (1-5 scale)
    scores = np.array([
        [3, 4, 2, 3, 5, 5],  # Random Forest
        [4, 3, 2, 4, 4, 3],  # AdaBoost
        [3, 4, 2, 3, 3, 3],  # Gradient Boosting
        [5, 5, 2, 3, 3, 4],  # XGBoost
        [5, 5, 4, 5, 3, 4],  # LightGBM
        [4, 5, 5, 2, 4, 4],  # CatBoost
    ])

    # Create heatmap
    im = ax.imshow(scores, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_yticklabels(algorithms, fontsize=11)

    # Add text annotations
    for i in range(len(algorithms)):
        for j in range(len(metrics)):
            text = ax.text(j, i, scores[i, j], ha='center', va='center',
                          fontsize=12, fontweight='bold',
                          color='white' if scores[i, j] < 3 else 'black')

    ax.set_title('Ensemble Algorithm Comparison\n(Score: 1=Poor, 5=Excellent)',
                 fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Score', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_algorithm_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: ensemble_algorithm_comparison.png")


if __name__ == '__main__':
    print("Generating Ensemble Methods visualizations...")
    plot_bagging_vs_boosting_architecture()
    plot_bias_variance_ensemble()
    plot_random_forest_feature_sampling()
    plot_boosting_weight_updates()
    plot_algorithm_comparison()
    print("\nAll visualizations generated successfully!")
