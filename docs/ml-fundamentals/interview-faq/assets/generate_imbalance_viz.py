"""
Generate visualizations for Class Imbalance FAQ
Covers: class distribution, SMOTE, under/oversampling comparison,
decision boundary shift, and metric comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

# Set consistent styling
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

# Output directory (same as script location)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_class_distribution():
    """
    Figure 1: Class distribution bar chart showing imbalance
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)

    # Left: Balanced dataset
    classes = ['Legitimate', 'Fraud']
    balanced_counts = [5000, 5000]
    colors = ['#3498db', '#e74c3c']

    bars1 = axes[0].bar(classes, balanced_counts, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_title('Balanced Dataset', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_ylim(0, 12000)

    # Add value labels
    for bar, count in zip(bars1, balanced_counts):
        axes[0].annotate(f'{count:,}\n(50%)',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Right: Imbalanced dataset
    imbalanced_counts = [9900, 100]
    bars2 = axes[1].bar(classes, imbalanced_counts, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_title('Imbalanced Dataset (99:1)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Number of Samples', fontsize=12)
    axes[1].set_ylim(0, 12000)

    # Add value labels
    for bar, count, pct in zip(bars2, imbalanced_counts, ['99%', '1%']):
        axes[1].annotate(f'{count:,}\n({pct})',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add imbalance ratio annotation
    axes[1].annotate('Imbalance Ratio: 99:1',
                    xy=(0.5, 0.85), xycoords='axes fraction',
                    ha='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: class_distribution.png")


def plot_smote_visualization():
    """
    Figure 2: SMOTE visualization showing original + synthetic points in 2D
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)

    # Generate majority class (blue cluster)
    n_majority = 100
    majority_x = np.random.randn(n_majority) * 0.8 + 2
    majority_y = np.random.randn(n_majority) * 0.8 + 2

    # Generate minority class (red cluster - sparse)
    n_minority = 8
    minority_x = np.array([5.0, 5.5, 6.0, 5.2, 5.8, 6.2, 5.4, 6.5])
    minority_y = np.array([5.0, 5.3, 5.8, 5.5, 6.0, 5.2, 6.2, 5.6])

    # Left plot: Before SMOTE
    axes[0].scatter(majority_x, majority_y, c='#3498db', s=50, alpha=0.6,
                   label='Majority Class', edgecolors='white', linewidth=0.5)
    axes[0].scatter(minority_x, minority_y, c='#e74c3c', s=80, alpha=0.9,
                   label='Minority Class', edgecolors='black', linewidth=1.5, marker='s')

    axes[0].set_title('Before SMOTE', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature 1', fontsize=12)
    axes[0].set_ylabel('Feature 2', fontsize=12)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].set_xlim(-1, 8)
    axes[0].set_ylim(-1, 8)

    # Add count annotation
    axes[0].annotate(f'Majority: {n_majority}\nMinority: {n_minority}',
                    xy=(0.95, 0.05), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right plot: After SMOTE
    axes[1].scatter(majority_x, majority_y, c='#3498db', s=50, alpha=0.6,
                   label='Majority Class', edgecolors='white', linewidth=0.5)
    axes[1].scatter(minority_x, minority_y, c='#e74c3c', s=80, alpha=0.9,
                   label='Original Minority', edgecolors='black', linewidth=1.5, marker='s')

    # Generate SMOTE synthetic samples
    synthetic_x = []
    synthetic_y = []
    n_synthetic = 92  # To balance classes

    for _ in range(n_synthetic):
        # Pick random minority sample
        idx1 = np.random.randint(0, n_minority)
        # Pick another random minority sample as neighbor
        idx2 = np.random.randint(0, n_minority)
        while idx2 == idx1:
            idx2 = np.random.randint(0, n_minority)

        # Interpolate
        alpha = np.random.random()
        new_x = minority_x[idx1] + alpha * (minority_x[idx2] - minority_x[idx1])
        new_y = minority_y[idx1] + alpha * (minority_y[idx2] - minority_y[idx1])
        synthetic_x.append(new_x)
        synthetic_y.append(new_y)

    axes[1].scatter(synthetic_x, synthetic_y, c='#f39c12', s=40, alpha=0.7,
                   label='Synthetic (SMOTE)', edgecolors='black', linewidth=0.5, marker='D')

    # Draw some interpolation lines to show SMOTE mechanism
    for i in range(3):
        idx1 = i
        idx2 = (i + 1) % n_minority
        axes[1].plot([minority_x[idx1], minority_x[idx2]],
                    [minority_y[idx1], minority_y[idx2]],
                    'k--', alpha=0.3, linewidth=1)

    axes[1].set_title('After SMOTE', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Feature 1', fontsize=12)
    axes[1].set_ylabel('Feature 2', fontsize=12)
    axes[1].legend(loc='upper left', fontsize=9)
    axes[1].set_xlim(-1, 8)
    axes[1].set_ylim(-1, 8)

    # Add count annotation
    axes[1].annotate(f'Majority: {n_majority}\nMinority: {n_minority + n_synthetic}\n(+{n_synthetic} synthetic)',
                    xy=(0.95, 0.05), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'smote_visualization.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: smote_visualization.png")


def plot_sampling_comparison():
    """
    Figure 3: Undersampling vs oversampling comparison
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 6), dpi=150)

    # Original imbalanced data
    n_maj = 100
    n_min = 10

    maj_x = np.random.randn(n_maj) * 1.0 + 2
    maj_y = np.random.randn(n_maj) * 1.0 + 2
    min_x = np.random.randn(n_min) * 0.5 + 5
    min_y = np.random.randn(n_min) * 0.5 + 5

    # Left: Original
    axes[0].scatter(maj_x, maj_y, c='#3498db', s=40, alpha=0.6,
                   edgecolors='white', linewidth=0.5)
    axes[0].scatter(min_x, min_y, c='#e74c3c', s=60, alpha=0.9,
                   edgecolors='black', linewidth=1)
    axes[0].set_title('Original Data\n(100:10)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Feature 1', fontsize=10)
    axes[0].set_ylabel('Feature 2', fontsize=10)

    # Middle: Undersampling
    # Randomly select 10 majority samples
    undersample_idx = np.random.choice(n_maj, 10, replace=False)
    axes[1].scatter(maj_x[undersample_idx], maj_y[undersample_idx], c='#3498db', s=40, alpha=0.6,
                   edgecolors='white', linewidth=0.5, label='Kept')
    # Show removed samples as faded
    kept_mask = np.zeros(n_maj, dtype=bool)
    kept_mask[undersample_idx] = True
    axes[1].scatter(maj_x[~kept_mask], maj_y[~kept_mask], c='#3498db', s=30, alpha=0.15,
                   edgecolors='gray', linewidth=0.5, label='Removed')
    axes[1].scatter(min_x, min_y, c='#e74c3c', s=60, alpha=0.9,
                   edgecolors='black', linewidth=1)
    axes[1].set_title('Undersampling\n(10:10)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Feature 1', fontsize=10)

    # Add annotation
    axes[1].annotate('90 samples\nremoved', xy=(2, 0.5), fontsize=9,
                    ha='center', color='gray', fontstyle='italic')

    # Right: Oversampling
    axes[2].scatter(maj_x, maj_y, c='#3498db', s=40, alpha=0.6,
                   edgecolors='white', linewidth=0.5)
    # Duplicate minority samples randomly
    oversample_x = np.tile(min_x, 10) + np.random.randn(100) * 0.08
    oversample_y = np.tile(min_y, 10) + np.random.randn(100) * 0.08
    axes[2].scatter(oversample_x, oversample_y, c='#e74c3c', s=30, alpha=0.5,
                   edgecolors='black', linewidth=0.3)
    axes[2].scatter(min_x, min_y, c='#e74c3c', s=60, alpha=0.9,
                   edgecolors='black', linewidth=1.5, marker='s', label='Original')
    axes[2].set_title('Oversampling\n(100:100)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Feature 1', fontsize=10)

    # Add annotation
    axes[2].annotate('90 samples\nduplicated', xy=(5, 3.5), fontsize=9,
                    ha='center', color='#c0392b', fontstyle='italic')

    # Set same limits for all
    for ax in axes:
        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 8)

    # Add legend
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', alpha=0.6, label='Majority'),
        Patch(facecolor='#e74c3c', edgecolor='black', alpha=0.9, label='Minority'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
              bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(OUTPUT_DIR, 'sampling_comparison.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: sampling_comparison.png")


def plot_decision_boundary_shift():
    """
    Figure 4: Decision boundary shift before/after resampling
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)

    # Generate data
    np.random.seed(42)
    n_maj = 200
    n_min = 20

    # Majority class (spread out)
    maj_x = np.random.randn(n_maj) * 1.5 + 1
    maj_y = np.random.randn(n_maj) * 1.5 + 1

    # Minority class (compact)
    min_x = np.random.randn(n_min) * 0.8 + 5
    min_y = np.random.randn(n_min) * 0.8 + 5

    # Create meshgrid for decision boundary visualization
    xx, yy = np.meshgrid(np.linspace(-3, 8, 200), np.linspace(-3, 8, 200))

    # Left: Imbalanced (boundary shifted toward minority)
    # Simulated decision boundary - shifted toward minority class
    boundary_biased = 0.6 * xx + 0.4 * yy - 4.5

    axes[0].contourf(xx, yy, boundary_biased, levels=[-10, 0, 10],
                    colors=['#d5e8f7', '#fce4e4'], alpha=0.7)
    axes[0].contour(xx, yy, boundary_biased, levels=[0], colors=['black'],
                   linewidths=2, linestyles='--')

    axes[0].scatter(maj_x, maj_y, c='#3498db', s=40, alpha=0.7,
                   edgecolors='white', linewidth=0.5, label='Majority')
    axes[0].scatter(min_x, min_y, c='#e74c3c', s=60, alpha=0.9,
                   edgecolors='black', linewidth=1, label='Minority')

    axes[0].set_title('Imbalanced Training\n(Boundary Shifted)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Feature 1', fontsize=10)
    axes[0].set_ylabel('Feature 2', fontsize=10)
    axes[0].set_xlim(-3, 8)
    axes[0].set_ylim(-3, 8)

    # Add arrow showing shift
    axes[0].annotate('', xy=(4.2, 4.2), xytext=(3.2, 3.2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    axes[0].text(2.5, 2.5, 'Boundary\nshifted', fontsize=9, color='red',
                ha='center', fontstyle='italic')

    # Right: After resampling (balanced boundary)
    boundary_balanced = 0.6 * xx + 0.4 * yy - 3.5

    axes[1].contourf(xx, yy, boundary_balanced, levels=[-10, 0, 10],
                    colors=['#d5e8f7', '#fce4e4'], alpha=0.7)
    axes[1].contour(xx, yy, boundary_balanced, levels=[0], colors=['black'],
                   linewidths=2, linestyles='-')

    axes[1].scatter(maj_x, maj_y, c='#3498db', s=40, alpha=0.7,
                   edgecolors='white', linewidth=0.5, label='Majority')
    axes[1].scatter(min_x, min_y, c='#e74c3c', s=60, alpha=0.9,
                   edgecolors='black', linewidth=1, label='Minority')

    # Add some synthetic samples to show resampling effect
    synth_x = np.random.randn(n_maj - n_min) * 0.8 + 5 + np.random.randn(n_maj - n_min) * 0.3
    synth_y = np.random.randn(n_maj - n_min) * 0.8 + 5 + np.random.randn(n_maj - n_min) * 0.3
    axes[1].scatter(synth_x, synth_y, c='#f39c12', s=25, alpha=0.4,
                   edgecolors='black', linewidth=0.3, marker='D', label='Synthetic')

    axes[1].set_title('After Resampling\n(Balanced Boundary)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Feature 1', fontsize=10)
    axes[1].set_ylabel('Feature 2', fontsize=10)
    axes[1].set_xlim(-3, 8)
    axes[1].set_ylim(-3, 8)

    # Add annotation showing better boundary
    axes[1].text(1.5, 5.5, 'Better\nseparation', fontsize=9, color='green',
                ha='center', fontweight='bold')

    # Common legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
              bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(os.path.join(OUTPUT_DIR, 'decision_boundary_shift.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: decision_boundary_shift.png")


def plot_metric_comparison():
    """
    Figure 5: Metric comparison on imbalanced data (accuracy vs F1 vs AUC)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)

    # Left: Bar comparison of metrics for different models
    models = ['Always\nNegative', 'Random\nGuess', 'Basic\nModel', 'Tuned\nModel']

    # Metrics for each model (simulated for 99:1 imbalance)
    accuracy = [0.99, 0.50, 0.92, 0.88]
    precision = [0.00, 0.01, 0.40, 0.65]
    recall = [0.00, 0.50, 0.30, 0.75]
    f1 = [0.00, 0.02, 0.34, 0.70]

    x = np.arange(len(models))
    width = 0.2

    bars1 = axes[0].bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#3498db', edgecolor='black')
    bars2 = axes[0].bar(x - 0.5*width, precision, width, label='Precision', color='#2ecc71', edgecolor='black')
    bars3 = axes[0].bar(x + 0.5*width, recall, width, label='Recall', color='#e74c3c', edgecolor='black')
    bars4 = axes[0].bar(x + 1.5*width, f1, width, label='F1-Score', color='#9b59b6', edgecolor='black')

    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Metrics Comparison\n(99:1 Imbalanced Dataset)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=10)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].set_ylim(0, 1.1)
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Add annotation
    axes[0].annotate('99% accuracy\nbut useless!', xy=(0, 0.99), xytext=(0.5, 0.75),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Right: Why different metrics matter - visual explanation
    # Confusion matrix breakdown
    categories = ['True Neg\n(TN)', 'False Pos\n(FP)', 'False Neg\n(FN)', 'True Pos\n(TP)']

    # For "Always Negative" model
    always_neg = [990, 0, 10, 0]  # Out of 1000
    # For tuned model
    tuned = [960, 30, 2, 8]

    x2 = np.arange(len(categories))
    width2 = 0.35

    bars_neg = axes[1].bar(x2 - width2/2, always_neg, width2, label='Always Negative',
                          color='#95a5a6', edgecolor='black')
    bars_tuned = axes[1].bar(x2 + width2/2, tuned, width2, label='Tuned Model',
                            color='#27ae60', edgecolor='black')

    axes[1].set_ylabel('Count (out of 1000)', fontsize=12)
    axes[1].set_title('Confusion Matrix Breakdown\n(Why F1 > Accuracy)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(categories, fontsize=10)
    axes[1].legend(loc='upper right', fontsize=9)

    # Add value labels
    for bar in bars_neg:
        height = bar.get_height()
        if height > 0:
            axes[1].annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                           ha='center', va='bottom', fontsize=9)
    for bar in bars_tuned:
        height = bar.get_height()
        if height > 0:
            axes[1].annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Highlight the important difference
    axes[1].annotate('Catches 8/10\nfraud cases!', xy=(3.2, 8), xytext=(3.2, 300),
                    fontsize=9, ha='center', color='green', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'metric_comparison.png'),
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated: metric_comparison.png")


def main():
    """Generate all visualizations."""
    print("Generating Class Imbalance visualizations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    plot_class_distribution()
    plot_smote_visualization()
    plot_sampling_comparison()
    plot_decision_boundary_shift()
    plot_metric_comparison()

    print("-" * 50)
    print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()
