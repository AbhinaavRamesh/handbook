"""
Cross-Validation Visualizations
Generates plots for ML interview preparation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12

output_dir = os.path.dirname(os.path.abspath(__file__))


def plot_kfold_visualization():
    """Visualize K-Fold cross-validation splits"""
    fig, ax = plt.subplots(figsize=(14, 8))

    k = 5
    n_samples = 100

    ax.set_xlim(0, k + 1)
    ax.set_ylim(0, k + 2)

    # Title row
    ax.text(0.5, k + 1.5, 'Data\nSplit', ha='center', va='center', fontsize=12, fontweight='bold')
    for i in range(k):
        ax.text(i + 1.5, k + 1.5, f'Fold {i+1}', ha='center', va='center', fontsize=11, fontweight='bold')

    # Each iteration
    for iteration in range(k):
        y = k - iteration

        # Iteration label
        ax.text(0.5, y + 0.5, f'Iter {iteration + 1}', ha='center', va='center', fontsize=11)

        # Draw folds
        for fold in range(k):
            x = fold + 1

            if fold == iteration:
                # Test fold
                rect = Rectangle((x + 0.05, y + 0.1), 0.9, 0.8,
                                 facecolor='salmon', edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                ax.text(x + 0.5, y + 0.5, 'TEST', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='darkred')
            else:
                # Train fold
                rect = Rectangle((x + 0.05, y + 0.1), 0.9, 0.8,
                                 facecolor='lightgreen', edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                ax.text(x + 0.5, y + 0.5, 'TRAIN', ha='center', va='center',
                       fontsize=9, color='darkgreen')

    # Add annotations
    ax.annotate('Each fold serves as\ntest set exactly once',
                xy=(k + 0.5, k/2), fontsize=11, ha='left',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    ax.axis('off')
    ax.set_title('K-Fold Cross-Validation (K=5)\nEach iteration trains on K-1 folds, tests on 1 fold',
                 fontsize=14, fontweight='bold', y=1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_kfold.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: cv_kfold.png")


def plot_stratified_vs_regular():
    """Compare stratified vs regular K-fold for imbalanced data"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    np.random.seed(42)

    # Imbalanced dataset: 80% class 0, 20% class 1
    n_samples = 100
    n_class_0 = 80
    n_class_1 = 20

    # Regular K-Fold (random split)
    ax1 = axes[0]

    k = 5
    fold_size = n_samples // k

    # Simulate random split (uneven class distribution)
    regular_folds = [
        {'class_0': 18, 'class_1': 2},
        {'class_0': 14, 'class_1': 6},
        {'class_0': 17, 'class_1': 3},
        {'class_0': 19, 'class_1': 1},
        {'class_0': 12, 'class_1': 8},
    ]

    x = np.arange(k)
    width = 0.35

    class_0_counts = [f['class_0'] for f in regular_folds]
    class_1_counts = [f['class_1'] for f in regular_folds]

    bars1 = ax1.bar(x - width/2, class_0_counts, width, label='Class 0 (Majority)', color='steelblue')
    bars2 = ax1.bar(x + width/2, class_1_counts, width, label='Class 1 (Minority)', color='coral')

    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('Sample Count', fontsize=12)
    ax1.set_title('Regular K-Fold\n(Uneven class distribution across folds)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Fold {i+1}' for i in range(k)])
    ax1.legend(loc='upper right')
    ax1.axhline(y=16, color='blue', linestyle='--', alpha=0.5, label='Expected Class 0')
    ax1.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='Expected Class 1')

    # Stratified K-Fold (maintains ratio)
    ax2 = axes[1]

    stratified_folds = [
        {'class_0': 16, 'class_1': 4},
        {'class_0': 16, 'class_1': 4},
        {'class_0': 16, 'class_1': 4},
        {'class_0': 16, 'class_1': 4},
        {'class_0': 16, 'class_1': 4},
    ]

    class_0_counts = [f['class_0'] for f in stratified_folds]
    class_1_counts = [f['class_1'] for f in stratified_folds]

    bars1 = ax2.bar(x - width/2, class_0_counts, width, label='Class 0 (Majority)', color='steelblue')
    bars2 = ax2.bar(x + width/2, class_1_counts, width, label='Class 1 (Minority)', color='coral')

    ax2.set_xlabel('Fold', fontsize=12)
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.set_title('Stratified K-Fold\n(Class ratio preserved in each fold)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Fold {i+1}' for i in range(k)])
    ax2.legend(loc='upper right')
    ax2.axhline(y=16, color='blue', linestyle='--', alpha=0.5)
    ax2.axhline(y=4, color='red', linestyle='--', alpha=0.5)

    plt.suptitle('Stratified vs Regular K-Fold for Imbalanced Data (80/20 split)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_stratified.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: cv_stratified.png")


def plot_data_leakage():
    """Visualize data leakage concept"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # WRONG approach (data leakage)
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)

    # Full data box
    rect = Rectangle((1, 6), 8, 1.5, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(5, 6.75, 'Full Dataset', ha='center', va='center', fontsize=12, fontweight='bold')

    # Preprocessing arrow
    ax1.annotate('', xy=(5, 5), xytext=(5, 6),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax1.text(6.5, 5.5, 'Scale/Normalize\n(ALL data)', ha='left', fontsize=11, color='red', fontweight='bold')

    # Preprocessed data
    rect = Rectangle((1, 3.5), 8, 1, facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(5, 4, 'Preprocessed Data', ha='center', va='center', fontsize=11)

    # Split arrow
    ax1.annotate('', xy=(3, 2.5), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax1.annotate('', xy=(7, 2.5), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Train/Test
    rect_train = Rectangle((1, 1.5), 3, 1, facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax1.add_patch(rect_train)
    ax1.text(2.5, 2, 'Train', ha='center', va='center', fontsize=11, fontweight='bold')

    rect_test = Rectangle((6, 1.5), 3, 1, facecolor='salmon', edgecolor='black', linewidth=2)
    ax1.add_patch(rect_test)
    ax1.text(7.5, 2, 'Test', ha='center', va='center', fontsize=11, fontweight='bold')

    # Warning
    ax1.text(5, 0.5, '⚠️ TEST DATA LEAKED INTO PREPROCESSING!', ha='center', fontsize=11,
             color='red', fontweight='bold')

    ax1.axis('off')
    ax1.set_title('❌ WRONG: Data Leakage\n(Preprocessing before split)', fontsize=13, fontweight='bold', color='red')

    # CORRECT approach
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)

    # Full data box
    rect = Rectangle((1, 6), 8, 1.5, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax2.add_patch(rect)
    ax2.text(5, 6.75, 'Full Dataset', ha='center', va='center', fontsize=12, fontweight='bold')

    # Split first arrow
    ax2.annotate('', xy=(3, 5), xytext=(5, 6),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax2.annotate('', xy=(7, 5), xytext=(5, 6),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax2.text(5, 5.3, 'SPLIT FIRST', ha='center', fontsize=11, color='green', fontweight='bold')

    # Train/Test raw
    rect_train = Rectangle((1, 4), 3, 1, facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax2.add_patch(rect_train)
    ax2.text(2.5, 4.5, 'Train (raw)', ha='center', va='center', fontsize=10)

    rect_test = Rectangle((6, 4), 3, 1, facecolor='salmon', edgecolor='black', linewidth=2)
    ax2.add_patch(rect_test)
    ax2.text(7.5, 4.5, 'Test (raw)', ha='center', va='center', fontsize=10)

    # fit_transform on train
    ax2.annotate('', xy=(2.5, 2.5), xytext=(2.5, 4),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax2.text(0.5, 3.2, 'fit_transform()\n(Learn stats)', ha='left', fontsize=9, color='green')

    # transform only on test
    ax2.annotate('', xy=(7.5, 2.5), xytext=(7.5, 4),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax2.text(8.5, 3.2, 'transform()\nONLY', ha='left', fontsize=9, color='green')

    # Preprocessed splits
    rect_train = Rectangle((1, 1.5), 3, 1, facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax2.add_patch(rect_train)
    ax2.text(2.5, 2, 'Train (scaled)', ha='center', va='center', fontsize=10, fontweight='bold')

    rect_test = Rectangle((6, 1.5), 3, 1, facecolor='salmon', edgecolor='black', linewidth=2)
    ax2.add_patch(rect_test)
    ax2.text(7.5, 2, 'Test (scaled)', ha='center', va='center', fontsize=10, fontweight='bold')

    ax2.text(5, 0.5, '✓ NO LEAKAGE: Test data never seen during training', ha='center', fontsize=11,
             color='green', fontweight='bold')

    ax2.axis('off')
    ax2.set_title('✓ CORRECT: No Data Leakage\n(Split before preprocessing)', fontsize=13, fontweight='bold', color='green')

    plt.suptitle('Data Leakage: A Common Interview Topic',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_data_leakage.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: cv_data_leakage.png")


def plot_cv_types_comparison():
    """Compare different cross-validation types"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def draw_cv_split(ax, n_folds, title, split_type):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, n_folds + 1)

        if split_type == 'loocv':
            # Leave-One-Out
            n_samples = 8
            for i in range(n_samples):
                y = n_samples - i - 0.5
                for j in range(n_samples):
                    x = j + 1
                    if j == i:
                        rect = Rectangle((x + 0.05, y - 0.35), 0.9, 0.7,
                                        facecolor='salmon', edgecolor='black', linewidth=1)
                        ax.add_patch(rect)
                    else:
                        rect = Rectangle((x + 0.05, y - 0.35), 0.9, 0.7,
                                        facecolor='lightgreen', edgecolor='black', linewidth=1)
                        ax.add_patch(rect)
            ax.text(5, n_samples + 0.3, 'Sample 1    Sample 2    Sample 3    ...', ha='center', fontsize=9)

        elif split_type == 'timeseries':
            # Time Series Split
            for i in range(n_folds):
                y = n_folds - i - 0.5
                train_end = i + 2
                test_start = train_end
                test_end = test_start + 1

                # Train
                for j in range(train_end):
                    x = j + 1
                    rect = Rectangle((x + 0.05, y - 0.35), 0.9, 0.7,
                                    facecolor='lightgreen', edgecolor='black', linewidth=1)
                    ax.add_patch(rect)

                # Test
                rect = Rectangle((test_start + 1.05, y - 0.35), 0.9, 0.7,
                                facecolor='salmon', edgecolor='black', linewidth=2)
                ax.add_patch(rect)

                # Future (unused)
                for j in range(test_end + 1, 8):
                    x = j + 1
                    rect = Rectangle((x + 0.05, y - 0.35), 0.9, 0.7,
                                    facecolor='lightgray', edgecolor='black', linewidth=1, alpha=0.5)
                    ax.add_patch(rect)

            ax.text(5, n_folds + 0.3, 'Time →', ha='center', fontsize=11, style='italic')

        elif split_type == 'group':
            # Group K-Fold
            groups = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']
            colors = ['#ff9999', '#ff9999', '#99ff99', '#99ff99', '#9999ff', '#9999ff', '#ffff99', '#ffff99']

            for i in range(n_folds):
                y = n_folds - i - 0.5
                test_group = ['A', 'B', 'C', 'D'][i]

                for j, (group, color) in enumerate(zip(groups, colors)):
                    x = j + 1
                    if group == test_group:
                        rect = Rectangle((x + 0.05, y - 0.35), 0.9, 0.7,
                                        facecolor='salmon', edgecolor='black', linewidth=2)
                    else:
                        rect = Rectangle((x + 0.05, y - 0.35), 0.9, 0.7,
                                        facecolor=color, edgecolor='black', linewidth=1)
                    ax.add_patch(rect)

            ax.text(5, n_folds + 0.3, 'Group:  A    A    B    B    C    C    D    D', ha='center', fontsize=9)

        elif split_type == 'repeated':
            # Repeated K-Fold
            repeats = 2
            k = 3

            for rep in range(repeats):
                for fold in range(k):
                    i = rep * k + fold
                    y = (repeats * k) - i - 0.5

                    for j in range(6):
                        x = j + 2
                        test_indices = [(fold * 2, fold * 2 + 1)]
                        if rep == 1:  # Different shuffle
                            test_indices = [((fold * 2 + 1) % 6, (fold * 2 + 2) % 6)]

                        is_test = j in test_indices[0]

                        if is_test:
                            rect = Rectangle((x + 0.05, y - 0.35), 0.9, 0.7,
                                            facecolor='salmon', edgecolor='black', linewidth=2)
                        else:
                            rect = Rectangle((x + 0.05, y - 0.35), 0.9, 0.7,
                                            facecolor='lightgreen', edgecolor='black', linewidth=1)
                        ax.add_patch(rect)

                    ax.text(1, y, f'Rep{rep+1}', fontsize=9, va='center')

        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold')

    # LOOCV
    draw_cv_split(axes[0, 0], 8, 'Leave-One-Out CV (LOOCV)\nn iterations for n samples', 'loocv')

    # Time Series
    draw_cv_split(axes[0, 1], 5, 'Time Series Split\n(Respects temporal order)', 'timeseries')

    # Group K-Fold
    draw_cv_split(axes[1, 0], 4, 'Group K-Fold\n(Keeps groups together)', 'group')

    # Repeated K-Fold
    draw_cv_split(axes[1, 1], 6, 'Repeated K-Fold\n(Multiple shuffled iterations)', 'repeated')

    # Legend
    fig.text(0.5, 0.02, '🟢 Train | 🔴 Test | ⬜ Unused/Future',
             ha='center', fontsize=12, fontweight='bold')

    plt.suptitle('Cross-Validation Types Comparison',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'cv_types_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: cv_types_comparison.png")


def plot_k_selection():
    """Visualize effect of K value selection"""
    fig, ax = plt.subplots(figsize=(10, 7))

    k_values = np.arange(2, 21)

    # Simulated bias and variance curves
    bias = 1 / (k_values ** 0.5)
    variance = (k_values - 2) / 50 + 0.1
    total_error = bias + variance

    # Computational cost (linear)
    comp_cost = k_values / 20

    ax.plot(k_values, bias, 'b-', linewidth=2.5, label='Bias (of performance estimate)')
    ax.plot(k_values, variance, 'r-', linewidth=2.5, label='Variance (of performance estimate)')
    ax.plot(k_values, total_error, 'g-', linewidth=3, label='Total Estimation Error')
    ax.plot(k_values, comp_cost, 'k--', linewidth=2, label='Computational Cost')

    # Mark common choices
    ax.axvline(x=5, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax.axvline(x=10, color='purple', linestyle=':', linewidth=2, alpha=0.7)

    ax.scatter([5], [total_error[3]], s=150, c='orange', zorder=5, edgecolors='black', linewidth=2)
    ax.scatter([10], [total_error[8]], s=150, c='purple', zorder=5, edgecolors='black', linewidth=2)

    ax.annotate('K=5\n(Fast, good balance)', xy=(5.2, total_error[3] + 0.05),
                fontsize=11, color='orange', fontweight='bold')
    ax.annotate('K=10\n(Lower bias)', xy=(10.2, total_error[8] + 0.05),
                fontsize=11, color='purple', fontweight='bold')

    ax.set_xlabel('K (Number of Folds)', fontsize=13)
    ax.set_ylabel('Value', fontsize=13)
    ax.set_title('Choosing K: Bias-Variance Tradeoff in Cross-Validation',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(2, 20)
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_k_selection.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: cv_k_selection.png")


if __name__ == '__main__':
    print("Generating Cross-Validation visualizations...")
    plot_kfold_visualization()
    plot_stratified_vs_regular()
    plot_data_leakage()
    plot_cv_types_comparison()
    plot_k_selection()
    print("\nAll visualizations generated successfully!")
