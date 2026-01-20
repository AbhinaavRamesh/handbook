"""
Generate visualizations for Cross-Validation FAQ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
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
    'train': '#93c5fd',
    'val': '#fca5a5',
    'test': '#86efac'
}


def plot_kfold():
    """Generate K-Fold cross-validation structure."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)

    k = 5
    n_samples = 100
    fold_size = n_samples // k

    for fold in range(k):
        y_pos = k - fold - 1

        for i in range(k):
            x_start = i * fold_size
            width = fold_size

            if i == fold:
                color = COLORS['val']
                label = 'Validation' if fold == 0 and i == 0 else ''
            else:
                color = COLORS['train']
                label = 'Training' if fold == 0 and i == 1 else ''

            rect = Rectangle((x_start, y_pos), width - 1, 0.8,
                              facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)

        ax.text(-8, y_pos + 0.4, f'Fold {fold + 1}', fontsize=11, va='center', ha='right', fontweight='bold')

    # Add legend
    train_patch = mpatches.Patch(color=COLORS['train'], label='Training Data')
    val_patch = mpatches.Patch(color=COLORS['val'], label='Validation Data')
    ax.legend(handles=[train_patch, val_patch], loc='upper right', fontsize=11)

    ax.set_xlim(-15, n_samples + 5)
    ax.set_ylim(-0.5, k + 0.5)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_title('K-Fold Cross-Validation (K=5)\nEach sample is used for validation exactly once',
                 fontsize=14, fontweight='bold')
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('cv_kfold.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: cv_kfold.png")


def plot_stratified():
    """Generate stratified vs regular K-Fold comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI)

    # Regular K-Fold (imbalanced folds)
    ax = axes[0]
    k = 5

    # Simulate imbalanced data (20% positive)
    class_ratios_regular = [
        [0.85, 0.15],  # Fold 1
        [0.75, 0.25],  # Fold 2
        [0.90, 0.10],  # Fold 3
        [0.70, 0.30],  # Fold 4
        [0.80, 0.20],  # Fold 5
    ]

    for fold, ratios in enumerate(class_ratios_regular):
        y_pos = k - fold - 1
        x_start = 0
        for i, ratio in enumerate(ratios):
            color = COLORS['primary'] if i == 0 else COLORS['secondary']
            width = ratio * 100
            rect = Rectangle((x_start, y_pos), width, 0.8,
                              facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            if width > 15:
                ax.text(x_start + width/2, y_pos + 0.4, f'{ratio:.0%}',
                        ha='center', va='center', fontsize=10, color='white', fontweight='bold')
            x_start += width
        ax.text(-5, y_pos + 0.4, f'Fold {fold + 1}', fontsize=10, va='center', ha='right')

    ax.set_xlim(-10, 105)
    ax.set_ylim(-0.5, k + 0.5)
    ax.set_title('Regular K-Fold\n(Class ratios vary across folds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Percentage', fontsize=11)
    ax.set_yticks([])

    # Stratified K-Fold (balanced folds)
    ax = axes[1]

    for fold in range(k):
        y_pos = k - fold - 1
        x_start = 0
        for i, ratio in enumerate([0.80, 0.20]):  # All folds maintain 80-20 ratio
            color = COLORS['primary'] if i == 0 else COLORS['secondary']
            width = ratio * 100
            rect = Rectangle((x_start, y_pos), width, 0.8,
                              facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(x_start + width/2, y_pos + 0.4, f'{ratio:.0%}',
                    ha='center', va='center', fontsize=10, color='white', fontweight='bold')
            x_start += width
        ax.text(-5, y_pos + 0.4, f'Fold {fold + 1}', fontsize=10, va='center', ha='right')

    ax.set_xlim(-10, 105)
    ax.set_ylim(-0.5, k + 0.5)
    ax.set_title('Stratified K-Fold\n(Class ratios preserved in each fold)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Percentage', fontsize=11)
    ax.set_yticks([])

    # Legend
    class0_patch = mpatches.Patch(color=COLORS['primary'], label='Class 0 (Majority)')
    class1_patch = mpatches.Patch(color=COLORS['secondary'], label='Class 1 (Minority)')
    fig.legend(handles=[class0_patch, class1_patch], loc='lower center', ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Stratified vs Regular K-Fold for Imbalanced Data', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('cv_stratified.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: cv_stratified.png")


def plot_data_leakage():
    """Generate data leakage illustration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI)

    # Wrong way (leakage)
    ax = axes[0]
    steps = [
        ('Full Data', 0, COLORS['primary']),
        ('Normalize\n(fit on ALL data)', 1, COLORS['secondary']),
        ('Split Train/Val', 2, 'orange'),
        ('Train Model', 3, COLORS['tertiary']),
    ]

    for label, y, color in steps:
        rect = Rectangle((0.2, 3-y), 0.6, 0.6, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, 3-y+0.3, label, ha='center', va='center', fontsize=11, fontweight='bold')
        if y < 3:
            ax.annotate('', xy=(0.5, 3-y-0.05), xytext=(0.5, 3-y+0.55),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.text(0.5, -0.5, '❌ WRONG: Scaler sees validation data!', ha='center', fontsize=12,
            color=COLORS['secondary'], fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 4)
    ax.set_title('Data Leakage (Wrong)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Right way (no leakage)
    ax = axes[1]
    steps = [
        ('Full Data', 0, COLORS['primary']),
        ('Split Train/Val', 1, 'orange'),
        ('Normalize\n(fit on TRAIN only)', 2, COLORS['tertiary']),
        ('Apply transform\nto Val', 2.7, COLORS['tertiary']),
        ('Train Model', 3.4, COLORS['tertiary']),
    ]

    rect = Rectangle((0.2, 3), 0.6, 0.6, facecolor=COLORS['primary'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(0.5, 3.3, 'Full Data', ha='center', va='center', fontsize=11, fontweight='bold')

    ax.annotate('', xy=(0.5, 2.4), xytext=(0.5, 2.95),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Split into train and val
    rect_train = Rectangle((0.1, 1.8), 0.35, 0.5, facecolor=COLORS['train'], edgecolor='black', linewidth=2)
    rect_val = Rectangle((0.55, 1.8), 0.35, 0.5, facecolor=COLORS['val'], edgecolor='black', linewidth=2)
    ax.add_patch(rect_train)
    ax.add_patch(rect_val)
    ax.text(0.275, 2.05, 'Train', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0.725, 2.05, 'Val', ha='center', va='center', fontsize=10, fontweight='bold')

    # Fit scaler on train
    ax.annotate('', xy=(0.275, 1.2), xytext=(0.275, 1.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    rect = Rectangle((0.05, 0.6), 0.45, 0.5, facecolor=COLORS['tertiary'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(0.275, 0.85, 'fit() on Train\ntransform() Train', ha='center', va='center', fontsize=9, fontweight='bold')

    # Transform val
    ax.annotate('', xy=(0.725, 1.2), xytext=(0.725, 1.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    rect = Rectangle((0.5, 0.6), 0.45, 0.5, facecolor=COLORS['tertiary'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(0.725, 0.85, 'transform()\nVal only', ha='center', va='center', fontsize=9, fontweight='bold')

    ax.text(0.5, -0.3, '✓ CORRECT: Scaler never sees validation data', ha='center', fontsize=12,
            color=COLORS['tertiary'], fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, 4)
    ax.set_title('No Data Leakage (Correct)', fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.suptitle('Data Leakage in Cross-Validation: fit() only on Training Data!',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('cv_data_leakage.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: cv_data_leakage.png")


def plot_types_comparison():
    """Generate CV types comparison."""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)

    cv_types = [
        ('K-Fold', 'Standard splitting', 'General purpose', 'Medium', 'Medium'),
        ('Stratified K-Fold', 'Preserves class ratios', 'Imbalanced data', 'Medium', 'Medium'),
        ('Leave-One-Out', 'N-1 train, 1 test', 'Small datasets', 'Very High', 'Low'),
        ('Time Series Split', 'Respects temporal order', 'Time series data', 'High', 'Low'),
        ('Group K-Fold', 'Keeps groups together', 'Grouped data', 'Medium', 'Medium'),
    ]

    columns = ['CV Type', 'How it Works', 'Best For', 'Compute Cost', 'Variance']

    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=cv_types,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=[COLORS['primary']] * 5,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Cross-Validation Types Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('cv_types_comparison.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: cv_types_comparison.png")


def plot_k_selection():
    """Generate K selection tradeoffs."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    k_values = np.arange(2, 21)

    # Bias decreases with K
    bias = 1.5 / np.sqrt(k_values)

    # Variance increases with K
    variance = 0.1 * np.sqrt(k_values)

    # Computational cost increases linearly with K
    cost = k_values / 10

    ax.plot(k_values, bias, label='Estimation Bias', color=COLORS['primary'], linewidth=2.5, marker='o', markersize=4)
    ax.plot(k_values, variance, label='Estimation Variance', color=COLORS['secondary'], linewidth=2.5, marker='s', markersize=4)
    ax.plot(k_values, cost, label='Computational Cost', color=COLORS['tertiary'], linewidth=2.5, marker='^', markersize=4)

    # Mark common choices
    for k, name in [(5, 'K=5'), (10, 'K=10')]:
        ax.axvline(x=k, color='gray', linestyle='--', alpha=0.5)
        ax.text(k, 2.3, name, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Number of Folds (K)', fontsize=12)
    ax.set_ylabel('Relative Magnitude', fontsize=12)
    ax.set_title('Choosing K: Trade-offs Between Bias, Variance, and Cost', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(2, 20)
    ax.set_ylim(0, 2.5)

    # Add annotations
    ax.annotate('K=5 or K=10 are\ncommon choices', xy=(7.5, 1.8), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig('cv_k_selection.png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: cv_k_selection.png")


if __name__ == '__main__':
    print("Generating Cross-Validation visualizations...")
    plot_kfold()
    plot_stratified()
    plot_data_leakage()
    plot_types_comparison()
    plot_k_selection()
    print("Done!")
