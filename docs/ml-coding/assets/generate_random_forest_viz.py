#!/usr/bin/env python3
"""
Random Forest Visualization Generator
======================================

This script generates various Random Forest visualizations for educational purposes.
It creates multiple diagrams showing different aspects of Random Forest including:
- Bootstrap sampling visualization (overlapping samples)
- Single tree vs forest decision boundary comparison
- Feature importance bar chart (Gini vs Permutation)
- OOB error convergence curve (n_estimators vs error)
- Ensemble voting visualization

Requirements:
-------------
    pip install matplotlib scikit-learn numpy

Usage:
------
    python generate_random_forest_viz.py

Output:
-------
    All images are saved to the same directory as this script (assets folder).

Author: Auto-generated for GoogleSDE_InterviewPrep
"""

# =============================================================================
# REQUIREMENTS
# =============================================================================
# matplotlib>=3.5.0
# scikit-learn>=1.0.0
# numpy>=1.21.0

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np

from sklearn.datasets import make_classification, load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directory (same as script location)
OUTPUT_DIR = Path(__file__).parent
DPI = 150  # Resolution for PNG files
FIGSIZE = (10, 6)
FIGSIZE_WIDE = (12, 6)
FIGSIZE_LARGE = (14, 8)

# Use seaborn style
plt.style.use('seaborn-v0_8-whitegrid')

# Color scheme for visualizations
COLORS = {
    'class_0': '#3498db',  # Blue
    'class_1': '#e74c3c',  # Red
    'primary': '#2c3e50',
    'secondary': '#95a5a6',
    'accent': '#27ae60',
    'highlight': '#f39c12',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_figure(fig, filename):
    """
    Save figure as PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        Base filename without extension
    """
    filepath = OUTPUT_DIR / f"{filename}.png"
    fig.savefig(filepath, format='png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved: {filepath}")


def create_sample_data(n_samples=200, random_state=42):
    """
    Create sample classification data for examples.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    X : ndarray
        Feature matrix
    y : ndarray
        Target labels
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=random_state,
        class_sep=1.5
    )
    return X, y


# =============================================================================
# VISUALIZATION 1: BOOTSTRAP SAMPLING
# =============================================================================

def generate_bootstrap_sampling():
    """
    Generate visualization of bootstrap sampling showing overlapping samples.
    Demonstrates how each tree gets a different subset with replacement.
    """
    print("\n[1/5] Generating bootstrap sampling visualization...")

    np.random.seed(42)
    n_samples = 20
    n_trees = 3

    fig, axes = plt.subplots(1, 4, figsize=FIGSIZE_LARGE)

    # Original data
    original_indices = list(range(n_samples))

    # Color code original samples
    colors = plt.cm.tab20(np.linspace(0, 1, n_samples))

    # Plot original data
    ax = axes[0]
    for i in range(n_samples):
        ax.barh(i, 1, color=colors[i], edgecolor='black', linewidth=0.5)
        ax.text(0.5, i, str(i), ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n_samples - 0.5)
    ax.set_title('Original Data\n(n=20 samples)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sample Index')
    ax.set_xticks([])
    ax.invert_yaxis()

    # Generate bootstrap samples for each tree
    for tree_idx in range(n_trees):
        ax = axes[tree_idx + 1]
        np.random.seed(42 + tree_idx)
        bootstrap_indices = np.random.randint(0, n_samples, size=n_samples)

        # Count occurrences
        counts = np.bincount(bootstrap_indices, minlength=n_samples)
        oob_indices = np.where(counts == 0)[0]

        # Plot bootstrap sample
        for pos, idx in enumerate(bootstrap_indices):
            ax.barh(pos, 1, color=colors[idx], edgecolor='black', linewidth=0.5, alpha=0.8)
            ax.text(0.5, pos, str(idx), ha='center', va='center', fontsize=7)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, n_samples - 0.5)

        unique_count = len(np.unique(bootstrap_indices))
        oob_pct = len(oob_indices) / n_samples * 100

        ax.set_title(f'Tree {tree_idx + 1} Bootstrap\n'
                     f'Unique: {unique_count}/20 ({100-oob_pct:.0f}%)\n'
                     f'OOB: {len(oob_indices)} samples ({oob_pct:.0f}%)',
                     fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.invert_yaxis()

        if tree_idx == 0:
            ax.set_ylabel('Position in Sample')

    # Add annotation box
    textstr = ('Bootstrap Sampling:\n'
               '• Sample WITH replacement\n'
               '• Same size as original (n)\n'
               '• ~63.2% unique samples\n'
               '• ~36.8% Out-of-Bag (OOB)\n'
               '• P(selected) = 1 - (1-1/n)^n')

    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange')
    fig.text(0.5, -0.08, textstr, ha='center', fontsize=10,
             bbox=props, family='monospace')

    plt.suptitle('Bootstrap Sampling: Each Tree Gets Different Data Subset',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'rf_bootstrap_sampling')
    plt.close(fig)


# =============================================================================
# VISUALIZATION 2: SINGLE TREE VS FOREST DECISION BOUNDARY
# =============================================================================

def generate_decision_boundary_comparison():
    """
    Generate comparison of single tree vs forest decision boundaries.
    Shows how ensemble averaging creates smoother boundaries.
    """
    print("\n[2/5] Generating single tree vs forest decision boundary comparison...")

    # Create sample data
    X, y = create_sample_data(n_samples=200, random_state=42)

    # Create mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # Custom colormaps
    cmap_light = ListedColormap(['#aed6f1', '#f5b7b1'])
    cmap_bold = ListedColormap([COLORS['class_0'], COLORS['class_1']])

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_LARGE)

    # Single Decision Tree
    clf_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf_tree.fit(X, y)
    Z_tree = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axes[0].contourf(xx, yy, Z_tree, alpha=0.4, cmap=cmap_light)
    axes[0].contour(xx, yy, Z_tree, colors='black', linewidths=0.5, alpha=0.5)
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='black', s=30, alpha=0.8)
    axes[0].set_title(f'Single Decision Tree\n(Depth=10, Acc={clf_tree.score(X, y):.2%})',
                      fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')

    # Random Forest (10 trees)
    clf_rf10 = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
    clf_rf10.fit(X, y)
    Z_rf10 = clf_rf10.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axes[1].contourf(xx, yy, Z_rf10, alpha=0.4, cmap=cmap_light)
    axes[1].contour(xx, yy, Z_rf10, colors='black', linewidths=0.5, alpha=0.5)
    axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='black', s=30, alpha=0.8)
    axes[1].set_title(f'Random Forest (10 Trees)\n(Acc={clf_rf10.score(X, y):.2%})',
                      fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Feature 1')

    # Random Forest (100 trees)
    clf_rf100 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf_rf100.fit(X, y)
    Z_rf100 = clf_rf100.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axes[2].contourf(xx, yy, Z_rf100, alpha=0.4, cmap=cmap_light)
    axes[2].contour(xx, yy, Z_rf100, colors='black', linewidths=0.5, alpha=0.5)
    axes[2].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='black', s=30, alpha=0.8)
    axes[2].set_title(f'Random Forest (100 Trees)\n(Acc={clf_rf100.score(X, y):.2%})',
                      fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Feature 1')

    # Add legend
    class_0_patch = mpatches.Patch(color=COLORS['class_0'], label='Class 0')
    class_1_patch = mpatches.Patch(color=COLORS['class_1'], label='Class 1')
    fig.legend(handles=[class_0_patch, class_1_patch], loc='lower center',
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Decision Boundaries: Single Tree vs Random Forest\n'
                 '(Ensemble averaging creates smoother, more robust boundaries)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'rf_decision_boundary_comparison')
    plt.close(fig)


# =============================================================================
# VISUALIZATION 3: FEATURE IMPORTANCE (GINI VS PERMUTATION)
# =============================================================================

def generate_feature_importance():
    """
    Generate feature importance comparison between Gini and Permutation methods.
    """
    print("\n[3/5] Generating feature importance comparison...")

    # Load iris dataset for meaningful feature names
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Gini importance (built-in)
    gini_importance = rf.feature_importances_

    # Permutation importance
    perm_result = permutation_importance(rf, X, y, n_repeats=30, random_state=42)
    perm_importance = perm_result.importances_mean
    perm_std = perm_result.importances_std

    # Normalize both for comparison
    gini_norm = gini_importance / gini_importance.sum()
    perm_norm = perm_importance / perm_importance.sum()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Gini importance
    x_pos = np.arange(len(feature_names))
    bars1 = axes[0].barh(x_pos, gini_norm, color=COLORS['class_0'], edgecolor='black', alpha=0.8)
    axes[0].set_yticks(x_pos)
    axes[0].set_yticklabels(feature_names)
    axes[0].set_xlabel('Normalized Importance')
    axes[0].set_title('Gini Importance\n(Mean Decrease in Impurity)', fontsize=11, fontweight='bold')
    axes[0].set_xlim(0, max(gini_norm) * 1.15)

    # Add value labels
    for bar, val in zip(bars1, gini_norm):
        axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=9)

    # Permutation importance with error bars
    bars2 = axes[1].barh(x_pos, perm_norm, xerr=perm_std/perm_importance.sum(),
                          color=COLORS['class_1'], edgecolor='black', alpha=0.8, capsize=3)
    axes[1].set_yticks(x_pos)
    axes[1].set_yticklabels(feature_names)
    axes[1].set_xlabel('Normalized Importance')
    axes[1].set_title('Permutation Importance\n(Mean Decrease in Accuracy)', fontsize=11, fontweight='bold')
    axes[1].set_xlim(0, max(perm_norm) * 1.25)

    # Add value labels
    for bar, val in zip(bars2, perm_norm):
        axes[1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=9)

    # Add comparison note
    textstr = ('Gini Importance:\n'
               '  + Fast (computed during training)\n'
               '  - Biased toward high-cardinality features\n\n'
               'Permutation Importance:\n'
               '  + More reliable, accounts for interactions\n'
               '  - Slower (requires re-evaluation)')

    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange')
    fig.text(0.5, -0.12, textstr, ha='center', fontsize=9,
             bbox=props, family='monospace')

    plt.suptitle('Feature Importance: Gini vs Permutation Method\n(Iris Dataset - Random Forest)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'rf_feature_importance')
    plt.close(fig)


# =============================================================================
# VISUALIZATION 4: OOB ERROR CONVERGENCE
# =============================================================================

def generate_oob_convergence():
    """
    Generate OOB error convergence curve showing how error decreases with more trees.
    """
    print("\n[4/5] Generating OOB error convergence curve...")

    # Create sample data
    X, y = create_sample_data(n_samples=500, random_state=42)

    # Track OOB error for different numbers of estimators
    n_estimator_range = list(range(1, 201, 5))
    oob_errors = []
    train_errors = []

    for n_est in n_estimator_range:
        rf = RandomForestClassifier(n_estimators=n_est, oob_score=True,
                                     random_state=42, warm_start=False)
        rf.fit(X, y)
        oob_errors.append(1 - rf.oob_score_)
        train_errors.append(1 - rf.score(X, y))

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Plot curves
    ax.plot(n_estimator_range, oob_errors, 'o-', color=COLORS['class_1'],
            linewidth=2, markersize=4, label='OOB Error', alpha=0.8)
    ax.plot(n_estimator_range, train_errors, 's--', color=COLORS['class_0'],
            linewidth=2, markersize=4, label='Training Error', alpha=0.8)

    # Add horizontal lines for reference
    min_oob = min(oob_errors)
    ax.axhline(y=min_oob, color='gray', linestyle=':', alpha=0.7)
    ax.text(205, min_oob, f'Min OOB: {min_oob:.3f}', va='center', fontsize=9)

    # Add annotations
    ax.annotate('Diminishing returns\n(more trees = marginal gain)',
                xy=(100, oob_errors[20]), xytext=(130, oob_errors[20] + 0.02),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, ha='left')

    ax.set_xlabel('Number of Trees (n_estimators)', fontsize=11)
    ax.set_ylabel('Error Rate', fontsize=11)
    ax.set_title('OOB Error Convergence: More Trees = Better Generalization\n'
                 '(Error stabilizes after ~50-100 trees)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, 210)
    ax.set_ylim(0, max(oob_errors) * 1.1)

    # Add explanation box
    textstr = ('Why OOB Error?\n'
               '• ~36.8% samples unused per tree\n'
               '• Free cross-validation estimate\n'
               '• No separate validation set needed\n'
               '• Similar to leave-one-out CV')

    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange')
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props, family='monospace')

    plt.tight_layout()
    save_figure(fig, 'rf_oob_convergence')
    plt.close(fig)


# =============================================================================
# VISUALIZATION 5: ENSEMBLE VOTING
# =============================================================================

def generate_ensemble_voting():
    """
    Generate visualization of ensemble voting mechanism.
    Shows how individual trees vote and final prediction is determined.
    """
    print("\n[5/5] Generating ensemble voting visualization...")

    np.random.seed(42)
    n_trees = 7
    n_samples = 5

    # Simulate tree predictions (0 or 1)
    # Make it somewhat realistic - trees should mostly agree but not always
    true_labels = [0, 1, 1, 0, 1]
    tree_predictions = []

    for _ in range(n_trees):
        preds = []
        for true_label in true_labels:
            # 80% chance of correct prediction per tree
            if np.random.random() < 0.8:
                preds.append(true_label)
            else:
                preds.append(1 - true_label)
        tree_predictions.append(preds)

    tree_predictions = np.array(tree_predictions)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Create grid
    cell_colors = []
    for i in range(n_trees):
        row_colors = []
        for j in range(n_samples):
            if tree_predictions[i, j] == 0:
                row_colors.append(COLORS['class_0'])
            else:
                row_colors.append(COLORS['class_1'])
        cell_colors.append(row_colors)

    # Plot grid
    for i in range(n_trees):
        for j in range(n_samples):
            color = cell_colors[i][j]
            rect = mpatches.FancyBboxPatch((j, n_trees - 1 - i), 0.9, 0.9,
                                            boxstyle="round,pad=0.02",
                                            facecolor=color, edgecolor='black',
                                            linewidth=1.5, alpha=0.8)
            ax.add_patch(rect)
            ax.text(j + 0.45, n_trees - 0.55 - i, str(tree_predictions[i, j]),
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white')

    # Calculate majority votes
    majority_votes = []
    for j in range(n_samples):
        votes = tree_predictions[:, j]
        vote_0 = np.sum(votes == 0)
        vote_1 = np.sum(votes == 1)
        majority_votes.append((0, vote_0) if vote_0 > vote_1 else (1, vote_1))

    # Plot majority vote row
    for j in range(n_samples):
        pred, count = majority_votes[j]
        color = COLORS['class_0'] if pred == 0 else COLORS['class_1']
        rect = mpatches.FancyBboxPatch((j, -1.2), 0.9, 0.9,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor='black',
                                        linewidth=2.5, alpha=1.0)
        ax.add_patch(rect)
        ax.text(j + 0.45, -0.75, f'{pred}', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
        ax.text(j + 0.45, -1.3, f'({count}/{n_trees})', ha='center', va='center',
                fontsize=8, color='gray')

    # Add labels
    for i in range(n_trees):
        ax.text(-0.3, n_trees - 0.55 - i, f'Tree {i+1}', ha='right', va='center',
                fontsize=10, fontweight='bold')

    ax.text(-0.3, -0.75, 'Final Vote', ha='right', va='center', fontsize=11,
            fontweight='bold', color=COLORS['primary'])

    for j in range(n_samples):
        ax.text(j + 0.45, n_trees + 0.3, f'Sample {j+1}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Add arrow
    ax.annotate('', xy=(2.45, -0.2), xytext=(2.45, 0.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    ax.text(3.5, 0.1, 'Majority\nVoting', fontsize=10, ha='left', va='center',
            color=COLORS['primary'], fontweight='bold')

    # Legend
    class_0_patch = mpatches.Patch(color=COLORS['class_0'], label='Class 0')
    class_1_patch = mpatches.Patch(color=COLORS['class_1'], label='Class 1')
    ax.legend(handles=[class_0_patch, class_1_patch], loc='upper right',
              fontsize=10, title='Predictions')

    ax.set_xlim(-0.8, n_samples + 0.5)
    ax.set_ylim(-2, n_trees + 0.8)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.title('Ensemble Voting: Each Tree Votes, Majority Wins\n'
              '(Classification: Mode | Regression: Mean)',
              fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    save_figure(fig, 'rf_ensemble_voting')
    plt.close(fig)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to generate all Random Forest visualizations.
    """
    print("=" * 60)
    print("Random Forest Visualization Generator")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Resolution: {DPI} DPI")
    print("\nGenerating visualizations...")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate all visualizations
    generate_bootstrap_sampling()
    generate_decision_boundary_comparison()
    generate_feature_importance()
    generate_oob_convergence()
    generate_ensemble_voting()

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob('rf_*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
