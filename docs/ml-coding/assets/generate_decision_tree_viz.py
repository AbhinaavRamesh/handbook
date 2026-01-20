#!/usr/bin/env python3
"""
Decision Tree Visualization Generator
======================================

Generates matplotlib visualizations for decision tree concepts:
- Tree structure diagram showing splits and leaf nodes
- Decision boundary evolution as tree grows (GIF)
- Gini vs Entropy comparison curves
- Feature importance bar chart
- Pruning effect on accuracy

Requirements:
    pip install matplotlib scikit-learn numpy pillow

Usage:
    python generate_decision_tree_viz.py

Output:
    All images are saved to the same directory as this script (assets folder).
"""

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import numpy as np

from sklearn.datasets import make_classification, load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path(__file__).parent
DPI = 150
FIGSIZE = (10, 6)

# Use seaborn style
plt.style.use('seaborn-v0_8-whitegrid')

# Color scheme
COLORS = {
    'class_0': '#3498db',
    'class_1': '#e74c3c',
    'class_2': '#2ecc71',
    'gini': '#3498db',
    'entropy': '#e74c3c',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_figure(fig, filename):
    """Save figure as PNG."""
    filepath = OUTPUT_DIR / f"{filename}.png"
    fig.savefig(filepath, format='png', dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved: {filepath}")


def create_sample_data(n_samples=200, random_state=42):
    """Create sample classification data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=random_state
    )
    return X, y


# =============================================================================
# VISUALIZATION 1: TREE STRUCTURE DIAGRAM
# =============================================================================

def generate_tree_structure():
    """
    Generate a clean tree structure diagram showing splits and leaf nodes.
    """
    print("\n[1/5] Generating tree structure diagram...")

    # Load iris dataset
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target

    # Train decision tree with limited depth
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Plot the tree
    plot_tree(
        clf,
        feature_names=['Sepal Length', 'Sepal Width'],
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=9,
        proportion=False,
        impurity=True
    )

    ax.set_title('Decision Tree Structure\n(Iris Dataset, max_depth=3)',
                 fontsize=14, fontweight='bold', pad=20)

    save_figure(fig, 'decision_tree_structure')
    plt.close(fig)


# =============================================================================
# VISUALIZATION 2: DECISION BOUNDARY EVOLUTION GIF
# =============================================================================

def generate_decision_boundary_gif():
    """
    Generate GIF showing decision boundary evolution as tree grows.
    """
    print("\n[2/5] Generating decision boundary evolution GIF...")

    # Create sample data
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )

    # Create mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # Custom colormap
    cmap_light = ListedColormap(['#aed6f1', '#f5b7b1'])
    cmap_bold = ListedColormap(['#3498db', '#e74c3c'])

    depths = [1, 2, 3, 4, 5, 6, 7, 8]

    # Create figure for animation
    fig, ax = plt.subplots(figsize=FIGSIZE)

    def animate(frame):
        ax.clear()
        depth = depths[frame]

        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X, y)

        # Predict on mesh grid
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
        ax.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)

        # Plot data points
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                   edgecolors='black', s=30, alpha=0.8)

        accuracy = clf.score(X, y)
        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.set_title(f'Decision Boundary Evolution\nDepth = {depth} | Accuracy = {accuracy:.1%}',
                     fontsize=14, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Add legend
        class_0_patch = mpatches.Patch(color='#3498db', label='Class 0')
        class_1_patch = mpatches.Patch(color='#e74c3c', label='Class 1')
        ax.legend(handles=[class_0_patch, class_1_patch], loc='upper right')

        return ax,

    anim = animation.FuncAnimation(fig, animate, frames=len(depths),
                                   interval=1000, blit=False)

    # Save as GIF
    gif_path = OUTPUT_DIR / 'decision_boundary_evolution.gif'
    anim.save(str(gif_path), writer='pillow', fps=1, dpi=DPI)
    print(f"  Saved: {gif_path}")
    plt.close(fig)


# =============================================================================
# VISUALIZATION 3: GINI VS ENTROPY COMPARISON
# =============================================================================

def generate_gini_entropy_comparison():
    """
    Generate comparison curves of Gini impurity vs Entropy.
    """
    print("\n[3/5] Generating Gini vs Entropy comparison curves...")

    # Create probability range for binary classification
    p = np.linspace(0.001, 0.999, 500)

    # Gini impurity: 1 - p^2 - (1-p)^2 = 2p(1-p)
    gini = 1 - p**2 - (1-p)**2

    # Entropy: -p*log2(p) - (1-p)*log2(1-p)
    entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)

    # Normalize entropy to [0, 0.5] for comparison (divide by 2)
    entropy_scaled = entropy / 2

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Raw values
    axes[0].plot(p, gini, color=COLORS['gini'], linewidth=2.5, label='Gini Impurity')
    axes[0].plot(p, entropy, color=COLORS['entropy'], linewidth=2.5, label='Entropy')
    axes[0].set_xlabel('Probability of Class 1 (p)', fontsize=11)
    axes[0].set_ylabel('Impurity Value', fontsize=11)
    axes[0].set_title('Gini vs Entropy\n(Original Scale)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].annotate('Max impurity\nat p=0.5', xy=(0.5, 0.5), xytext=(0.65, 0.35),
                     fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    # Right plot: Normalized comparison
    axes[1].plot(p, gini, color=COLORS['gini'], linewidth=2.5, label='Gini: 2p(1-p)')
    axes[1].plot(p, entropy_scaled, color=COLORS['entropy'], linewidth=2.5,
                 label='Entropy/2 (scaled)')
    axes[1].set_xlabel('Probability of Class 1 (p)', fontsize=11)
    axes[1].set_ylabel('Impurity Value', fontsize=11)
    axes[1].set_title('Gini vs Entropy (Scaled)\n(Both normalized to [0, 0.5])',
                      fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].fill_between(p, gini, entropy_scaled, alpha=0.2, color='gray',
                         label='Difference')

    # Add text box with formulas
    formula_text = (
        "Gini = 1 - p\u00b2 - (1-p)\u00b2\n"
        "Entropy = -p\u00b7log\u2082(p) - (1-p)\u00b7log\u2082(1-p)"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axes[0].text(0.02, 0.98, formula_text, transform=axes[0].transAxes, fontsize=9,
                 verticalalignment='top', bbox=props, family='monospace')

    plt.suptitle('Comparing Splitting Criteria: Gini Impurity vs Entropy',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'gini_vs_entropy')
    plt.close(fig)


# =============================================================================
# VISUALIZATION 4: FEATURE IMPORTANCE BAR CHART
# =============================================================================

def generate_feature_importance():
    """
    Generate feature importance bar chart.
    """
    print("\n[4/5] Generating feature importance bar chart...")

    # Load iris dataset with all features
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names

    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)

    # Get feature importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Create bar chart
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    bars = ax.barh(range(len(importances)), importances[indices],
                   color=[colors[i % len(colors)] for i in range(len(importances))],
                   edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, importances[indices]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)

    # Set labels
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
    ax.set_xlabel('Feature Importance (Gini)', fontsize=11)
    ax.set_title('Feature Importance in Decision Tree\n(Iris Dataset)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(importances) + 0.15)
    ax.invert_yaxis()

    # Add explanation text
    explanation = (
        "Importance = weighted sum of\n"
        "impurity decrease at splits"
    )
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.98, 0.02, explanation, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    save_figure(fig, 'feature_importance')
    plt.close(fig)


# =============================================================================
# VISUALIZATION 5: PRUNING EFFECT ON ACCURACY
# =============================================================================

def generate_pruning_effect():
    """
    Generate chart showing pruning effect on train/test accuracy.
    """
    print("\n[5/5] Generating pruning effect on accuracy chart...")

    # Create sample data with noise
    np.random.seed(42)
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_redundant=3,
        n_informative=5,
        n_clusters_per_class=2,
        flip_y=0.1,
        random_state=42
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Test different max_depth values
    depths = range(1, 16)
    train_accuracies = []
    test_accuracies = []

    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        train_accuracies.append(clf.score(X_train, y_train))
        test_accuracies.append(clf.score(X_test, y_test))

    # Find optimal depth
    best_depth = depths[np.argmax(test_accuracies)]
    best_test_acc = max(test_accuracies)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Plot accuracies
    ax.plot(depths, train_accuracies, 'o-', color=COLORS['gini'],
            linewidth=2, markersize=6, label='Training Accuracy')
    ax.plot(depths, test_accuracies, 's-', color=COLORS['entropy'],
            linewidth=2, markersize=6, label='Test Accuracy')

    # Mark optimal depth
    ax.axvline(x=best_depth, color='green', linestyle='--', alpha=0.7,
               label=f'Optimal depth = {best_depth}')
    ax.scatter([best_depth], [best_test_acc], color='green', s=150,
               zorder=5, marker='*')

    # Add shaded regions
    ax.axvspan(1, best_depth, alpha=0.1, color='blue', label='Underfitting region')
    ax.axvspan(best_depth, 15, alpha=0.1, color='red', label='Overfitting region')

    # Labels and title
    ax.set_xlabel('Maximum Tree Depth', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Effect of Tree Depth (Pruning) on Accuracy\n'
                 'Demonstrating Bias-Variance Tradeoff',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xticks(depths)
    ax.set_ylim(0.5, 1.05)

    # Add annotation
    gap = train_accuracies[-1] - test_accuracies[-1]
    ax.annotate(f'Overfitting gap: {gap:.1%}',
                xy=(14, (train_accuracies[-1] + test_accuracies[-1])/2),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_figure(fig, 'pruning_effect')
    plt.close(fig)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Generate all decision tree visualizations."""
    print("=" * 60)
    print("Decision Tree Visualization Generator")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Resolution: {DPI} DPI")
    print("\nGenerating visualizations...")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate all visualizations
    generate_tree_structure()
    generate_decision_boundary_gif()
    generate_gini_entropy_comparison()
    generate_feature_importance()
    generate_pruning_effect()

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob('decision_*')):
        if f.suffix in ['.png', '.gif']:
            print(f"  - {f.name}")


if __name__ == '__main__':
    main()
