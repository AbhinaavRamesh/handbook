#!/usr/bin/env python3
"""
Decision Tree Visualization Generator
======================================

This script generates various decision tree visualizations for educational purposes.
It creates multiple diagrams showing different aspects of decision trees including:
- Simple tree structure
- Trees with Gini impurity values
- Trees with sample counts
- Pruned vs unpruned comparison
- Step-by-step tree building visualization

Requirements:
-------------
    pip install matplotlib scikit-learn graphviz numpy

Optional (for graphviz export):
    brew install graphviz  # macOS
    apt-get install graphviz  # Ubuntu/Debian

Usage:
------
    python generate_decision_tree_viz.py

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
# graphviz>=0.20.0  (optional, for DOT format export)
# numpy>=1.21.0

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np

from sklearn.datasets import make_classification, load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directory (same as script location)
OUTPUT_DIR = Path(__file__).parent
DPI = 150  # Resolution for PNG files
FIGSIZE_SMALL = (10, 8)
FIGSIZE_LARGE = (16, 12)
FIGSIZE_WIDE = (20, 8)

# Color scheme for visualizations
COLORS = {
    'class_0': '#3498db',  # Blue
    'class_1': '#e74c3c',  # Red
    'class_2': '#2ecc71',  # Green
    'node_fill': '#f0f0f0',
    'leaf_class_0': '#aed6f1',
    'leaf_class_1': '#f5b7b1',
    'leaf_class_2': '#abebc6',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_figure(fig, filename, formats=('png', 'svg')):
    """
    Save figure in multiple formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        Base filename without extension
    formats : tuple
        File formats to save (default: png and svg)
    """
    for fmt in formats:
        filepath = OUTPUT_DIR / f"{filename}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=DPI, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Saved: {filepath}")


def create_sample_data(n_samples=200, random_state=42):
    """
    Create sample classification data for decision tree examples.

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
    feature_names : list
        Names of features
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=random_state
    )
    feature_names = ['Feature 1', 'Feature 2']
    return X, y, feature_names


# =============================================================================
# VISUALIZATION 1: SIMPLE TREE STRUCTURE
# =============================================================================

def generate_simple_tree_structure():
    """
    Generate a simple, clean decision tree visualization.
    Shows basic tree structure with decision rules.
    """
    print("\n[1/6] Generating simple tree structure...")

    # Load iris dataset for a classic example
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target  # Use first 2 features for simplicity

    # Train a simple decision tree (limited depth for clarity)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)

    # Plot the tree
    plot_tree(
        clf,
        feature_names=['Sepal Length', 'Sepal Width'],
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=10,
        proportion=False,
        impurity=False  # Hide impurity for simplicity
    )

    ax.set_title('Simple Decision Tree Structure\n(Iris Dataset - Depth 3)',
                 fontsize=14, fontweight='bold', pad=20)

    save_figure(fig, 'decision_tree_simple')
    plt.close(fig)


# =============================================================================
# VISUALIZATION 2: TREE WITH GINI IMPURITY VALUES
# =============================================================================

def generate_tree_with_gini():
    """
    Generate a decision tree showing Gini impurity at each node.
    Educational visualization for understanding impurity metrics.
    """
    print("\n[2/6] Generating tree with Gini impurity values...")

    # Create sample data
    X, y, feature_names = create_sample_data()

    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=4, random_state=42, criterion='gini')
    clf.fit(X, y)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    # Plot tree with impurity values
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=['Class 0', 'Class 1'],
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=9,
        impurity=True,  # Show Gini impurity
        proportion=False
    )

    ax.set_title('Decision Tree with Gini Impurity Values\n'
                 '(Lower Gini = Purer Node)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add legend explaining Gini
    gini_text = ("Gini Impurity Formula:\n"
                 "Gini = 1 - Σ(pᵢ)²\n\n"
                 "Where pᵢ is the probability\n"
                 "of class i in the node.\n\n"
                 "Gini = 0: Pure node\n"
                 "Gini = 0.5: Maximum impurity\n"
                 "(for binary classification)")

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, gini_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')

    save_figure(fig, 'decision_tree_gini')
    plt.close(fig)


# =============================================================================
# VISUALIZATION 3: TREE WITH SAMPLE COUNTS
# =============================================================================

def generate_tree_with_samples():
    """
    Generate a decision tree showing sample counts at each node.
    Helps visualize how data is split at each decision point.
    """
    print("\n[3/6] Generating tree with sample counts...")

    # Load iris for multi-class example
    iris = load_iris()
    X, y = iris.data[:, [0, 2]], iris.target  # Sepal length & Petal length

    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Left: Tree with proportions
    plot_tree(
        clf,
        feature_names=['Sepal Length', 'Petal Length'],
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        ax=axes[0],
        fontsize=8,
        proportion=True,  # Show proportions
        impurity=False
    )
    axes[0].set_title('Tree with Sample Proportions\n(Percentage in Each Class)',
                      fontsize=12, fontweight='bold')

    # Right: Tree with sample counts
    plot_tree(
        clf,
        feature_names=['Sepal Length', 'Petal Length'],
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        ax=axes[1],
        fontsize=8,
        proportion=False,  # Show counts
        impurity=False
    )
    axes[1].set_title('Tree with Sample Counts\n(Number of Samples per Class)',
                      fontsize=12, fontweight='bold')

    plt.suptitle('Understanding Sample Distribution in Decision Trees',
                 fontsize=14, fontweight='bold', y=1.02)

    save_figure(fig, 'decision_tree_samples')
    plt.close(fig)


# =============================================================================
# VISUALIZATION 4: PRUNED VS UNPRUNED COMPARISON
# =============================================================================

def generate_pruning_comparison():
    """
    Generate comparison between pruned and unpruned trees.
    Demonstrates the effect of different pruning strategies.
    """
    print("\n[4/6] Generating pruned vs unpruned comparison...")

    # Create sample data with some noise
    np.random.seed(42)
    X, y = make_classification(
        n_samples=150,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        flip_y=0.1,  # Add 10% noise
        random_state=42
    )
    feature_names = ['Feature 1', 'Feature 2']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train multiple trees with different pruning
    trees = {
        'Unpruned (Overfit)': DecisionTreeClassifier(random_state=42),
        'max_depth=3': DecisionTreeClassifier(max_depth=3, random_state=42),
        'min_samples_leaf=10': DecisionTreeClassifier(min_samples_leaf=10, random_state=42),
        'ccp_alpha=0.02': DecisionTreeClassifier(ccp_alpha=0.02, random_state=42)
    }

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for idx, (name, clf) in enumerate(trees.items()):
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        plot_tree(
            clf,
            feature_names=feature_names,
            class_names=['Class 0', 'Class 1'],
            filled=True,
            rounded=True,
            ax=axes[idx],
            fontsize=7,
            impurity=True
        )

        title = f'{name}\n'
        title += f'Nodes: {clf.tree_.node_count} | '
        title += f'Depth: {clf.get_depth()}\n'
        title += f'Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}'
        axes[idx].set_title(title, fontsize=11, fontweight='bold')

    plt.suptitle('Effect of Pruning on Decision Trees\n'
                 '(Comparing Different Regularization Strategies)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    save_figure(fig, 'decision_tree_pruning_comparison')
    plt.close(fig)


# =============================================================================
# VISUALIZATION 5: STEP-BY-STEP TREE BUILDING
# =============================================================================

def generate_tree_building_steps():
    """
    Generate step-by-step visualization of tree construction.
    Shows how the tree grows from depth 1 to depth 4.
    """
    print("\n[5/6] Generating step-by-step tree building visualization...")

    # Create sample data
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target
    feature_names = ['Sepal Length', 'Sepal Width']

    # Create figure with 4 subplots showing tree at different depths
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
    axes = axes.flatten()

    depths = [1, 2, 3, 4]

    for idx, depth in enumerate(depths):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X, y)

        plot_tree(
            clf,
            feature_names=feature_names,
            class_names=iris.target_names,
            filled=True,
            rounded=True,
            ax=axes[idx],
            fontsize=8 if depth <= 2 else 7,
            impurity=True
        )

        accuracy = clf.score(X, y)
        n_leaves = clf.get_n_leaves()

        axes[idx].set_title(
            f'Step {idx + 1}: Depth = {depth}\n'
            f'Leaves: {n_leaves} | Accuracy: {accuracy:.2%}',
            fontsize=11, fontweight='bold'
        )

    plt.suptitle('Building a Decision Tree Step-by-Step\n'
                 '(Observing Growth from Depth 1 to 4)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    save_figure(fig, 'decision_tree_building_steps')
    plt.close(fig)


# =============================================================================
# VISUALIZATION 6: DECISION BOUNDARY VISUALIZATION
# =============================================================================

def generate_decision_boundary():
    """
    Generate visualization of decision boundaries.
    Shows how the tree partitions the feature space.
    """
    print("\n[6/6] Generating decision boundary visualization...")

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

    # Train trees with different depths
    depths = [1, 2, 3, 5]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
    axes = axes.flatten()

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

    for idx, depth in enumerate(depths):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X, y)

        # Predict on mesh grid
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
        axes[idx].contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)

        # Plot data points
        scatter = axes[idx].scatter(
            X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolors='black', s=30, alpha=0.8
        )

        accuracy = clf.score(X, y)
        axes[idx].set_xlabel('Feature 1', fontsize=10)
        axes[idx].set_ylabel('Feature 2', fontsize=10)
        axes[idx].set_title(
            f'Depth = {depth} (Accuracy: {accuracy:.2%})\n'
            f'Nodes: {clf.tree_.node_count} | Leaves: {clf.get_n_leaves()}',
            fontsize=11, fontweight='bold'
        )
        axes[idx].set_xlim(x_min, x_max)
        axes[idx].set_ylim(y_min, y_max)

    # Add legend
    class_0_patch = mpatches.Patch(color='#3498db', label='Class 0')
    class_1_patch = mpatches.Patch(color='#e74c3c', label='Class 1')
    fig.legend(handles=[class_0_patch, class_1_patch],
               loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Decision Boundaries at Different Tree Depths\n'
                 '(Showing How Trees Partition Feature Space)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    save_figure(fig, 'decision_tree_boundaries')
    plt.close(fig)


# =============================================================================
# BONUS: EXPORT TO GRAPHVIZ DOT FORMAT
# =============================================================================

def export_graphviz_format():
    """
    Export decision tree to Graphviz DOT format.
    This can be used with external tools for high-quality renders.
    """
    print("\n[Bonus] Exporting to Graphviz DOT format...")

    iris = load_iris()
    X, y = iris.data[:, :2], iris.target

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    # Export to DOT format
    dot_filepath = OUTPUT_DIR / 'decision_tree.dot'
    export_graphviz(
        clf,
        out_file=str(dot_filepath),
        feature_names=['Sepal Length', 'Sepal Width'],
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    print(f"  Saved: {dot_filepath}")
    print("  Note: Convert to PNG using: dot -Tpng decision_tree.dot -o decision_tree_graphviz.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to generate all decision tree visualizations.
    """
    print("=" * 60)
    print("Decision Tree Visualization Generator")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Resolution: {DPI} DPI")
    print("\nGenerating visualizations...")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate all visualizations
    generate_simple_tree_structure()
    generate_tree_with_gini()
    generate_tree_with_samples()
    generate_pruning_comparison()
    generate_tree_building_steps()
    generate_decision_boundary()
    export_graphviz_format()

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob('decision_tree*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
