"""
Generate visualizations for Decision Trees documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split

# Set style for clean visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

def plot_decision_boundary():
    """Generate decision boundary visualization."""
    # Create a synthetic dataset with non-linear boundary
    np.random.seed(42)
    X, y = make_moons(n_samples=200, noise=0.25, random_state=42)

    # Train a decision tree
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)

    # Create mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict on mesh grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Decision regions
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='gray', linewidths=0.5, alpha=0.5)

    # Data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                         edgecolors='black', linewidths=0.5, s=50)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Tree: Axis-Aligned Decision Boundaries')

    # Add annotation
    ax.annotate('Axis-aligned\nsplits create\nrectangular\nregions',
                xy=(0.5, 0.3), fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/images/decision-tree-boundary.svg',
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: decision-tree-boundary.svg")


def plot_gini_vs_entropy():
    """Plot Gini impurity vs Entropy curves."""
    # Probability of class 1
    p = np.linspace(0.001, 0.999, 100)

    # Gini impurity: 1 - p^2 - (1-p)^2 = 2p(1-p)
    gini = 2 * p * (1 - p)

    # Entropy: -p*log2(p) - (1-p)*log2(1-p)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    # Normalized entropy (to compare with Gini)
    entropy_normalized = entropy / 2  # Divide by max entropy for binary case

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(p, gini, 'b-', linewidth=2.5, label='Gini Impurity')
    ax.plot(p, entropy, 'r-', linewidth=2.5, label='Entropy')
    ax.plot(p, entropy_normalized, 'r--', linewidth=2, alpha=0.7,
            label='Entropy (scaled to [0, 0.5])')

    # Mark maximum points
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7)
    ax.scatter([0.5], [0.5], color='blue', s=80, zorder=5, marker='o')
    ax.scatter([0.5], [1.0], color='red', s=80, zorder=5, marker='o')

    # Annotations
    ax.annotate('Max Gini = 0.5', xy=(0.52, 0.48), fontsize=10, color='blue')
    ax.annotate('Max Entropy = 1.0', xy=(0.52, 0.95), fontsize=10, color='red')
    ax.annotate('Pure\n(p=0)', xy=(0.02, 0.02), fontsize=9, ha='left')
    ax.annotate('Pure\n(p=1)', xy=(0.98, 0.02), fontsize=9, ha='right')

    ax.set_xlabel('Probability of Class 1 (p)')
    ax.set_ylabel('Impurity Value')
    ax.set_title('Gini Impurity vs Entropy for Binary Classification')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/images/gini-vs-entropy.svg',
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: gini-vs-entropy.svg")


def plot_depth_vs_accuracy():
    """Plot tree depth vs accuracy to illustrate overfitting."""
    np.random.seed(42)

    # Generate dataset with some noise
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                               n_redundant=2, n_clusters_per_class=2,
                               random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    depths = range(1, 21)
    train_scores = []
    test_scores = []

    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        train_scores.append(clf.score(X_train, y_train))
        test_scores.append(clf.score(X_test, y_test))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(depths, train_scores, 'b-o', linewidth=2, markersize=6, label='Training Accuracy')
    ax.plot(depths, test_scores, 'r-s', linewidth=2, markersize=6, label='Test Accuracy')

    # Find optimal depth
    optimal_depth = depths[np.argmax(test_scores)]
    max_test_acc = max(test_scores)

    # Mark optimal point
    ax.axvline(x=optimal_depth, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.scatter([optimal_depth], [max_test_acc], color='green', s=150, zorder=5, marker='*')

    # Shade regions
    ax.axvspan(1, optimal_depth, alpha=0.1, color='blue', label='_nolegend_')
    ax.axvspan(optimal_depth, 20, alpha=0.1, color='red', label='_nolegend_')

    # Annotations
    ax.annotate('Underfitting\nregion', xy=(2.5, 0.72), fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
    ax.annotate('Overfitting\nregion', xy=(15, 0.72), fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.5))
    ax.annotate(f'Optimal depth = {optimal_depth}',
                xy=(optimal_depth + 0.3, max_test_acc - 0.03), fontsize=10, color='green')

    # Gap annotation
    gap_depth = 18
    train_at_gap = train_scores[gap_depth - 1]
    test_at_gap = test_scores[gap_depth - 1]
    ax.annotate('', xy=(gap_depth, test_at_gap), xytext=(gap_depth, train_at_gap),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.annotate('Overfitting\ngap', xy=(gap_depth + 0.5, (train_at_gap + test_at_gap) / 2),
                fontsize=9, color='purple')

    ax.set_xlabel('Tree Depth (max_depth)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Tree Depth vs Accuracy: Overfitting Illustration')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim(1, 20)
    ax.set_ylim(0.65, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/images/tree-depth-vs-accuracy.svg',
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved: tree-depth-vs-accuracy.svg")


if __name__ == '__main__':
    print("Generating Decision Tree visualizations...")
    plot_decision_boundary()
    plot_gini_vs_entropy()
    plot_depth_vs_accuracy()
    print("All visualizations generated successfully!")
