"""
Generate KNN visualizations for documentation.
Creates SVG files for:
1. KNN classification with different k values (k=1, k=5, k=15)
2. Decision boundary comparison for different k
3. Distance metrics comparison (Euclidean vs Manhattan)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons, make_classification
import os

# Set style for clean visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Output directory
OUTPUT_DIR = "/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colors
COLORS = ['#3b82f6', '#ef4444']  # Blue, Red
CMAP_LIGHT = ListedColormap(['#dbeafe', '#fee2e2'])  # Light blue, Light red
CMAP_BOLD = ListedColormap(COLORS)


def generate_decision_boundary_comparison():
    """
    Generate decision boundary comparison for different k values.
    Shows k=1, k=5, k=15 side by side.
    """
    # Generate moon-shaped data for interesting boundaries
    np.random.seed(42)
    X, y = make_moons(n_samples=150, noise=0.25, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    k_values = [1, 5, 15]

    for ax, k in zip(axes, k_values):
        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Train and predict
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, cmap=CMAP_LIGHT, alpha=0.8)
        ax.contour(xx, yy, Z, colors='gray', linewidths=0.5, alpha=0.5)

        # Plot points
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c=COLORS[0], edgecolors='white',
                   s=50, label='Class 0', linewidths=0.5)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c=COLORS[1], edgecolors='white',
                   s=50, label='Class 1', linewidths=0.5)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(f'k = {k}', fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        # Add description
        if k == 1:
            desc = "High variance\nOverfitting risk"
        elif k == 5:
            desc = "Balanced\nGood generalization"
        else:
            desc = "High bias\nSmooth boundary"
        ax.text(0.05, 0.95, desc, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'knn-decision-boundary.svg'),
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated: knn-decision-boundary.svg")


def generate_distance_metrics_comparison():
    """
    Generate visualization comparing Euclidean vs Manhattan distance.
    Shows how the metrics create different "neighborhoods".
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Generate sample data
    np.random.seed(123)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               class_sep=1.2, random_state=42)

    metrics = ['euclidean', 'manhattan']
    titles = ['Euclidean Distance (L2)', 'Manhattan Distance (L1)']
    formulas = [r'$d = \sqrt{\sum(x_i - y_i)^2}$', r'$d = \sum|x_i - y_i|$']

    for ax, metric, title, formula in zip(axes, metrics, titles, formulas):
        # Create mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Train and predict
        clf = KNeighborsClassifier(n_neighbors=5, metric=metric)
        clf.fit(X, y)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot
        ax.contourf(xx, yy, Z, cmap=CMAP_LIGHT, alpha=0.8)
        ax.contour(xx, yy, Z, colors='gray', linewidths=0.5, alpha=0.5)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c=COLORS[0], edgecolors='white',
                   s=50, linewidths=0.5)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c=COLORS[1], edgecolors='white',
                   s=50, linewidths=0.5)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(f'{title}\n{formula}', fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'knn-distance-metrics.svg'),
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated: knn-distance-metrics.svg")


def generate_k_effect_visualization():
    """
    Generate a visualization showing how k affects prediction for a single query point.
    Shows the same query point with k=1, k=5, k=15 and which neighbors are selected.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Create simple dataset
    np.random.seed(42)
    n_points = 30

    # Class 0: blue points (left cluster)
    X0 = np.random.randn(n_points // 2, 2) * 0.5 + np.array([-1, 0])
    # Class 1: red points (right cluster)
    X1 = np.random.randn(n_points // 2, 2) * 0.5 + np.array([1, 0])

    X = np.vstack([X0, X1])
    y = np.array([0] * (n_points // 2) + [1] * (n_points // 2))

    # Query point (in the middle, slightly toward red)
    query = np.array([[0.3, 0.1]])

    k_values = [1, 5, 15]

    for ax, k in zip(axes, k_values):
        # Train KNN
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X, y)

        # Get neighbors
        distances, indices = clf.kneighbors(query)
        prediction = clf.predict(query)[0]

        # Plot all points
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c=COLORS[0], edgecolors='white',
                   s=80, label='Class 0', linewidths=0.5, alpha=0.6)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c=COLORS[1], edgecolors='white',
                   s=80, label='Class 1', linewidths=0.5, alpha=0.6)

        # Highlight k nearest neighbors
        neighbor_indices = indices[0]
        ax.scatter(X[neighbor_indices, 0], X[neighbor_indices, 1],
                   c=[COLORS[y[i]] for i in neighbor_indices],
                   edgecolors='black', s=120, linewidths=2, zorder=5)

        # Draw lines to neighbors
        for idx in neighbor_indices:
            ax.plot([query[0, 0], X[idx, 0]], [query[0, 1], X[idx, 1]],
                    'k--', alpha=0.3, linewidth=1)

        # Plot query point
        pred_color = COLORS[prediction]
        ax.scatter(query[0, 0], query[0, 1], c='white', edgecolors='black',
                   s=200, marker='*', linewidths=2, zorder=10, label='Query')

        # Count votes
        neighbor_labels = y[neighbor_indices]
        votes_0 = np.sum(neighbor_labels == 0)
        votes_1 = np.sum(neighbor_labels == 1)

        ax.set_title(f'k = {k}', fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        # Add vote count
        vote_text = f"Votes: {votes_0} blue, {votes_1} red\nPrediction: {'Blue' if prediction == 0 else 'Red'}"
        ax.text(0.05, 0.95, vote_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2, 2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'knn-k-effect.svg'),
                format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated: knn-k-effect.svg")


if __name__ == "__main__":
    print("Generating KNN visualizations...")
    generate_decision_boundary_comparison()
    generate_distance_metrics_comparison()
    generate_k_effect_visualization()
    print("\nAll visualizations generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
