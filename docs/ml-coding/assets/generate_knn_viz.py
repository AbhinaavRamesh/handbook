#!/usr/bin/env python3
"""
K-Nearest Neighbors (KNN) Visualization Generator

This script generates educational visualizations for KNN:
1. Query point with K nearest neighbors highlighted (circles)
2. Decision boundary comparison for K=1,3,5,15 (2x2 subplot)
3. Distance metric comparison (Euclidean circle vs Manhattan diamond)
4. Weighted vs uniform voting visualization

Outputs:
- knn_query_neighbors.png: Query point with K nearest neighbors
- knn_decision_boundaries.png: Decision boundaries for different K values
- knn_distance_metrics.png: Euclidean vs Manhattan distance visualization
- knn_weighted_voting.png: Weighted vs uniform voting comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.colors import ListedColormap
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Set random seed for reproducibility
np.random.seed(42)

# Color palette (colorblind-friendly)
COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']
BACKGROUND_COLOR = '#F8F9FA'
QUERY_COLOR = '#9B5DE5'  # Purple for query point

# Output directory
OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-coding/assets/'


def generate_classification_data(n_samples=150, random_state=42):
    """Generate sample 2D classification data with 3 classes."""
    np.random.seed(random_state)

    # Class 0: Bottom-left cluster
    X0 = np.random.randn(n_samples // 3, 2) * 0.8 + np.array([2, 2])

    # Class 1: Top-right cluster
    X1 = np.random.randn(n_samples // 3, 2) * 0.8 + np.array([6, 6])

    # Class 2: Bottom-right cluster
    X2 = np.random.randn(n_samples // 3, 2) * 0.8 + np.array([6, 2])

    X = np.vstack([X0, X1, X2])
    y = np.array([0] * (n_samples // 3) + [1] * (n_samples // 3) + [2] * (n_samples // 3))

    return X, y


def euclidean_distance(x1, x2):
    """Compute Euclidean distance between two points."""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    """Compute Manhattan distance between two points."""
    return np.sum(np.abs(x1 - x2))


def knn_predict(X_train, y_train, x_query, k=3, weighted=False, metric='euclidean'):
    """
    KNN prediction with optional weighting.

    Returns:
        prediction, k_nearest_indices, k_nearest_distances
    """
    # Compute distances
    if metric == 'euclidean':
        distances = np.array([euclidean_distance(x, x_query) for x in X_train])
    else:  # manhattan
        distances = np.array([manhattan_distance(x, x_query) for x in X_train])

    # Find k nearest
    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_distances = distances[k_nearest_indices]
    k_nearest_labels = y_train[k_nearest_indices]

    if weighted:
        # Weight = 1 / distance (with epsilon for numerical stability)
        weights = 1.0 / (k_nearest_distances + 1e-8)

        # Weighted vote
        unique_labels = np.unique(y_train)
        weighted_votes = {}
        for label in unique_labels:
            mask = k_nearest_labels == label
            weighted_votes[label] = np.sum(weights[mask])

        prediction = max(weighted_votes, key=weighted_votes.get)
    else:
        # Majority vote
        prediction = Counter(k_nearest_labels).most_common(1)[0][0]

    return prediction, k_nearest_indices, k_nearest_distances


def create_query_neighbors_visualization(X, y, output_path=None):
    """
    Create visualization of query point with K nearest neighbors highlighted.

    Shows:
    - Training data points colored by class
    - Query point prominently marked
    - K nearest neighbors connected with lines
    - Circles showing distance from query
    """
    print("Generating query neighbors visualization...")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Query point
    x_query = np.array([4.5, 4.0])
    k = 5

    # Get k nearest neighbors
    _, k_nearest_indices, k_nearest_distances = knn_predict(X, y, x_query, k=k)

    # Plot all training points
    for class_idx in range(3):
        mask = y == class_idx
        ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[class_idx], s=80, alpha=0.6,
                   label=f'Class {class_idx}', edgecolors='white', linewidth=0.5)

    # Highlight k nearest neighbors
    ax.scatter(X[k_nearest_indices, 0], X[k_nearest_indices, 1],
               c=[COLORS[y[i]] for i in k_nearest_indices], s=200,
               edgecolors='black', linewidth=2, zorder=5)

    # Draw lines from query to k nearest neighbors
    for i, idx in enumerate(k_nearest_indices):
        ax.plot([x_query[0], X[idx, 0]], [x_query[1], X[idx, 1]],
                'k--', alpha=0.5, linewidth=1.5)
        # Add distance label
        mid_x = (x_query[0] + X[idx, 0]) / 2
        mid_y = (x_query[1] + X[idx, 1]) / 2
        ax.annotate(f'd={k_nearest_distances[i]:.2f}', (mid_x, mid_y),
                    fontsize=8, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Plot query point
    ax.scatter(x_query[0], x_query[1], c=QUERY_COLOR, s=300, marker='*',
               edgecolors='black', linewidth=2, zorder=10, label='Query Point')

    # Draw concentric circles showing distance
    max_dist = max(k_nearest_distances) * 1.1
    for r in [max_dist * 0.5, max_dist]:
        circle = Circle((x_query[0], x_query[1]), r, fill=False,
                         color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.add_patch(circle)

    # Count votes
    k_labels = y[k_nearest_indices]
    vote_counts = Counter(k_labels)
    vote_str = ', '.join([f'Class {c}: {cnt}' for c, cnt in sorted(vote_counts.items())])

    ax.set_title(f'KNN Query with K={k} Nearest Neighbors\nVotes: {vote_str} -> Prediction: Class {vote_counts.most_common(1)[0][0]}',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_decision_boundary_comparison(X, y, output_path=None):
    """
    Create 2x2 subplot showing decision boundaries for K=1,3,5,15.

    Shows:
    - How decision boundary smoothness changes with K
    - Overfitting (K=1) vs underfitting (K=15)
    """
    print("Generating decision boundary comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=150)
    axes = axes.flatten()
    k_values = [1, 3, 5, 15]

    # Create mesh grid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Custom colormap
    cmap_light = ListedColormap(['#FFCCCC', '#CCE5FF', '#CCFFCC'])
    cmap_bold = ListedColormap(COLORS[:3])

    for ax, k in zip(axes, k_values):
        # Predict for each point in the mesh
        Z = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                x_query = np.array([xx[i, j], yy[i, j]])
                pred, _, _ = knn_predict(X, y, x_query, k=k)
                Z[i, j] = pred

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
        ax.contour(xx, yy, Z, colors='gray', linewidths=0.5, alpha=0.5)

        # Plot training points
        for class_idx in range(3):
            mask = y == class_idx
            ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[class_idx], s=30,
                       edgecolors='white', linewidth=0.5, alpha=0.8)

        # Add subtitle with interpretation
        if k == 1:
            subtitle = '(High variance, overfitting)'
        elif k == 15:
            subtitle = '(High bias, underfitting)'
        else:
            subtitle = '(Balanced)'

        ax.set_title(f'K = {k}\n{subtitle}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=9)
        ax.set_ylabel('Feature 2', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('KNN Decision Boundaries: Effect of K Value', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_distance_metric_comparison(output_path=None):
    """
    Create visualization comparing Euclidean (circle) vs Manhattan (diamond) distance.

    Shows:
    - Same query point with both metrics
    - Circle for Euclidean equidistant points
    - Diamond for Manhattan equidistant points
    """
    print("Generating distance metric comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)

    # Query point at origin for clarity
    query = np.array([0, 0])
    radius = 3

    # Left plot: Euclidean distance
    ax = axes[0]
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # Draw circle (all points at Euclidean distance = radius)
    circle = Circle((0, 0), radius, fill=False, color=COLORS[0], linewidth=3,
                    linestyle='-', label=f'Distance = {radius}')
    ax.add_patch(circle)

    # Fill the circle lightly
    circle_fill = Circle((0, 0), radius, fill=True, color=COLORS[0], alpha=0.1)
    ax.add_patch(circle_fill)

    # Plot query point
    ax.scatter(0, 0, c=QUERY_COLOR, s=200, marker='*', edgecolors='black',
               linewidth=2, zorder=10, label='Query Point')

    # Add some example points
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for angle in angles:
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        ax.scatter(x, y, c='gray', s=50, edgecolors='black', linewidth=1)

    ax.set_title('Euclidean Distance\n$d = \\sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)

    # Right plot: Manhattan distance
    ax = axes[1]
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # Draw diamond (all points at Manhattan distance = radius)
    diamond_vertices = np.array([
        [radius, 0],
        [0, radius],
        [-radius, 0],
        [0, -radius],
        [radius, 0]  # Close the shape
    ])
    ax.plot(diamond_vertices[:, 0], diamond_vertices[:, 1],
            color=COLORS[1], linewidth=3, label=f'Distance = {radius}')
    ax.fill(diamond_vertices[:, 0], diamond_vertices[:, 1],
            color=COLORS[1], alpha=0.1)

    # Plot query point
    ax.scatter(0, 0, c=QUERY_COLOR, s=200, marker='*', edgecolors='black',
               linewidth=2, zorder=10, label='Query Point')

    # Add some example points on the diamond
    for x, y in [(radius, 0), (0, radius), (-radius, 0), (0, -radius),
                 (radius/2, radius/2), (-radius/2, radius/2)]:
        ax.scatter(x, y, c='gray', s=50, edgecolors='black', linewidth=1)

    ax.set_title('Manhattan Distance\n$d = |x_2-x_1| + |y_2-y_1|$',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)

    fig.suptitle('Distance Metric Comparison: Equidistant Points from Query',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_weighted_voting_visualization(output_path=None):
    """
    Create visualization comparing weighted vs uniform voting.

    Shows:
    - Same k neighbors with different distances
    - How weights change the prediction
    - Bar charts showing vote weights
    """
    print("Generating weighted vs uniform voting visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=150)

    # Create a scenario where weighted and uniform give different results
    # Query point
    x_query = np.array([3.0, 3.0])

    # Neighbors: 1 close Class A, 2 farther Class B
    neighbors = np.array([
        [3.2, 3.1],   # Class 0, very close
        [1.0, 1.0],   # Class 1, far
        [1.5, 1.2],   # Class 1, far
    ])
    neighbor_labels = np.array([0, 1, 1])

    # Calculate distances
    distances = np.array([euclidean_distance(n, x_query) for n in neighbors])
    weights = 1.0 / (distances + 1e-8)
    normalized_weights = weights / weights.sum()

    for ax_idx, (ax, title, is_weighted) in enumerate(zip(
            axes, ['Uniform Voting', 'Distance-Weighted Voting'], [False, True])):

        # Plot the query point and neighbors
        ax.scatter(x_query[0], x_query[1], c=QUERY_COLOR, s=300, marker='*',
                   edgecolors='black', linewidth=2, zorder=10, label='Query')

        for i, (neighbor, label) in enumerate(zip(neighbors, neighbor_labels)):
            # Draw connection line
            ax.plot([x_query[0], neighbor[0]], [x_query[1], neighbor[1]],
                    'k--', alpha=0.5, linewidth=1)

            # Plot neighbor with size proportional to weight (if weighted)
            if is_weighted:
                size = 100 + normalized_weights[i] * 400
            else:
                size = 150

            ax.scatter(neighbor[0], neighbor[1], c=COLORS[label], s=size,
                       edgecolors='black', linewidth=2,
                       label=f'Class {label} (d={distances[i]:.2f})')

        # Calculate votes
        if is_weighted:
            class_votes = {0: 0, 1: 0}
            for label, weight in zip(neighbor_labels, weights):
                class_votes[label] += weight
            vote_str = f'Class 0: {class_votes[0]:.2f}, Class 1: {class_votes[1]:.2f}'
            prediction = 0 if class_votes[0] > class_votes[1] else 1
        else:
            vote_counts = Counter(neighbor_labels)
            vote_str = f'Class 0: {vote_counts[0]}, Class 1: {vote_counts[1]}'
            prediction = 1  # Majority vote: 2 Class 1 vs 1 Class 0

        ax.set_title(f'{title}\nVotes: {vote_str}\nPrediction: Class {prediction}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=10)
        ax.set_ylabel('Feature 2', fontsize=10)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Add explanation text box
        if is_weighted:
            explanation = 'Close neighbor has\nhigher influence'
        else:
            explanation = 'All neighbors have\nequal influence'
        ax.text(0.05, 0.05, explanation, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.8))

    fig.suptitle('Weighted vs Uniform KNN Voting\n(K=3, Same Neighbors, Different Outcomes)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")

    plt.close(fig)


def main():
    """Generate all KNN visualizations."""
    print("=" * 60)
    print("KNN Visualization Generator")
    print("=" * 60)
    print()

    # Generate sample data
    print("Generating sample data...")
    X, y = generate_classification_data(n_samples=150)
    print(f"  Created {len(X)} data points in 3 classes")
    print()

    # 1. Query point with K nearest neighbors highlighted
    create_query_neighbors_visualization(X, y,
        output_path=f'{OUTPUT_DIR}knn_query_neighbors.png')

    # 2. Decision boundary comparison for K=1,3,5,15
    create_decision_boundary_comparison(X, y,
        output_path=f'{OUTPUT_DIR}knn_decision_boundaries.png')

    # 3. Distance metric comparison (Euclidean vs Manhattan)
    create_distance_metric_comparison(
        output_path=f'{OUTPUT_DIR}knn_distance_metrics.png')

    # 4. Weighted vs uniform voting visualization
    create_weighted_voting_visualization(
        output_path=f'{OUTPUT_DIR}knn_weighted_voting.png')

    print()
    print("=" * 60)
    print("All KNN visualizations generated successfully!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"  1. {OUTPUT_DIR}knn_query_neighbors.png")
    print(f"  2. {OUTPUT_DIR}knn_decision_boundaries.png")
    print(f"  3. {OUTPUT_DIR}knn_distance_metrics.png")
    print(f"  4. {OUTPUT_DIR}knn_weighted_voting.png")


if __name__ == '__main__':
    main()
