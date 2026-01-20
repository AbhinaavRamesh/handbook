#!/usr/bin/env python3
"""
K-Means Clustering Visualization Generator

This script generates educational visualizations for K-Means clustering:
1. K-Means iteration animation (centroids moving, assignments changing)
2. Elbow method plot
3. Silhouette analysis visualization
4. K-Means++ initialization comparison with random init
5. Final clustering result with decision boundaries (Voronoi)

Outputs:
- kmeans_iteration.gif: Animation showing K-Means convergence
- elbow_method.png: Elbow method for optimal K selection
- kmeans_pp_comparison.gif: Comparison of random vs K-Means++ initialization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from scipy.spatial import Voronoi
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# Use consistent styling
plt.style.use('seaborn-v0_8-whitegrid')

# Set random seed for reproducibility
np.random.seed(42)

# Color palette for clusters (colorblind-friendly)
COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261', '#9B5DE5']
CENTROID_COLOR = '#1D3557'
BACKGROUND_COLOR = '#F8F9FA'

# Consistent figure settings
DEFAULT_FIGSIZE = (10, 6)
DEFAULT_DPI = 150

# Output directory
OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-coding/assets/'


def generate_sample_data(n_samples=300, n_clusters=4, cluster_std=0.8, random_state=42):
    """Generate sample clustered data for visualization."""
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return X, y_true


def initialize_centroids_random(X, k):
    """Initialize centroids randomly from data points."""
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices].copy()


def initialize_centroids_kmeans_pp(X, k):
    """Initialize centroids using K-Means++ algorithm."""
    centroids = []

    # Choose first centroid randomly
    first_idx = np.random.randint(len(X))
    centroids.append(X[first_idx].copy())

    for _ in range(1, k):
        # Compute distances to nearest centroid
        distances = np.min([np.sum((X - c)**2, axis=1) for c in centroids], axis=0)

        # Choose next centroid with probability proportional to distance squared
        probabilities = distances / distances.sum()
        next_idx = np.random.choice(len(X), p=probabilities)
        centroids.append(X[next_idx].copy())

    return np.array(centroids)


def assign_clusters(X, centroids):
    """Assign each point to the nearest centroid."""
    distances = np.sqrt(((X[:, np.newaxis] - centroids)**2).sum(axis=2))
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, k):
    """Update centroids as the mean of assigned points."""
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        mask = labels == i
        if mask.sum() > 0:
            new_centroids[i] = X[mask].mean(axis=0)
    return new_centroids


def run_kmeans(X, k, init_method='kmeans++', max_iter=20):
    """
    Run K-Means clustering and record history for animation.

    Returns:
        history: List of (centroids, labels) tuples for each iteration
    """
    # Initialize centroids
    if init_method == 'kmeans++':
        centroids = initialize_centroids_kmeans_pp(X, k)
    else:
        centroids = initialize_centroids_random(X, k)

    history = []

    for iteration in range(max_iter):
        # Assign clusters
        labels = assign_clusters(X, centroids)
        history.append((centroids.copy(), labels.copy()))

        # Update centroids
        new_centroids = update_centroids(X, labels, k)

        # Check convergence
        if np.allclose(centroids, new_centroids, rtol=1e-4):
            # Add final state
            history.append((new_centroids.copy(), labels.copy()))
            break

        centroids = new_centroids

    return history


def compute_voronoi_regions(centroids, xlim, ylim):
    """
    Compute Voronoi regions for decision boundaries.
    Add dummy points far away to bound the regions.
    """
    # Add far away points to bound the Voronoi diagram
    far = 100
    dummy_points = np.array([
        [xlim[0] - far, ylim[0] - far],
        [xlim[0] - far, ylim[1] + far],
        [xlim[1] + far, ylim[0] - far],
        [xlim[1] + far, ylim[1] + far]
    ])

    all_points = np.vstack([centroids, dummy_points])
    vor = Voronoi(all_points)

    return vor


def get_voronoi_polygons(vor, xlim, ylim, n_centroids):
    """Extract bounded polygons for each centroid's Voronoi region."""
    polygons = []

    for i in range(n_centroids):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if -1 in region or len(region) == 0:
            polygons.append(None)
            continue

        # Get vertices of this region
        vertices = vor.vertices[region]

        # Clip to bounds
        vertices = np.clip(vertices, [xlim[0], ylim[0]], [xlim[1], ylim[1]])

        polygons.append(vertices)

    return polygons


def create_kmeans_animation(X, k=4, output_path=None):
    """
    Create animation showing K-Means iteration process (Lloyd's algorithm).

    Shows:
    - Data points colored by cluster assignment
    - Centroids moving towards cluster centers
    - Decision boundaries (Voronoi regions) updating
    """
    print("Generating K-Means iteration animation (Lloyd's algorithm)...")

    # Run K-Means and get history
    history = run_kmeans(X, k, init_method='kmeans++', max_iter=15)

    # Set up the figure with consistent styling
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, facecolor=BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    # Compute bounds
    margin = 1.5
    xlim = (X[:, 0].min() - margin, X[:, 0].max() + margin)
    ylim = (X[:, 1].min() - margin, X[:, 1].max() + margin)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Initialize plot elements
    scatter = ax.scatter([], [], s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    centroid_scatter = ax.scatter([], [], s=300, marker='X', c=CENTROID_COLOR,
                                   edgecolors='white', linewidth=2, zorder=10)

    # For Voronoi regions
    voronoi_patches = []

    title = ax.set_title("Lloyd's Algorithm: K-Means Iteration 0", fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)

    # Add legend
    legend_elements = [plt.scatter([], [], c=COLORS[i], s=100, label=f'Cluster {i+1}')
                      for i in range(k)]
    legend_elements.append(plt.scatter([], [], marker='X', c=CENTROID_COLOR, s=150,
                                       label='Centroids'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        centroid_scatter.set_offsets(np.empty((0, 2)))
        return scatter, centroid_scatter

    def animate(frame):
        # Clear previous Voronoi patches
        for patch in voronoi_patches:
            patch.remove()
        voronoi_patches.clear()

        # Determine which history frame to show (repeat last frame for pause)
        if frame < len(history):
            centroids, labels = history[frame]
            iteration = frame
        else:
            centroids, labels = history[-1]
            iteration = len(history) - 1

        # Draw Voronoi regions
        try:
            vor = compute_voronoi_regions(centroids, xlim, ylim)
            polygons = get_voronoi_polygons(vor, xlim, ylim, k)

            for i, polygon in enumerate(polygons):
                if polygon is not None and len(polygon) >= 3:
                    # Create polygon patch with light cluster color
                    patch = Polygon(polygon, facecolor=COLORS[i], alpha=0.15,
                                   edgecolor=COLORS[i], linewidth=2)
                    ax.add_patch(patch)
                    voronoi_patches.append(patch)
        except Exception:
            pass  # Skip Voronoi if it fails

        # Update scatter plot colors based on labels
        colors = [COLORS[label] for label in labels]
        scatter.set_offsets(X)
        scatter.set_facecolors(colors)

        # Update centroid positions
        centroid_scatter.set_offsets(centroids)

        # Update title
        if frame >= len(history) - 1:
            title.set_text(f"Lloyd's Algorithm: Converged! (Iteration {iteration})")
        else:
            title.set_text(f"Lloyd's Algorithm: K-Means Iteration {iteration}")

        return scatter, centroid_scatter, title

    # Create animation with pause at end
    n_frames = len(history) + 5  # Extra frames to pause at the end
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                                   interval=800, blit=False, repeat=True)

    if output_path:
        anim.save(output_path, writer='pillow', fps=1.5, dpi=100)
        print(f"  Saved: {output_path}")

    plt.close(fig)
    return anim


def create_elbow_plot(X, max_k=10, output_path=None):
    """
    Create elbow method plot for optimal K selection.

    Shows:
    - Inertia (within-cluster sum of squares) vs K
    - Highlighted "elbow" point
    - Silhouette scores on secondary axis
    """
    print("Generating elbow method plot...")

    k_range = range(2, max_k + 1)
    inertias = []
    silhouette_scores_list = []

    for k in k_range:
        # Run K-Means multiple times and take best
        best_inertia = float('inf')
        best_labels = None

        for _ in range(5):
            history = run_kmeans(X, k, init_method='kmeans++')
            centroids, labels = history[-1]

            # Compute inertia
            inertia = 0
            for i in range(k):
                mask = labels == i
                if mask.sum() > 0:
                    inertia += ((X[mask] - centroids[i])**2).sum()

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels

        inertias.append(best_inertia)

        # Compute silhouette score
        if len(np.unique(best_labels)) > 1:
            sil_score = silhouette_score(X, best_labels)
        else:
            sil_score = 0
        silhouette_scores_list.append(sil_score)

    # Create figure with two y-axes using consistent styling
    fig, ax1 = plt.subplots(figsize=DEFAULT_FIGSIZE, facecolor=BACKGROUND_COLOR)
    ax1.set_facecolor(BACKGROUND_COLOR)

    # Plot inertia
    color1 = '#E63946'
    ax1.plot(k_range, inertias, 'o-', color=color1, linewidth=2, markersize=10,
             label='Inertia (WCSS)')
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=11, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(list(k_range))

    # Find elbow point (using simple heuristic)
    # Compute the angle at each point
    inertias_normalized = np.array(inertias) / max(inertias)
    k_normalized = (np.array(list(k_range)) - min(k_range)) / (max(k_range) - min(k_range))

    # Second derivative approach
    second_derivative = np.diff(np.diff(inertias))
    elbow_idx = np.argmax(second_derivative) + 2  # +2 because of two diffs and 0-indexing
    elbow_k = list(k_range)[elbow_idx]

    # Highlight elbow
    ax1.axvline(x=elbow_k, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.annotate(f'Elbow at K={elbow_k}', xy=(elbow_k, inertias[elbow_idx]),
                xytext=(elbow_k + 0.5, inertias[elbow_idx] + max(inertias)*0.1),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray'))

    # Plot silhouette scores on secondary axis
    ax2 = ax1.twinx()
    color2 = '#457B9D'
    ax2.plot(k_range, silhouette_scores_list, 's--', color=color2, linewidth=2,
             markersize=8, label='Silhouette Score', alpha=0.8)
    ax2.set_ylabel('Silhouette Score', fontsize=11, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1)

    # Add title and legend
    ax1.set_title('Elbow Method for Optimal K Selection\nwith Silhouette Score',
                  fontsize=14, fontweight='bold', pad=15)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    # Add grid
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_silhouette_plot(X, k=4, output_path=None):
    """
    Create silhouette analysis visualization.

    Shows:
    - Silhouette coefficient for each sample
    - Grouped by cluster
    - Average silhouette score
    """
    print("Generating silhouette analysis plot...")

    # Run K-Means
    history = run_kmeans(X, k, init_method='kmeans++')
    centroids, labels = history[-1]

    # Compute silhouette scores
    sample_silhouette_values = silhouette_samples(X, labels)
    silhouette_avg = silhouette_score(X, labels)

    # Create figure with two subplots using consistent styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor=BACKGROUND_COLOR)

    # Left plot: Silhouette plot
    ax1.set_facecolor(BACKGROUND_COLOR)
    ax1.set_xlim([-0.1, 1])

    y_lower = 10
    for i in range(k):
        # Get silhouette scores for cluster i
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()

        cluster_size = cluster_silhouette_values.shape[0]
        y_upper = y_lower + cluster_size

        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                          facecolor=COLORS[i], edgecolor=COLORS[i], alpha=0.7)

        # Label the cluster
        ax1.text(-0.05, y_lower + 0.5 * cluster_size, str(i + 1), fontsize=10, fontweight='bold')

        y_lower = y_upper + 10

    # Add average line
    ax1.axvline(x=silhouette_avg, color=CENTROID_COLOR, linestyle="--", linewidth=2,
                label=f'Average: {silhouette_avg:.3f}')

    ax1.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Silhouette Coefficient', fontsize=11)
    ax1.set_ylabel('Cluster (sorted by silhouette score)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_yticks([])

    # Right plot: Scatter with cluster colors
    ax2.set_facecolor(BACKGROUND_COLOR)
    for i in range(k):
        mask = labels == i
        ax2.scatter(X[mask, 0], X[mask, 1], c=COLORS[i], s=50, alpha=0.7,
                   label=f'Cluster {i+1}', edgecolors='white', linewidth=0.5)

    ax2.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c=CENTROID_COLOR,
               edgecolors='white', linewidth=2, label='Centroids', zorder=10)

    ax2.set_title('Cluster Assignments', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Feature 1', fontsize=11)
    ax2.set_ylabel('Feature 2', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_initialization_comparison(X, k=4, output_path=None):
    """
    Create animation comparing random initialization vs K-Means++.

    Shows:
    - Side-by-side comparison
    - Both methods converging simultaneously
    - Final inertia comparison
    """
    print("Generating K-Means++ vs Random initialization comparison...")

    # Set seed for reproducibility but different initializations
    np.random.seed(123)

    # Run both methods
    history_random = run_kmeans(X, k, init_method='random', max_iter=15)

    np.random.seed(456)
    history_kmeans_pp = run_kmeans(X, k, init_method='kmeans++', max_iter=15)

    # Pad shorter history to match lengths
    max_len = max(len(history_random), len(history_kmeans_pp))
    while len(history_random) < max_len:
        history_random.append(history_random[-1])
    while len(history_kmeans_pp) < max_len:
        history_kmeans_pp.append(history_kmeans_pp[-1])

    # Set up figure with consistent styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor=BACKGROUND_COLOR)

    for ax in [ax1, ax2]:
        ax.set_facecolor(BACKGROUND_COLOR)

    # Compute bounds
    margin = 1.5
    xlim = (X[:, 0].min() - margin, X[:, 0].max() + margin)
    ylim = (X[:, 1].min() - margin, X[:, 1].max() + margin)

    for ax in [ax1, ax2]:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)

    # Initialize plot elements
    scatter1 = ax1.scatter([], [], s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    centroid1 = ax1.scatter([], [], s=300, marker='X', c=CENTROID_COLOR,
                            edgecolors='white', linewidth=2, zorder=10)

    scatter2 = ax2.scatter([], [], s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    centroid2 = ax2.scatter([], [], s=300, marker='X', c=CENTROID_COLOR,
                            edgecolors='white', linewidth=2, zorder=10)

    title1 = ax1.set_title('Random Initialization', fontsize=12, fontweight='bold')
    title2 = ax2.set_title('K-Means++ Initialization', fontsize=12, fontweight='bold')

    # Add main title
    fig.suptitle('Initialization Method Comparison', fontsize=14, fontweight='bold', y=1.02)

    # Text for inertia
    inertia_text1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10,
                             verticalalignment='top', bbox=dict(boxstyle='round',
                             facecolor='white', alpha=0.8))
    inertia_text2 = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=10,
                             verticalalignment='top', bbox=dict(boxstyle='round',
                             facecolor='white', alpha=0.8))

    def compute_inertia(X, centroids, labels, k):
        inertia = 0
        for i in range(k):
            mask = labels == i
            if mask.sum() > 0:
                inertia += ((X[mask] - centroids[i])**2).sum()
        return inertia

    def init():
        scatter1.set_offsets(np.empty((0, 2)))
        centroid1.set_offsets(np.empty((0, 2)))
        scatter2.set_offsets(np.empty((0, 2)))
        centroid2.set_offsets(np.empty((0, 2)))
        return scatter1, centroid1, scatter2, centroid2

    def animate(frame):
        # Get current state
        if frame < len(history_random):
            centroids_r, labels_r = history_random[frame]
            centroids_pp, labels_pp = history_kmeans_pp[frame]
            iteration = frame
        else:
            centroids_r, labels_r = history_random[-1]
            centroids_pp, labels_pp = history_kmeans_pp[-1]
            iteration = len(history_random) - 1

        # Update random init plot
        colors_r = [COLORS[label] for label in labels_r]
        scatter1.set_offsets(X)
        scatter1.set_facecolors(colors_r)
        centroid1.set_offsets(centroids_r)

        # Update K-Means++ plot
        colors_pp = [COLORS[label] for label in labels_pp]
        scatter2.set_offsets(X)
        scatter2.set_facecolors(colors_pp)
        centroid2.set_offsets(centroids_pp)

        # Compute and display inertias
        inertia_r = compute_inertia(X, centroids_r, labels_r, k)
        inertia_pp = compute_inertia(X, centroids_pp, labels_pp, k)

        inertia_text1.set_text(f'Inertia: {inertia_r:.1f}')
        inertia_text2.set_text(f'Inertia: {inertia_pp:.1f}')

        # Update titles
        if frame >= len(history_random) - 1:
            title1.set_text(f'Random Init: Converged (Iter {iteration})')
            title2.set_text(f'K-Means++: Converged (Iter {iteration})')
        else:
            title1.set_text(f'Random Init: Iteration {iteration}')
            title2.set_text(f'K-Means++: Iteration {iteration}')

        return scatter1, centroid1, scatter2, centroid2, inertia_text1, inertia_text2

    # Create animation
    n_frames = max_len + 5
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                                   interval=800, blit=False, repeat=True)

    plt.tight_layout()

    if output_path:
        anim.save(output_path, writer='pillow', fps=1.5, dpi=100)
        print(f"  Saved: {output_path}")

    plt.close(fig)
    return anim


def create_voronoi_final(X, k=4, output_path=None):
    """
    Create final clustering result with Voronoi decision boundaries.

    Shows:
    - Final cluster assignments
    - Voronoi regions as decision boundaries
    - Centroids prominently marked
    """
    print("Generating Voronoi decision boundary plot...")

    # Run K-Means
    history = run_kmeans(X, k, init_method='kmeans++')
    centroids, labels = history[-1]

    # Set up figure with consistent styling
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, facecolor=BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    # Compute bounds
    margin = 1.5
    xlim = (X[:, 0].min() - margin, X[:, 0].max() + margin)
    ylim = (X[:, 1].min() - margin, X[:, 1].max() + margin)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Draw Voronoi regions
    try:
        vor = compute_voronoi_regions(centroids, xlim, ylim)

        for i in range(k):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]

            if -1 not in region and len(region) > 0:
                vertices = vor.vertices[region]
                vertices = np.clip(vertices, [xlim[0], ylim[0]], [xlim[1], ylim[1]])

                # Draw polygon
                polygon = Polygon(vertices, facecolor=COLORS[i], alpha=0.2,
                                 edgecolor=COLORS[i], linewidth=2)
                ax.add_patch(polygon)
    except Exception:
        pass

    # Plot data points
    for i in range(k):
        mask = labels == i
        ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[i], s=60, alpha=0.8,
                  label=f'Cluster {i+1} (n={mask.sum()})', edgecolors='white', linewidth=0.5)

    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=400, c=CENTROID_COLOR,
              edgecolors='white', linewidth=3, label='Centroids', zorder=10)

    # Add centroid labels
    for i, (cx, cy) in enumerate(centroids):
        ax.annotate(f'C{i+1}', (cx, cy), xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=CENTROID_COLOR, alpha=0.8))

    ax.set_title('K-Means Final Clustering with Voronoi Decision Boundaries',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_kmeans_pp_init_viz(X, k=4, output_path=None):
    """
    Create visualization showing K-Means++ initialization process step by step.

    Shows how centroids are selected with probability proportional to distance squared.
    """
    print("Generating K-Means++ initialization visualization...")

    np.random.seed(42)
    n_samples = len(X)
    centroids = []

    # Step 1: Choose first centroid randomly
    first_idx = np.random.randint(n_samples)
    centroids.append(X[first_idx].copy())

    # Create figure with subplots for each step
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor=BACKGROUND_COLOR)
    axes = axes.flatten()

    # Compute bounds
    margin = 1.5
    xlim = (X[:, 0].min() - margin, X[:, 0].max() + margin)
    ylim = (X[:, 1].min() - margin, X[:, 1].max() + margin)

    step_titles = [
        'Step 1: Random first centroid',
        'Step 2: Second centroid (prob ~ dist^2)',
        'Step 3: Third centroid (prob ~ dist^2)',
        'Step 4: Fourth centroid (prob ~ dist^2)'
    ]

    for step in range(k):
        ax = axes[step]
        ax.set_facecolor(BACKGROUND_COLOR)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if step > 0:
            # Compute distances to nearest existing centroid
            distances = np.min([np.sum((X - c)**2, axis=1) for c in centroids], axis=0)
            probabilities = distances / distances.sum()

            # Choose next centroid
            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[next_idx].copy())

            # Color points by distance (probability)
            scatter = ax.scatter(X[:, 0], X[:, 1], c=probabilities, cmap='YlOrRd',
                               s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Selection Probability')
        else:
            # First step: all points gray
            ax.scatter(X[:, 0], X[:, 1], c='gray', s=50, alpha=0.5,
                      edgecolors='white', linewidth=0.5)

        # Plot existing centroids
        centroids_arr = np.array(centroids)
        ax.scatter(centroids_arr[:, 0], centroids_arr[:, 1], marker='X', s=300,
                  c=[COLORS[i] for i in range(len(centroids))],
                  edgecolors='white', linewidth=2, zorder=10)

        # Label centroids
        for i, (cx, cy) in enumerate(centroids_arr):
            ax.annotate(f'C{i+1}', (cx, cy), xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS[i], alpha=0.9))

        ax.set_title(step_titles[step], fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=10)
        ax.set_ylabel('Feature 2', fontsize=10)

    fig.suptitle('K-Means++ Initialization: Selecting Spread-Out Centroids',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        print(f"  Saved: {output_path}")

    plt.close(fig)


def main():
    """Generate all K-Means visualizations."""
    print("=" * 60)
    print("K-Means Clustering Visualization Generator")
    print("=" * 60)
    print()

    # Generate sample data
    print("Generating sample data...")
    X, y_true = generate_sample_data(n_samples=300, n_clusters=4, cluster_std=0.8)
    print(f"  Created {len(X)} data points in {len(np.unique(y_true))} clusters")
    print()

    # Generate all visualizations
    k = 4  # Number of clusters

    # 1. K-Means iteration animation
    create_kmeans_animation(X, k=k, output_path=f'{OUTPUT_DIR}kmeans_iteration.gif')

    # 2. Elbow method plot
    create_elbow_plot(X, max_k=10, output_path=f'{OUTPUT_DIR}elbow_method.png')

    # 3. Silhouette analysis
    create_silhouette_plot(X, k=k, output_path=f'{OUTPUT_DIR}silhouette_analysis.png')

    # 4. K-Means++ vs Random initialization comparison
    create_initialization_comparison(X, k=k, output_path=f'{OUTPUT_DIR}kmeans_pp_comparison.gif')

    # 5. Final Voronoi plot
    create_voronoi_final(X, k=k, output_path=f'{OUTPUT_DIR}voronoi_boundaries.png')

    # 6. K-Means++ initialization step-by-step
    create_kmeans_pp_init_viz(X, k=k, output_path=f'{OUTPUT_DIR}kmeans_pp_init.png')

    print()
    print("=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"  1. {OUTPUT_DIR}kmeans_iteration.gif")
    print(f"  2. {OUTPUT_DIR}elbow_method.png")
    print(f"  3. {OUTPUT_DIR}silhouette_analysis.png")
    print(f"  4. {OUTPUT_DIR}kmeans_pp_comparison.gif")
    print(f"  5. {OUTPUT_DIR}voronoi_boundaries.png")
    print(f"  6. {OUTPUT_DIR}kmeans_pp_init.png")


if __name__ == '__main__':
    main()
