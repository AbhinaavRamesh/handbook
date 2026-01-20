#!/usr/bin/env python3
"""
PCA (Principal Component Analysis) Visualization Generator

This script generates educational visualizations for PCA:
1. Principal components as vectors on 2D data scatter
2. Scree plot (eigenvalues + cumulative variance)
3. 3D to 2D projection animation (GIF)
4. Reconstruction error vs number of components
5. Biplot with loading vectors

Outputs:
- pca_principal_components.png: Principal components as vectors on 2D scatter
- pca_scree_plot.png: Eigenvalues and cumulative variance explained
- pca_3d_projection.gif: Animation of 3D to 2D projection
- pca_reconstruction_error.png: Reconstruction error vs number of components
- pca_biplot.png: Biplot with loading vectors
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import warnings
warnings.filterwarnings('ignore')

# Use the specified style
plt.style.use('seaborn-v0_8-whitegrid')

# Set random seed for reproducibility
np.random.seed(42)

# Color palette (consistent with other visualization scripts)
COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261', '#9B5DE5']
PRIMARY_COLOR = '#E63946'
SECONDARY_COLOR = '#457B9D'
TERTIARY_COLOR = '#2A9D8F'
ACCENT_COLOR = '#1D3557'
BACKGROUND_COLOR = '#F8F9FA'

# Output directory
OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/ml-coding/assets/'


class PCASimple:
    """Simple PCA implementation for visualization purposes."""

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        """Fit PCA using SVD."""
        n_samples, n_features = X.shape

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Store results
        self.components_ = Vt
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        if self.n_components is None:
            self.n_components_ = min(n_samples, n_features)
        else:
            self.n_components_ = self.n_components

        return self

    def transform(self, X):
        """Project data onto principal components."""
        X_centered = X - self.mean_
        return X_centered @ self.components_[:self.n_components_].T

    def inverse_transform(self, X_transformed):
        """Reconstruct data from principal components."""
        return X_transformed @ self.components_[:X_transformed.shape[1]] + self.mean_

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


def generate_2d_correlated_data(n_samples=200, correlation=0.8, noise=0.3):
    """Generate 2D data with correlation for visualization."""
    # Generate along a main axis with some perpendicular noise
    t = np.random.randn(n_samples)

    # Main direction
    x = t + noise * np.random.randn(n_samples)
    y = correlation * t + np.sqrt(1 - correlation**2) * np.random.randn(n_samples)

    # Scale and shift
    X = np.column_stack([x, y]) * 2

    return X


def generate_3d_data(n_samples=300):
    """Generate 3D data that lies mostly on a 2D plane for projection demo."""
    # Create data that lies mostly on a 2D plane embedded in 3D
    t1 = np.random.randn(n_samples)
    t2 = np.random.randn(n_samples)
    noise = np.random.randn(n_samples) * 0.3

    # Data on a tilted plane
    x = t1 + 0.3 * t2
    y = 0.5 * t1 + t2
    z = 0.3 * t1 + 0.3 * t2 + noise

    X = np.column_stack([x, y, z])
    return X


def generate_multi_feature_data(n_samples=200, n_features=10):
    """Generate multi-dimensional data with intrinsic lower dimensionality."""
    # True underlying factors
    n_factors = 3
    factors = np.random.randn(n_samples, n_factors)

    # Random loadings
    loadings = np.random.randn(n_factors, n_features)

    # Generate data with some noise
    X = factors @ loadings + 0.5 * np.random.randn(n_samples, n_features)

    return X


def create_principal_components_plot(output_path=None):
    """
    Create visualization showing principal components as vectors on 2D data.

    Shows:
    - Scatter plot of 2D data
    - PC1 and PC2 as vectors from the mean
    - Vector lengths proportional to explained variance
    """
    print("Generating principal components visualization...")

    # Generate data
    X = generate_2d_correlated_data(n_samples=200, correlation=0.85)

    # Fit PCA
    pca = PCASimple()
    pca.fit(X)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot data points
    ax.scatter(X[:, 0], X[:, 1], c=SECONDARY_COLOR, alpha=0.5, s=50,
               edgecolors='white', linewidth=0.5, label='Data points')

    # Plot mean point
    ax.scatter(pca.mean_[0], pca.mean_[1], c=ACCENT_COLOR, s=200, marker='*',
               edgecolors='white', linewidth=2, zorder=10, label='Mean')

    # Plot principal components as arrows
    # Scale vectors by sqrt of explained variance for visual clarity
    scale_factor = 3.0

    for i, (variance, color, label) in enumerate([
        (pca.explained_variance_[0], PRIMARY_COLOR, f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)'),
        (pca.explained_variance_[1], TERTIARY_COLOR, f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
    ]):
        # Component direction scaled by variance
        length = np.sqrt(variance) * scale_factor
        direction = pca.components_[i]

        # Draw arrow
        ax.annotate('',
                    xy=(pca.mean_[0] + direction[0] * length,
                        pca.mean_[1] + direction[1] * length),
                    xytext=(pca.mean_[0], pca.mean_[1]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=3),
                    zorder=9)

        # Add label at arrow tip
        ax.text(pca.mean_[0] + direction[0] * length * 1.15,
                pca.mean_[1] + direction[1] * length * 1.15,
                label, fontsize=11, fontweight='bold', color=color,
                ha='center', va='center')

    # Draw dashed lines showing data projections onto PC1
    n_proj = 20
    indices = np.random.choice(len(X), n_proj, replace=False)
    pc1_direction = pca.components_[0]

    for idx in indices:
        point = X[idx]
        centered = point - pca.mean_
        projection_length = np.dot(centered, pc1_direction)
        projection = pca.mean_ + projection_length * pc1_direction

        ax.plot([point[0], projection[0]], [point[1], projection[1]],
                'k--', alpha=0.2, linewidth=0.8)

    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title('Principal Component Analysis: Direction of Maximum Variance',
                 fontsize=14, fontweight='bold', pad=15)

    # Add explanation text box
    textstr = 'PC1: Direction of maximum variance\nPC2: Orthogonal to PC1'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.legend(loc='lower right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_scree_plot(output_path=None):
    """
    Create scree plot showing eigenvalues and cumulative variance.

    Shows:
    - Bar chart of individual explained variance
    - Line plot of cumulative explained variance
    - Threshold lines (e.g., 90%, 95%)
    """
    print("Generating scree plot...")

    # Generate multi-dimensional data
    X = generate_multi_feature_data(n_samples=200, n_features=10)

    # Fit PCA
    pca = PCASimple()
    pca.fit(X)

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

    n_components = len(pca.explained_variance_ratio_)
    components = np.arange(1, n_components + 1)

    # Bar chart for individual variance
    bars = ax1.bar(components, pca.explained_variance_ratio_ * 100,
                   color=SECONDARY_COLOR, alpha=0.7, edgecolor='white', linewidth=1.5,
                   label='Individual explained variance')

    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance (%)', fontsize=12, color=SECONDARY_COLOR)
    ax1.tick_params(axis='y', labelcolor=SECONDARY_COLOR)
    ax1.set_xticks(components)
    ax1.set_ylim(0, max(pca.explained_variance_ratio_ * 100) * 1.2)

    # Line plot for cumulative variance on secondary axis
    ax2 = ax1.twinx()
    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
    ax2.plot(components, cumulative, 'o-', color=PRIMARY_COLOR, linewidth=2.5,
             markersize=8, label='Cumulative explained variance')
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=12, color=PRIMARY_COLOR)
    ax2.tick_params(axis='y', labelcolor=PRIMARY_COLOR)
    ax2.set_ylim(0, 105)

    # Add threshold lines
    for threshold, style in [(90, '--'), (95, ':')]:
        ax2.axhline(y=threshold, color='gray', linestyle=style, alpha=0.7, linewidth=1.5)
        # Find component that crosses threshold
        crossing = np.searchsorted(cumulative, threshold) + 1
        if crossing <= n_components:
            ax2.annotate(f'{threshold}% at PC{crossing}',
                        xy=(crossing, threshold),
                        xytext=(crossing + 0.5, threshold + 3),
                        fontsize=9, color='gray')

    # Find elbow point
    diffs = np.diff(pca.explained_variance_ratio_)
    second_diffs = np.diff(diffs)
    if len(second_diffs) > 0:
        elbow_idx = np.argmax(second_diffs) + 2
        ax1.axvline(x=elbow_idx, color=TERTIARY_COLOR, linestyle='--', alpha=0.7, linewidth=2)
        ax1.annotate(f'Elbow at PC{elbow_idx}',
                    xy=(elbow_idx, pca.explained_variance_ratio_[elbow_idx-1] * 100),
                    xytext=(elbow_idx + 0.5, pca.explained_variance_ratio_[elbow_idx-1] * 100 + 5),
                    fontsize=10, fontweight='bold', color=TERTIARY_COLOR,
                    arrowprops=dict(arrowstyle='->', color=TERTIARY_COLOR))

    ax1.set_title('Scree Plot: Variance Explained by Principal Components',
                  fontsize=14, fontweight='bold', pad=15)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

    ax1.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")

    plt.close(fig)


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def create_3d_projection_animation(output_path=None):
    """
    Create animation showing 3D data being projected to 2D.

    Shows:
    - 3D scatter plot of original data
    - Principal component directions
    - Smooth transition to 2D projection
    - Final 2D scatter
    """
    print("Generating 3D to 2D projection animation...")

    # Generate 3D data
    X = generate_3d_data(n_samples=200)

    # Fit PCA
    pca = PCASimple(n_components=2)
    pca.fit(X)
    X_2d = pca.transform(X)

    # Create figure
    fig = plt.figure(figsize=(10, 6), dpi=100)

    # Number of frames
    n_frames = 60
    pause_frames = 15
    total_frames = n_frames + pause_frames * 2

    def animate(frame):
        fig.clear()

        # Determine animation phase
        if frame < pause_frames:
            # Initial 3D view
            t = 0
            phase = 'start'
        elif frame < pause_frames + n_frames:
            # Transition
            t = (frame - pause_frames) / n_frames
            phase = 'transition'
        else:
            # Final 2D view
            t = 1
            phase = 'end'

        if t < 1:
            # 3D view (morphing to 2D)
            ax = fig.add_subplot(111, projection='3d')

            # Calculate interpolated elevation angle
            elev = 30 * (1 - t) + 90 * t  # Goes from 30 to 90 (top-down)
            azim = 45 * (1 - t)  # Goes from 45 to 0

            ax.view_init(elev=elev, azim=azim)

            # Interpolate z values towards projection
            z_interp = X[:, 2] * (1 - t)

            # Color by original z for visual continuity
            colors = plt.cm.viridis((X[:, 2] - X[:, 2].min()) / (X[:, 2].max() - X[:, 2].min()))

            ax.scatter(X[:, 0], X[:, 1], z_interp, c=colors, s=30, alpha=0.7)

            # Draw principal component vectors (fading out)
            if t < 0.5:
                alpha = 1 - 2 * t
                scale = 2.0
                for i, color in enumerate([PRIMARY_COLOR, SECONDARY_COLOR]):
                    direction = pca.components_[i]
                    # Project 2D direction back to 3D (on z=0 plane)
                    end = pca.mean_ + direction * scale * np.sqrt(pca.explained_variance_[i])

                    ax.plot([pca.mean_[0], end[0]],
                           [pca.mean_[1], end[1]],
                           [0, 0],
                           color=color, linewidth=3, alpha=alpha)

            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            ax.set_zlabel('Z', fontsize=10)

            if phase == 'start':
                ax.set_title('Original 3D Data', fontsize=14, fontweight='bold')
            else:
                ax.set_title(f'Projecting to 2D... ({int(t*100)}%)',
                            fontsize=14, fontweight='bold')

            # Set consistent limits
            ax.set_xlim([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
            ax.set_ylim([X[:, 1].min() - 0.5, X[:, 1].max() + 0.5])
            ax.set_zlim([min(X[:, 2].min(), 0) - 0.5, X[:, 2].max() + 0.5])

        else:
            # Final 2D view
            ax = fig.add_subplot(111)

            colors = plt.cm.viridis((X[:, 2] - X[:, 2].min()) / (X[:, 2].max() - X[:, 2].min()))

            ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=50, alpha=0.7,
                      edgecolors='white', linewidth=0.5)

            # Draw PC axes
            scale = 2.0
            for i, (color, label) in enumerate([(PRIMARY_COLOR, 'PC1'), (SECONDARY_COLOR, 'PC2')]):
                length = scale * np.sqrt(pca.explained_variance_[i])
                ax.annotate('', xy=(length, 0) if i == 0 else (0, length),
                           xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
                ax.text(length * 1.1 if i == 0 else 0.2,
                       0.2 if i == 0 else length * 1.1,
                       f'{label} ({pca.explained_variance_ratio_[i]*100:.1f}%)',
                       fontsize=10, fontweight='bold', color=color)

            ax.set_xlabel('Principal Component 1', fontsize=11)
            ax.set_ylabel('Principal Component 2', fontsize=11)
            ax.set_title('2D PCA Projection (Color = Original Z)',
                        fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            # Add variance explained info
            total_var = sum(pca.explained_variance_ratio_[:2]) * 100
            ax.text(0.02, 0.98, f'Total variance explained: {total_var:.1f}%',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        return []

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                   interval=80, blit=False, repeat=True)

    if output_path:
        anim.save(output_path, writer='pillow', fps=12, dpi=100)
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_reconstruction_error_plot(output_path=None):
    """
    Create plot showing reconstruction error vs number of components.

    Shows:
    - MSE reconstruction error for different component counts
    - Comparison with total variance
    - Elbow point for optimal selection
    """
    print("Generating reconstruction error plot...")

    # Generate multi-dimensional data
    X = generate_multi_feature_data(n_samples=200, n_features=10)

    # Compute reconstruction error for each number of components
    n_features = X.shape[1]
    n_components_range = range(1, n_features + 1)

    errors = []
    variance_retained = []

    pca_full = PCASimple()
    pca_full.fit(X)

    for n_comp in n_components_range:
        pca = PCASimple(n_components=n_comp)
        pca.fit(X)

        # Transform and reconstruct
        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)

        # Compute MSE
        mse = np.mean((X - X_reconstructed) ** 2)
        errors.append(mse)

        # Variance retained
        var_ret = np.sum(pca_full.explained_variance_ratio_[:n_comp])
        variance_retained.append(var_ret)

    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot reconstruction error
    ax1.plot(n_components_range, errors, 'o-', color=PRIMARY_COLOR, linewidth=2.5,
             markersize=10, label='Reconstruction Error (MSE)')
    ax1.fill_between(n_components_range, errors, alpha=0.2, color=PRIMARY_COLOR)

    ax1.set_xlabel('Number of Principal Components', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12, color=PRIMARY_COLOR)
    ax1.tick_params(axis='y', labelcolor=PRIMARY_COLOR)
    ax1.set_xticks(list(n_components_range))

    # Plot variance retained on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(n_components_range, np.array(variance_retained) * 100, 's--',
             color=TERTIARY_COLOR, linewidth=2, markersize=8,
             label='Variance Retained (%)')
    ax2.set_ylabel('Variance Retained (%)', fontsize=12, color=TERTIARY_COLOR)
    ax2.tick_params(axis='y', labelcolor=TERTIARY_COLOR)
    ax2.set_ylim(0, 105)

    # Find and mark elbow point
    # Using the ratio of error reduction
    error_ratios = np.array(errors[:-1]) / np.array(errors[1:])
    if len(error_ratios) > 1:
        elbow_idx = np.argmax(np.diff(error_ratios) < 0) + 1
        if elbow_idx == 0:
            elbow_idx = 3  # Default to 3 if no clear elbow

        ax1.axvline(x=elbow_idx + 1, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.scatter([elbow_idx + 1], [errors[elbow_idx]], s=200, c='yellow',
                   edgecolors=PRIMARY_COLOR, linewidth=3, zorder=10)
        ax1.annotate(f'Optimal: {elbow_idx + 1} components\nMSE: {errors[elbow_idx]:.4f}',
                    xy=(elbow_idx + 1, errors[elbow_idx]),
                    xytext=(elbow_idx + 2, errors[elbow_idx] + max(errors) * 0.1),
                    fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax1.set_title('Reconstruction Error vs Number of Components',
                  fontsize=14, fontweight='bold', pad=15)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

    ax1.grid(True, alpha=0.3)

    # Add interpretation
    textstr = 'Lower error = better reconstruction\nFind elbow where error stabilizes'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax1.text(0.02, 0.3, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")

    plt.close(fig)


def create_biplot(output_path=None):
    """
    Create biplot showing both data projections and feature loadings.

    Shows:
    - Data points projected onto first two PCs
    - Loading vectors showing feature contributions
    - Feature labels at vector tips
    """
    print("Generating biplot with loading vectors...")

    # Generate data with named features
    np.random.seed(42)
    n_samples = 150

    # Create meaningful correlations
    feature_names = ['Height', 'Weight', 'Age', 'Income', 'Education', 'Experience']
    n_features = len(feature_names)

    # Generate correlated features
    base = np.random.randn(n_samples, 2)
    X = np.zeros((n_samples, n_features))

    X[:, 0] = base[:, 0] + 0.3 * np.random.randn(n_samples)  # Height
    X[:, 1] = 0.8 * base[:, 0] + 0.3 * np.random.randn(n_samples)  # Weight (correlated with height)
    X[:, 2] = base[:, 1] + 0.3 * np.random.randn(n_samples)  # Age
    X[:, 3] = 0.7 * base[:, 1] + 0.4 * np.random.randn(n_samples)  # Income (correlated with age)
    X[:, 4] = 0.5 * base[:, 0] + 0.5 * base[:, 1] + 0.3 * np.random.randn(n_samples)  # Education
    X[:, 5] = 0.6 * base[:, 1] + 0.4 * np.random.randn(n_samples)  # Experience

    # Standardize
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Fit PCA
    pca = PCASimple(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    # Compute loadings (correlations between variables and components)
    loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Plot data points
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=SECONDARY_COLOR, alpha=0.4, s=40,
                        edgecolors='white', linewidth=0.3, label='Data points')

    # Scaling factor for loading vectors
    # Scale to make vectors visible relative to data spread
    data_scale = max(np.abs(X_pca).max(axis=0))
    loading_scale = max(np.abs(loadings).max(axis=0))
    scale_factor = data_scale / loading_scale * 0.8

    # Plot loading vectors
    colors_loading = plt.cm.tab10(np.linspace(0, 1, n_features))

    for i, (loading, name, color) in enumerate(zip(loadings, feature_names, colors_loading)):
        # Draw arrow
        ax.annotate('', xy=(loading[0] * scale_factor, loading[1] * scale_factor),
                   xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2.5))

        # Add feature label
        # Position label slightly beyond arrow tip
        offset_x = 1.1 if loading[0] >= 0 else 0.9
        offset_y = 1.1 if loading[1] >= 0 else 0.9
        ha = 'left' if loading[0] >= 0 else 'right'
        va = 'bottom' if loading[1] >= 0 else 'top'

        ax.text(loading[0] * scale_factor * offset_x,
               loading[1] * scale_factor * offset_y,
               name, fontsize=10, fontweight='bold', color=color,
               ha=ha, va=va,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=color))

    # Add reference circle
    circle = plt.Circle((0, 0), scale_factor, fill=False, color='gray',
                        linestyle='--', alpha=0.5)
    ax.add_patch(circle)

    # Set axis labels with variance explained
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('PCA Biplot: Data Scores and Feature Loadings',
                 fontsize=14, fontweight='bold', pad=15)

    # Add origin lines
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Add interpretation guide
    textstr = ('Loading vectors show feature contributions:\n'
               '- Arrow direction: correlation with PCs\n'
               '- Arrow length: contribution strength\n'
               '- Close arrows: correlated features')
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")

    plt.close(fig)


def main():
    """Generate all PCA visualizations."""
    print("=" * 60)
    print("PCA Visualization Generator")
    print("=" * 60)
    print()

    # 1. Principal components visualization
    create_principal_components_plot(
        output_path=f'{OUTPUT_DIR}pca_principal_components.png'
    )

    # 2. Scree plot
    create_scree_plot(
        output_path=f'{OUTPUT_DIR}pca_scree_plot.png'
    )

    # 3. 3D to 2D projection animation
    create_3d_projection_animation(
        output_path=f'{OUTPUT_DIR}pca_3d_projection.gif'
    )

    # 4. Reconstruction error plot
    create_reconstruction_error_plot(
        output_path=f'{OUTPUT_DIR}pca_reconstruction_error.png'
    )

    # 5. Biplot with loading vectors
    create_biplot(
        output_path=f'{OUTPUT_DIR}pca_biplot.png'
    )

    print()
    print("=" * 60)
    print("All PCA visualizations generated successfully!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"  1. {OUTPUT_DIR}pca_principal_components.png")
    print(f"  2. {OUTPUT_DIR}pca_scree_plot.png")
    print(f"  3. {OUTPUT_DIR}pca_3d_projection.gif")
    print(f"  4. {OUTPUT_DIR}pca_reconstruction_error.png")
    print(f"  5. {OUTPUT_DIR}pca_biplot.png")


if __name__ == '__main__':
    main()
