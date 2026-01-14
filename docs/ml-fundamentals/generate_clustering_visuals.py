"""
Generate visualizations for Clustering page
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

# Set style for clean visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

OUTPUT_DIR = '/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/public/images/'

# Color palette
COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

# ============================================================
# 1. K-Means Clustering with Centroids
# ============================================================
def plot_kmeans_clustering():
    """Generate K-Means visualization showing clusters and centroids."""
    np.random.seed(42)

    # Generate clustered data
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

    # Fit K-Means
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot points colored by cluster
    for i in range(4):
        mask = labels == i
        ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[i], s=50, alpha=0.7,
                   label=f'Cluster {i+1}', edgecolors='white', linewidth=0.5)

    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, marker='X',
               edgecolors='white', linewidth=2, label='Centroids', zorder=10)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('K-Means Clustering (k=4)')
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}clustering-kmeans.svg', format='svg', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: clustering-kmeans.svg")

# ============================================================
# 2. Elbow Method for Optimal K
# ============================================================
def plot_elbow_method():
    """Generate Elbow method visualization for choosing optimal k."""
    np.random.seed(42)

    # Generate data with 4 natural clusters
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

    # Calculate WCSS for different k values
    k_range = range(1, 11)
    wcss = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot WCSS curve
    ax.plot(k_range, wcss, 'b-o', linewidth=2, markersize=8,
            markerfacecolor='white', markeredgewidth=2)

    # Highlight the elbow point (k=4)
    elbow_k = 4
    ax.scatter([elbow_k], [wcss[elbow_k-1]], c='#e74c3c', s=200, zorder=10,
               edgecolors='white', linewidth=2)
    ax.annotate('Elbow point\n(optimal k=4)', xy=(elbow_k, wcss[elbow_k-1]),
                xytext=(elbow_k+1.5, wcss[elbow_k-1]+200),
                fontsize=11, ha='left',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax.set_title('Elbow Method for Optimal k')
    ax.set_xticks(range(1, 11))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}clustering-elbow.svg', format='svg', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: clustering-elbow.svg")

# ============================================================
# 3. DBSCAN vs K-Means Comparison
# ============================================================
def plot_dbscan_vs_kmeans():
    """Compare DBSCAN and K-Means on non-spherical clusters."""
    np.random.seed(42)

    # Generate non-spherical (moon-shaped) clusters with noise
    X_moons, y_moons = make_moons(n_samples=300, noise=0.08, random_state=42)

    # Add some noise points
    noise = np.random.uniform(-1.5, 2.5, size=(30, 2))
    X = np.vstack([X_moons, noise])

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit K-Means (k=2)
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)

    # Fit DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X_scaled)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # K-Means subplot
    ax1 = axes[0]
    for i in range(2):
        mask = labels_kmeans == i
        ax1.scatter(X_scaled[mask, 0], X_scaled[mask, 1], c=COLORS[i], s=50,
                    alpha=0.7, edgecolors='white', linewidth=0.5)
    centroids = kmeans.cluster_centers_
    ax1.scatter(centroids[:, 0], centroids[:, 1], c='black', s=150, marker='X',
                edgecolors='white', linewidth=2, zorder=10)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title('K-Means (k=2)\nStruggles with non-spherical shapes')

    # DBSCAN subplot
    ax2 = axes[1]
    unique_labels = set(labels_dbscan)
    for i, label in enumerate(unique_labels):
        mask = labels_dbscan == label
        if label == -1:
            # Noise points
            ax2.scatter(X_scaled[mask, 0], X_scaled[mask, 1], c='gray', s=30,
                        alpha=0.5, marker='x', label='Noise')
        else:
            ax2.scatter(X_scaled[mask, 0], X_scaled[mask, 1], c=COLORS[label % len(COLORS)],
                        s=50, alpha=0.7, edgecolors='white', linewidth=0.5,
                        label=f'Cluster {label+1}')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('DBSCAN\nHandles arbitrary shapes + detects noise')
    ax2.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}clustering-dbscan-comparison.svg', format='svg', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: clustering-dbscan-comparison.svg")

# ============================================================
# Run all visualizations
# ============================================================
if __name__ == '__main__':
    print("Generating clustering visualizations...")
    plot_kmeans_clustering()
    plot_elbow_method()
    plot_dbscan_vs_kmeans()
    print("\nAll visualizations generated successfully!")
