#!/usr/bin/env python3
"""
Embedding Space Visualization
Demonstrates t-SNE and UMAP visualizations of embedding spaces for RAG systems.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'images', 'rag-systems')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_mock_embeddings(n_samples=200, n_dims=384, n_clusters=5):
    """Generate mock embeddings for visualization."""
    np.random.seed(42)

    embeddings = []
    labels = []
    cluster_names = ['Technical Docs', 'FAQs', 'Tutorials', 'API Reference', 'Troubleshooting']

    for i in range(n_clusters):
        # Create cluster center
        center = np.random.randn(n_dims) * 0.5
        # Generate points around center
        n_points = n_samples // n_clusters
        cluster_points = center + np.random.randn(n_points, n_dims) * 0.3
        embeddings.extend(cluster_points)
        labels.extend([cluster_names[i]] * n_points)

    return np.array(embeddings), labels


def create_tsne_visualization():
    """Create t-SNE visualization of embedding space."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Generate mock data
    embeddings, labels = generate_mock_embeddings()

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot 1: Colored by category
    ax1 = axes[0]
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = [l == label for l in labels]
        ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[color], label=label, alpha=0.7, s=50)

    ax1.set_title('t-SNE Visualization of Document Embeddings', fontsize=12, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Query similarity visualization
    ax2 = axes[1]

    # Add a query point
    query_2d = np.mean(embeddings_2d[np.array(labels) == 'FAQs'], axis=0)

    # Calculate distances from query
    distances = np.sqrt(np.sum((embeddings_2d - query_2d) ** 2, axis=1))
    max_dist = np.max(distances)

    # Color by relevance (inverse distance)
    relevance = 1 - (distances / max_dist)

    scatter = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=relevance, cmap='RdYlGn', alpha=0.7, s=50)
    ax2.scatter(query_2d[0], query_2d[1], c='blue', marker='*',
               s=300, edgecolors='black', linewidths=2, label='Query')

    ax2.set_title('Semantic Similarity to Query', fontsize=12, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Similarity Score')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'embedding_space_tsne.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_embedding_comparison():
    """Compare different embedding model dimensionalities."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    np.random.seed(42)

    # Simulate different embedding dimensions
    dim_configs = [
        (384, 'MiniLM (384d)', 'Faster, Less Precise'),
        (768, 'BGE-base (768d)', 'Balanced'),
        (1536, 'OpenAI (1536d)', 'Higher Quality')
    ]

    for ax, (dim, title, subtitle) in zip(axes, dim_configs):
        # Generate clustered data
        n_points = 150
        n_clusters = 4

        # Higher dims = better cluster separation (simulated)
        separation_factor = dim / 384

        data_2d = []
        colors_list = []
        cluster_colors = ['#E53935', '#43A047', '#1E88E5', '#FB8C00']

        for i in range(n_clusters):
            angle = 2 * np.pi * i / n_clusters
            center_x = np.cos(angle) * separation_factor * 2
            center_y = np.sin(angle) * separation_factor * 2

            points = np.random.randn(n_points // n_clusters, 2) * 0.8
            points[:, 0] += center_x
            points[:, 1] += center_y

            data_2d.extend(points)
            colors_list.extend([cluster_colors[i]] * (n_points // n_clusters))

        data_2d = np.array(data_2d)

        ax.scatter(data_2d[:, 0], data_2d[:, 1], c=colors_list, alpha=0.6, s=40)
        ax.set_title(f'{title}\n{subtitle}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('Embedding Dimension Impact on Cluster Separation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'embedding_dimension_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_similarity_heatmap():
    """Create heatmap showing embedding similarities."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sample documents/queries
    items = [
        'Query: How to reset password?',
        'Doc: Password recovery steps',
        'Doc: Account security settings',
        'Query: Billing questions',
        'Doc: Payment methods guide',
        'Doc: Invoice history',
        'Query: Technical integration',
        'Doc: API documentation',
        'Doc: SDK installation'
    ]

    # Generate mock similarity matrix
    np.random.seed(42)
    n = len(items)
    similarity = np.eye(n)

    # Create realistic similarity patterns
    # Password cluster
    similarity[0:3, 0:3] = 0.7 + np.random.rand(3, 3) * 0.25
    # Billing cluster
    similarity[3:6, 3:6] = 0.7 + np.random.rand(3, 3) * 0.25
    # Technical cluster
    similarity[6:9, 6:9] = 0.7 + np.random.rand(3, 3) * 0.25

    # Low cross-cluster similarity
    for i in range(n):
        for j in range(n):
            if similarity[i, j] == 0:
                similarity[i, j] = 0.1 + np.random.rand() * 0.2

    # Ensure diagonal is 1 and symmetric
    np.fill_diagonal(similarity, 1.0)
    similarity = (similarity + similarity.T) / 2

    # Create heatmap
    im = ax.imshow(similarity, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Add labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([item[:20] + '...' if len(item) > 20 else item for item in items],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([item[:20] + '...' if len(item) > 20 else item for item in items],
                       fontsize=8)

    # Add values to cells
    for i in range(n):
        for j in range(n):
            text_color = 'white' if similarity[i, j] > 0.6 else 'black'
            ax.text(j, i, f'{similarity[i, j]:.2f}',
                   ha='center', va='center', fontsize=7, color=text_color)

    ax.set_title('Embedding Similarity Matrix\n(Query-Document Relationships)', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'embedding_similarity_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    create_tsne_visualization()
    create_embedding_comparison()
    create_similarity_heatmap()
    print("Embedding space visualizations complete!")
