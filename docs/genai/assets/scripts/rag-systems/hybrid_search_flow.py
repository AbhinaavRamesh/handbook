#!/usr/bin/env python3
"""
Hybrid Search Flow Visualization
Demonstrates dense + sparse retrieval fusion in RAG systems.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'images', 'rag-systems')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_hybrid_search_diagram():
    """Create hybrid search architecture diagram."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    colors = {
        'query': '#E3F2FD',
        'dense': '#C8E6C9',
        'sparse': '#FFF3E0',
        'fusion': '#F3E5F5',
        'result': '#E0F2F1'
    }

    # Title
    ax.text(8, 9.5, 'Hybrid Search Architecture', fontsize=16, fontweight='bold',
            ha='center', va='center')

    # Query input
    query_box = FancyBboxPatch((6.5, 7.5), 3, 1.2, boxstyle="round,pad=0.05",
                                facecolor=colors['query'], edgecolor='#333',
                                linewidth=2)
    ax.add_patch(query_box)
    ax.text(8, 8.3, 'User Query', fontsize=11, fontweight='bold', ha='center')
    ax.text(8, 7.9, '"How to optimize database queries?"', fontsize=8,
            ha='center', style='italic')

    # Dense path (left)
    # Embedding
    dense_embed = FancyBboxPatch((2, 5), 3, 1.2, boxstyle="round,pad=0.05",
                                  facecolor=colors['dense'], edgecolor='#333',
                                  linewidth=1.5)
    ax.add_patch(dense_embed)
    ax.text(3.5, 5.8, 'Embedding Model', fontsize=10, fontweight='bold', ha='center')
    ax.text(3.5, 5.3, 'Query -> [0.23, -0.45, ...]', fontsize=8, ha='center')

    # Vector Search
    dense_search = FancyBboxPatch((2, 3), 3, 1.2, boxstyle="round,pad=0.05",
                                   facecolor=colors['dense'], edgecolor='#333',
                                   linewidth=1.5)
    ax.add_patch(dense_search)
    ax.text(3.5, 3.8, 'Vector Search', fontsize=10, fontweight='bold', ha='center')
    ax.text(3.5, 3.3, 'HNSW / ANN', fontsize=8, ha='center')

    # Dense Results
    dense_results = FancyBboxPatch((2, 1), 3, 1.2, boxstyle="round,pad=0.05",
                                    facecolor=colors['dense'], edgecolor='#333',
                                    linewidth=1.5)
    ax.add_patch(dense_results)
    ax.text(3.5, 1.8, 'Semantic Matches', fontsize=10, fontweight='bold', ha='center')
    ax.text(3.5, 1.3, 'Conceptually similar', fontsize=8, ha='center')

    # Sparse path (right)
    # Tokenization
    sparse_token = FancyBboxPatch((11, 5), 3, 1.2, boxstyle="round,pad=0.05",
                                   facecolor=colors['sparse'], edgecolor='#333',
                                   linewidth=1.5)
    ax.add_patch(sparse_token)
    ax.text(12.5, 5.8, 'Tokenization', fontsize=10, fontweight='bold', ha='center')
    ax.text(12.5, 5.3, 'BM25 / TF-IDF', fontsize=8, ha='center')

    # Keyword Search
    sparse_search = FancyBboxPatch((11, 3), 3, 1.2, boxstyle="round,pad=0.05",
                                    facecolor=colors['sparse'], edgecolor='#333',
                                    linewidth=1.5)
    ax.add_patch(sparse_search)
    ax.text(12.5, 3.8, 'Inverted Index', fontsize=10, fontweight='bold', ha='center')
    ax.text(12.5, 3.3, 'Term matching', fontsize=8, ha='center')

    # Sparse Results
    sparse_results = FancyBboxPatch((11, 1), 3, 1.2, boxstyle="round,pad=0.05",
                                     facecolor=colors['sparse'], edgecolor='#333',
                                     linewidth=1.5)
    ax.add_patch(sparse_results)
    ax.text(12.5, 1.8, 'Keyword Matches', fontsize=10, fontweight='bold', ha='center')
    ax.text(12.5, 1.3, 'Exact term overlap', fontsize=8, ha='center')

    # Fusion
    fusion_box = FancyBboxPatch((6.5, 1), 3, 1.5, boxstyle="round,pad=0.05",
                                 facecolor=colors['fusion'], edgecolor='#333',
                                 linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(8, 2.1, 'RRF / Weighted', fontsize=10, fontweight='bold', ha='center')
    ax.text(8, 1.7, 'Score Fusion', fontsize=10, fontweight='bold', ha='center')
    ax.text(8, 1.3, 'alpha * dense + (1-alpha) * sparse', fontsize=8, ha='center')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='#666', lw=1.5,
                       connectionstyle='arc3,rad=0')

    # Query to branches
    ax.annotate('', xy=(3.5, 6.2), xytext=(7, 7.5),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2,
                               connectionstyle='arc3,rad=0.2'))
    ax.annotate('', xy=(12.5, 6.2), xytext=(9, 7.5),
                arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2,
                               connectionstyle='arc3,rad=-0.2'))

    # Dense path arrows
    ax.annotate('', xy=(3.5, 4.2), xytext=(3.5, 5), arrowprops=arrow_style)
    ax.annotate('', xy=(3.5, 2.2), xytext=(3.5, 3), arrowprops=arrow_style)

    # Sparse path arrows
    ax.annotate('', xy=(12.5, 4.2), xytext=(12.5, 5), arrowprops=arrow_style)
    ax.annotate('', xy=(12.5, 2.2), xytext=(12.5, 3), arrowprops=arrow_style)

    # To fusion
    ax.annotate('', xy=(6.5, 1.75), xytext=(5, 1.5),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))
    ax.annotate('', xy=(9.5, 1.75), xytext=(11, 1.5),
                arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2))

    # Labels
    ax.text(1.5, 6, 'Dense Path\n(Semantic)', fontsize=10, fontweight='bold',
            color='#2E7D32', ha='center')
    ax.text(14.5, 6, 'Sparse Path\n(Keyword)', fontsize=10, fontweight='bold',
            color='#E65100', ha='center')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['dense'], edgecolor='#333',
                       label='Dense (Semantic)'),
        mpatches.Patch(facecolor=colors['sparse'], edgecolor='#333',
                       label='Sparse (Keyword)'),
        mpatches.Patch(facecolor=colors['fusion'], edgecolor='#333',
                       label='Fusion'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', framealpha=0.9)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'hybrid_search_flow.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_fusion_comparison():
    """Compare different fusion methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sample documents with their scores
    docs = ['Doc A', 'Doc B', 'Doc C', 'Doc D', 'Doc E']

    # Dense scores (semantic)
    dense_scores = [0.92, 0.85, 0.78, 0.65, 0.55]
    # Sparse scores (BM25)
    sparse_scores = [0.45, 0.88, 0.72, 0.90, 0.35]

    # Plot 1: Individual scores
    ax1 = axes[0]
    x = np.arange(len(docs))
    width = 0.35

    bars1 = ax1.bar(x - width/2, dense_scores, width, label='Dense (Semantic)',
                    color='#4CAF50', alpha=0.8)
    bars2 = ax1.bar(x + width/2, sparse_scores, width, label='Sparse (BM25)',
                    color='#FF9800', alpha=0.8)

    ax1.set_ylabel('Score')
    ax1.set_xlabel('Document')
    ax1.set_xticks(x)
    ax1.set_xticklabels(docs)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.set_title('Individual Retrieval Scores', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Weighted fusion (alpha = 0.7)
    ax2 = axes[1]
    alpha = 0.7
    weighted_scores = [alpha * d + (1 - alpha) * s
                       for d, s in zip(dense_scores, sparse_scores)]

    # Sort by weighted score
    sorted_indices = np.argsort(weighted_scores)[::-1]
    sorted_docs = [docs[i] for i in sorted_indices]
    sorted_weighted = [weighted_scores[i] for i in sorted_indices]
    sorted_dense = [dense_scores[i] for i in sorted_indices]
    sorted_sparse = [sparse_scores[i] for i in sorted_indices]

    x = np.arange(len(docs))
    bars = ax2.bar(x, sorted_weighted, color='#9C27B0', alpha=0.8)

    # Add component breakdown
    for i, (d, s, w) in enumerate(zip(sorted_dense, sorted_sparse, sorted_weighted)):
        ax2.text(i, w + 0.02, f'{w:.2f}', ha='center', fontsize=9)

    ax2.set_ylabel('Weighted Score')
    ax2.set_xlabel('Document (sorted)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sorted_docs)
    ax2.set_ylim(0, 1)
    ax2.set_title(f'Weighted Fusion (alpha={alpha})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: RRF comparison
    ax3 = axes[2]

    # Calculate RRF scores
    k = 60
    dense_ranks = np.argsort(np.argsort(dense_scores)[::-1]) + 1
    sparse_ranks = np.argsort(np.argsort(sparse_scores)[::-1]) + 1

    rrf_scores = [1/(k + dr) + 1/(k + sr)
                  for dr, sr in zip(dense_ranks, sparse_ranks)]

    # Normalize RRF scores
    max_rrf = max(rrf_scores)
    rrf_normalized = [s / max_rrf for s in rrf_scores]

    # Sort by RRF
    rrf_sorted_indices = np.argsort(rrf_normalized)[::-1]
    rrf_sorted_docs = [docs[i] for i in rrf_sorted_indices]
    rrf_sorted_scores = [rrf_normalized[i] for i in rrf_sorted_indices]

    bars = ax3.bar(np.arange(len(docs)), rrf_sorted_scores, color='#00BCD4', alpha=0.8)

    for i, s in enumerate(rrf_sorted_scores):
        ax3.text(i, s + 0.02, f'{s:.2f}', ha='center', fontsize=9)

    ax3.set_ylabel('RRF Score (normalized)')
    ax3.set_xlabel('Document (sorted)')
    ax3.set_xticks(np.arange(len(docs)))
    ax3.set_xticklabels(rrf_sorted_docs)
    ax3.set_ylim(0, 1.2)
    ax3.set_title('Reciprocal Rank Fusion', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'fusion_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_alpha_sensitivity():
    """Visualize impact of alpha parameter on hybrid search."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Alpha values
    alphas = np.linspace(0, 1, 11)

    # Simulated performance for different query types
    np.random.seed(42)

    # Semantic queries benefit from higher alpha
    semantic_perf = 0.5 + 0.4 * alphas + np.random.randn(len(alphas)) * 0.02

    # Keyword queries benefit from lower alpha
    keyword_perf = 0.9 - 0.3 * alphas + np.random.randn(len(alphas)) * 0.02

    # Mixed queries are balanced
    mixed_perf = 0.6 + 0.3 * np.sin(np.pi * alphas) + np.random.randn(len(alphas)) * 0.02

    ax.plot(alphas, semantic_perf, 'o-', label='Semantic Queries',
            linewidth=2, markersize=8, color='#4CAF50')
    ax.plot(alphas, keyword_perf, 's-', label='Keyword Queries',
            linewidth=2, markersize=8, color='#FF9800')
    ax.plot(alphas, mixed_perf, '^-', label='Mixed Queries',
            linewidth=2, markersize=8, color='#9C27B0')

    ax.set_xlabel('Alpha (Dense Weight)', fontsize=11)
    ax.set_ylabel('Recall@5', fontsize=11)
    ax.set_title('Hybrid Search: Alpha Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.4, 1.0)

    # Add annotations
    ax.annotate('Pure BM25\n(Sparse only)', xy=(0, 0.9), xytext=(0.1, 0.75),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate('Pure Vector\n(Dense only)', xy=(1, 0.9), xytext=(0.9, 0.75),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    # Highlight optimal region
    ax.axvspan(0.5, 0.7, alpha=0.2, color='green', label='Typical Optimal Range')
    ax.text(0.6, 0.45, 'Typical\nOptimal\nRange', ha='center', fontsize=9, color='green')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'hybrid_alpha_sensitivity.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    create_hybrid_search_diagram()
    create_fusion_comparison()
    create_alpha_sensitivity()
    print("Hybrid search visualizations complete!")
