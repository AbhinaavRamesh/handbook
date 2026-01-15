#!/usr/bin/env python3
"""
Chunking Strategy Comparison Visualization
Compares different chunking strategies and their impact on RAG quality.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'images', 'rag-systems')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_chunking_strategy_comparison():
    """Compare different chunking strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Strategy characteristics
    strategies = ['Fixed-Size', 'Recursive', 'Semantic', 'Document-Aware']
    metrics = {
        'Speed': [0.95, 0.85, 0.50, 0.60],
        'Context Preservation': [0.50, 0.70, 0.90, 0.85],
        'Consistency': [0.95, 0.80, 0.60, 0.70],
        'Retrieval Quality': [0.65, 0.75, 0.88, 0.82]
    }

    # Plot 1: Radar chart comparison
    ax1 = axes[0, 0]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    colors = ['#E53935', '#43A047', '#1E88E5', '#FB8C00']

    for i, strategy in enumerate(strategies):
        values = [metrics[m][i] for m in metrics.keys()]
        values += values[:1]
        ax1.plot(angles, values, 'o-', linewidth=2, label=strategy, color=colors[i])
        ax1.fill(angles, values, alpha=0.1, color=colors[i])

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics.keys(), fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.set_title('Chunking Strategy Radar Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bar comparison of retrieval quality
    ax2 = axes[0, 1]
    x = np.arange(len(strategies))
    quality = metrics['Retrieval Quality']
    bars = ax2.bar(x, quality, color=colors, alpha=0.8, edgecolor='black')

    ax2.set_ylabel('Retrieval Quality Score')
    ax2.set_xlabel('Chunking Strategy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=15)
    ax2.set_ylim(0, 1)
    ax2.set_title('Retrieval Quality by Strategy', fontsize=12, fontweight='bold')
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Target')
    ax2.legend()

    for bar, val in zip(bars, quality):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=10)

    # Plot 3: Chunk size distribution
    ax3 = axes[1, 0]

    np.random.seed(42)
    fixed_sizes = np.random.normal(500, 10, 100)  # Very consistent
    recursive_sizes = np.random.normal(450, 80, 100)  # Some variation
    semantic_sizes = np.random.exponential(400, 100) + 100  # Variable
    docaware_sizes = np.random.normal(500, 50, 100)  # Moderate variation

    data = [fixed_sizes, recursive_sizes, semantic_sizes, docaware_sizes]

    bp = ax3.boxplot(data, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_xticklabels(strategies, rotation=15)
    ax3.set_ylabel('Chunk Size (tokens)')
    ax3.set_title('Chunk Size Distribution by Strategy', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Processing time vs quality tradeoff
    ax4 = axes[1, 1]

    processing_time = [10, 25, 150, 80]  # ms per document
    quality_scores = metrics['Retrieval Quality']

    for i, (t, q, s) in enumerate(zip(processing_time, quality_scores, strategies)):
        ax4.scatter(t, q, s=200, c=colors[i], alpha=0.8, edgecolors='black', linewidths=2)
        ax4.annotate(s, (t, q), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax4.set_xlabel('Processing Time (ms/document)')
    ax4.set_ylabel('Retrieval Quality Score')
    ax4.set_title('Speed vs Quality Tradeoff', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 180)
    ax4.set_ylim(0.5, 1.0)

    # Add quadrant labels
    ax4.axhline(y=0.75, color='gray', linestyle='--', alpha=0.3)
    ax4.axvline(x=80, color='gray', linestyle='--', alpha=0.3)
    ax4.text(40, 0.55, 'Fast & Low Quality', fontsize=8, color='gray', ha='center')
    ax4.text(130, 0.55, 'Slow & Low Quality', fontsize=8, color='gray', ha='center')
    ax4.text(40, 0.95, 'Fast & High Quality', fontsize=8, color='gray', ha='center')
    ax4.text(130, 0.95, 'Slow & High Quality', fontsize=8, color='gray', ha='center')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'chunking_strategy_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_chunk_size_impact():
    """Visualize impact of chunk size on retrieval."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Chunk sizes to test
    chunk_sizes = [128, 256, 512, 768, 1024, 1536, 2048]

    # Simulated metrics
    np.random.seed(42)
    recall = [0.92, 0.95, 0.97, 0.95, 0.90, 0.85, 0.78]
    precision = [0.75, 0.82, 0.88, 0.85, 0.80, 0.72, 0.65]
    f1 = [2 * (r * p) / (r + p) for r, p in zip(recall, precision)]

    # Plot 1: Metrics by chunk size
    ax1 = axes[0]
    ax1.plot(chunk_sizes, recall, 'o-', label='Recall@5', linewidth=2, markersize=8, color='#4CAF50')
    ax1.plot(chunk_sizes, precision, 's-', label='Precision@5', linewidth=2, markersize=8, color='#2196F3')
    ax1.plot(chunk_sizes, f1, '^-', label='F1@5', linewidth=2, markersize=8, color='#FF9800')

    ax1.set_xlabel('Chunk Size (tokens)')
    ax1.set_ylabel('Score')
    ax1.set_title('Retrieval Metrics vs Chunk Size', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 1.0)

    # Highlight optimal range
    ax1.axvspan(400, 700, alpha=0.2, color='green', label='Optimal Range')

    # Plot 2: Number of chunks vs chunk size
    ax2 = axes[1]

    # Assuming 100,000 tokens total
    total_tokens = 100000
    overlap_ratio = 0.1

    num_chunks = [int(total_tokens / (size * (1 - overlap_ratio))) for size in chunk_sizes]
    storage_size = [n * (size + 100) / 1000 for n, size in zip(num_chunks, chunk_sizes)]  # KB

    ax2_twin = ax2.twinx()

    line1 = ax2.bar(np.arange(len(chunk_sizes)) - 0.2, num_chunks, 0.4,
                    label='Number of Chunks', color='#673AB7', alpha=0.7)
    line2 = ax2_twin.bar(np.arange(len(chunk_sizes)) + 0.2, storage_size, 0.4,
                         label='Index Size (KB)', color='#FF5722', alpha=0.7)

    ax2.set_xlabel('Chunk Size (tokens)')
    ax2.set_ylabel('Number of Chunks', color='#673AB7')
    ax2_twin.set_ylabel('Approximate Index Size (KB)', color='#FF5722')

    ax2.set_xticks(np.arange(len(chunk_sizes)))
    ax2.set_xticklabels(chunk_sizes)
    ax2.set_title('Chunk Count & Storage vs Chunk Size', fontsize=12, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'chunk_size_impact.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_overlap_visualization():
    """Visualize chunk overlap patterns."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Simulate document as colored segments
    doc_length = 1000
    chunk_size = 200
    overlap = 40

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    y_positions = [3, 2, 1]  # No overlap, 10% overlap, 20% overlap
    overlap_amounts = [0, 20, 40]
    labels = ['No Overlap (0%)', '10% Overlap', '20% Overlap']

    for y, overlap_amt, label in zip(y_positions, overlap_amounts, labels):
        step = chunk_size - overlap_amt
        start = 0
        chunk_idx = 0

        while start < doc_length - chunk_size // 2:
            end = min(start + chunk_size, doc_length)
            color = colors[chunk_idx % len(colors)]

            rect = plt.Rectangle((start, y - 0.3), end - start, 0.6,
                                 facecolor=color, edgecolor='black',
                                 linewidth=1, alpha=0.7)
            ax.add_patch(rect)

            # Add chunk number
            ax.text((start + end) / 2, y, f'C{chunk_idx+1}',
                   ha='center', va='center', fontsize=8, fontweight='bold')

            start += step
            chunk_idx += 1

        ax.text(-50, y, label, ha='right', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(-150, doc_length + 50)
    ax.set_ylim(0.2, 3.8)
    ax.set_xlabel('Document Position (tokens)', fontsize=11)
    ax.set_title('Chunk Overlap Strategies', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)

    # Add annotation for overlap benefit
    ax.annotate('Overlap preserves context\nat chunk boundaries',
               xy=(180, 1), xytext=(400, 0.7),
               fontsize=9, color='gray',
               arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'chunk_overlap_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    create_chunking_strategy_comparison()
    create_chunk_size_impact()
    create_overlap_visualization()
    print("Chunking comparison visualizations complete!")
