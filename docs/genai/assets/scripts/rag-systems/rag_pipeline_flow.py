#!/usr/bin/env python3
"""
RAG Pipeline Flow Visualization
Generates an end-to-end RAG pipeline diagram showing both indexing and query phases.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'images', 'rag-systems')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_rag_pipeline_diagram():
    """Create a comprehensive RAG pipeline flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')

    # Colors
    colors = {
        'documents': '#E3F2FD',    # Light blue
        'processing': '#FFF3E0',   # Light orange
        'embedding': '#F3E5F5',    # Light purple
        'storage': '#E8F5E9',      # Light green
        'retrieval': '#FCE4EC',    # Light pink
        'generation': '#E0F2F1',   # Light teal
        'border': '#333333'
    }

    # Title
    ax.text(8, 9.5, 'RAG System Architecture', fontsize=18, fontweight='bold',
            ha='center', va='center')

    # === INDEXING PHASE (Top) ===
    # Background box for indexing
    indexing_bg = FancyBboxPatch((0.5, 5.5), 15, 3.5, boxstyle="round,pad=0.1",
                                  facecolor='#F5F5F5', edgecolor='#9E9E9E',
                                  linewidth=2, linestyle='--')
    ax.add_patch(indexing_bg)
    ax.text(1, 8.7, 'Indexing Phase (Offline)', fontsize=12, fontweight='bold',
            style='italic', color='#666666')

    # Documents
    doc_box = FancyBboxPatch((1, 6.5), 2, 1.5, boxstyle="round,pad=0.05",
                              facecolor=colors['documents'], edgecolor=colors['border'],
                              linewidth=1.5)
    ax.add_patch(doc_box)
    ax.text(2, 7.5, 'Documents', fontsize=10, fontweight='bold', ha='center')
    ax.text(2, 7.0, 'PDF, HTML, TXT', fontsize=8, ha='center', color='#666666')

    # Chunking
    chunk_box = FancyBboxPatch((4, 6.5), 2, 1.5, boxstyle="round,pad=0.05",
                                facecolor=colors['processing'], edgecolor=colors['border'],
                                linewidth=1.5)
    ax.add_patch(chunk_box)
    ax.text(5, 7.5, 'Chunking', fontsize=10, fontweight='bold', ha='center')
    ax.text(5, 7.0, 'Split & Overlap', fontsize=8, ha='center', color='#666666')

    # Embedding
    embed_box = FancyBboxPatch((7, 6.5), 2, 1.5, boxstyle="round,pad=0.05",
                                facecolor=colors['embedding'], edgecolor=colors['border'],
                                linewidth=1.5)
    ax.add_patch(embed_box)
    ax.text(8, 7.5, 'Embedding', fontsize=10, fontweight='bold', ha='center')
    ax.text(8, 7.0, 'Text → Vector', fontsize=8, ha='center', color='#666666')

    # Vector DB
    vdb_box = FancyBboxPatch((10, 6.5), 2.5, 1.5, boxstyle="round,pad=0.05",
                              facecolor=colors['storage'], edgecolor=colors['border'],
                              linewidth=1.5)
    ax.add_patch(vdb_box)
    ax.text(11.25, 7.5, 'Vector Database', fontsize=10, fontweight='bold', ha='center')
    ax.text(11.25, 7.0, 'HNSW Index', fontsize=8, ha='center', color='#666666')

    # Arrows for indexing
    arrow_style = dict(arrowstyle='->', color='#666666', lw=2,
                       connectionstyle='arc3,rad=0')
    ax.annotate('', xy=(4, 7.25), xytext=(3, 7.25),
                arrowprops=arrow_style)
    ax.annotate('', xy=(7, 7.25), xytext=(6, 7.25),
                arrowprops=arrow_style)
    ax.annotate('', xy=(10, 7.25), xytext=(9, 7.25),
                arrowprops=arrow_style)

    # === QUERY PHASE (Bottom) ===
    # Background box for query
    query_bg = FancyBboxPatch((0.5, 0.5), 15, 4.5, boxstyle="round,pad=0.1",
                               facecolor='#FAFAFA', edgecolor='#9E9E9E',
                               linewidth=2, linestyle='--')
    ax.add_patch(query_bg)
    ax.text(1, 4.7, 'Query Phase (Online)', fontsize=12, fontweight='bold',
            style='italic', color='#666666')

    # User Query
    query_box = FancyBboxPatch((1, 2.5), 2, 1.5, boxstyle="round,pad=0.05",
                                facecolor=colors['documents'], edgecolor=colors['border'],
                                linewidth=1.5)
    ax.add_patch(query_box)
    ax.text(2, 3.5, 'User Query', fontsize=10, fontweight='bold', ha='center')
    ax.text(2, 3.0, '"How does...?"', fontsize=8, ha='center', color='#666666')

    # Query Embedding
    qembed_box = FancyBboxPatch((4, 2.5), 2, 1.5, boxstyle="round,pad=0.05",
                                 facecolor=colors['embedding'], edgecolor=colors['border'],
                                 linewidth=1.5)
    ax.add_patch(qembed_box)
    ax.text(5, 3.5, 'Embed Query', fontsize=10, fontweight='bold', ha='center')
    ax.text(5, 3.0, 'Same Model', fontsize=8, ha='center', color='#666666')

    # Retrieval
    retrieve_box = FancyBboxPatch((7, 2.5), 2, 1.5, boxstyle="round,pad=0.05",
                                   facecolor=colors['retrieval'], edgecolor=colors['border'],
                                   linewidth=1.5)
    ax.add_patch(retrieve_box)
    ax.text(8, 3.5, 'Retrieval', fontsize=10, fontweight='bold', ha='center')
    ax.text(8, 3.0, 'Top-K Similar', fontsize=8, ha='center', color='#666666')

    # Context Assembly
    context_box = FancyBboxPatch((10, 2.5), 2, 1.5, boxstyle="round,pad=0.05",
                                  facecolor=colors['processing'], edgecolor=colors['border'],
                                  linewidth=1.5)
    ax.add_patch(context_box)
    ax.text(11, 3.5, 'Context', fontsize=10, fontweight='bold', ha='center')
    ax.text(11, 3.0, 'Assembly', fontsize=8, ha='center', color='#666666')

    # LLM Generation
    llm_box = FancyBboxPatch((13, 2.5), 2, 1.5, boxstyle="round,pad=0.05",
                              facecolor=colors['generation'], edgecolor=colors['border'],
                              linewidth=1.5)
    ax.add_patch(llm_box)
    ax.text(14, 3.5, 'LLM', fontsize=10, fontweight='bold', ha='center')
    ax.text(14, 3.0, 'Generation', fontsize=8, ha='center', color='#666666')

    # Arrows for query phase
    ax.annotate('', xy=(4, 3.25), xytext=(3, 3.25), arrowprops=arrow_style)
    ax.annotate('', xy=(7, 3.25), xytext=(6, 3.25), arrowprops=arrow_style)
    ax.annotate('', xy=(10, 3.25), xytext=(9, 3.25), arrowprops=arrow_style)
    ax.annotate('', xy=(13, 3.25), xytext=(12, 3.25), arrowprops=arrow_style)

    # Connection from Vector DB to Retrieval
    ax.annotate('', xy=(8, 4), xytext=(11.25, 6.5),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2,
                               connectionstyle='arc3,rad=-0.2'))

    # Response output
    ax.annotate('', xy=(15.5, 3.25), xytext=(15, 3.25),
                arrowprops=dict(arrowstyle='->', color='#2196F3', lw=2))
    ax.text(15.7, 3.25, 'Response', fontsize=10, fontweight='bold',
            ha='left', va='center', color='#2196F3')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['documents'], edgecolor=colors['border'], label='Input/Output'),
        mpatches.Patch(facecolor=colors['processing'], edgecolor=colors['border'], label='Processing'),
        mpatches.Patch(facecolor=colors['embedding'], edgecolor=colors['border'], label='Embedding'),
        mpatches.Patch(facecolor=colors['storage'], edgecolor=colors['border'], label='Storage'),
        mpatches.Patch(facecolor=colors['retrieval'], edgecolor=colors['border'], label='Retrieval'),
        mpatches.Patch(facecolor=colors['generation'], edgecolor=colors['border'], label='Generation'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'rag_pipeline_flow.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def create_rag_components_detail():
    """Create detailed component breakdown."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Chunking strategies
    ax1 = axes[0, 0]
    ax1.set_title('Chunking Strategies', fontsize=12, fontweight='bold')
    strategies = ['Fixed-Size', 'Recursive', 'Semantic', 'Document-Aware']
    scores = [0.65, 0.75, 0.85, 0.80]
    colors = ['#E3F2FD', '#FFF3E0', '#C8E6C9', '#F3E5F5']
    bars = ax1.barh(strategies, scores, color=colors, edgecolor='#333333')
    ax1.set_xlim(0, 1)
    ax1.set_xlabel('Relative Quality Score')
    for bar, score in zip(bars, scores):
        ax1.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}', va='center', fontsize=10)

    # Embedding dimensions comparison
    ax2 = axes[0, 1]
    ax2.set_title('Embedding Model Comparison', fontsize=12, fontweight='bold')
    models = ['MiniLM\n(384d)', 'BGE-base\n(768d)', 'BGE-large\n(1024d)', 'OpenAI\n(1536d)', 'OpenAI-3\n(3072d)']
    quality = [0.56, 0.62, 0.64, 0.62, 0.65]
    cost = [0.1, 0.2, 0.3, 0.5, 0.75]

    x = np.arange(len(models))
    width = 0.35
    bars1 = ax2.bar(x - width/2, quality, width, label='Quality (MTEB)', color='#4CAF50', alpha=0.8)
    bars2 = ax2.bar(x + width/2, cost, width, label='Relative Cost', color='#F44336', alpha=0.8)
    ax2.set_ylabel('Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=8)
    ax2.legend()
    ax2.set_ylim(0, 1)

    # Retrieval pipeline stages
    ax3 = axes[1, 0]
    ax3.set_title('Multi-Stage Retrieval', fontsize=12, fontweight='bold')
    stages = ['Initial\n(100 docs)', 'Reranked\n(20 docs)', 'MMR\n(5 docs)', 'Context\n(Final)']
    docs = [100, 20, 5, 5]
    latency = [50, 150, 20, 10]

    ax3_twin = ax3.twinx()
    x = np.arange(len(stages))
    bars1 = ax3.bar(x - 0.2, docs, 0.4, label='Documents', color='#2196F3', alpha=0.8)
    bars2 = ax3_twin.bar(x + 0.2, latency, 0.4, label='Latency (ms)', color='#FF9800', alpha=0.8)

    ax3.set_ylabel('Number of Documents', color='#2196F3')
    ax3_twin.set_ylabel('Latency (ms)', color='#FF9800')
    ax3.set_xticks(x)
    ax3.set_xticklabels(stages)
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')

    # Evaluation metrics
    ax4 = axes[1, 1]
    ax4.set_title('RAG Evaluation Metrics (RAGAS)', fontsize=12, fontweight='bold')
    metrics = ['Faithfulness', 'Answer\nRelevance', 'Context\nPrecision', 'Context\nRecall']
    baseline = [0.72, 0.68, 0.65, 0.70]
    optimized = [0.88, 0.82, 0.80, 0.85]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax4.bar(x - width/2, baseline, width, label='Baseline RAG', color='#9E9E9E', alpha=0.8)
    bars2 = ax4.bar(x + width/2, optimized, width, label='Optimized RAG', color='#4CAF50', alpha=0.8)
    ax4.set_ylabel('Score')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.set_ylim(0, 1)
    ax4.axhline(y=0.8, color='#F44336', linestyle='--', alpha=0.5, label='Target')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'rag_components_detail.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    create_rag_pipeline_diagram()
    create_rag_components_detail()
    print("RAG pipeline visualizations complete!")
