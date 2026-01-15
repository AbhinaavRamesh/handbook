"""
Enterprise RAG Architecture Visualization
Generates diagrams for enterprise RAG system design
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          'images', 'system-design')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_rag_pipeline():
    """Create RAG pipeline architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Enterprise RAG Pipeline Architecture', fontsize=16, fontweight='bold', pad=20)

    # Color scheme
    colors = {
        'ingest': '#e3f2fd',
        'storage': '#e8f5e9',
        'query': '#fff3e0',
        'generation': '#fce4ec',
        'security': '#f3e5f5'
    }

    # Ingestion Pipeline
    ingest_box = FancyBboxPatch((0.5, 5.5), 4, 2, boxstyle="round,pad=0.1",
                                 facecolor=colors['ingest'], edgecolor='#1976d2', linewidth=2)
    ax.add_patch(ingest_box)
    ax.text(2.5, 7, 'Ingestion Pipeline', ha='center', va='center', fontsize=11, fontweight='bold')

    ingest_steps = ['Documents', 'Processing', 'Chunking', 'Embedding']
    for i, step in enumerate(ingest_steps):
        ax.text(1 + i * 1, 6, step, ha='center', va='center', fontsize=7,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#1976d2'))
        if i < 3:
            ax.annotate("", xy=(1.5 + i * 1, 6), xytext=(1.3 + i * 1, 6),
                       arrowprops=dict(arrowstyle='->', color='#1976d2', lw=1))

    # Storage Layer
    storage_box = FancyBboxPatch((5, 5.5), 4, 2, boxstyle="round,pad=0.1",
                                  facecolor=colors['storage'], edgecolor='#388e3c', linewidth=2)
    ax.add_patch(storage_box)
    ax.text(7, 7, 'Storage Layer', ha='center', va='center', fontsize=11, fontweight='bold')

    storage_items = [('Vector DB', 5.5, 6), ('Metadata', 7, 6), ('Docs', 8.5, 6)]
    for name, x, y in storage_items:
        ax.text(x, y, name, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#388e3c'))

    # Query Pipeline
    query_box = FancyBboxPatch((0.5, 2.5), 4, 2, boxstyle="round,pad=0.1",
                                facecolor=colors['query'], edgecolor='#f57c00', linewidth=2)
    ax.add_patch(query_box)
    ax.text(2.5, 4, 'Query Pipeline', ha='center', va='center', fontsize=11, fontweight='bold')

    query_steps = ['Query', 'Hybrid\nSearch', 'Rerank', 'Context']
    for i, step in enumerate(query_steps):
        ax.text(1 + i * 1, 3, step, ha='center', va='center', fontsize=7,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#f57c00'))
        if i < 3:
            ax.annotate("", xy=(1.5 + i * 1, 3), xytext=(1.3 + i * 1, 3),
                       arrowprops=dict(arrowstyle='->', color='#f57c00', lw=1))

    # Generation Layer
    gen_box = FancyBboxPatch((5, 2.5), 4, 2, boxstyle="round,pad=0.1",
                              facecolor=colors['generation'], edgecolor='#c2185b', linewidth=2)
    ax.add_patch(gen_box)
    ax.text(7, 4, 'Generation Layer', ha='center', va='center', fontsize=11, fontweight='bold')

    gen_steps = ['Prompt', 'LLM', 'Citations']
    for i, step in enumerate(gen_steps):
        ax.text(5.5 + i * 1.3, 3, step, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#c2185b'))

    # Access Control
    acl_box = FancyBboxPatch((10, 3.5), 3.5, 4, boxstyle="round,pad=0.1",
                              facecolor=colors['security'], edgecolor='#7b1fa2', linewidth=2)
    ax.add_patch(acl_box)
    ax.text(11.75, 7, 'Access Control', ha='center', va='center', fontsize=11, fontweight='bold')

    acl_items = ['Authentication', 'Authorization', 'ACL Filtering', 'Audit Logging']
    for i, item in enumerate(acl_items):
        ax.text(11.75, 6 - i * 0.7, f'• {item}', ha='center', va='center', fontsize=8)

    # Arrows connecting components
    # Ingest to Storage
    ax.annotate("", xy=(5, 6.5), xytext=(4.5, 6.5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Storage to Query
    ax.annotate("", xy=(4.5, 4.5), xytext=(5, 5.5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Query to Generation
    ax.annotate("", xy=(5, 3.5), xytext=(4.5, 3.5),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # ACL connections
    ax.annotate("", xy=(10, 5.5), xytext=(9, 6.5),
               arrowprops=dict(arrowstyle='<->', color='#7b1fa2', lw=1.5, ls='--'))
    ax.annotate("", xy=(10, 4), xytext=(9, 3.5),
               arrowprops=dict(arrowstyle='<->', color='#7b1fa2', lw=1.5, ls='--'))

    # User input/output
    ax.text(0.5, 3, 'User\nQuery', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#e0e0e0', edgecolor='gray'))
    ax.text(9, 3, 'Response', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#e0e0e0', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'enterprise_rag_pipeline.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_chunking_comparison():
    """Create chunking strategies comparison."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    strategies = [
        ('Fixed Size', ['Chunk 1\n(500 tokens)', 'Chunk 2\n(500 tokens)', 'Chunk 3\n(500 tokens)'],
         '#e3f2fd'),
        ('Semantic', ['Introduction\n(variable)', 'Main Topic\n(variable)', 'Conclusion\n(variable)'],
         '#e8f5e9'),
        ('Recursive', ['Chapter 1\n├─ Section 1\n└─ Section 2', 'Chapter 2\n├─ Section 1', ''],
         '#fff3e0'),
        ('Document\nAware', ['Text Block', 'Table\n(preserved)', 'Code Block\n(preserved)'],
         '#fce4ec')
    ]

    for ax, (name, chunks, color) in zip(axes, strategies):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 5)
        ax.axis('off')
        ax.set_title(name, fontsize=11, fontweight='bold')

        for i, chunk in enumerate(chunks):
            if chunk:
                rect = FancyBboxPatch((0.5, 4 - i * 1.3), 3, 1, boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor='gray', linewidth=1)
                ax.add_patch(rect)
                ax.text(2, 4.5 - i * 1.3, chunk, ha='center', va='center', fontsize=7)

    plt.suptitle('Chunking Strategies Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'chunking_strategies.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_hybrid_search():
    """Create hybrid search architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Hybrid Search Architecture', fontsize=14, fontweight='bold', pad=20)

    # Query input
    ax.text(1, 6, 'User Query', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#e0e0e0', edgecolor='gray', linewidth=2))

    # Search methods
    methods = [
        ('Vector Search\n(Semantic)', 3, 4.5, '#e3f2fd'),
        ('Keyword Search\n(BM25)', 6, 4.5, '#e8f5e9'),
        ('Structured\n(Filters)', 9, 4.5, '#fff3e0')
    ]

    for name, x, y, color in methods:
        rect = FancyBboxPatch((x-1, y-0.5), 2, 1.2, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='gray', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=9)
        ax.annotate("", xy=(x, y-0.5), xytext=(1, 5.5),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Fusion
    fusion_box = FancyBboxPatch((4.5, 2), 3, 1, boxstyle="round,pad=0.1",
                                 facecolor='#fce4ec', edgecolor='#c2185b', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(6, 2.5, 'Reciprocal Rank\nFusion', ha='center', va='center', fontsize=10, fontweight='bold')

    for x in [3, 6, 9]:
        ax.annotate("", xy=(6, 3), xytext=(x, 4),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Reranking
    rerank_box = FancyBboxPatch((4.5, 0.5), 3, 1, boxstyle="round,pad=0.1",
                                 facecolor='#f3e5f5', edgecolor='#7b1fa2', linewidth=2)
    ax.add_patch(rerank_box)
    ax.text(6, 1, 'Cross-Encoder\nReranking', ha='center', va='center', fontsize=10, fontweight='bold')

    ax.annotate("", xy=(6, 1.5), xytext=(6, 2),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Output
    ax.text(11, 1, 'Top-K\nResults', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#c8e6c9', edgecolor='#388e3c', linewidth=2))
    ax.annotate("", xy=(9.5, 1), xytext=(7.5, 1),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_search.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


def create_access_control_flow():
    """Create access control flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Permission model comparison
    models = ['RBAC', 'ABAC', 'ACL', 'Hierarchical']
    complexity = [2, 4, 5, 3]
    flexibility = [2, 5, 4, 3]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, complexity, width, label='Complexity', color='#ef5350')
    bars2 = ax.bar(x + width/2, flexibility, width, label='Flexibility', color='#42a5f5')

    ax.set_ylabel('Score (1-5)', fontsize=11)
    ax.set_title('Access Control Models: Complexity vs Flexibility', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'access_control_models.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    print("Generating enterprise RAG visualizations...")
    create_rag_pipeline()
    create_chunking_comparison()
    create_hybrid_search()
    create_access_control_flow()
    print(f"Visualizations saved to {OUTPUT_DIR}")
