#!/usr/bin/env python3
"""
Multimodal RAG Pipeline Visualization

Generates a visualization showing the complete multimodal RAG pipeline
including document processing, indexing, retrieval, and generation.

Output: ../images/multimodal/multimodal_rag_flow.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'multimodal')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_multimodal_rag_diagram():
    """Create a comprehensive multimodal RAG pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Color scheme
    colors = {
        'document': '#E3F2FD',
        'processing': '#BBDEFB',
        'embedding': '#F3E5F5',
        'storage': '#E8F5E9',
        'query': '#FFF3E0',
        'retrieval': '#FFECB3',
        'generation': '#FCE4EC',
        'border': '#37474F',
        'arrow': '#546E7A'
    }

    # Title
    ax.text(9, 11.5, 'Multimodal RAG Pipeline', fontsize=18, fontweight='bold', ha='center')

    # === SECTION 1: Document Ingestion (Left) ===
    section1_x = 1

    # Section label
    ax.text(section1_x + 1.5, 10.8, '1. Document Ingestion', fontsize=12, fontweight='bold', ha='center')

    # Document sources
    doc_types = [('PDF', 9.5), ('Web', 8.5), ('Images', 7.5)]
    for doc_type, y in doc_types:
        box = FancyBboxPatch((section1_x, y), 1.2, 0.7, boxstyle="round,pad=0.05",
                              facecolor=colors['document'], edgecolor=colors['border'], linewidth=1.5)
        ax.add_patch(box)
        ax.text(section1_x + 0.6, y + 0.35, doc_type, ha='center', va='center', fontsize=9)

    # Arrows from documents
    for _, y in doc_types:
        ax.annotate('', xy=(section1_x + 2.2, 8.5), xytext=(section1_x + 1.3, y + 0.35),
                    arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5,
                                   connectionstyle='arc3,rad=0.1'))

    # === SECTION 2: Content Extraction ===
    section2_x = 3.5

    # Extraction box
    extract_box = FancyBboxPatch((section2_x, 7.5), 2.5, 2.5, boxstyle="round,pad=0.1",
                                  facecolor=colors['processing'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(extract_box)
    ax.text(section2_x + 1.25, 9.7, 'Content\nExtraction', ha='center', va='center', fontsize=10, fontweight='bold')

    # Sub-components
    components = [('Text', 9.0), ('Images', 8.4), ('Tables', 7.8)]
    for comp, y in components:
        small_box = FancyBboxPatch((section2_x + 0.2, y), 2.1, 0.4, boxstyle="round,pad=0.03",
                                    facecolor='white', edgecolor=colors['border'], linewidth=1)
        ax.add_patch(small_box)
        ax.text(section2_x + 1.25, y + 0.2, comp, ha='center', va='center', fontsize=8)

    # === SECTION 3: Processing ===
    section3_x = 6.5

    ax.text(section3_x + 1.25, 10.8, '2. Processing', fontsize=12, fontweight='bold', ha='center')

    # Text processing
    text_proc = FancyBboxPatch((section3_x, 9.0), 2.5, 1.5, boxstyle="round,pad=0.1",
                                facecolor=colors['processing'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(text_proc)
    ax.text(section3_x + 1.25, 10.1, 'Text Processing', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(section3_x + 1.25, 9.6, 'Chunking', ha='center', va='center', fontsize=8)
    ax.text(section3_x + 1.25, 9.2, '+ Captions', ha='center', va='center', fontsize=8)

    # Image processing
    img_proc = FancyBboxPatch((section3_x, 7.0), 2.5, 1.5, boxstyle="round,pad=0.1",
                               facecolor=colors['processing'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(img_proc)
    ax.text(section3_x + 1.25, 8.1, 'Image Processing', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(section3_x + 1.25, 7.6, 'BLIP Captioning', ha='center', va='center', fontsize=8)
    ax.text(section3_x + 1.25, 7.2, '+ Resize', ha='center', va='center', fontsize=8)

    # Arrows
    ax.annotate('', xy=(section3_x, 9.75), xytext=(section2_x + 2.5, 8.75),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(section3_x, 7.75), xytext=(section2_x + 2.5, 8.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # === SECTION 4: Embedding ===
    section4_x = 9.5

    ax.text(section4_x + 1.25, 10.8, '3. Embedding', fontsize=12, fontweight='bold', ha='center')

    # Text encoder
    text_enc = FancyBboxPatch((section4_x, 9.0), 2.5, 1.5, boxstyle="round,pad=0.1",
                               facecolor=colors['embedding'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(text_enc)
    ax.text(section4_x + 1.25, 10.1, 'Text Encoder', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(section4_x + 1.25, 9.6, 'E5 / BGE', ha='center', va='center', fontsize=8)
    ax.text(section4_x + 1.25, 9.2, '[768-dim]', ha='center', va='center', fontsize=8, style='italic')

    # Image encoder
    img_enc = FancyBboxPatch((section4_x, 7.0), 2.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor=colors['embedding'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(img_enc)
    ax.text(section4_x + 1.25, 8.1, 'Image Encoder', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(section4_x + 1.25, 7.6, 'CLIP ViT-L', ha='center', va='center', fontsize=8)
    ax.text(section4_x + 1.25, 7.2, '[768-dim]', ha='center', va='center', fontsize=8, style='italic')

    # Arrows
    ax.annotate('', xy=(section4_x, 9.75), xytext=(section3_x + 2.5, 9.75),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(section4_x, 7.75), xytext=(section3_x + 2.5, 7.75),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # === SECTION 5: Vector Store ===
    section5_x = 13

    ax.text(section5_x + 1.5, 10.8, '4. Storage', fontsize=12, fontweight='bold', ha='center')

    # Vector store
    vs_box = FancyBboxPatch((section5_x, 7.0), 3, 3.5, boxstyle="round,pad=0.1",
                             facecolor=colors['storage'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(vs_box)
    ax.text(section5_x + 1.5, 10.0, 'Vector Store', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(section5_x + 1.5, 9.5, '(FAISS / Pinecone)', ha='center', va='center', fontsize=8, style='italic')

    # Index contents
    ax.text(section5_x + 0.3, 8.8, 'Text Index:', ha='left', va='center', fontsize=8, fontweight='bold')
    for i in range(3):
        small_rect = Rectangle((section5_x + 0.3 + i*0.6, 8.3), 0.5, 0.4,
                                facecolor='#C8E6C9', edgecolor=colors['border'], linewidth=1)
        ax.add_patch(small_rect)

    ax.text(section5_x + 0.3, 7.9, 'Image Index:', ha='left', va='center', fontsize=8, fontweight='bold')
    for i in range(3):
        small_rect = Rectangle((section5_x + 0.3 + i*0.6, 7.4), 0.5, 0.4,
                                facecolor='#F8BBD9', edgecolor=colors['border'], linewidth=1)
        ax.add_patch(small_rect)

    # Arrows to vector store
    ax.annotate('', xy=(section5_x, 9.25), xytext=(section4_x + 2.5, 9.25),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(section5_x, 8.0), xytext=(section4_x + 2.5, 7.75),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # === SECTION 6: Query Flow (Bottom) ===
    query_y = 4

    # Query input
    query_box = FancyBboxPatch((1, query_y - 0.5), 3, 1.5, boxstyle="round,pad=0.1",
                                facecolor=colors['query'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(query_box)
    ax.text(2.5, query_y + 0.7, 'User Query', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2.5, query_y + 0.2, '"Show me charts', ha='center', va='center', fontsize=8)
    ax.text(2.5, query_y - 0.1, 'about Q3 revenue"', ha='center', va='center', fontsize=8)

    # Query embedding
    query_enc = FancyBboxPatch((5, query_y - 0.3), 2.5, 1.1, boxstyle="round,pad=0.1",
                                facecolor=colors['embedding'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(query_enc)
    ax.text(6.25, query_y + 0.5, 'Query Encoder', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(6.25, query_y + 0.05, 'Text + CLIP', ha='center', va='center', fontsize=8)

    ax.annotate('', xy=(5, query_y + 0.25), xytext=(4, query_y + 0.25),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # Retrieval
    ret_box = FancyBboxPatch((8.5, query_y - 0.3), 2.5, 1.1, boxstyle="round,pad=0.1",
                              facecolor=colors['retrieval'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(ret_box)
    ax.text(9.75, query_y + 0.5, 'Hybrid Retrieval', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(9.75, query_y + 0.05, 'Top-K Results', ha='center', va='center', fontsize=8)

    ax.annotate('', xy=(8.5, query_y + 0.25), xytext=(7.5, query_y + 0.25),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # Arrow from vector store to retrieval
    ax.annotate('', xy=(9.75, query_y + 0.8), xytext=(14.5, 7.0),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5,
                               connectionstyle='arc3,rad=-0.3'))

    # Results
    results_box = FancyBboxPatch((8.5, query_y - 2.5), 2.5, 1.8, boxstyle="round,pad=0.1",
                                  facecolor='white', edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(results_box)
    ax.text(9.75, query_y - 1.0, 'Retrieved:', ha='center', va='center', fontsize=9, fontweight='bold')

    # Text result
    text_res = Rectangle((8.7, query_y - 1.8), 1.0, 0.5, facecolor='#C8E6C9',
                          edgecolor=colors['border'], linewidth=1)
    ax.add_patch(text_res)
    ax.text(9.2, query_y - 1.55, 'Text', ha='center', va='center', fontsize=7)

    # Image result
    img_res = Rectangle((10.0, query_y - 1.8), 0.8, 0.5, facecolor='#F8BBD9',
                         edgecolor=colors['border'], linewidth=1)
    ax.add_patch(img_res)
    ax.text(10.4, query_y - 1.55, 'Img', ha='center', va='center', fontsize=7)

    ax.annotate('', xy=(9.75, query_y - 2.5), xytext=(9.75, query_y - 0.3),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # LLM
    llm_box = FancyBboxPatch((12, query_y - 1.5), 3, 2.5, boxstyle="round,pad=0.1",
                              facecolor=colors['generation'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(llm_box)
    ax.text(13.5, query_y + 0.7, 'Multimodal LLM', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(13.5, query_y + 0.2, '(GPT-4V / Gemini)', ha='center', va='center', fontsize=8, style='italic')
    ax.text(13.5, query_y - 0.3, 'Context + Query', ha='center', va='center', fontsize=8)
    ax.text(13.5, query_y - 0.7, '= Response', ha='center', va='center', fontsize=8)

    ax.annotate('', xy=(12, query_y - 0.25), xytext=(11, query_y - 0.25),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # Original query to LLM
    ax.annotate('', xy=(12, query_y + 0.5), xytext=(4, query_y + 0.5),
                arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1.5, linestyle='dashed',
                               connectionstyle='arc3,rad=0.2'))
    ax.text(8, query_y + 1.3, 'Original Query', ha='center', va='center', fontsize=8, color='#1976D2')

    # Output
    output_box = FancyBboxPatch((15.5, query_y - 1.2), 2, 2, boxstyle="round,pad=0.1",
                                 facecolor=colors['query'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(output_box)
    ax.text(16.5, query_y + 0.5, 'Response', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(16.5, query_y, '"Q3 revenue', ha='center', va='center', fontsize=8)
    ax.text(16.5, query_y - 0.3, 'increased 15%', ha='center', va='center', fontsize=8)
    ax.text(16.5, query_y - 0.6, 'as shown in..."', ha='center', va='center', fontsize=8)

    ax.annotate('', xy=(15.5, query_y - 0.2), xytext=(15, query_y - 0.2),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    # === Legend ===
    legend_y = 0.8
    legend_items = [
        ('Document Input', colors['document']),
        ('Processing', colors['processing']),
        ('Embedding', colors['embedding']),
        ('Vector Storage', colors['storage']),
        ('Query/Output', colors['query']),
        ('Retrieval', colors['retrieval']),
        ('Generation', colors['generation']),
    ]

    for i, (label, color) in enumerate(legend_items):
        x = 1.5 + i * 2.3
        rect = Rectangle((x, legend_y), 0.4, 0.3, facecolor=color, edgecolor=colors['border'])
        ax.add_patch(rect)
        ax.text(x + 0.5, legend_y + 0.15, label, ha='left', va='center', fontsize=8)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'multimodal_rag_flow.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved multimodal RAG diagram to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_multimodal_rag_diagram()
