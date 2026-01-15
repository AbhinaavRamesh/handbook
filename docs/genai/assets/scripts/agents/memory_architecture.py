"""
Memory Architecture Visualization

Creates diagrams showing different memory types and architectures
for LLM agents.

Output: /docs/genai/assets/images/agents/memory_architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

# Output directory
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "images", "agents"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_memory_box(ax, x, y, width, height, label, sublabel=None,
                    color='#e3f2fd', border='#1976d2'):
    """Draw a memory type box."""
    rect = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color,
        edgecolor=border,
        linewidth=2
    )
    ax.add_patch(rect)

    if sublabel:
        ax.text(x, y + 0.15, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='#333')
        ax.text(x, y - 0.15, sublabel, ha='center', va='center',
                fontsize=8, color='#666', style='italic')
    else:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='#333')


def draw_arrow(ax, start, end, color='#666', label=None, curved=0):
    """Draw an arrow with optional label."""
    if curved:
        style = FancyArrowPatch(
            start, end,
            arrowstyle='-|>',
            color=color,
            lw=1.5,
            mutation_scale=15,
            connectionstyle=f"arc3,rad={curved}"
        )
        ax.add_patch(style)
    else:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5,
                                    mutation_scale=15))

    if label:
        mid = ((start[0] + end[0])/2, (start[1] + end[1])/2)
        ax.text(mid[0], mid[1] + 0.15, label, ha='center', fontsize=7, color='#666')


def create_memory_types_diagram():
    """Create diagram showing different memory types."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, "Agent Memory Architecture", fontsize=16,
            ha='center', fontweight='bold', color='#1a237e')

    # Agent/LLM box at center
    draw_memory_box(ax, 7, 5, 2.5, 1.5, "LLM Agent", "Processing",
                    color='#fff3e0', border='#f57c00')

    # Short-term Memory (Working Memory)
    draw_memory_box(ax, 2.5, 7.5, 3.5, 2, "Short-term Memory",
                    "Conversation Buffer",
                    color='#e3f2fd', border='#1976d2')

    # Details for short-term
    ax.text(2.5, 6.2, "• Current conversation\n• Recent context\n• Token-limited",
            ha='center', fontsize=8, color='#666')

    # Long-term Memory section
    rect = FancyBboxPatch(
        (8, 6.5), 5, 3,
        boxstyle="round,pad=0.05",
        facecolor='#f5f5f5',
        edgecolor='#9e9e9e',
        linewidth=1,
        linestyle='--'
    )
    ax.add_patch(rect)
    ax.text(10.5, 9.2, "Long-term Memory", fontsize=11, fontweight='bold',
            ha='center', color='#333')

    # Episodic Memory
    draw_memory_box(ax, 9.3, 8, 2, 1, "Episodic",
                    color='#e8f5e9', border='#388e3c')
    ax.text(9.3, 7.2, "Past experiences\nLessons learned", ha='center',
            fontsize=7, color='#666')

    # Semantic Memory
    draw_memory_box(ax, 11.7, 8, 2, 1, "Semantic",
                    color='#fce4ec', border='#c2185b')
    ax.text(11.7, 7.2, "Facts, knowledge\nUser preferences", ha='center',
            fontsize=7, color='#666')

    # Vector Store
    draw_memory_box(ax, 10.5, 5, 2.5, 1.2, "Vector Store",
                    "Embeddings + Index",
                    color='#e8eaf6', border='#5c6bc0')

    # Tools/External Memory
    draw_memory_box(ax, 2.5, 2.5, 3, 1.5, "External Memory",
                    "Databases, Files, APIs",
                    color='#fff9c4', border='#ffc107')

    # State Manager
    draw_memory_box(ax, 7, 2, 2.5, 1.2, "State Manager",
                    "Task/Session State",
                    color='#e0f2f1', border='#00897b')

    # Draw connections
    # Short-term <-> Agent
    draw_arrow(ax, (4.25, 7), (5.75, 5.5), label="context")
    draw_arrow(ax, (5.75, 5.2), (4.25, 6.5), label="update")

    # Long-term -> Agent (retrieval)
    draw_arrow(ax, (9, 5), (8.25, 5), label="retrieve")

    # Agent -> Long-term (store)
    draw_arrow(ax, (8.25, 4.7), (9.2, 4.7), label="store")

    # Vector Store connections
    draw_arrow(ax, (10.5, 5.6), (10.5, 7.2))
    draw_arrow(ax, (10.5, 7.5), (10.5, 5.8), curved=0.3)

    # External Memory <-> Agent
    draw_arrow(ax, (4, 3), (5.75, 4.5), label="query")
    draw_arrow(ax, (5.75, 4.2), (4, 2.7), label="results", curved=0.2)

    # State Manager <-> Agent
    draw_arrow(ax, (7, 2.6), (7, 4.25), label="state")

    # Legend
    legend_items = [
        ("Short-term", '#e3f2fd', "Session-scoped, fast access"),
        ("Episodic", '#e8f5e9', "Experience-based learning"),
        ("Semantic", '#fce4ec', "Factual knowledge"),
        ("External", '#fff9c4', "Persistent storage")
    ]

    for i, (label, color, desc) in enumerate(legend_items):
        y = 1.5 - i * 0.35
        rect = patches.Rectangle((0.5, y - 0.1), 0.4, 0.2,
                                  facecolor=color, edgecolor='#666')
        ax.add_patch(rect)
        ax.text(1.0, y, f"{label}: {desc}", fontsize=7, va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'memory_architecture.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_memory_flow_diagram():
    """Create diagram showing memory read/write flow."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6, 7.5, "Memory Read/Write Flow", fontsize=14,
            ha='center', fontweight='bold', color='#1a237e')

    # User Input
    draw_memory_box(ax, 1.5, 6, 2, 1, "User Input",
                    color='#e3f2fd', border='#1976d2')

    # Memory Retrieval
    draw_memory_box(ax, 4.5, 6, 2.5, 1, "Memory\nRetrieval",
                    color='#e8f5e9', border='#388e3c')

    # Context Assembly
    draw_memory_box(ax, 7.5, 6, 2.5, 1, "Context\nAssembly",
                    color='#fff3e0', border='#f57c00')

    # LLM Processing
    draw_memory_box(ax, 10.5, 6, 2, 1, "LLM\nProcessing",
                    color='#fce4ec', border='#c2185b')

    # Memory Store (below)
    draw_memory_box(ax, 4.5, 3, 3, 2, "Memory Store",
                    "Vector DB + Metadata",
                    color='#e8eaf6', border='#5c6bc0')

    # Response
    draw_memory_box(ax, 10.5, 3, 2, 1, "Response",
                    color='#e3f2fd', border='#1976d2')

    # Memory Update
    draw_memory_box(ax, 7.5, 1.5, 2.5, 1, "Memory\nUpdate",
                    color='#e8f5e9', border='#388e3c')

    # Draw flow arrows
    draw_arrow(ax, (2.5, 6), (3.25, 6))
    draw_arrow(ax, (5.75, 6), (6.25, 6))
    draw_arrow(ax, (8.75, 6), (9.5, 6))
    draw_arrow(ax, (10.5, 5.5), (10.5, 3.5))

    # Memory retrieval from store
    draw_arrow(ax, (4.5, 4), (4.5, 5.5), label="embed & search")

    # Memory update to store
    draw_arrow(ax, (7.5, 2), (6, 3), label="store new")
    draw_arrow(ax, (10.5, 2.5), (8.75, 1.7), label="extract important", curved=-0.3)

    # Add process descriptions
    descriptions = [
        (1.5, 5.2, "Query"),
        (4.5, 5.2, "Semantic\nsearch"),
        (7.5, 5.2, "Combine\ncontext"),
        (10.5, 5.2, "Generate"),
        (4.5, 1.8, "Embeddings\n+ Metadata"),
    ]

    for x, y, text in descriptions:
        ax.text(x, y, text, ha='center', fontsize=7, color='#666')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'memory_flow.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_vector_memory_detail():
    """Create detailed diagram of vector memory operations."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6, 7.5, "Vector Memory: Semantic Retrieval", fontsize=14,
            ha='center', fontweight='bold', color='#1a237e')

    # Write Path (left side)
    ax.text(3, 7, "Write Path", fontsize=12, fontweight='bold',
            ha='center', color='#388e3c')

    write_boxes = [
        (3, 6, "Text Content"),
        (3, 4.8, "Embedding Model"),
        (3, 3.6, "Vector [0.1, 0.3, ...]"),
        (3, 2.4, "Vector Store"),
    ]

    for i, (x, y, label) in enumerate(write_boxes):
        color = '#e8f5e9' if i % 2 == 0 else '#c8e6c9'
        draw_memory_box(ax, x, y, 2.8, 0.8, label, color=color, border='#388e3c')
        if i < len(write_boxes) - 1:
            draw_arrow(ax, (x, y - 0.4), (x, write_boxes[i+1][1] + 0.4))

    # Read Path (right side)
    ax.text(9, 7, "Read Path", fontsize=12, fontweight='bold',
            ha='center', color='#1976d2')

    read_boxes = [
        (9, 6, "Query"),
        (9, 4.8, "Embedding Model"),
        (9, 3.6, "Query Vector"),
        (9, 2.4, "Similarity Search"),
        (9, 1.2, "Top-K Results"),
    ]

    for i, (x, y, label) in enumerate(read_boxes):
        color = '#e3f2fd' if i % 2 == 0 else '#bbdefb'
        draw_memory_box(ax, x, y, 2.8, 0.8, label, color=color, border='#1976d2')
        if i < len(read_boxes) - 1:
            draw_arrow(ax, (x, y - 0.4), (x, read_boxes[i+1][1] + 0.4))

    # Connection from store to search
    draw_arrow(ax, (4.4, 2.4), (7.6, 2.4), label="index lookup")

    # Similarity illustration
    ax.text(6, 1.5, "cosine_similarity(query_vec, doc_vec)", fontsize=8,
            ha='center', color='#666', family='monospace')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'vector_memory_detail.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    print("Creating memory architecture diagrams...")
    create_memory_types_diagram()
    print(f"Memory types diagram saved to: {OUTPUT_DIR}/memory_architecture.png")
    create_memory_flow_diagram()
    print(f"Memory flow diagram saved to: {OUTPUT_DIR}/memory_flow.png")
    create_vector_memory_detail()
    print(f"Vector memory diagram saved to: {OUTPUT_DIR}/vector_memory_detail.png")
    print("Done!")
