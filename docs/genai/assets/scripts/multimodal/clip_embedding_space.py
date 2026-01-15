#!/usr/bin/env python3
"""
CLIP Cross-Modal Embedding Space Visualization

Generates visualizations showing:
1. How CLIP aligns image and text embeddings in a shared space
2. The contrastive learning objective
3. Cross-modal retrieval concept

Output: ../images/multimodal/clip_embedding_space.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
import numpy as np
import os

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'multimodal')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_clip_embedding_visualization():
    """Create a comprehensive CLIP embedding space visualization."""
    fig = plt.figure(figsize=(16, 14))

    # Color scheme
    colors = {
        'image': '#1976D2',
        'text': '#D32F2F',
        'match': '#4CAF50',
        'space': '#F5F5F5',
        'border': '#37474F'
    }

    # === Panel 1: CLIP Architecture Overview ===
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    ax1.set_title('CLIP Dual Encoder Architecture', fontsize=12, fontweight='bold', pad=10)

    # Image encoder
    img_enc = FancyBboxPatch((0.5, 4), 3, 3, boxstyle="round,pad=0.1",
                              facecolor='#BBDEFB', edgecolor=colors['border'], linewidth=2)
    ax1.add_patch(img_enc)
    ax1.text(2, 6.5, 'Image Encoder', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(2, 6.0, '(ViT-L/14)', ha='center', va='center', fontsize=9, style='italic')

    # Image encoder components
    components = ['Patch Embed', 'Transformer', 'Projection']
    for i, comp in enumerate(components):
        box = FancyBboxPatch((0.8, 5.2 - i*0.6), 2.4, 0.5, boxstyle="round,pad=0.03",
                              facecolor='white', edgecolor=colors['border'], linewidth=1)
        ax1.add_patch(box)
        ax1.text(2, 5.45 - i*0.6, comp, ha='center', va='center', fontsize=8)

    # Text encoder
    txt_enc = FancyBboxPatch((6.5, 4), 3, 3, boxstyle="round,pad=0.1",
                              facecolor='#FFCDD2', edgecolor=colors['border'], linewidth=2)
    ax1.add_patch(txt_enc)
    ax1.text(8, 6.5, 'Text Encoder', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(8, 6.0, '(Transformer)', ha='center', va='center', fontsize=9, style='italic')

    # Text encoder components
    components = ['Token Embed', 'Transformer', 'Projection']
    for i, comp in enumerate(components):
        box = FancyBboxPatch((6.8, 5.2 - i*0.6), 2.4, 0.5, boxstyle="round,pad=0.03",
                              facecolor='white', edgecolor=colors['border'], linewidth=1)
        ax1.add_patch(box)
        ax1.text(8, 5.45 - i*0.6, comp, ha='center', va='center', fontsize=8)

    # Inputs
    img_input = FancyBboxPatch((1, 7.2), 2, 0.6, boxstyle="round,pad=0.05",
                                facecolor='#E3F2FD', edgecolor=colors['border'], linewidth=1)
    ax1.add_patch(img_input)
    ax1.text(2, 7.5, 'Image', ha='center', va='center', fontsize=9)

    txt_input = FancyBboxPatch((7, 7.2), 2, 0.6, boxstyle="round,pad=0.05",
                                facecolor='#FFEBEE', edgecolor=colors['border'], linewidth=1)
    ax1.add_patch(txt_input)
    ax1.text(8, 7.5, 'Text', ha='center', va='center', fontsize=9)

    # Arrows to encoders
    ax1.annotate('', xy=(2, 7.0), xytext=(2, 7.2),
                 arrowprops=dict(arrowstyle='->', color=colors['border'], lw=1.5))
    ax1.annotate('', xy=(8, 7.0), xytext=(8, 7.2),
                 arrowprops=dict(arrowstyle='->', color=colors['border'], lw=1.5))

    # Shared embedding space
    space_ellipse = Ellipse((5, 1.5), 6, 2, facecolor=colors['space'],
                             edgecolor=colors['border'], linewidth=2, linestyle='--')
    ax1.add_patch(space_ellipse)
    ax1.text(5, 1.5, 'Shared Embedding Space\n(768-dim)', ha='center', va='center', fontsize=10)

    # Arrows to embedding space
    ax1.annotate('', xy=(3.5, 2.2), xytext=(2, 4.0),
                 arrowprops=dict(arrowstyle='->', color=colors['image'], lw=2))
    ax1.annotate('', xy=(6.5, 2.2), xytext=(8, 4.0),
                 arrowprops=dict(arrowstyle='->', color=colors['text'], lw=2))

    # Embedding labels
    ax1.text(2.5, 3.0, 'Image\nEmbed', ha='center', va='center', fontsize=8, color=colors['image'])
    ax1.text(7.5, 3.0, 'Text\nEmbed', ha='center', va='center', fontsize=8, color=colors['text'])

    # === Panel 2: Contrastive Learning Matrix ===
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Contrastive Learning: Similarity Matrix', fontsize=12, fontweight='bold', pad=10)

    # Draw similarity matrix
    matrix_size = 5
    cell_size = 1.2
    start_x, start_y = 2, 2

    # Matrix cells
    for i in range(matrix_size):
        for j in range(matrix_size):
            if i == j:
                color = colors['match']
                alpha = 0.8
            else:
                color = '#FFEBEE'
                alpha = 0.4

            rect = plt.Rectangle(
                (start_x + j * cell_size, start_y + (matrix_size - 1 - i) * cell_size),
                cell_size, cell_size,
                facecolor=color, edgecolor=colors['border'], linewidth=1, alpha=alpha
            )
            ax2.add_patch(rect)

            # Add values
            if i == j:
                val = f'{0.8 + np.random.rand() * 0.15:.2f}'
            else:
                val = f'{np.random.rand() * 0.3:.2f}'
            ax2.text(
                start_x + j * cell_size + cell_size/2,
                start_y + (matrix_size - 1 - i) * cell_size + cell_size/2,
                val, ha='center', va='center', fontsize=8
            )

    # Labels
    ax2.text(start_x + matrix_size * cell_size / 2, start_y - 0.5,
             'Text Embeddings (T)', ha='center', va='center', fontsize=10)
    ax2.text(start_x - 0.5, start_y + matrix_size * cell_size / 2,
             'Image\nEmbeddings\n(I)', ha='center', va='center', fontsize=10, rotation=0)

    # Sample labels
    for i in range(matrix_size):
        ax2.text(start_x + i * cell_size + cell_size/2, start_y + matrix_size * cell_size + 0.2,
                 f'T{i+1}', ha='center', va='center', fontsize=8)
        ax2.text(start_x - 0.3, start_y + (matrix_size - 1 - i) * cell_size + cell_size/2,
                 f'I{i+1}', ha='center', va='center', fontsize=8)

    # Loss formula
    ax2.text(5, 9, r'$\mathcal{L} = \frac{1}{2}(CE(I \cdot T^T / \tau, y) + CE(T \cdot I^T / \tau, y))$',
             ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='none'))

    # Legend
    ax2.text(5, 0.5, 'Diagonal = matching pairs (maximize)\nOff-diagonal = non-matching (minimize)',
             ha='center', va='center', fontsize=9, style='italic')

    # === Panel 3: Embedding Space Visualization ===
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-3, 3)
    ax3.set_title('Aligned Embedding Space (t-SNE view)', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Dimension 1', fontsize=10)
    ax3.set_ylabel('Dimension 2', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Generate clustered points
    np.random.seed(42)

    # Clusters representing different concepts
    concepts = [
        ('cat', (1.5, 1.5)),
        ('dog', (-1.5, 1.5)),
        ('car', (1.5, -1.5)),
        ('tree', (-1.5, -1.5)),
    ]

    for concept, (cx, cy) in concepts:
        # Image embeddings (circles)
        img_x = cx + np.random.randn(3) * 0.2
        img_y = cy + np.random.randn(3) * 0.2
        ax3.scatter(img_x, img_y, c=colors['image'], s=100, marker='o',
                   label='Image' if concept == 'cat' else '', alpha=0.8, edgecolors='white', linewidth=1)

        # Text embeddings (squares)
        txt_x = cx + np.random.randn(3) * 0.2
        txt_y = cy + np.random.randn(3) * 0.2
        ax3.scatter(txt_x, txt_y, c=colors['text'], s=100, marker='s',
                   label='Text' if concept == 'cat' else '', alpha=0.8, edgecolors='white', linewidth=1)

        # Label the cluster
        ax3.text(cx, cy + 0.7, f'"{concept}"', ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

        # Draw cluster boundary
        circle = plt.Circle((cx, cy), 0.6, fill=False, linestyle='--',
                            edgecolor='gray', linewidth=1, alpha=0.5)
        ax3.add_patch(circle)

    ax3.legend(loc='upper left', fontsize=9)

    # === Panel 4: Cross-Modal Retrieval ===
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 8)
    ax4.axis('off')
    ax4.set_title('Cross-Modal Retrieval', fontsize=12, fontweight='bold', pad=10)

    # Text-to-Image retrieval
    ax4.text(5, 7.5, 'Text Query -> Image Results', ha='center', va='center',
             fontsize=11, fontweight='bold')

    # Query
    query_box = FancyBboxPatch((0.5, 5.5), 3, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#FFEBEE', edgecolor=colors['border'], linewidth=2)
    ax4.add_patch(query_box)
    ax4.text(2, 6.7, 'Text Query', ha='center', va='center', fontsize=9, fontweight='bold')
    ax4.text(2, 6.1, '"a photo of', ha='center', va='center', fontsize=8)
    ax4.text(2, 5.7, 'a cute cat"', ha='center', va='center', fontsize=8)

    # Arrow
    ax4.annotate('', xy=(4.5, 6.25), xytext=(3.5, 6.25),
                 arrowprops=dict(arrowstyle='->', color=colors['border'], lw=2))
    ax4.text(4, 6.7, 'encode\n& search', ha='center', va='center', fontsize=8)

    # Results
    for i, score in enumerate([0.92, 0.87, 0.81]):
        box = FancyBboxPatch((5 + i*1.5, 5.5), 1.3, 1.5, boxstyle="round,pad=0.05",
                              facecolor='#BBDEFB', edgecolor=colors['border'], linewidth=1)
        ax4.add_patch(box)
        ax4.text(5.65 + i*1.5, 6.6, f'Image {i+1}', ha='center', va='center', fontsize=8)
        ax4.text(5.65 + i*1.5, 5.8, f'{score:.2f}', ha='center', va='center', fontsize=8,
                fontweight='bold', color=colors['match'])

    # Image-to-Text retrieval
    ax4.text(5, 4.5, 'Image Query -> Text Results', ha='center', va='center',
             fontsize=11, fontweight='bold')

    # Query
    query_box2 = FancyBboxPatch((0.5, 2.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#BBDEFB', edgecolor=colors['border'], linewidth=2)
    ax4.add_patch(query_box2)
    ax4.text(2, 3.7, 'Image Query', ha='center', va='center', fontsize=9, fontweight='bold')
    ax4.text(2, 3.1, '[cat photo]', ha='center', va='center', fontsize=8, style='italic')

    # Arrow
    ax4.annotate('', xy=(4.5, 3.25), xytext=(3.5, 3.25),
                 arrowprops=dict(arrowstyle='->', color=colors['border'], lw=2))
    ax4.text(4, 3.7, 'encode\n& search', ha='center', va='center', fontsize=8)

    # Results
    texts = ['"cute cat"', '"kitten"', '"feline"']
    scores = [0.89, 0.84, 0.79]
    for i, (txt, score) in enumerate(zip(texts, scores)):
        box = FancyBboxPatch((5 + i*1.5, 2.5), 1.3, 1.5, boxstyle="round,pad=0.05",
                              facecolor='#FFCDD2', edgecolor=colors['border'], linewidth=1)
        ax4.add_patch(box)
        ax4.text(5.65 + i*1.5, 3.6, txt, ha='center', va='center', fontsize=7)
        ax4.text(5.65 + i*1.5, 2.8, f'{score:.2f}', ha='center', va='center', fontsize=8,
                fontweight='bold', color=colors['match'])

    # Key insight
    ax4.text(5, 1.0, 'Key: Same embedding space enables bidirectional search!',
             ha='center', va='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='none'))

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'clip_embedding_space.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved CLIP embedding space diagram to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_clip_embedding_visualization()
