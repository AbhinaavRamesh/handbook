#!/usr/bin/env python3
"""
Vision-Language Model (VLM) Architecture Diagram

Generates a visualization showing the architecture of modern VLMs like LLaVA,
including the vision encoder, projection layer, and language model components.

Output: ../images/multimodal/vlm_architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'multimodal')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_vlm_architecture():
    """Create a detailed VLM architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Color scheme
    colors = {
        'input': '#E3F2FD',
        'vision': '#BBDEFB',
        'projection': '#F3E5F5',
        'llm': '#E8F5E9',
        'output': '#FFF3E0',
        'attention': '#FFECB3',
        'border': '#37474F'
    }

    # Title
    ax.text(8, 9.5, 'Vision-Language Model Architecture (LLaVA-style)',
            ha='center', va='center', fontsize=16, fontweight='bold')

    # === Input Section ===
    # Image input
    image_box = FancyBboxPatch((0.5, 6), 2, 2.5, boxstyle="round,pad=0.1",
                                facecolor=colors['input'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(image_box)
    ax.text(1.5, 7.8, 'Image Input', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.5, 7.2, '224x224x3', ha='center', va='center', fontsize=9)

    # Draw image grid
    for i in range(3):
        for j in range(3):
            rect = plt.Rectangle((0.7 + i*0.5, 6.2 + j*0.3), 0.4, 0.25,
                                  facecolor='#90CAF9', edgecolor='white', linewidth=1)
            ax.add_patch(rect)

    # Text input
    text_box = FancyBboxPatch((0.5, 2.5), 2, 2.5, boxstyle="round,pad=0.1",
                               facecolor=colors['input'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(text_box)
    ax.text(1.5, 4.5, 'Text Input', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.5, 3.9, '"What is in', ha='center', va='center', fontsize=9)
    ax.text(1.5, 3.5, 'this image?"', ha='center', va='center', fontsize=9)

    # === Vision Encoder Section ===
    vision_box = FancyBboxPatch((3.5, 5.5), 3, 3.5, boxstyle="round,pad=0.1",
                                 facecolor=colors['vision'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(vision_box)
    ax.text(5, 8.5, 'Vision Encoder', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, 8.0, '(CLIP ViT-L/14)', ha='center', va='center', fontsize=9, style='italic')

    # Patch embedding
    patch_box = FancyBboxPatch((3.8, 7.0), 2.4, 0.7, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor=colors['border'], linewidth=1)
    ax.add_patch(patch_box)
    ax.text(5, 7.35, 'Patch Embedding', ha='center', va='center', fontsize=9)

    # Transformer blocks
    for i, y in enumerate([6.6, 6.1, 5.6]):
        block = FancyBboxPatch((3.8, y), 2.4, 0.4, boxstyle="round,pad=0.03",
                                facecolor='white', edgecolor=colors['border'], linewidth=1)
        ax.add_patch(block)
        if i < 2:
            ax.text(5, y + 0.2, f'Transformer Block', ha='center', va='center', fontsize=8)
        else:
            ax.text(5, y + 0.2, '... x N layers', ha='center', va='center', fontsize=8)

    # Output: visual features
    ax.text(5, 5.2, '[257, 1024]', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#E1BEE7', edgecolor='none'))

    # === Projection Layer ===
    proj_box = FancyBboxPatch((7, 5.5), 2.5, 3.5, boxstyle="round,pad=0.1",
                               facecolor=colors['projection'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(proj_box)
    ax.text(8.25, 8.5, 'Visual Projector', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(8.25, 8.0, '(MLP)', ha='center', va='center', fontsize=9, style='italic')

    # MLP layers
    for i, (y, label) in enumerate([(7.2, 'Linear (1024 -> 4096)'),
                                     (6.6, 'GELU'),
                                     (6.0, 'Linear (4096 -> 4096)')]):
        mlp_block = FancyBboxPatch((7.3, y), 1.9, 0.5, boxstyle="round,pad=0.03",
                                    facecolor='white', edgecolor=colors['border'], linewidth=1)
        ax.add_patch(mlp_block)
        ax.text(8.25, y + 0.25, label, ha='center', va='center', fontsize=7)

    # Output: visual tokens
    ax.text(8.25, 5.2, '[257, 4096]', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#E1BEE7', edgecolor='none'))

    # === Tokenizer ===
    tok_box = FancyBboxPatch((3.5, 2.5), 3, 2, boxstyle="round,pad=0.1",
                              facecolor=colors['input'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(tok_box)
    ax.text(5, 4.0, 'Tokenizer', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 3.4, 'BPE / SentencePiece', ha='center', va='center', fontsize=9)
    ax.text(5, 2.8, '[seq_len, vocab]', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='none'))

    # === LLM Section ===
    llm_box = FancyBboxPatch((10, 2), 4.5, 7, boxstyle="round,pad=0.1",
                              facecolor=colors['llm'], edgecolor=colors['border'], linewidth=2)
    ax.add_patch(llm_box)
    ax.text(12.25, 8.5, 'Large Language Model', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(12.25, 8.0, '(LLaMA-2 7B)', ha='center', va='center', fontsize=9, style='italic')

    # Token embedding
    emb_box = FancyBboxPatch((10.3, 7.0), 3.9, 0.7, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor=colors['border'], linewidth=1)
    ax.add_patch(emb_box)
    ax.text(12.25, 7.35, 'Token Embedding Layer', ha='center', va='center', fontsize=9)

    # Concatenation point
    concat_circle = plt.Circle((12.25, 6.5), 0.25, facecolor=colors['attention'],
                                edgecolor=colors['border'], linewidth=2)
    ax.add_patch(concat_circle)
    ax.text(12.25, 6.5, '+', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(13.0, 6.5, 'Concat', ha='left', va='center', fontsize=8)

    # Transformer decoder blocks
    for i, y in enumerate([5.8, 5.2, 4.6, 4.0]):
        block = FancyBboxPatch((10.3, y), 3.9, 0.5, boxstyle="round,pad=0.03",
                                facecolor='white', edgecolor=colors['border'], linewidth=1)
        ax.add_patch(block)
        if i < 3:
            ax.text(12.25, y + 0.25, 'Self-Attention + FFN', ha='center', va='center', fontsize=8)
        else:
            ax.text(12.25, y + 0.25, '... x 32 layers', ha='center', va='center', fontsize=8)

    # LM Head
    lm_head = FancyBboxPatch((10.3, 3.2), 3.9, 0.6, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor=colors['border'], linewidth=1)
    ax.add_patch(lm_head)
    ax.text(12.25, 3.5, 'LM Head (Linear)', ha='center', va='center', fontsize=9)

    # Output
    output_box = FancyBboxPatch((10.3, 2.3), 3.9, 0.7, boxstyle="round,pad=0.05",
                                 facecolor=colors['output'], edgecolor=colors['border'], linewidth=1)
    ax.add_patch(output_box)
    ax.text(12.25, 2.65, 'Output: "A cat sitting..."', ha='center', va='center', fontsize=9)

    # === Arrows ===
    arrow_style = dict(arrowstyle='->', color=colors['border'], lw=2,
                       connectionstyle='arc3,rad=0')

    # Image to Vision Encoder
    ax.annotate('', xy=(3.5, 7.25), xytext=(2.5, 7.25),
                arrowprops=arrow_style)

    # Vision Encoder to Projector
    ax.annotate('', xy=(7, 7.25), xytext=(6.5, 7.25),
                arrowprops=arrow_style)

    # Projector to LLM (visual tokens)
    ax.annotate('', xy=(10, 6.5), xytext=(9.5, 6.5),
                arrowprops=arrow_style)

    # Text to Tokenizer
    ax.annotate('', xy=(3.5, 3.5), xytext=(2.5, 3.5),
                arrowprops=arrow_style)

    # Tokenizer to LLM (through embedding)
    ax.annotate('', xy=(10, 3.5), xytext=(6.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=colors['border'], lw=2,
                               connectionstyle='arc3,rad=-0.2'))

    # === Legend ===
    legend_y = 1.2
    legend_items = [
        ('Vision Components', colors['vision']),
        ('Projection Layer', colors['projection']),
        ('Language Model', colors['llm']),
        ('Input/Output', colors['input']),
    ]

    for i, (label, color) in enumerate(legend_items):
        x = 2 + i * 3.5
        rect = plt.Rectangle((x, legend_y), 0.4, 0.3, facecolor=color, edgecolor=colors['border'])
        ax.add_patch(rect)
        ax.text(x + 0.5, legend_y + 0.15, label, ha='left', va='center', fontsize=9)

    # Add dimension annotations
    ax.text(8, 0.5, 'Typical dimensions: Image patches = 256+1 (CLS), Vision dim = 1024, LLM dim = 4096',
            ha='center', va='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'vlm_architecture.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved VLM architecture diagram to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_vlm_architecture()
