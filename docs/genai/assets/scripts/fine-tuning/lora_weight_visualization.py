"""
LoRA Weight Injection Visualization

This script creates a diagram showing how LoRA adapters inject
low-rank updates into transformer layers.

Output: lora_injection_diagram.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "images" / "fine-tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_matrix(ax, x, y, width, height, label, color='lightblue', alpha=0.7):
    """Draw a rectangle representing a matrix"""
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='black',
        alpha=alpha,
        linewidth=2
    )
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label,
            ha='center', va='center', fontsize=11, fontweight='bold')


def draw_arrow(ax, start, end, color='black', style='->', linewidth=2):
    """Draw an arrow between two points"""
    ax.annotate(
        '',
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=style, color=color, lw=linewidth)
    )


def create_lora_diagram():
    """Create the main LoRA injection diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 12)
    ax.axis('off')

    # Title
    ax.text(7, 11.5, 'LoRA: Low-Rank Adaptation',
            ha='center', fontsize=16, fontweight='bold')
    ax.text(7, 10.8, 'Injecting trainable low-rank matrices into frozen weights',
            ha='center', fontsize=11, style='italic', color='gray')

    # Input
    draw_matrix(ax, 0, 4.5, 1.5, 2, 'Input\nx', color='#e8f5e9')

    # Original weight matrix (frozen)
    draw_matrix(ax, 3, 3.5, 2.5, 4, 'W\n(frozen)\nd × d', color='#e0e0e0')

    # Add snowflake symbol for frozen
    ax.text(4.25, 8, '❄', fontsize=20, ha='center')

    # LoRA branch - Down projection A
    draw_matrix(ax, 3, 0, 2.5, 2, 'A\nr × d', color='#c8e6c9')
    ax.text(4.25, -0.8, 'Down-project', ha='center', fontsize=9, style='italic')

    # LoRA branch - Up projection B
    draw_matrix(ax, 7, 0, 2.5, 2, 'B\nd × r', color='#c8e6c9')
    ax.text(8.25, -0.8, 'Up-project', ha='center', fontsize=9, style='italic')

    # Scaling factor
    ax.text(6.25, 1, '×', fontsize=16, ha='center', va='center')
    draw_matrix(ax, 10.5, 0, 1.5, 2, 'α/r', color='#fff9c4')
    ax.text(11.25, -0.8, 'Scale', ha='center', fontsize=9, style='italic')

    # Addition circle
    circle = plt.Circle((11.25, 5.5), 0.5, fill=True, facecolor='white',
                        edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(11.25, 5.5, '+', fontsize=20, ha='center', va='center')

    # Output
    draw_matrix(ax, 13, 4.5, 1.5, 2, 'Output\ny', color='#e3f2fd')

    # Arrows for main path
    draw_arrow(ax, (1.5, 5.5), (3, 5.5))  # Input to W
    draw_arrow(ax, (5.5, 5.5), (10.75, 5.5))  # W to +

    # Arrows for LoRA path
    draw_arrow(ax, (1.5, 5.5), (2.5, 3), color='#2e7d32')  # Input to A
    draw_arrow(ax, (2.5, 1), (3, 1), color='#2e7d32')  # bend to A
    draw_arrow(ax, (5.5, 1), (7, 1), color='#2e7d32')  # A to B
    draw_arrow(ax, (9.5, 1), (10.5, 1), color='#2e7d32')  # B to scale
    draw_arrow(ax, (12, 1), (12.5, 4), color='#2e7d32')  # scale to +
    draw_arrow(ax, (12.5, 5), (11.75, 5.5), color='#2e7d32')  # to + circle

    # Output arrow
    draw_arrow(ax, (11.75, 5.5), (13, 5.5))  # + to output

    # Legend box
    legend_x, legend_y = 0, 8.5
    draw_matrix(ax, legend_x, legend_y, 3, 2.5, '', color='white', alpha=0.9)
    ax.text(legend_x + 1.5, legend_y + 2.1, 'Legend', ha='center', fontsize=10, fontweight='bold')

    # Legend items
    ax.add_patch(patches.Rectangle((legend_x + 0.2, legend_y + 1.3), 0.4, 0.4,
                                   facecolor='#e0e0e0', edgecolor='black'))
    ax.text(legend_x + 0.8, legend_y + 1.5, 'Frozen', fontsize=9, va='center')

    ax.add_patch(patches.Rectangle((legend_x + 0.2, legend_y + 0.7), 0.4, 0.4,
                                   facecolor='#c8e6c9', edgecolor='black'))
    ax.text(legend_x + 0.8, legend_y + 0.9, 'Trainable', fontsize=9, va='center')

    ax.add_patch(patches.Rectangle((legend_x + 1.6, legend_y + 1.3), 0.4, 0.4,
                                   facecolor='#fff9c4', edgecolor='black'))
    ax.text(legend_x + 2.2, legend_y + 1.5, 'Scale', fontsize=9, va='center')

    # Dimension annotations
    ax.text(7, 9.5, 'Typical dimensions:', ha='center', fontsize=10, fontweight='bold')
    ax.text(7, 9, 'd = 4096 (hidden dim)    r = 16-64 (rank)', ha='center', fontsize=9)
    ax.text(7, 8.5, 'Parameters: 2 × d × r ≈ 0.1% of original', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lora_injection_diagram.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'lora_injection_diagram.png'}")


def create_lora_comparison():
    """Create comparison of different LoRA configurations"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ranks = [4, 16, 64]
    colors = ['#ffcdd2', '#fff9c4', '#c8e6c9']
    titles = ['Low Rank (r=4)', 'Medium Rank (r=16)', 'High Rank (r=64)']

    for ax, rank, color, title in zip(axes, ranks, colors, titles):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Title
        ax.text(5, 9.5, title, ha='center', fontsize=12, fontweight='bold')

        # Draw W matrix
        w_size = 6
        w_rect = patches.Rectangle((2, 2), w_size, w_size,
                                   facecolor='#e0e0e0', edgecolor='black', linewidth=2)
        ax.add_patch(w_rect)
        ax.text(5, 5, 'W', ha='center', va='center', fontsize=14, fontweight='bold')

        # Draw LoRA overlay (proportional to rank)
        lora_height = min(w_size * (rank / 64), w_size)
        lora_rect = patches.Rectangle((2, 2), w_size, lora_height,
                                      facecolor=color, edgecolor='black',
                                      linewidth=2, alpha=0.7)
        ax.add_patch(lora_rect)

        # Annotations
        params = 2 * 4096 * rank
        ax.text(5, 0.8, f'Trainable params: {params:,}', ha='center', fontsize=10)
        ax.text(5, 0.2, f'({params/16777216*100:.2f}% of 7B model)', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lora_rank_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'lora_rank_comparison.png'}")


def create_lora_math_diagram():
    """Create a diagram showing the LoRA mathematical formula"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'LoRA Mathematical Formulation',
            ha='center', va='top', fontsize=16, fontweight='bold',
            transform=ax.transAxes)

    # Formula
    formula_text = r"$W' = W + \Delta W = W + BA$"
    ax.text(0.5, 0.75, formula_text,
            ha='center', va='center', fontsize=24,
            transform=ax.transAxes)

    # Breakdown
    breakdown = [
        (r"$W$", "Original pretrained weights (frozen)", "#e0e0e0"),
        (r"$B$", f"Low-rank matrix: d × r", "#c8e6c9"),
        (r"$A$", f"Low-rank matrix: r × d", "#c8e6c9"),
        (r"$\alpha/r$", "Scaling factor (applied to BA)", "#fff9c4"),
    ]

    y_pos = 0.55
    for symbol, desc, color in breakdown:
        ax.add_patch(patches.FancyBboxPatch(
            (0.15, y_pos - 0.05), 0.1, 0.08,
            transform=ax.transAxes,
            facecolor=color, edgecolor='black',
            boxstyle="round,pad=0.01"
        ))
        ax.text(0.2, y_pos, symbol, ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
        ax.text(0.28, y_pos, desc, ha='left', va='center',
                fontsize=12, transform=ax.transAxes)
        y_pos -= 0.12

    # Key insight box
    ax.add_patch(patches.FancyBboxPatch(
        (0.55, 0.1), 0.4, 0.35,
        transform=ax.transAxes,
        facecolor='#e3f2fd', edgecolor='#1976d2',
        boxstyle="round,pad=0.02", linewidth=2
    ))
    ax.text(0.75, 0.4, 'Key Insight', ha='center', va='top',
            fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.75, 0.32,
            'Weight updates have low\n"intrinsic rank" - most\ninformation can be captured\nwith r << d',
            ha='center', va='top', fontsize=10, transform=ax.transAxes)
    ax.text(0.75, 0.15,
            'Typical: d=4096, r=16\nCompression: 256×',
            ha='center', va='top', fontsize=10, style='italic',
            transform=ax.transAxes)

    plt.savefig(OUTPUT_DIR / 'lora_math_diagram.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'lora_math_diagram.png'}")


if __name__ == "__main__":
    create_lora_diagram()
    create_lora_comparison()
    create_lora_math_diagram()
    print("\nAll LoRA visualizations created successfully!")
