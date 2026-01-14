#!/usr/bin/env python3
"""Generate factorial recursion tree visualization."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def generate_factorial_tree(n=5):
    """Generate a visualization of factorial recursion tree."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n + 2)
    ax.axis('off')
    ax.set_title(f'Factorial Recursion Tree: factorial({n})', fontsize=16, fontweight='bold', pad=20)

    # Calculate positions for each recursive call
    x_center = 5
    y_positions = []

    for i in range(n, -1, -1):
        level = n - i
        y = n + 0.5 - level * 0.9
        y_positions.append((i, y))

    # Draw nodes and arrows
    prev_pos = None
    for i, (val, y) in enumerate(y_positions):
        x = x_center

        # Color based on base case
        is_base = (val == 0)
        color = '#90EE90' if is_base else '#FFE4B5'
        edge_color = '#228B22' if is_base else '#DAA520'

        # Draw node
        box = FancyBboxPatch(
            (x - 1.2, y - 0.25), 2.4, 0.5,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2
        )
        ax.add_patch(box)

        # Text content
        if is_base:
            text = f'factorial(0) = 1'
        else:
            text = f'factorial({val})'
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')

        # Draw arrow from previous
        if prev_pos:
            ax.annotate(
                '', xy=(x, y + 0.25), xytext=(prev_pos[0], prev_pos[1] - 0.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2)
            )
            # Add call annotation
            ax.text(x + 1.5, (y + prev_pos[1]) / 2, f'calls', fontsize=9,
                   ha='left', va='center', color='gray', style='italic')

        prev_pos = (x, y)

    # Add return arrows on the right side
    for i, (val, y) in enumerate(y_positions[:-1]):
        if i < len(y_positions) - 1:
            next_y = y_positions[i + 1][1]
            ax.annotate(
                '', xy=(x_center + 1.8, y - 0.25),
                xytext=(x_center + 1.8, next_y + 0.25),
                arrowprops=dict(arrowstyle='->', color='#4169E1', lw=1.5,
                               connectionstyle="arc3,rad=-0.3")
            )

    # Add return values on the right
    ax.text(8.5, n + 0.5 - 5 * 0.9, 'Return values:', fontsize=10, fontweight='bold')
    return_vals = [1, 1, 2, 6, 24, 120]  # factorial values 0! to 5!
    for i, (val, y) in enumerate(y_positions):
        if i < len(return_vals):
            idx = len(y_positions) - 1 - i
            ax.text(8.5, y, f'{idx}! = {return_vals[idx]}', fontsize=9,
                   ha='left', va='center', color='#4169E1')

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#FFE4B5', edgecolor='#DAA520',
                      label='Recursive Call'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#90EE90', edgecolor='#228B22',
                      label='Base Case (returns 1)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Add complexity note
    ax.text(0.5, 0.3, f'Time Complexity: O(n)\nSpace Complexity: O(n) - call stack depth',
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = generate_factorial_tree(5)
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/sde-coding/recursion-backtracking/assets/factorial_tree.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated factorial_tree.png")
