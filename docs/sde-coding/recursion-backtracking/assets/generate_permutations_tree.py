#!/usr/bin/env python3
"""Generate permutations backtracking tree visualization."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def generate_permutations_tree(elements=[1, 2, 3]):
    """Generate a visualization of permutations backtracking tree."""
    fig, ax = plt.subplots(figsize=(16, 10))

    n = len(elements)
    nodes = []
    edges = []

    def build_tree(path, remaining, x, y, x_spread, parent_pos=None, choice=None):
        """Recursively build the tree structure."""
        is_leaf = len(remaining) == 0
        nodes.append((path[:], remaining[:], x, y, is_leaf))
        if parent_pos:
            edges.append((parent_pos, (x, y), choice))

        if is_leaf:
            return

        num_children = len(remaining)
        new_spread = x_spread / max(num_children, 1)
        start_x = x - x_spread * (num_children - 1) / (2 * num_children) if num_children > 1 else x

        for i, elem in enumerate(remaining):
            child_x = start_x + i * x_spread / num_children if num_children > 1 else x
            child_y = y - 1.5
            new_remaining = remaining[:i] + remaining[i+1:]
            build_tree(path + [elem], new_remaining, child_x, child_y, new_spread, (x, y), str(elem))

    # Build tree
    build_tree([], elements, 8, n * 1.5 + 0.5, 6)

    # Find bounds
    min_x = min(node[2] for node in nodes) - 1.5
    max_x = max(node[2] for node in nodes) + 1.5
    min_y = min(node[3] for node in nodes) - 1
    max_y = max(node[3] for node in nodes) + 1

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.axis('off')
    ax.set_title(f'Permutations Backtracking Tree: {elements}', fontsize=16, fontweight='bold', pad=20)

    # Draw edges first
    for start, end, choice in edges:
        ax.plot([start[0], end[0]], [start[1] - 0.25, end[1] + 0.25],
               'gray', linewidth=1.5, zorder=1)
        # Add choice label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x + 0.1, mid_y + 0.1, f'pick {choice}', fontsize=7, ha='left', va='bottom',
               color='#1976D2', fontweight='bold')

    # Draw nodes
    for path, remaining, x, y, is_leaf in nodes:
        if is_leaf:
            color = '#90EE90'
            edge_color = '#228B22'
        else:
            color = '#E3F2FD'
            edge_color = '#1976D2'

        path_str = str(path) if path else '[]'
        width = max(1.0, len(path_str) * 0.13 + 0.4)

        box = FancyBboxPatch(
            (x - width / 2, y - 0.25), width, 0.5,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2,
            zorder=2
        )
        ax.add_patch(box)

        ax.text(x, y, path_str, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

        # Show remaining elements for non-leaf nodes
        if not is_leaf:
            remaining_str = f'remaining: {remaining}'
            ax.text(x, y - 0.4, remaining_str, fontsize=6, ha='center', va='top',
                   color='#666', style='italic')

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#E3F2FD', edgecolor='#1976D2',
                      label='Partial permutation'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#90EE90', edgecolor='#228B22',
                      label='Complete permutation'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Add complexity note
    import math
    results = [p for p, _, _, _, is_leaf in nodes if is_leaf]
    ax.text(0.02, 0.02, f'Total permutations: {len(results)} = {n}!\n'
            f'Time Complexity: O(n * n!)\n'
            f'Space Complexity: O(n)',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = generate_permutations_tree([1, 2, 3])
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/sde-coding/recursion-backtracking/assets/permutations_tree.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated permutations_tree.png")
