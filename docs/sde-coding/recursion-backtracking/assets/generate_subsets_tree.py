#!/usr/bin/env python3
"""Generate subsets backtracking tree visualization."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def generate_subsets_tree(elements=[1, 2, 3]):
    """Generate a visualization of subsets backtracking tree."""
    fig, ax = plt.subplots(figsize=(16, 10))

    n = len(elements)
    nodes = []
    edges = []

    def build_tree(subset, level, x, x_spread, parent_pos=None, decision=None):
        """Recursively build the tree structure."""
        nodes.append((subset[:], level, x, parent_pos, level == n))
        if parent_pos:
            edges.append((parent_pos, (x, n - level + 0.5), decision))

        if level == n:
            return

        y = n - level + 0.5
        new_spread = x_spread / 2

        # Don't include element
        build_tree(subset, level + 1, x - x_spread / 2, new_spread, (x, y), "skip")

        # Include element
        build_tree(subset + [elements[level]], level + 1, x + x_spread / 2, new_spread, (x, y), "include")

    # Build tree
    initial_spread = 6
    build_tree([], 0, 8, initial_spread)

    ax.set_xlim(0, 16)
    ax.set_ylim(-0.5, n + 2)
    ax.axis('off')
    ax.set_title(f'Subsets Backtracking Tree: {elements}', fontsize=16, fontweight='bold', pad=20)

    # Draw edges first
    for start, end, decision in edges:
        color = '#4CAF50' if decision == 'include' else '#FF5722'
        ax.plot([start[0], end[0]], [start[1] - 0.25, end[1] + 0.25],
               color=color, linewidth=1.5, zorder=1, alpha=0.7)
        # Add decision label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, decision, fontsize=7, ha='center', va='center',
               color=color, fontweight='bold')

    # Draw nodes
    for subset, level, x, parent_pos, is_leaf in nodes:
        y = n - level + 0.5

        is_result = is_leaf
        if is_result:
            color = '#90EE90'
            edge_color = '#228B22'
        else:
            color = '#E3F2FD'
            edge_color = '#1976D2'

        subset_str = str(subset) if subset else '[]'
        width = max(0.8, len(subset_str) * 0.12 + 0.3)

        box = FancyBboxPatch(
            (x - width / 2, y - 0.22), width, 0.44,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2,
            zorder=2
        )
        ax.add_patch(box)

        ax.text(x, y, subset_str, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

        # Add level indicator
        if not is_leaf and level < n:
            ax.text(x, y + 0.35, f'Consider {elements[level]}?', fontsize=7,
                   ha='center', va='bottom', color='#666', style='italic')

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#E3F2FD', edgecolor='#1976D2',
                      label='Decision Node'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#90EE90', edgecolor='#228B22',
                      label='Result Subset'),
        plt.Line2D([0], [0], color='#4CAF50', linewidth=2, label='Include element'),
        plt.Line2D([0], [0], color='#FF5722', linewidth=2, label='Skip element'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Add complexity note
    results = [s for s, l, _, _, is_leaf in nodes if is_leaf]
    ax.text(0.02, 0.02, f'Total subsets: {len(results)} = 2^{n}\n'
            f'Time Complexity: O(n * 2^n)\n'
            f'Space Complexity: O(n)',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = generate_subsets_tree([1, 2, 3])
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/sde-coding/recursion-backtracking/assets/subsets_tree.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated subsets_tree.png")
