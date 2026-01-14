#!/usr/bin/env python3
"""Generate climbing stairs recursion tree visualization."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def generate_climbing_stairs_tree(n=4):
    """Generate a visualization of climbing stairs recursion tree."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Store all nodes to draw
    nodes = []
    edges = []

    def build_tree(remaining, x, y, level, x_spread, parent_pos=None, edge_label=None):
        """Recursively build the tree structure."""
        nodes.append((remaining, x, y, level))
        if parent_pos:
            edges.append((parent_pos, (x, y), edge_label))

        if remaining <= 0:
            return

        # Calculate spread for children
        new_spread = x_spread * 0.5

        # Take 1 step
        left_x = x - x_spread
        left_y = y - 1.4
        build_tree(remaining - 1, left_x, left_y, level + 1, new_spread, (x, y), "1 step")

        # Take 2 steps
        right_x = x + x_spread
        right_y = y - 1.4
        build_tree(remaining - 2, right_x, right_y, level + 1, new_spread, (x, y), "2 steps")

    # Build tree structure
    initial_spread = 2 ** (n - 1) * 0.6
    build_tree(n, 7, n + 1, 0, initial_spread)

    # Find bounds
    min_x = min(node[1] for node in nodes) - 1.5
    max_x = max(node[1] for node in nodes) + 1.5
    min_y = min(node[2] for node in nodes) - 1
    max_y = max(node[2] for node in nodes) + 1

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.axis('off')
    ax.set_title(f'Climbing Stairs Recursion Tree: climbStairs({n})', fontsize=16, fontweight='bold', pad=20)

    # Draw edges first
    for start, end, label in edges:
        ax.plot([start[0], end[0]], [start[1] - 0.25, end[1] + 0.25],
               'gray', linewidth=1.5, zorder=1)
        # Add edge label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.1, label, fontsize=7, ha='center', va='bottom',
               color='#666666', style='italic')

    # Draw nodes
    for remaining, x, y, level in nodes:
        is_base = (remaining <= 0)
        is_success = (remaining == 0)

        if is_success:
            color = '#90EE90'  # Green for success
            edge_color = '#228B22'
        elif remaining < 0:
            color = '#FFB6C1'  # Light red for invalid
            edge_color = '#DC143C'
        else:
            color = '#FFE4B5'
            edge_color = '#DAA520'

        box = FancyBboxPatch(
            (x - 0.6, y - 0.25), 1.2, 0.5,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=2,
            zorder=2
        )
        ax.add_patch(box)

        if is_success:
            text = f'climb(0)\n= 1 way!'
        elif remaining < 0:
            text = f'climb({remaining})\n= 0 (invalid)'
        else:
            text = f'climb({remaining})'
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)

    # Count successful paths
    success_count = sum(1 for r, _, _, _ in nodes if r == 0)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#FFE4B5', edgecolor='#DAA520',
                      label='Recursive Call'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#90EE90', edgecolor='#228B22',
                      label=f'Success (reached step {n})'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#FFB6C1', edgecolor='#DC143C',
                      label='Invalid (overshot)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Add complexity note
    ax.text(0.02, 0.02, f'Total ways to climb {n} stairs: {success_count}\n'
            f'Time Complexity: O(2^n) without memoization\n'
            f'With memoization: O(n) time, O(n) space',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = generate_climbing_stairs_tree(4)
    plt.savefig('/Users/abhinaavramesh/Desktop/GoogleSDE_InterviewPrep/docs/sde-coding/recursion-backtracking/assets/climbing_stairs_tree.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Generated climbing_stairs_tree.png")
